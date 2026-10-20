/*
 * hvo_pg: HyperVector-Optimizer as a PostgreSQL planner hook for pgvector.
 *
 * For a query of the form
 *     SELECT ... FROM <table> WHERE <conjunction of range / = / IN predicates>
 *            [AND fk IN (SELECT id FROM <dims> WHERE <IN / = predicates on dimension columns>)]
 *     ORDER BY embedding <#> '<vector>' LIMIT k
 * the hook encodes the predicate, runs the native HVO planner (HPBS estimate, residual calibrator,
 * minimax cost routing, recall check; ../hvo/kernels/est.c) and steers the standard planner to the
 * chosen plan by setting plan-type GUCs for the duration of planning (as pg_hint_plan does):
 *   pre-filter  -> enable_indexscan = off        (filter first, then exact top-k sort)
 *   post-filter -> enable_seqscan = enable_bitmapscan = off   (pgvector iterative HNSW scan)
 * pgvector has no in-traversal (fused) plan, so routing is two-way here.
 *
 * GUCs: hvo.enabled (bool), hvo.model (path to the binary model written by experiments/pg_export.py),
 *       hvo.table (relation name the model belongs to), hvo.dims (dimension table name).
 */
#include "postgres.h"
#include "fmgr.h"
#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "nodes/nodeFuncs.h"
#include "nodes/parsenodes.h"
#include "optimizer/planner.h"
#include "optimizer/optimizer.h"
#include "optimizer/clauses.h"
#include "parser/parsetree.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "extension/vector/vector.h"

#include <stdio.h>
#include <math.h>
#include <time.h>

/* the native planner and sketch query path, shared with the Python benchmark */
#include "../hvo/kernels/est.c"

PG_MODULE_MAGIC;

void _PG_init(void);
PG_FUNCTION_INFO_V1(hvo_last);

static planner_hook_type prev_planner = NULL;
static bool hvo_enabled = false;
static char *hvo_model_path = NULL;
static char *hvo_table = NULL;
static char *hvo_dims = NULL;
static double hvo_recall_tau = -1;	/* < 0: use the model's threshold; 0: ignore recall */

/* ---------------------------------------------------------------- model file */
typedef struct { char name[96]; int dtype; int ndim; int64 dims[3]; void *data; } Entry;
static Entry *entries = NULL;
static int n_entries = 0;
static char *loaded_path = NULL;

static Entry *find(const char *name)
{
	for (int i = 0; i < n_entries; i++)
		if (strcmp(entries[i].name, name) == 0)
			return &entries[i];
	return NULL;
}

static const int dtype_size[] = {4, 8, 4, 1, 2};

static bool load_model(const char *path)
{
	FILE	   *f = fopen(path, "rb");
	char		magic[4];
	uint32		cnt;

	if (!f)
		return false;
	if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "HVO1", 4) != 0 || fread(&cnt, 4, 1, f) != 1)
	{
		fclose(f);
		return false;
	}
	entries = calloc(cnt, sizeof(Entry));
	n_entries = cnt;
	for (uint32 i = 0; i < cnt; i++)
	{
		uint32		len, nd;
		uint8		dt;
		int64		total = 1;

		if (fread(&len, 4, 1, f) != 1 || len >= 96 || fread(entries[i].name, 1, len, f) != len ||
			fread(&dt, 1, 1, f) != 1 || fread(&nd, 4, 1, f) != 1 || nd > 3)
		{
			fclose(f);
			return false;
		}
		entries[i].dtype = dt;
		entries[i].ndim = nd;
		for (uint32 j = 0; j < nd; j++)
		{
			if (fread(&entries[i].dims[j], 8, 1, f) != 1)
			{
				fclose(f);
				return false;
			}
			total *= entries[i].dims[j];
		}
		entries[i].data = malloc(total * dtype_size[dt] + 16);
		if (fread(entries[i].data, dtype_size[dt], total, f) != (size_t) total)
		{
			fclose(f);
			return false;
		}
	}
	fclose(f);
	return true;
}

/* model state built from the entries */
static int	L, M, D, B1, n_num, n_cat, n_dimcols, n_dimrows;
static char num_names[16][64], cat_names[16][64], dim_names[8][64];
static Part parts[8];
static const float *num_cum[8][16], *cat_frac[8][16];
static int	catK[16];
static const float *edges[16];
static const uint8 *integral;
static const int16 *dim_vals[8];
static CalW cw;
static CostP cp;
static double model_N, model_tau;
static float *work_f;
static int *work_i;
static uint8 *validkey;

static void split_names(Entry *e, char out[][64], int *n)
{
	char	   *s = (char *) e->data;
	int			len = (int) e->dims[0], j = 0, c = 0;

	*n = 0;
	for (int i = 0; i <= len; i++)
	{
		if (i == len || s[i] == '\n')
		{
			out[*n][c] = 0;
			(*n)++;
			c = 0;
		}
		else if (c < 63)
			out[*n][c++] = s[i];
	}
	(void) j;
}

static bool build_state(void)
{
	char		key[96];
	const double *cost;

	L = *(int32 *) find("L")->data;
	M = *(int32 *) find("m")->data;
	D = *(int32 *) find("d")->data;
	B1 = *(int32 *) find("B1")->data;
	model_N = *(double *) find("N")->data;
	split_names(find("num_names"), num_names, &n_num);
	split_names(find("cat_names"), cat_names, &n_cat);
	split_names(find("dim_names"), dim_names, &n_dimcols);
	integral = (const uint8 *) find("integral")->data;
	for (int c = 0; c < n_num; c++)
	{
		snprintf(key, sizeof(key), "edges.%d", c);
		edges[c] = find(key)->data;
	}
	for (int c = 0; c < n_dimcols; c++)
	{
		snprintf(key, sizeof(key), "dim.%d", c);
		dim_vals[c] = find(key)->data;
		n_dimrows = (int) find(key)->dims[0];
	}
	for (int l = 0; l < L; l++)
	{
		Entry	   *e;

		parts[l].m = M;
		parts[l].d = D;
		parts[l].B1 = B1;
		snprintf(key, sizeof(key), "p%d.cent", l); parts[l].cent = find(key)->data;
		snprintf(key, sizeof(key), "p%d.c2", l); parts[l].c2 = find(key)->data;
		snprintf(key, sizeof(key), "p%d.Nb", l); parts[l].Nb = find(key)->data;
		snprintf(key, sizeof(key), "p%d.rq", l); parts[l].rq = find(key)->data;
		for (int c = 0; c < n_num; c++)
		{
			snprintf(key, sizeof(key), "p%d.num%d", l, c);
			num_cum[l][c] = find(key)->data;
		}
		for (int c = 0; c < n_cat; c++)
		{
			snprintf(key, sizeof(key), "p%d.cat%d", l, c);
			e = find(key);
			cat_frac[l][c] = e->data;
			catK[c] = (int) e->dims[1];
		}
		parts[l].nnum = n_num;
		parts[l].numcum = num_cum[l];
		parts[l].ncat = n_cat;
		parts[l].catf = cat_frac[l];
		parts[l].catK = catK;
		snprintf(key, sizeof(key), "p%d.keys", l); e = find(key);
		parts[l].keys = e->data;
		parts[l].kmv = (int) e->dims[1];
		snprintf(key, sizeof(key), "p%d.kcnt", l); parts[l].kcnt = find(key)->data;
	}
	cw.nin = 12;
	cw.nh = (int) find("cal.b0")->dims[0];
	cw.mu = find("cal.mu")->data;
	cw.sd = find("cal.sd")->data;
	cw.W0 = find("cal.W0")->data;
	cw.b0 = find("cal.b0")->data;
	cw.W1 = find("cal.W1")->data;
	cw.b1 = find("cal.b1")->data;
	cw.W2 = find("cal.W2")->data;
	cw.b2 = *(float *) find("cal.b2")->data;
	cost = find("cost")->data;
	memset(&cp, 0, sizeof(cp));
	cp.rel0 = cost[0]; cp.rel1 = cost[1]; cp.rel2 = cost[2]; cp.a0 = cost[3]; cp.cscan = cost[4];
	cp.post[0] = cost[5]; cp.post[1] = cost[6]; cp.post[2] = cost[7];
	cp.fused[0] = 0; cp.fused[1] = 700; cp.fused[2] = 1;	/* no fused plan in pgvector: cost ~ e^700 */
	cp.cf = 1; cp.beta = 1; cp.ef0 = 64; cp.cap = 131072;
	cp.eta = cost[8];
	cp.tau_rec = model_tau = cost[9];
	cp.rec_w0[0] = NAN; cp.rec_c[0] = 0;
	if (isnan(cost[10]))
	{
		cp.rec_w0[1] = NAN; cp.rec_c[1] = cost[11];
	}
	else
	{
		cp.rec_w0[1] = cost[10];
		for (int i = 0; i < 4; i++)
			cp.rec_w[1][i] = cost[11 + i];
	}
	work_f = malloc(sizeof(float) * M * 18);
	work_i = malloc(sizeof(int) * M * 18);
	validkey = malloc(n_dimrows + 16);
	return true;
}

static bool ensure_model(void)
{
	if (hvo_model_path == NULL || hvo_model_path[0] == 0)
		return false;
	if (loaded_path && strcmp(loaded_path, hvo_model_path) == 0)
		return true;
	if (!load_model(hvo_model_path))
	{
		elog(WARNING, "hvo: cannot load model %s", hvo_model_path);
		return false;
	}
	loaded_path = strdup(hvo_model_path);
	return build_state();
}

/* ------------------------------------------------------------- query parsing */
#define MAXT 16
typedef struct
{
	double		lo[16], hi[16];
	bool		has_range[16];
	uint64		in_mask[16];
	bool		has_in[16];
	uint64		dim_mask[8];
	bool		has_dim[8];
	bool		ok;
} Pred;

static int name_index(char names[][64], int n, const char *s)
{
	for (int i = 0; i < n; i++)
		if (strcmp(names[i], s) == 0)
			return i;
	return -1;
}

/* strip relabelings and single-argument cast functions (e.g. real -> double precision) */
static Node *strip_casts(Node *n)
{
	for (;;)
	{
		if (n && IsA(n, RelabelType))
			n = (Node *) ((RelabelType *) n)->arg;
		else if (n && IsA(n, FuncExpr) && ((FuncExpr *) n)->funcformat != COERCE_EXPLICIT_CALL &&
				 list_length(((FuncExpr *) n)->args) == 1)
			n = linitial(((FuncExpr *) n)->args);
		else
			return n;
	}
}

static bool const_double(Node *n, double *out)
{
	Const	   *c;

	/* fold casts of literals, e.g. float8(2001.0::numeric), into a Const */
	if (n && !IsA(n, Const))
		n = eval_const_expressions(NULL, n);
	if (IsA(n, RelabelType))
		n = (Node *) ((RelabelType *) n)->arg;
	if (!IsA(n, Const) || ((Const *) n)->constisnull)
		return false;
	c = (Const *) n;
	switch (c->consttype)
	{
		case INT2OID: *out = DatumGetInt16(c->constvalue); return true;
		case INT4OID: *out = DatumGetInt32(c->constvalue); return true;
		case INT8OID: *out = (double) DatumGetInt64(c->constvalue); return true;
		case FLOAT4OID: *out = DatumGetFloat4(c->constvalue); return true;
		case FLOAT8OID: *out = DatumGetFloat8(c->constvalue); return true;
		case NUMERICOID:
			*out = DatumGetFloat8(DirectFunctionCall1(numeric_float8, c->constvalue));
			return true;
		default: return false;
	}
}

static const char *var_name(Query *q, Var *v)
{
	RangeTblEntry *rte = rt_fetch(v->varno, q->rtable);

	if (rte->rtekind != RTE_RELATION)
		return NULL;
	return get_attname(rte->relid, v->varattno, true);
}

static bool add_qual(Query *q, Node *n, Pred *p, bool on_dims);

static bool add_list(Query *q, Node *quals, Pred *p, bool on_dims)
{
	if (quals == NULL)
		return true;
	if (IsA(quals, BoolExpr) && ((BoolExpr *) quals)->boolop == AND_EXPR)
	{
		ListCell   *lc;

		foreach(lc, ((BoolExpr *) quals)->args)
			if (!add_qual(q, lfirst(lc), p, on_dims))
				return false;
		return true;
	}
	if (IsA(quals, List))
	{
		ListCell   *lc;

		foreach(lc, (List *) quals)
			if (!add_qual(q, lfirst(lc), p, on_dims))
				return false;
		return true;
	}
	return add_qual(q, quals, p, on_dims);
}

static bool add_qual(Query *q, Node *n, Pred *p, bool on_dims)
{
	if (IsA(n, BoolExpr))
		return add_list(q, n, p, on_dims);
	if (IsA(n, OpExpr) && list_length(((OpExpr *) n)->args) == 2)
	{
		OpExpr	   *op = (OpExpr *) n;
		Node	   *a = linitial(op->args), *b = lsecond(op->args);
		char	   *opname = get_opname(op->opno);
		const char *col;
		double		v;
		bool		flip = false;
		int			ci;

		if (IsA(strip_casts(a), Var))
			a = strip_casts(a);
		else if (IsA(strip_casts(b), Var))
		{
			Node	   *t = a; a = strip_casts(b); b = t; flip = true;
		}
		if (!IsA(a, Var) || !const_double(b, &v) || opname == NULL || (col = var_name(q, (Var *) a)) == NULL)
			return false;
		if (flip)
		{
			if (strcmp(opname, ">=") == 0) opname = "<=";
			else if (strcmp(opname, "<=") == 0) opname = ">=";
			else if (strcmp(opname, ">") == 0) opname = "<";
			else if (strcmp(opname, "<") == 0) opname = ">";
		}
		if (on_dims)
		{
			int			di = name_index(dim_names, n_dimcols, col);

			if (di < 0 || strcmp(opname, "=") != 0)
				return false;
			p->dim_mask[di] = (p->has_dim[di] ? p->dim_mask[di] : ~0ULL) & (1ULL << (int) v);
			p->has_dim[di] = true;
			return true;
		}
		if ((ci = name_index(cat_names, n_cat, col)) >= 0 && strcmp(opname, "=") == 0)
		{
			p->in_mask[ci] = (p->has_in[ci] ? p->in_mask[ci] : ~0ULL) & (1ULL << (int) v);
			p->has_in[ci] = true;
			return true;
		}
		if ((ci = name_index(num_names, n_num, col)) >= 0)
		{
			if (!p->has_range[ci])
			{
				p->lo[ci] = -INFINITY; p->hi[ci] = INFINITY; p->has_range[ci] = true;
			}
			if (integral[ci] == 1 && strcmp(opname, ">") == 0) { v = floor(v) + 1; opname = ">="; }
			if (integral[ci] == 1 && strcmp(opname, "<") == 0) { v = ceil(v) - 1; opname = "<="; }
			if (opname[0] == '>' && v > p->lo[ci]) p->lo[ci] = v;
			else if (opname[0] == '<' && v < p->hi[ci]) p->hi[ci] = v;
			else if (strcmp(opname, "=") == 0) { p->lo[ci] = v; p->hi[ci] = v; }
			return true;
		}
		return false;
	}
	if (IsA(n, ScalarArrayOpExpr))
	{
		ScalarArrayOpExpr *sa = (ScalarArrayOpExpr *) n;
		Node	   *a = linitial(sa->args), *b = lsecond(sa->args);
		const char *col;
		ArrayType  *arr;
		Datum	   *vals;
		bool	   *nulls;
		int			nv, ci;
		int16		typlen;
		bool		typbyval;
		char		typalign;
		uint64		mask = 0;

		a = strip_casts(a);
		if (!IsA(b, Const))
			b = eval_const_expressions(NULL, b);
		if (!sa->useOr || !IsA(a, Var) || !IsA(b, Const) || (col = var_name(q, (Var *) a)) == NULL)
			return false;
		arr = DatumGetArrayTypeP(((Const *) b)->constvalue);
		get_typlenbyvalalign(ARR_ELEMTYPE(arr), &typlen, &typbyval, &typalign);
		deconstruct_array(arr, ARR_ELEMTYPE(arr), typlen, typbyval, typalign, &vals, &nulls, &nv);
		for (int i = 0; i < nv; i++)
		{
			int			x = ARR_ELEMTYPE(arr) == INT2OID ? DatumGetInt16(vals[i]) : DatumGetInt32(vals[i]);

			if (x >= 0 && x < 64)
				mask |= 1ULL << x;
		}
		if (on_dims)
		{
			int			di = name_index(dim_names, n_dimcols, col);

			if (di < 0)
				return false;
			p->dim_mask[di] = (p->has_dim[di] ? p->dim_mask[di] : ~0ULL) & mask;
			p->has_dim[di] = true;
			return true;
		}
		if ((ci = name_index(cat_names, n_cat, col)) < 0)
			return false;
		p->in_mask[ci] = (p->has_in[ci] ? p->in_mask[ci] : ~0ULL) & mask;
		p->has_in[ci] = true;
		return true;
	}
	if (IsA(n, SubLink) && ((SubLink *) n)->subLinkType == ANY_SUBLINK && !on_dims)
	{
		/* fk IN (SELECT id FROM dims WHERE ...) */
		Query	   *sub = (Query *) ((SubLink *) n)->subselect;

		return add_list(sub, sub->jointree->quals, p, true);
	}
	return false;
}

/* ------------------------------------------------------------------ planning */
static int	last_choice = -1;
static double last_us = 0, last_sloc = 0, last_sg = 0;

static int searchsorted_right(const float *e, int n, double x)
{
	int			lo = 0, hi = n;

	while (lo < hi)
	{
		int			mid = (lo + hi) >> 1;

		if (e[mid] <= x) lo = mid + 1;
		else hi = mid;
	}
	return lo;
}

static int decide(Query *parse, Vector *vec, int k)
{
	Pred		p;
	int			rc[16], jl[16], jh[16], cc[16];
	float		tl[16], th[16];
	uint64_t	al[16];
	int			nr = 0, nc = 0, ndim = 0;
	HQuery		Qs;
	double		est[3];
	struct timespec t0, t1;

	memset(&p, 0, sizeof(p));
	clock_gettime(CLOCK_MONOTONIC, &t0);
	if (!add_list(parse, parse->jointree->quals, &p, false))
		return -1;
	for (int c = 0; c < n_num; c++)
	{
		double		lo, hi;

		if (!p.has_range[c])
			continue;
		lo = p.lo[c]; hi = p.hi[c];
		if (integral[c] == 1)			/* integer column: half-integer bin edges */
		{
			lo = isinf(lo) ? lo : ceil(lo) - 0.5;
			hi = isinf(hi) ? hi : floor(hi) + 0.5;
		}
		else if (integral[c] == 2 && !isinf(hi))	/* point mass at hi: include its exact bin */
			hi = nextafterf((float) hi, INFINITY);
		for (int s = 0; s < 2; s++)
		{
			double		x = s == 0 ? lo : hi;
			int			j = searchsorted_right(edges[c], B1, x) - 1;
			double		w, t;

			if (j < 0) j = 0;
			if (j > B1 - 2) j = B1 - 2;
			w = edges[c][j + 1] - edges[c][j];
			t = w <= 0 ? 0 : (x - edges[c][j]) / w;
			t = t < 0 ? 0 : (t > 1 ? 1 : t);
			if (s == 0) { jl[nr] = j; tl[nr] = t; }
			else { jh[nr] = j; th[nr] = t; }
		}
		rc[nr++] = c;
	}
	for (int c = 0; c < n_cat; c++)
		if (p.has_in[c])
		{
			cc[nc] = c;
			al[nc++] = p.in_mask[c];
		}
	for (int i = 0; i < n_dimrows; i++)
		validkey[i] = 1;
	for (int c = 0; c < n_dimcols; c++)
		if (p.has_dim[c])
		{
			ndim++;
			for (int i = 0; i < n_dimrows; i++)
				validkey[i] &= (p.dim_mask[c] >> dim_vals[c][i]) & 1;
		}
	Qs.nr = nr; Qs.rcol = rc; Qs.jlo = jl; Qs.jhi = jh; Qs.tlo = tl; Qs.thi = th;
	Qs.nc = nc; Qs.ccol = cc; Qs.allowed = al;
	Qs.validkey = ndim ? validkey : NULL;
	last_choice = hvo_plan(parts, L, &Qs, vec->x, (double) k, model_N, (double) (nr + nc + ndim),
						   ndim ? 1.0 : 0.0, (double) nr, (double) nc, &cw, &cp, est, work_f, work_i);
	clock_gettime(CLOCK_MONOTONIC, &t1);
	last_us = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
	last_sg = est[0];
	last_sloc = est[2];
	return last_choice;
}

static bool match_query(Query *parse, Vector **vec, int *k)
{
	TargetEntry *te;
	OpExpr	   *op;
	Node	   *a, *b;
	RangeTblEntry *rte;
	SortGroupClause *sc;
	double		kv;

	if (parse->commandType != CMD_SELECT || list_length(parse->sortClause) != 1 || parse->limitCount == NULL ||
		list_length(parse->rtable) < 1)
		return false;
	rte = rt_fetch(1, parse->rtable);
	if (rte->rtekind != RTE_RELATION || hvo_table == NULL || strcmp(get_rel_name(rte->relid), hvo_table) != 0)
		return false;
	if (!const_double(parse->limitCount, &kv))
		return false;
	sc = linitial(parse->sortClause);
	te = get_sortgroupclause_tle(sc, parse->targetList);
	if (!IsA(te->expr, OpExpr))
		return false;
	op = (OpExpr *) te->expr;
	if (strcmp(get_opname(op->opno), "<#>") != 0 || list_length(op->args) != 2)
		return false;
	a = linitial(op->args);
	b = lsecond(op->args);
	a = strip_casts(a);
	b = strip_casts(b);
	if (IsA(a, Const)) { Node *t = a; a = b; b = t; }
	if (!IsA(b, Const))
		b = eval_const_expressions(NULL, b);
	if (!IsA(a, Var) || !IsA(b, Const) || ((Const *) b)->constisnull)
		return false;
	*vec = (Vector *) PG_DETOAST_DATUM(((Const *) b)->constvalue);
	if ((*vec)->dim != D)
		return false;
	*k = (int) kv;
	return true;
}

static PlannedStmt *
hvo_planner(Query *parse, const char *query_string, int cursorOptions, ParamListInfo boundParams)
{
	PlannedStmt *stmt;
	Vector	   *vec;
	int			k, choice = -1, nest = -1;

	if (hvo_enabled && ensure_model() && match_query(parse, &vec, &k))
	{
		last_choice = -1;
		cp.tau_rec = hvo_recall_tau >= 0 ? hvo_recall_tau : model_tau;
		choice = decide(parse, vec, k);
		if (choice >= 0)
		{
			nest = NewGUCNestLevel();
			if (choice == 0)
				set_config_option("enable_indexscan", "off", PGC_USERSET, PGC_S_SESSION, GUC_ACTION_SAVE, true, 0, false);
			else
			{
				set_config_option("enable_seqscan", "off", PGC_USERSET, PGC_S_SESSION, GUC_ACTION_SAVE, true, 0, false);
				set_config_option("enable_bitmapscan", "off", PGC_USERSET, PGC_S_SESSION, GUC_ACTION_SAVE, true, 0, false);
			}
		}
	}
	if (prev_planner)
		stmt = prev_planner(parse, query_string, cursorOptions, boundParams);
	else
		stmt = standard_planner(parse, query_string, cursorOptions, boundParams);
	if (nest >= 0)
		AtEOXact_GUC(true, nest);
	return stmt;
}

/* SELECT * FROM hvo_last() -> (choice, planning_us, s_g, s_loc) of the last planned query */
Datum
hvo_last(PG_FUNCTION_ARGS)
{
	char		buf[256];

	snprintf(buf, sizeof(buf), "%d %.3f %.6g %.6g", last_choice, last_us, last_sg, last_sloc);
	PG_RETURN_TEXT_P(cstring_to_text(buf));
}

void
_PG_init(void)
{
	DefineCustomBoolVariable("hvo.enabled", "Route filtered vector queries with HVO.", NULL, &hvo_enabled,
							 false, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomStringVariable("hvo.model", "Path of the HVO model file.", NULL, &hvo_model_path, "",
							   PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomStringVariable("hvo.table", "Table the model describes.", NULL, &hvo_table, "items",
							   PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomStringVariable("hvo.dims", "Dimension table.", NULL, &hvo_dims, "dims",
							   PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("hvo.recall_tau", "Minimum predicted P(recall >= 0.9) for the post-filter plan "
							 "(-1: model default, 0: ignore recall).", NULL, &hvo_recall_tau, -1, -1, 1,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	prev_planner = planner_hook;
	planner_hook = hvo_planner;
}
