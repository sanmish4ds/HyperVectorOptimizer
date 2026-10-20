// HQuery-time HPBS estimation (one partition per call). Mirrors HPBS._ball_weights /
// HPBS._cell_fraction in hvo/sketch.py; the Python version is the reference implementation.
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <arm_neon.h>
// cblas_sgemv from Accelerate (declared here: the SDK's umbrella header pulls in unrelated frameworks)
enum { CblasRowMajor = 101 }; enum { CblasNoTrans = 111 };
void cblas_sgemv(int order, int trans, int m, int n, float alpha, const float *A, int lda,
                 const float *x, int incx, float beta, float *y, int incy);

#define NRAD 16

typedef struct {
    int m, d, B1;                 // cells, dimension, numeric bins + 1
    const float *cent, *c2, *Nb, *rq;
    int nnum; const float *const *numcum;   // each m x B1 cumulative fractions
    int ncat; const float *const *catf; const int *catK;  // each m x K fractions
    int kmv; const int32_t *keys; const float *kcnt;
} Part;

typedef struct {
    int nr; const int *rcol, *jlo, *jhi; const float *tlo, *thi;
    int nc; const int *ccol; const uint64_t *allowed;
    const uint8_t *validkey;      // NULL when no join predicate
} HQuery;

static inline float dotf(const float *a, const float *b, int d) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < d; i++) s += a[i] * b[i];
    return s;
}

static float cell_frac(const Part *P, const HQuery *Q, int b) {
    float f = 1.0f;
    for (int t = 0; t < Q->nr; t++) {
        const float *cum = P->numcum[Q->rcol[t]] + (int64_t)b * P->B1;
        float hi = cum[Q->jhi[t]] * (1 - Q->thi[t]) + cum[Q->jhi[t] + 1] * Q->thi[t];
        float lo = cum[Q->jlo[t]] * (1 - Q->tlo[t]) + cum[Q->jlo[t] + 1] * Q->tlo[t];
        float v = hi - lo; f *= v < 0 ? 0 : (v > 1 ? 1 : v);
    }
    for (int t = 0; t < Q->nc; t++) {
        int c = Q->ccol[t], K = P->catK[c];
        const float *row = P->catf[c] + (int64_t)b * K;
        uint64_t a = Q->allowed[t]; float s = 0;
        while (a) { int v = __builtin_ctzll(a); if (v < K) s += row[v]; a &= a - 1; }
        f *= s;
    }
    if (Q->validkey) {
        const int32_t *ks = P->keys + (int64_t)b * P->kmv; const float *cs = P->kcnt + (int64_t)b * P->kmv;
        float num = 0, den = 0;
        for (int j = 0; j < P->kmv && ks[j] >= 0; j++) { den += cs[j]; if (Q->validkey[ks[j]]) num += cs[j]; }
        f *= den > 0 ? num / den : 0;
    }
    return f;
}

double hpbs_global(const Part *P, const HQuery *Q, double N) {
    double s = 0;
    for (int b = 0; b < P->m; b++) if (P->Nb[b] > 0) s += P->Nb[b] * cell_frac(P, Q, b);
    return s / N;
}

// number of a cell's radial quantiles (ascending) that are <= t
static inline int nle(const float *rq, float t) {
    int lo = 0, hi = NRAD;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (rq[mid] <= t) lo = mid + 1; else hi = mid; }
    return lo;
}

static double ball_count(const Part *P, const float *dq, const int *cand, int nc, float r2) {
    double c = 0;
    for (int i = 0; i < nc; i++) {
        int b = cand[i];
        if (dq[b] <= r2) c += nle(P->rq + (int64_t)b * NRAD, r2 - dq[b]) * (double)P->Nb[b];
    }
    return c / NRAD;
}

// returns local selectivity estimate for one partition; writes squared radius and active-cell count
double hpbs_local(const Part *P, const HQuery *Q, const float *q, double tau, float *out_r2, int *out_nact,
                  float *work_f, int *work_i) {
    int m = P->m;
    float *dq = work_f;
    int *cand = work_i;
    // dq_b = ||c_b||^2 - 2 <c_b, q> + 1   (q unit norm); BLAS sgemv on the centroid matrix
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, P->d, -2.0f, P->cent, P->d, q, 1, 0.0f, dq, 1);
    float lo = 0, hi = 0;
    for (int b = 0; b < m; b++) {
        float v = dq[b] + P->c2[b] + 1.0f;
        dq[b] = v > 0 ? v : 0;
        float top = dq[b] + P->rq[(int64_t)b * NRAD + NRAD - 1];
        if (top > hi) hi = top;
        cand[b] = b;
    }
    int nc = m;
    for (int it = 0; it < 24; it++) {          // smallest r2 whose estimated ball holds tau tuples
        float mid = 0.5f * (lo + hi);
        if (ball_count(P, dq, cand, nc, mid) < tau) lo = mid;
        else {
            hi = mid;                          // cells beyond the new upper radius can never contribute
            int w = 0;
            for (int i = 0; i < nc; i++) if (dq[cand[i]] <= hi) cand[w++] = cand[i];
            nc = w;
        }
    }
    float r2 = hi;
    double num = 0, den = 0; int nact = 0;
    for (int i = 0; i < nc; i++) {
        int b = cand[i];
        if (dq[b] > r2) continue;
        int c = nle(P->rq + (int64_t)b * NRAD, r2 - dq[b]);
        if (!c) continue;
        double w = c * (double)P->Nb[b] / NRAD;
        num += w * cell_frac(P, Q, b); den += w; nact++;
    }
    if (den <= 0) {                             // ball inside one cell: use the nearest cell
        int bb = 0; for (int b = 1; b < m; b++) if (dq[b] < dq[bb]) bb = b;
        num = cell_frac(P, Q, bb); den = 1; nact = 1;
    }
    *out_r2 = r2; *out_nact = nact;
    return num / den;
}


// k-th-match formulation: find the smallest radius whose estimated ball holds k matching tuples,
// return s_loc = k / (estimated ball size). Also returns this partition's global selectivity.
double hpbs_local_k(const Part *P, const HQuery *Q, const float *q, double k, double N,
                    double *out_sg, float *out_r2, int *out_nact, float *work_f, int *work_i) {
    int m = P->m;
    float *dq = work_f, *f = work_f + m;
    int *cand = work_i;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, P->d, -2.0f, P->cent, P->d, q, 1, 0.0f, dq, 1);
    float lo = 0, hi = 0; double sg = 0;
    int nc = 0;
    for (int b = 0; b < m; b++) {
        float v = dq[b] + P->c2[b] + 1.0f;
        dq[b] = v > 0 ? v : 0;
        f[b] = P->Nb[b] > 0 ? cell_frac(P, Q, b) : 0;
        sg += P->Nb[b] * f[b];
        if (f[b] > 0) {
            cand[nc++] = b;
            float top = dq[b] + P->rq[(int64_t)b * NRAD + NRAD - 1];
            if (top > hi) hi = top;
        }
    }
    *out_sg = sg / N;
    if (sg < k || nc == 0) { *out_r2 = hi; *out_nact = nc; return sg / N > 0 ? sg / N : 1.0 / N; }
    for (int it = 0; it < 26; it++) {
        float mid = 0.5f * (lo + hi);
        double c = 0;
        for (int i = 0; i < nc; i++) {
            int b = cand[i];
            if (dq[b] <= mid) c += nle(P->rq + (int64_t)b * NRAD, mid - dq[b]) * (double)P->Nb[b] * f[b];
        }
        if (c / NRAD < k) lo = mid;
        else {
            hi = mid;
            int w = 0;
            for (int i = 0; i < nc; i++) if (dq[cand[i]] <= hi) cand[w++] = cand[i];
            nc = w;
        }
    }
    // ball size at r2 over *all* cells (matching or not)
    double tot = 0; int nact = 0;
    for (int b = 0; b < m; b++) {
        if (dq[b] > hi) continue;
        int c = nle(P->rq + (int64_t)b * NRAD, hi - dq[b]);
        if (c) { tot += c * (double)P->Nb[b] / NRAD; nact++; }
    }
    *out_r2 = hi; *out_nact = nact;
    double s = tot > 0 ? k / tot : 1.0;
    return s > 1 ? 1 : s;
}

// All L partitions in one call. Writes per-partition s_loc, s_g, r2 and active-cell counts.
void hpbs_all_k(const Part *parts, int L, const HQuery *Q, const float *q, double k, double N,
                double *sloc, double *sg, float *r2, int *nact, float *work_f, int *work_i) {
    for (int l = 0; l < L; l++)
        sloc[l] = hpbs_local_k(&parts[l], Q, q, k, N, &sg[l], &r2[l], &nact[l], work_f, work_i);
}

// ------------------------------------------------------------------------------------------------
// Native HVO planner: sketch estimate -> residual calibrator -> recall check -> minimax cost routing.
// One call per query; mirrors experiments/analyze.py (Calibrator, RecallModel, CostModel, robust()).
typedef struct {
    int nin, nh;                 // 12 inputs, 16 hidden units per layer
    const float *mu, *sd;        // input standardisation
    const float *W0, *b0, *W1, *b1, *W2;   // W0: nin x nh, W1: nh x nh, W2: nh x 1 (row-major, sklearn layout)
    float b2;
} CalW;

typedef struct {
    double rel0, rel1, rel2;     // C_rel = rel0 + rel1 N terms + rel2 N join
    double a0, cscan;            // pre: C_rel + a0 + cscan s_g N
    double post[3], fused[3];    // staircase (log b0, log b1, gamma)
    double cf, beta;             // fused need = cf (k/s)^beta
    double ef0, cap;
    double rec_w0[2], rec_w[2][4];   // logistic P(recall ok) for fused, post; w0 = NaN -> constant rec_c
    double rec_c[2], tau_rec, eta;
    int n_iso;                       // > 0: fused cost = C_rel + piecewise-linear table in log(k/s)
    const double *iso_x, *iso_y;     // (isotonic regression thresholds, as in analyze.CostModel)
} CostP;

static double n_rounds(double need, double ef0, double cap) {
    if (need < ef0) need = ef0;
    if (need > cap) need = cap;
    double j = ceil(log2(need / ef0) - 1e-9);
    return j < 0 ? 0 : j;
}

static double stair(const double *p, double need, int extra, double ef0, double cap) {
    double b0 = exp(p[0]), b1 = exp(p[1]), g = p[2];
    double j = n_rounds(need, ef0, cap) + extra, J = log2(cap / ef0);
    if (j > J) j = J;
    double r = pow(2.0, g);
    return (j + 1) * b0 + b1 * pow(ef0, g) * (pow(r, j + 1) - 1) / (r - 1);
}

static void plan_costs(const CostP *c, double sg, double s, double k, double N, double nterm, double join,
                       double *out) {
    double lo = k / N;
    if (s < lo) s = lo;
    if (s > 1) s = 1;
    double rel = c->rel0 + c->rel1 * N * nterm + c->rel2 * N * join;
    double kexp = k / s; if (kexp > c->cap) kexp = c->cap;
    double nf = c->cf * pow(k / s, c->beta); if (nf > c->cap) nf = c->cap;
    out[0] = rel + c->a0 + c->cscan * sg * N;
    if (c->n_iso > 0) {
        double x = log(k / s < c->cap ? k / s : c->cap), y;
        if (x <= c->iso_x[0]) y = c->iso_y[0];
        else if (x >= c->iso_x[c->n_iso - 1]) y = c->iso_y[c->n_iso - 1];
        else {
            int i = 1;
            while (c->iso_x[i] < x) i++;
            double t = (x - c->iso_x[i - 1]) / (c->iso_x[i] - c->iso_x[i - 1]);
            y = c->iso_y[i - 1] + t * (c->iso_y[i] - c->iso_y[i - 1]);
        }
        out[1] = rel + y;
    } else
        out[1] = rel + stair(c->fused, nf, 1, c->ef0, c->cap);
    out[2] = stair(c->post, kexp, 0, c->ef0, c->cap);
}

static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static double median(double *v, int n) {
    qsort(v, n, sizeof(double), cmp_d);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Returns the chosen plan (0 pre, 1 fused, 2 post); writes s_g, raw and calibrated s_loc.
int hvo_plan(const Part *parts, int L, const HQuery *Q, const float *q, double k, double N,
             double nterm, double join, double nrange, double nin,
             const CalW *cw, const CostP *cp, double *out_est, float *work_f, int *work_i) {
    double sl[16], sg[16], lg[16], sl2[16], sg2[16];
    float r2[16]; int na[16];
    double r2m = 0, r2s = 0, nam = 0;
    for (int l = 0; l < L; l++) {
        sl[l] = hpbs_local_k(&parts[l], Q, q, k, N, &sg[l], &r2[l], &na[l], work_f, work_i);
        r2m += r2[l]; nam += na[l];
    }
    r2m /= L; nam /= L;
    for (int l = 0; l < L; l++) r2s += (r2[l] - r2m) * (r2[l] - r2m);
    r2s = sqrt(r2s / L);
    double lmean = 0, lmin = 1e300, lmax = -1e300;
    for (int l = 0; l < L; l++) {
        lg[l] = log(sl[l] > 1e-7 ? sl[l] : 1e-7);
        lmean += lg[l]; if (lg[l] < lmin) lmin = lg[l]; if (lg[l] > lmax) lmax = lg[l];
        sl2[l] = sl[l]; sg2[l] = sg[l];
    }
    lmean /= L;
    double lvar = 0;
    for (int l = 0; l < L; l++) lvar += (lg[l] - lmean) * (lg[l] - lmean);
    lvar /= L;
    double s_g = median(sg2, L); if (s_g < 1.0 / N) s_g = 1.0 / N;
    double s_raw = median(sl2, L);
    double sr = s_raw > 1e-7 ? s_raw : 1e-7;
    float f[12] = {(float)log(s_g), (float)log(sr), (float)lvar, (float)(lmax - lmin), (float)nterm, (float)join,
                   (float)nrange, (float)nin, (float)log(nam > 1 ? nam : 1), (float)r2m, (float)r2s,
                   (float)(log(sr) - log(s_g))};
    // calibrator forward pass
    float h0[64], h1[64];
    for (int j = 0; j < cw->nh; j++) {
        float a = cw->b0[j];
        for (int i = 0; i < cw->nin; i++) a += (f[i] - cw->mu[i]) / cw->sd[i] * cw->W0[i * cw->nh + j];
        h0[j] = a > 0 ? a : 0;
    }
    for (int j = 0; j < cw->nh; j++) {
        float a = cw->b1[j];
        for (int i = 0; i < cw->nh; i++) a += h0[i] * cw->W1[i * cw->nh + j];
        h1[j] = a > 0 ? a : 0;
    }
    float corr = cw->b2;
    for (int i = 0; i < cw->nh; i++) corr += h1[i] * cw->W2[i];
    double lo = k / N;
    double s_floor = s_raw < lo ? lo : (s_raw > 1 ? 1 : s_raw);
    double s_cal = s_floor * exp(corr);
    if (s_cal < lo) s_cal = lo;
    if (s_cal > 1) s_cal = 1;
    // minimax costs over [s/eta, s*eta]
    double ca[3], cb[3], best = 1e300; int choice = 0;
    plan_costs(cp, s_g, s_cal / cp->eta, k, N, nterm, join, ca);
    plan_costs(cp, s_g, s_cal * cp->eta, k, N, nterm, join, cb);
    double rf[4] = {log(s_cal > 1e-7 ? s_cal : 1e-7), log(s_g > 1e-7 ? s_g : 1e-7), nterm, join};
    for (int j = 0; j < 3; j++) {
        if (j > 0 && cp->tau_rec > 0) {                       // recall admissibility of graph plans
            double p;
            if (isnan(cp->rec_w0[j - 1])) p = cp->rec_c[j - 1];
            else {
                double z = cp->rec_w0[j - 1];
                for (int i = 0; i < 4; i++) z += cp->rec_w[j - 1][i] * rf[i];
                p = 1.0 / (1.0 + exp(-z));
            }
            if (p < cp->tau_rec) continue;
        }
        double c = ca[j] > cb[j] ? ca[j] : cb[j];
        if (c < best) { best = c; choice = j; }
    }
    out_est[0] = s_g; out_est[1] = s_raw; out_est[2] = s_cal;
    return choice;
}
