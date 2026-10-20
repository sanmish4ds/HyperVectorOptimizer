// Resumable HNSW scan for the post-filter plan (in the spirit of pgvector 0.8's iterative index scan).
//
// After a greedy descent through the upper layers, layer 0 is searched with HNSW's ef rule, kept
// resumable: the best `ef` visited-but-unemitted nodes form a window W (the rest wait in an overflow
// heap O). Candidates are expanded while the nearest unexpanded candidate is closer than the worst
// node in a full window -- HNSW's stopping condition -- and the closest node of W is then emitted and
// W refilled from O. Every emitted prefix therefore has the quality of an ef-bounded HNSW search, and
// the next batch resumes from the same state instead of restarting with a larger ef.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

typedef struct {
    int n, d;
    const float *X;          // n x d, row-major
    const int32_t *nbr;      // faiss HNSW neighbors
    const uint64_t *off;     // per-node offsets into nbr
    const int32_t *cum;      // cum_nneighbor_per_level
    const int32_t *levels;   // per-node level count (faiss: max level + 1)
    int entry, max_level;
} Graph;

typedef struct { float d; int32_t id; } Item;
typedef struct { Item *a; int n, cap; } Heap;   // min-heap on d

static void hpush(Heap *h, float d, int32_t id) {
    if (h->n == h->cap) { h->cap = h->cap ? 2 * h->cap : 1024; h->a = realloc(h->a, h->cap * sizeof(Item)); }
    int i = h->n++;
    while (i > 0) { int p = (i - 1) >> 1; if (h->a[p].d <= d) break; h->a[i] = h->a[p]; i = p; }
    h->a[i].d = d; h->a[i].id = id;
}
static Item hpop(Heap *h) {
    Item top = h->a[0], last = h->a[--h->n];
    int i = 0;
    for (;;) {
        int c = 2 * i + 1;
        if (c >= h->n) break;
        if (c + 1 < h->n && h->a[c + 1].d < h->a[c].d) c++;
        if (last.d <= h->a[c].d) break;
        h->a[i] = h->a[c]; i = c;
    }
    if (h->n) h->a[i] = last;
    return top;
}

typedef struct {
    const Graph *g;
    const float *q;
    uint32_t *stamp; uint32_t gen;
    Heap cand, over;          // candidates to expand (min-heap); overflow of visited nodes (min-heap)
    Item *win; int nwin, ef;  // window of the best <= ef visited, unemitted nodes (unsorted, ef small)
    int wi;                   // index of the window's worst node (valid when nwin > 0)
    int64_t ndis;
} Iter;

static void win_rescan(Iter *it) {
    int w = 0;
    for (int i = 1; i < it->nwin; i++) if (it->win[i].d > it->win[w].d) w = i;
    it->wi = w;
}
static int win_best(const Iter *it) {
    int b = 0;
    for (int i = 1; i < it->nwin; i++) if (it->win[i].d < it->win[b].d) b = i;
    return b;
}
static void win_add(Iter *it, float d, int32_t id) {
    if (it->nwin < it->ef) {
        int i = it->nwin++;
        it->win[i].d = d; it->win[i].id = id;
        if (i == 0 || d > it->win[it->wi].d) it->wi = i;
        return;
    }
    if (d < it->win[it->wi].d) {          // replace the worst; it moves to the overflow heap
        hpush(&it->over, it->win[it->wi].d, it->win[it->wi].id);
        it->win[it->wi].d = d; it->win[it->wi].id = id;
        win_rescan(it);
    } else {
        hpush(&it->over, d, id);
    }
}

static inline float dist(const Graph *g, const float *q, int32_t i) {   // -<x, q>, smaller is closer
    const float *a = g->X + (int64_t)i * g->d;
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    int j = 0;
    for (; j + 16 <= g->d; j += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + j), vld1q_f32(q + j));
        s1 = vfmaq_f32(s1, vld1q_f32(a + j + 4), vld1q_f32(q + j + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + j + 8), vld1q_f32(q + j + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + j + 12), vld1q_f32(q + j + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; j < g->d; j++) s += a[j] * q[j];
    return -s;
}

void *it_new(const Graph *g) {
    Iter *it = calloc(1, sizeof(Iter));
    it->g = g;
    it->stamp = calloc(g->n, sizeof(uint32_t));
    return it;
}

void it_free(void *p) {
    Iter *it = p;
    free(it->stamp); free(it->cand.a); free(it->over.a); free(it->win); free(it);
}

void it_start(void *p, const float *q, int ef) {
    Iter *it = p;
    const Graph *g = it->g;
    it->q = q; it->cand.n = it->over.n = 0; it->nwin = 0; it->ndis = 0;
    it->ef = ef;
    it->win = realloc(it->win, (size_t)ef * sizeof(Item));
    if (++it->gen == 0) { memset(it->stamp, 0, g->n * sizeof(uint32_t)); it->gen = 1; }
    int32_t cur = g->entry;
    float dcur = dist(g, q, cur); it->ndis++;
    for (int l = g->max_level; l >= 1; l--) {          // greedy descent on upper layers
        int improved = 1;
        while (improved) {
            improved = 0;
            const int32_t *nb = g->nbr + g->off[cur] + g->cum[l];
            int cnt = g->cum[l + 1] - g->cum[l];
            for (int j = 0; j < cnt && nb[j] >= 0; j++) {
                float dd = dist(g, q, nb[j]); it->ndis++;
                if (dd < dcur) { dcur = dd; cur = nb[j]; improved = 1; }
            }
        }
    }
    it->stamp[cur] = it->gen;
    hpush(&it->cand, dcur, cur);
    win_add(it, dcur, cur);
}

// Emit up to B more nodes in (approximate) distance order. Returns the number emitted (0 = exhausted).
int it_next(void *p, int B, int64_t *out, float *outd) {
    Iter *it = p;
    const Graph *g = it->g;
    int m = 0;
    while (m < B) {
        // HNSW rule: expand while the window is not full or a candidate is closer than its worst node
        while (it->cand.n && (it->nwin < it->ef || it->cand.a[0].d < it->win[it->wi].d)) {
            Item c = hpop(&it->cand);
            const int32_t *nb = g->nbr + g->off[c.id] + g->cum[0];
            int cnt = g->cum[1] - g->cum[0];
            int32_t fresh[256]; int nf = 0;
            for (int j = 0; j < cnt && nb[j] >= 0 && nf < 256; j++) {   // unvisited neighbours first...
                int32_t u = nb[j];
                if (it->stamp[u] == it->gen) continue;
                it->stamp[u] = it->gen;
                fresh[nf++] = u;
                const char *pv = (const char *)(g->X + (int64_t)u * g->d);
                for (int o = 0; o < g->d * 4; o += 128) __builtin_prefetch(pv + o);   // ...then prefetch
            }
            for (int j = 0; j < nf; j++) {
                float du = dist(g, it->q, fresh[j]); it->ndis++;
                hpush(&it->cand, du, fresh[j]);
                win_add(it, du, fresh[j]);
            }
        }
        if (!it->nwin) break;
        int b = win_best(it);                 // emit the closest node of the window
        out[m] = it->win[b].id; outd[m] = it->win[b].d; m++;
        it->win[b] = it->win[--it->nwin];
        if (it->nwin) win_rescan(it);
        if (it->over.n) { Item o = hpop(&it->over); win_add(it, o.d, o.id); }   // refill from overflow
    }
    return m;
}

int64_t it_ndis(void *p) { return ((Iter *)p)->ndis; }
