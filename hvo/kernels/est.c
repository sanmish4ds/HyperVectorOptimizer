// Query-time HPBS estimation (one partition per call). Mirrors HPBS._ball_weights /
// HPBS._cell_fraction in hvo/sketch.py; the Python version is the reference implementation.
#include <stdint.h>
#include <stdlib.h>
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
} Query;

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

static float cell_frac(const Part *P, const Query *Q, int b) {
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

double hpbs_global(const Part *P, const Query *Q, double N) {
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
double hpbs_local(const Part *P, const Query *Q, const float *q, double tau, float *out_r2, int *out_nact,
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
double hpbs_local_k(const Part *P, const Query *Q, const float *q, double k, double N,
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
void hpbs_all_k(const Part *parts, int L, const Query *Q, const float *q, double k, double N,
                double *sloc, double *sg, float *r2, int *nact, float *work_f, int *work_i) {
    for (int l = 0; l < L; l++)
        sloc[l] = hpbs_local_k(&parts[l], Q, q, k, N, &sg[l], &r2[l], &nact[l], work_f, work_i);
}
