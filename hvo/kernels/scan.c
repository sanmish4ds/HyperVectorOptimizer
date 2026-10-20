// Id-list-guided sequential top-k inner-product scan (the Pi_pre kernel).
// Built three ways from this one file:
//   scan_simd   : explicit NEON, 4 x float32x4 accumulators (16 FP32 lanes in flight)
//   scan_scalar : same loop, -fno-vectorize -fno-slp-vectorize, no intrinsics
//   scan_i8     : int8 vectors with NEON SDOT (vdotq_s32)
#include <stdint.h>
#include <string.h>
#ifndef PF
#define PF 16
#endif
#if defined(USE_NEON) || defined(USE_I8)
#include <arm_neon.h>
#endif

static inline void topk_push(float s, int64_t id, float *ts, int64_t *ti, int k, int *n) {
    if (*n < k) {
        int j = (*n)++;
        while (j > 0 && ts[j - 1] < s) { ts[j] = ts[j - 1]; ti[j] = ti[j - 1]; j--; }
        ts[j] = s; ti[j] = id;
    } else if (s > ts[k - 1]) {
        int j = k - 1;
        while (j > 0 && ts[j - 1] < s) { ts[j] = ts[j - 1]; ti[j] = ti[j - 1]; j--; }
        ts[j] = s; ti[j] = id;
    }
}

#ifndef USE_I8
static inline float dot(const float *a, const float *b, int d) {
#ifdef USE_NEON
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
#else
    float s = 0;
    for (int i = 0; i < d; i++) s += a[i] * b[i];
    return s;
#endif
}

int scan_topk(const float *X, int d, const int64_t *ids, int64_t n, const float *q, int k,
              int64_t *out_i, float *out_s) {
    int cnt = 0;
    for (int64_t j = 0; j < n; j++) {
        const float *x = X + ids[j] * (int64_t)d;
        if (j + PF < n) {  // prefetch every 128-byte line of the row PF rows ahead
            const char *p = (const char *)(X + ids[j + PF] * (int64_t)d);
            for (int o = 0; o < d * 4; o += 128) __builtin_prefetch(p + o);
        }
        topk_push(dot(x, q, d), ids[j], out_s, out_i, k, &cnt);
    }
    return cnt;
}
#else
int scan_topk(const int8_t *X, int d, const int64_t *ids, int64_t n, const int8_t *q, int k,
              int64_t *out_i, float *out_s) {
    int cnt = 0;
    for (int64_t j = 0; j < n; j++) {
        const int8_t *x = X + ids[j] * (int64_t)d;
        if (j + PF < n) {
            const char *p = (const char *)(X + ids[j + PF] * (int64_t)d);
            for (int o = 0; o < d; o += 128) __builtin_prefetch(p + o);
        }
        int32x4_t a0 = vdupq_n_s32(0), a1 = a0;
        int i = 0;
        for (; i + 32 <= d; i += 32) {
            a0 = vdotq_s32(a0, vld1q_s8(x + i), vld1q_s8(q + i));
            a1 = vdotq_s32(a1, vld1q_s8(x + i + 16), vld1q_s8(q + i + 16));
        }
        int32_t s = vaddvq_s32(vaddq_s32(a0, a1));
        for (; i < d; i++) s += (int32_t)x[i] * (int32_t)q[i];
        topk_push((float)s, ids[j], out_s, out_i, k, &cnt);
    }
    return cnt;
}
#endif
