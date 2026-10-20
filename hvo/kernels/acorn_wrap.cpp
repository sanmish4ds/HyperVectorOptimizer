// Thin C ABI over the ACORN reference implementation (stanford-futuredata/ACORN, a faiss fork),
// so that the fused plan can call ACORN-gamma's predicate-agnostic filtered search from Python.
#include <cstdint>
#include <vector>
#include <omp.h>
#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>
#include <faiss/impl/ACORN.h>

#define API __attribute__((visibility("default")))

static int64_t last_ndis = 0;

extern "C" {

API int64_t acorn_last_ndis() { return last_ndis; }

API void *acorn_build(int d, int M, int gamma, int M_beta, int64_t n, const float *X, int nthreads) {
    omp_set_num_threads(nthreads);
    std::vector<int> metadata(n, 0);  // unused: predicates are passed as a per-query filter map
    auto *idx = new faiss::IndexACORNFlat(d, M, gamma, metadata, M_beta, faiss::METRIC_L2);  // unit vectors: same order as IP
    idx->add(n, X);
    return idx;
}

API int acorn_write(void *idx, const char *path) {
    faiss::write_index(static_cast<faiss::Index *>(idx), path);
    return 0;
}

API void *acorn_read(const char *path) { return faiss::read_index(path); }

// One query; filter_map has ntotal bytes (1 = row satisfies the predicate). Returns #results found.
API int acorn_search(void *p, const float *q, int k, int efs, char *filter_map, float *D, int64_t *I) {
    omp_set_num_threads(1);
    auto *idx = static_cast<faiss::IndexACORNFlat *>(p);
    // hybrid_search reads the index's own efSearch member, not params->efSearch
    idx->acorn.efSearch = efs;
    faiss::SearchParametersACORN params;
    params.efSearch = efs;
    std::vector<faiss::idx_t> lab(k);
    // ACORN's hybrid (filtered) search counts distance computations in n3, not ndis
    size_t before = faiss::acorn_stats.n3;
    idx->search(1, q, k, D, lab.data(), filter_map, &params);
    last_ndis = (int64_t)(faiss::acorn_stats.n3 - before);
    int found = 0;
    for (int i = 0; i < k; i++) {
        I[i] = lab[i];
        if (lab[i] >= 0) found++;
    }
    return found;
}

}
