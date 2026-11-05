/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2025 Institute of Software, Chinese Academy of Sciences
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include <cmath>
#include <cstdio>
#include <string>

#include "oneapi/dnnl/dnnl_debug.h"

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/rv64/gemm/rvv_gemm_utils_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

#define BM_NOCOPY_RVV 64
#define BN_NOCOPY_RVV 48
#define BK_NOCOPY_RVV 384
#define BN_LARGE_NOCOPY_RVV 192
#define BM_SMALL_NOCOPY_RVV 16
#define BN_SMALL_NOCOPY_RVV 1
#define BK_SMALL_NOCOPY_RVV 4
// Determine number of threads for each dimension of a 3-D partitioning
// algorithm based on input parameters
// m/n/k - First/second/third parameter for GEMM
// nthrs - total available number of threads
// nthrs_m/nthrs_n/nthrs_k - number of threads to use in each dimension
// BM/BN/BK - blocking values
void calc_nthr_nocopy_rvv(dim_t m, dim_t n, dim_t k, int nthrs, int *nthrs_m,
        int *nthrs_n, int *nthrs_k, dim_t *BM, dim_t *BN, dim_t *BK) {

    // Quick exit for single thread.
    if (nthrs == 1) {
        *nthrs_m = 1;
        *nthrs_n = 1;
        *nthrs_k = 1;

        *BM = m;
        *BN = n;
        *BK = k;
        return;
    }

    int nthr, nthr_m, nthr_n, nthr_k;
    dim_t MB, NB, KB;

    nthr = nthrs;
    nthr_m = static_cast<int>((m + BM_NOCOPY_RVV - 1) / BM_NOCOPY_RVV);
    nthr_n = static_cast<int>((n + BN_NOCOPY_RVV - 1) / BN_NOCOPY_RVV);
    nthr_k = 1;

    // Partition along K dimension
    //  - if threading allows having barriers (e.g. OMP)
    //  - if there is not enough parallelism along M or N
    if (dnnl_thr_syncable()) {
        int nthr_other = nthr_k = 1;
        while ((nthr_m * nthr_n * nthr_other < nthr)
                && (k / (nthr_other + 1) > BK_NOCOPY_RVV)) {
            nthr_other++;
            if ((nthr / nthr_other) * nthr_other > 0.9 * nthr)
                nthr_k = nthr_other;
        }
    }
    nthr /= nthr_k;

    if (nthr_m == 1) nthr_n = nthr;
    if (nthr_n == 1) nthr_m = nthr;

    // Simple partition reduction
    while (nthr_m * nthr_n > nthr)
        if (nthr_m > nthr_n)
            nthr_m--;
        else
            nthr_n--;
    while (nthr_m * nthr_n < nthr)
        if (nthr_m < nthr_n)
            nthr_m++;
        else
            nthr_n++;

    if ((nthr_m * nthr_n > nthr) && (nthr_m > 1) && (nthr_n > 1)) {

        if (nthr_m <= nthr_n) {
            nthr_m = (int)sqrt((double)nthr);
            if (nthr_m > (m + BM_SMALL_NOCOPY_RVV - 1) / BM_SMALL_NOCOPY_RVV)
                nthr_m = static_cast<int>(
                        (m + BM_SMALL_NOCOPY_RVV - 1) / BM_SMALL_NOCOPY_RVV);
            nthr_n = nthr / nthr_m;

            while ((nthr_m > 1) && (nthr_m * nthr_n != nthr)) {
                nthr_m--;
                nthr_n = nthr / nthr_m;
            }
        } else {
            nthr_n = (int)sqrt((double)nthr);
            if (nthr_n > (n + BN_SMALL_NOCOPY_RVV - 1) / BN_SMALL_NOCOPY_RVV)
                nthr_n = static_cast<int>(
                        (n + BN_SMALL_NOCOPY_RVV - 1) / BN_SMALL_NOCOPY_RVV);
            nthr_m = nthr / nthr_n;

            while ((nthr_n > 1) && (nthr_m * nthr_n != nthr)) {
                nthr_n--;
                nthr_m = nthr / nthr_n;
            }
        }
    }

    MB = (m + nthr_m - 1) / nthr_m + BM_SMALL_NOCOPY_RVV - 1;
    MB -= MB % BM_SMALL_NOCOPY_RVV;
    NB = (n + nthr_n - 1) / nthr_n + BN_SMALL_NOCOPY_RVV - 1;
    NB -= NB % BN_SMALL_NOCOPY_RVV;
    KB = (k + nthr_k - 1) / nthr_k + BK_SMALL_NOCOPY_RVV - 1;
    KB -= KB % BK_SMALL_NOCOPY_RVV;

    if (MB * nthr_m > m) nthr_m = static_cast<int>((m + MB - 1) / MB);
    if (NB * nthr_n > n) nthr_n = static_cast<int>((n + NB - 1) / NB);
    if (KB * nthr_k > k) nthr_k = static_cast<int>((k + KB - 1) / KB);

    *nthrs_m = nthr_m;
    *nthrs_n = nthr_n;
    *nthrs_k = nthr_k;

    *BM = MB;
    *BN = NB;
    *BK = KB;
}
#undef BM_NOCOPY_RVV
#undef BN_NOCOPY_RVV
#undef BK_NOCOPY_RVV
#undef BN_LARGE_NOCOPY_RVV
#undef BM_SMALL_NOCOPY_RVV
#undef BN_SMALL_NOCOPY_RVV
#undef BK_SMALL_NOCOPY_RVV

namespace {

rvjit_part_scheme_t partition_scheme() {
    static const rvjit_part_scheme_t scheme
            = getenv_string_user("RV64_GEMM_PARTITION") == "legacy"
            ? rvjit_part_scheme_t::legacy
            : rvjit_part_scheme_t::balanced;
    return scheme;
}

// The size of one way of L1, the largest row stride of A still read in
// place at the fast rate. oneDNN knows the size on some CPUs and guesses
// 32 KiB elsewhere; the ways are not queried, 8 is common
int64_t stride_cliff() {
    constexpr unsigned l1_ways = 8;
    const unsigned l1 = platform::get_per_core_cache_size(1);
    return l1 ? l1 / l1_ways : 0;
}

} // namespace

gemm_partition_t rvjit_gemm_partition(rvjit_part_problem_t p, data_type_t dt) {
    if (!p.stride_cliff) p.stride_cliff = stride_cliff();
    rvjit_part_plan_t legacy {};
    dim_t MB, NB, KB;
    calc_nthr_nocopy_rvv(p.M, p.N, p.K, p.nthr, &legacy.nthr_m, &legacy.nthr_n,
            &legacy.nthr_k, &MB, &NB, &KB);
    legacy.MB = MB;
    legacy.NB = NB;
    legacy.KB = KB;
    const rvjit_part_scheme_t scheme = partition_scheme();
    const rvjit_part_plan_t r = rvjit_part_plan(p, scheme, legacy);

    static const bool verbose
            = getenv_int_user("RV64_GEMM_PARTITION_VERBOSE", 0) != 0;
    if (verbose)
        fprintf(stderr,
                "rvjit_gemm_partition,%s,dt:%s,M:%ld,N:%ld,K:%ld,lda:%ld,"
                "panels:%d,nthr:%d,plan:%dx%dx%d,MB:%ld,NB:%ld,KB:%ld,pack:%"
                "d\n",
                scheme == rvjit_part_scheme_t::legacy ? "legacy" : "balanced",
                dnnl_dt2str(dt), (long)p.M, (long)p.N, (long)p.K,
                (long)p.lda_bytes, (int)p.panels, p.nthr, r.nthr_m, r.nthr_n,
                r.nthr_k, (long)r.MB, (long)r.NB, (long)r.KB, (int)r.pack);
    return {r.nthr_m, r.nthr_n, r.nthr_k, r.MB, r.NB, r.KB, r.pack};
}

// Partition n values as equally as possible among nthr threads
// and set the offset (t_offset) and number of values (t_block) for ithr
// Assumption: 0 <= ithr < nthr
void partition_unit_diff(
        int ithr, int nthr, dim_t n, dim_t *t_offset, dim_t *t_block) {

    dim_t band = n / nthr;
    if (band == 0) band = 1;
    dim_t tail = n - band * nthr;
    if (tail < 0) tail = 0;

    if (ithr < tail) {
        band++;
        *t_offset = band * ithr;
        *t_block = band;
    } else {
        *t_offset = band * ithr + tail;
        *t_block = band;
    }

    if (*t_offset >= n) {
        *t_offset = 0;
        *t_block = 0;
    }

    if (*t_offset + *t_block > n) { *t_block = n - *t_offset; }
}

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
