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

#ifndef CPU_RV64_GEMM_RVV_GEMM_UTILS_F32_HPP
#define CPU_RV64_GEMM_RVV_GEMM_UTILS_F32_HPP

#include "common/c_types_map.hpp"

#include <cstddef>

#include "cpu/rv64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

template <typename T, bool isTransA, bool isTransB>
struct gemm_traits_t {};

template <bool isTransA, bool isTransB>
struct gemm_traits_t<float, isTransA, isTransB> {
    static constexpr dim_t BM = 4032;
    static constexpr dim_t BN = isTransA ? 96 : 256;
    static constexpr dim_t BK = isTransB ? 96 : 256;
};

// Sum the m*n values from p_src into p_dst, assuming the two-dimensional
// arrays have leading dimensions ld_src and ld_dst, respectively
template <typename data_t>
void sum_two_matrices(dim_t m, dim_t n, data_t *__restrict p_src, dim_t ld_src,
        data_t *__restrict p_dst, dim_t ld_dst) {

    for (dim_t j = 0; j < n; j++) {
        for (dim_t i = 0; i < m; i++) {
            p_dst[i + j * ld_dst] += p_src[i + j * ld_src];
        }
    }
}

void calc_nthr_nocopy_rvv(dim_t m, dim_t n, dim_t k, int nthrs, int *nthrs_m,
        int *nthrs_n, int *nthrs_k, dim_t *BM, dim_t *BN, dim_t *BK);

// Thread partition computed once at primitive initialization (with
// dnnl_get_max_threads()) and reused at execution time. Passing it to the GEMM
// drivers keeps the per-thread workspace offsets consistent with the capacity
// booked in the scratchpad, even when init and execute run under different
// threadpool contexts (e.g. --ctx-init=1 --ctx-exe=8). A nullptr leaves the
// driver to recompute the partition from dnnl_get_current_num_threads() and
// malloc its own workspace (the path used by inner_product / convolution).
struct gemm_partition_t {
    int nthr_m;
    int nthr_n;
    int nthr_k;
    dim_t MB;
    dim_t NB;
    dim_t KB;
};

void partition_unit_diff(
        int ithr, int nthr, dim_t n, dim_t *t_offset, dim_t *t_block);

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif // CPU_RV64_GEMM_RVV_GEMM_UTILS_F32_HPP
