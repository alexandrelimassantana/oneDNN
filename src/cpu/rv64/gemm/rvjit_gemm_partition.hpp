/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#ifndef CPU_RV64_GEMM_RVJIT_GEMM_PARTITION_HPP
#define CPU_RV64_GEMM_RVJIT_GEMM_PARTITION_HPP

// How the rv64 GEMM drivers split C over threads and whether a thread packs
// its strips of A. Only standard headers, so that the decision also builds on
// a host and can be checked offline.
//
// GEMM notation: C (M x N) = A (M x K) * B (K x N). A thread computes MB x NB
// of C, a strip of m_block rows of M at a time, and reads its strip of A once
// per n_block columns of N (a pass). A plain matmul's A is its weights, M
// their columns and N the source rows.

#include <cstdint>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

enum class rvjit_part_scheme_t {
    // calc_nthr_nocopy_rvv, and pack when NB / n_block > 3
    legacy,
    // The split that reuses the operand that is expensive to read again, as
    // evenly as the threads allow, and packing by A's row stride:
    // - A read in place or copied (strided): M first, in whole strips, so
    //   that every thread reads or copies its strips of A once; N with the
    //   threads left, the most even way.
    // - A in panels (cheap to stream): of the splits within 5% of the most
    //   even, the one that splits N the most, so that every thread's part of
    //   B, which it reads again for each strip, stays small.
    // - Packing: never panels, always pack_all; else pack when A's row
    //   stride is above stride_cliff and a thread makes at least 3 passes,
    //   or below it and more than one pass per 8 bytes of a strip's row
    //   (m_block * a_elem_bytes).
    balanced,
};

struct rvjit_part_problem_t {
    int64_t M, N, K;
    // Row strides of A and B, and the size of an element of A, in bytes
    int64_t lda_bytes, ldb_bytes, a_elem_bytes;
    // The kernel's strip rows and pass columns, and the driver's N block
    int64_t m_block, n_block, driver_bn;
    // A is packed ahead into panels (never copied, MB in whole panels), read
    // transposed, or always packed (a kernel the caller owns)
    bool panels, trans_a, pack_all;
    // The driver may split K over threads
    bool k_split;
    int nthr;
    // The largest row stride of A, in bytes, that is still read in place at
    // the fast rate: the size of one way of L1 (its size over its ways). 0
    // for the default, 4096 (32 KiB, 8 ways)
    int64_t stride_cliff;
};

struct rvjit_part_plan_t {
    int nthr_m, nthr_n, nthr_k;
    int64_t MB, NB, KB;
    bool pack;
};

// The plan of `scheme` for `p`, from `legacy`, the split of
// calc_nthr_nocopy_rvv (whose nthr_k and KB balanced keeps when the driver
// splits K)
rvjit_part_plan_t rvjit_part_plan(const rvjit_part_problem_t &p,
        rvjit_part_scheme_t scheme, const rvjit_part_plan_t &legacy);

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_GEMM_RVJIT_GEMM_PARTITION_HPP
