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

#ifndef CPU_RV64_GEMM_RVJIT_GEMM_HPP
#define CPU_RV64_GEMM_RVJIT_GEMM_HPP

#include <cassert>
#include <cstdint>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/rv64/rvjit/rvjit_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

// The configuration of a GEMM block kernel. The kernel computes one strip of
// at most `m_block()` rows of C over a whole block of N and K:
//   C = alpha * A * B + beta * C + bias
// with column-major A (M x K, K x M if transposed), B (K x N, N x K if
// transposed) and C, converting each operand between its memory and register
// types (see rvjit_matmul_desc_t).
struct rvjit_gemm_conf_t {
    rvjit_type_t a, b, c;
    data_type_t epilogue = data_type::undef;
    bool trans_a = false, trans_b = false;
    rvjit_bias_kind_t bias_kind = rvjit_bias_kind_t::per_m;
    // An s32 epilogue takes beta as the literal 1 its flag enables
    bool int_beta = false;
    // The most columns and the deepest K of a call, which size the kernel's N
    // and K loops to the problem; 0 leaves them unbounded
    dim_t n_bound = 0;
    dim_t k_bound = 0;
};

// The kernel of an f32 GEMM.
rvjit_gemm_conf_t rvjit_gemm_f32_conf();

// The kernel of a half-precision GEMM: `dt` inputs accumulating in f32,
// narrowed to `dst_dt` as they are stored.
rvjit_gemm_conf_t rvjit_gemm_hp_conf(data_type_t dt, data_type_t dst_dt);

// The kernel of an int8 GEMM: `a_dt` and `b_dt` inputs accumulating in s32.
// An s32 C takes the integer epilogue; any other C is scaled in f32 and
// converted to `dst_dt` as it is stored.
rvjit_gemm_conf_t rvjit_gemm_int8_conf(data_type_t a_dt, data_type_t b_dt,
        data_type_t dst_dt, bool bias_is_scalar);

// The arguments of one call, read by the kernel at their offsets.
struct rvjit_gemm_args_t {
    const void *a;
    const void *b;
    void *c;
    dim_t m, n, k;
    // In bytes
    dim_t lda, ldb, ldc;
    // The bit patterns of the epilogue's scalars
    uint64_t alpha, beta;
    // f32 elements, one per M (or a single one for a scalar bias)
    const void *bias;
    uint64_t rt_flags;
};

// A tuned and generated block kernel.
struct rvjit_gemm_t {
    status_t status = status::unimplemented;
    rvjit_matmul_blocks_t blocks;
    std::unique_ptr<rvjit_matmul_kernel_t> kernel;

    // Rows of one strip
    dim_t m_block() const { return blocks.m_block(); }

    // Runs one strip, enabling alpha unless it is 1, beta unless it is 0 and
    // the bias when `args.bias` is set
    void operator()(rvjit_gemm_args_t &args, float alpha, float beta) const;
};

// Decides the blocks of the kernel of `conf`, or returns
// status::unimplemented when rvjit cannot take it.
status_t rvjit_gemm_tune(
        const rvjit_gemm_conf_t &conf, rvjit_matmul_blocks_t &blocks);

// Tunes and generates the kernel of `conf` into `g`, for a client that owns
// it.
status_t rvjit_gemm_init(const rvjit_gemm_conf_t &conf, rvjit_gemm_t &g);

// The kernel of `conf`, tuned and generated on its first use and shared by
// every later one, for the clients that own none (the global sgemm).
const rvjit_gemm_t &rvjit_gemm(const rvjit_gemm_conf_t &conf);

// The kernels of a GEMM call and how it packs its strips of A, a strip being
// the rows one kernel call computes.
struct rvjit_gemm_strips_t {
    // The kernel of a transposed strip read in place, and that of an
    // untransposed or packed strip
    const rvjit_gemm_t *trans_a_ker = nullptr;
    const rvjit_gemm_t *nontrans_a_ker = nullptr;
    // Every strip is packed: a kernel the caller owns reads A untransposed
    bool pack_all = false;

    // Uses `kernel`, owned by the caller, or else the shared kernels of
    // `conf` (whose trans_a is ignored), for an A transposed when `trans_a`
    status_t init(
            rvjit_gemm_conf_t conf, bool trans_a, const rvjit_gemm_t *kernel);

    // The rows of a strip, which the per-thread A workspace holds
    dim_t m_block() const { return nontrans_a_ker->m_block(); }

    // Whether a thread computing `n` columns packs its whole strips
    bool packs(dim_t n) const {
        return pack_all || n / nontrans_a_ker->blocks.n_block() > 3;
    }

    // Computes the M rows of a block of C, a strip at a time, from `args`
    // (B, N, K, ldb and ldc set). A strip of A (`lda` elements of `a_size`
    // bytes between its K rows, or M rows when `trans_a`) is first packed
    // into `ws` with `copy_a` when `do_copy` holds for a whole strip, or
    // always under `pack_all`. C elements are `c_size` bytes, and every strip
    // reads the whole of a scalar bias.
    template <typename copy_a_t>
    void operator()(rvjit_gemm_args_t &args, dim_t M, const char *A, dim_t lda,
            dim_t a_size, bool trans_a, char *C, dim_t c_size, char *ws,
            bool do_copy, const float *bias, bool bias_is_scalar, float alpha,
            float beta, const copy_a_t &copy_a) const {
        const dim_t m_blk = m_block();
        for (dim_t i = 0; i < M; i += m_blk) {
            const dim_t tile_m = nstl::min(m_blk, M - i);
            const char *a = A + (trans_a ? i * lda : i) * a_size;
            const bool copy = (pack_all || (do_copy && tile_m == m_blk)) && ws;
            if (copy) copy_a(a, ws, tile_m);

            args.a = copy ? ws : a;
            args.lda = (copy ? tile_m : lda) * a_size;
            args.c = C + i * c_size;
            args.m = tile_m;
            args.bias = bias ? (bias_is_scalar ? bias : bias + i) : nullptr;
            const rvjit_gemm_t &ker
                    = copy || !trans_a ? *nontrans_a_ker : *trans_a_ker;
            ker(args, alpha, beta);
        }
    }
};

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_GEMM_RVJIT_GEMM_HPP
