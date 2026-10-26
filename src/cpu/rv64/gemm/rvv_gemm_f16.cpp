/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#include <cstring>

#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/rv64/gemm/rvjit_gemm.hpp"
#include "cpu/rv64/gemm/rvv_gemm_f16.hpp"
#include "cpu/rv64/gemm/rvv_gemm_utils_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::utils;
using namespace gemm_utils;

namespace {

// Scalar copy of A into workspace for cache-friendly access. Elements are
// 2 bytes (f16/bf16); after copy, ws is laid out as K blocks of m contiguous
// elements:
//   ws[k * m + i] = A_logical[i, k]
void copy_A_f16(
        bool isTransA, dim_t K, const char *A, dim_t lda, char *ws, dim_t m) {
    for (dim_t k = 0; k < K; k++) {
        if (isTransA) {
            for (dim_t i = 0; i < m; i++)
                std::memcpy(ws + i * 2, A + (i * lda + k) * 2, 2);
        } else {
            std::memcpy(ws, A + k * lda * 2, m * 2);
        }
        ws += m * 2;
    }
}

// Runs the block one strip of rows at a time
template <bool isTransA>
void block_ker_f16(const dim_t M, const dim_t N, const dim_t K, const char *A,
        const dim_t lda, const char *B, const dim_t ldb, char *C,
        const dim_t ldc, char *ws, bool do_copy, const rvjit_gemm_strips_t &sc,
        const float *bias, const rvjit_gemm_panels_t *panels) {
    rvjit_gemm_args_t args {};
    args.b = B;
    args.n = N;
    args.k = K;
    args.ldb = ldb * 2;
    args.ldc = ldc * 2;
    const auto copy_a = [&](const char *a, char *w, dim_t m) {
        copy_A_f16(isTransA, K, a, lda, w, m);
    };
    sc(args, M, A, lda, 2, isTransA, C, 2, ws, do_copy, bias, false, 1.0f, 0.0f,
            copy_a, panels);
}

template <bool isTransA>
void gemm_ithr_f16(const dim_t M, const dim_t N, const dim_t K, const char *A,
        const dim_t lda, const char *B, const dim_t ldb, char *C,
        const dim_t ldc, bool do_copy, char *ws, const rvjit_gemm_strips_t &sc,
        const float *bias, const rvjit_gemm_panels_t *panels) {

    // A block of A in panels starts at a panel
    constexpr dim_t BM_rows = gemm_traits_t<float, isTransA, false>::BM;
    const dim_t BM = panels ? panels->whole(BM_rows) : BM_rows;
    constexpr dim_t BN = gemm_traits_t<float, isTransA, false>::BN;

    if ((M <= 0) || (N <= 0)) return;

    // The JIT epilogue is overwrite-only, so the whole reduction runs in a
    // single K pass (no K blocking).
    for (dim_t Bm = 0; Bm < M; Bm += BM) {
        dim_t mb = nstl::min(M - Bm, BM);
        for (dim_t Bn = 0; Bn < N; Bn += BN) {
            dim_t nb = nstl::min(N - Bn, BN);
            const char *curA = panels ? A + panels->offset(Bm, 0) * 2
                    : isTransA        ? A + Bm * lda * 2
                                      : A + Bm * 2;
            const char *curB = B + Bn * ldb * 2;
            char *curC = C + (Bm + Bn * ldc) * 2;
            block_ker_f16<isTransA>(mb, nb, K, curA, lda, curB, ldb, curC, ldc,
                    ws, do_copy, sc, bias ? bias + Bm : nullptr, panels);
        }
    }
}
} // namespace

status_t rvv_gemm_f16(const char *transa_, const char *transb_, const dim_t *M_,
        const dim_t *N_, const dim_t *K_, const float *alpha_, const void *A,
        const dim_t *lda_, const void *B, const dim_t *ldb_, const float *beta_,
        void *C, const dim_t *ldc_, data_type_t dt, char *ws_buffers_in,
        const gemm_partition_t *part, const float *bias,
        const rvjit_gemm_t *kernel, const rvjit_gemm_panels_t *panels) {

    if (!(utils::one_of(*transa_, 'n', 'N', 't', 'T')
                && utils::one_of(*transb_, 'n', 'N')))
        return status::unimplemented;

    // Overwrite-only epilogue contract (see the kernel header).
    if (*alpha_ != 1.0f || *beta_ != 0.0f) return status::unimplemented;
    if (!utils::one_of(dt, data_type::f16, data_type::bf16))
        return status::unimplemented;

    bool isTransA = (*transa_ == 'T' || *transa_ == 't');

    const dim_t M = *M_, N = *N_, K = *K_;
    const dim_t lda = *lda_, ldb = *ldb_, ldc = *ldc_;

    // early out and avoid division by zero in partitioning
    if (utils::one_of(0, M, N)) return status::success;

    if (K <= 0) {
        auto *C_bytes = static_cast<char *>(C);
        for (dim_t j = 0; j < N; j++)
            std::memset(C_bytes + j * ldc * 2, 0, M * 2);
        return status::success;
    }

    int nthr_m, nthr_n, nthr_k;
    dim_t MB, NB, KB;
    if (part) {
        nthr_m = part->nthr_m;
        nthr_n = part->nthr_n;
        MB = part->MB;
        NB = part->NB;
    } else {
        int nthr = dnnl_get_current_num_threads();
        calc_nthr_nocopy_rvv(
                M, N, K, nthr, &nthr_m, &nthr_n, &nthr_k, &MB, &NB, &KB);
    }
    // The K axis is never split: the kernel stores the narrowed result once.
    const int nthr_mn = nthr_m * nthr_n;

    rvjit_gemm_strips_t sc;
    CHECK(sc.init(rvjit_gemm_hp_conf(dt, dt), isTransA, kernel));

    // Every thread's rows start at a panel of a packed A
    assert(IMPLICATION(
            panels, !isTransA && (nthr_m == 1 || MB % panels->m_block == 0)));
    const bool do_copy = !panels && sc.packs(NB);
    const size_t ws_elems_per_thr = K * sc.m_block() * 2;
    const size_t ws_size_per_thr = rnd_up(ws_elems_per_thr, PAGE_4K);

    // The per-thread A-copy workspace must be booked in the caller's
    // scratchpad (the matmul primitive books via pd_t::init_scratchpad()
    // whenever the copy can trigger).
    char *ws_buffers = ws_buffers_in;
    if (do_copy && !ws_buffers) return status::unimplemented;

    auto get_thr_block = [&](dim_t &from, dim_t &to, dim_t &myN, dim_t NB_,
                                 dim_t N_, int ithr) {
        from = NB_ * (ithr);
        to = NB_ * (ithr + 1);
        if (to > N_) to = N_;
        myN = to - from;
    };

    parallel(nthr_mn, [&](int ithr, int nthr) {
        assert(nthr_mn == nthr);
        MAYBE_UNUSED(nthr);

        int ithr_m = ithr % nthr_m;
        int ithr_n = ithr / nthr_m;

        char *ws = do_copy ? ws_buffers + ithr * ws_size_per_thr : nullptr;

        dim_t m_from = 0, m_to = 0, myM = 0, n_from = 0, n_to = 0, myN = 0;

        get_thr_block(m_from, m_to, myM, MB, M, ithr_m);
        get_thr_block(n_from, n_to, myN, NB, N, ithr_n);

        if (myM > 0 && myN > 0) {
            const char *A_bytes = static_cast<const char *>(A);
            const char *B_bytes = static_cast<const char *>(B);
            char *C_bytes = static_cast<char *>(C);
            const char *myA = panels ? A_bytes + panels->offset(m_from, 0) * 2
                    : isTransA       ? &A_bytes[m_from * lda * 2]
                                     : &A_bytes[m_from * 2];
            const char *myB = B_bytes + n_from * ldb * 2;
            char *myC = C_bytes + (m_from + n_from * ldc) * 2;

            if (!isTransA) {
                gemm_ithr_f16<false>(myM, myN, K, myA, lda, myB, ldb, myC, ldc,
                        do_copy, ws, sc, bias ? bias + m_from : nullptr,
                        panels);
            } else {
                gemm_ithr_f16<true>(myM, myN, K, myA, lda, myB, ldb, myC, ldc,
                        do_copy, ws, sc, bias ? bias + m_from : nullptr,
                        panels);
            }
        }
    });

    return status::success;
}
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
