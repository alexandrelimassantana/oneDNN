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

#include <algorithm>

#include "cpu/rv64/gemm/rvjit_gemm_partition.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

namespace {

// One way of a 32 KiB, 8-way L1: on the SpacemiT X60 the in-place reads of
// A slow down between 2304- and 4608-byte strides
constexpr int64_t default_stride_cliff = 4096;
// The passes from which packing a strip pays off above the stride cliff:
// the copy wins from 3 (1 or 2 tie)
constexpr int64_t passes_above_cliff = 3;
// Below the cliff, packing pays off after a pass per 8 bytes of a strip's
// row (m_block elements): 8 passes in f16, 16 in f32, where it crosses over
constexpr int64_t row_bytes_per_pass = 8;
// Splits within this share of the most even one count as equally even
constexpr double balance_slack = 0.05;

int64_t div_up(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

int64_t rnd_up(int64_t a, int64_t b) {
    return div_up(a, b) * b;
}

bool legacy_packs(const rvjit_part_problem_t &p, int64_t NB) {
    return !p.panels && (p.pack_all || NB / p.n_block > 3);
}

// The passes of a thread over NB columns of N, in the driver's N blocks
int64_t passes(const rvjit_part_problem_t &p, int64_t NB) {
    int64_t n = 0;
    for (int64_t j = 0; j < NB; j += p.driver_bn)
        n += div_up(std::min(p.driver_bn, NB - j), p.n_block);
    return n;
}

bool balanced_packs(const rvjit_part_problem_t &p, int64_t NB) {
    if (p.panels) return false;
    // Neither measured: a transposed A keeps the legacy rule
    if (p.pack_all || p.trans_a) return legacy_packs(p, NB);
    const int64_t cliff
            = p.stride_cliff > 0 ? p.stride_cliff : default_stride_cliff;
    const int64_t n = passes(p, std::min(NB, p.N));
    const int64_t below = p.m_block * p.a_elem_bytes / row_bytes_per_pass;
    return p.lda_bytes > cliff ? n >= passes_above_cliff : n > below;
}

} // namespace

rvjit_part_plan_t rvjit_part_plan(const rvjit_part_problem_t &p,
        rvjit_part_scheme_t scheme, const rvjit_part_plan_t &legacy) {
    rvjit_part_plan_t base = legacy;
    // A packed A starts every thread's rows at a panel
    if (p.panels) {
        base.MB = rnd_up(base.MB, p.m_block);
        base.nthr_m = static_cast<int>(div_up(p.M, base.MB));
    }
    if (scheme == rvjit_part_scheme_t::legacy) {
        base.pack = legacy_packs(p, base.NB);
        return base;
    }

    if (!p.k_split) {
        base.nthr_k = 1;
        base.KB = p.K;
    }
    const int nthr = std::max(p.nthr / base.nthr_k, 1);
    // The split of want_m x want_n threads, and its evenness: the work of an
    // even split over the threads available against that of the busiest
    // thread, so that idle threads count as imbalance
    auto split = [&](int want_m, int want_n, double &even) {
        rvjit_part_plan_t s = base;
        s.MB = rnd_up(div_up(p.M, want_m), p.m_block);
        s.nthr_m = static_cast<int>(div_up(p.M, s.MB));
        s.NB = div_up(p.N, want_n);
        s.nthr_n = static_cast<int>(div_up(p.N, s.NB));
        even = static_cast<double>(p.M) * p.N
                / (static_cast<double>(nthr) * std::min(s.MB, p.M)
                        * std::min(s.NB, p.N));
        return s;
    };
    double most_even = 0;
    for (int want_m = 1; want_m <= nthr; want_m++)
        for (int want_n = 1; want_n <= nthr / want_m; want_n++) {
            double even;
            split(want_m, want_n, even);
            most_even = std::max(most_even, even);
        }

    rvjit_part_plan_t best {};
    double best_even = 0;
    for (int want_m = 1; want_m <= nthr; want_m++)
        for (int want_n = 1; want_n <= nthr / want_m; want_n++) {
            double even;
            const rvjit_part_plan_t s = split(want_m, want_n, even);
            const int used = s.nthr_m * s.nthr_n;
            const int best_used = best.nthr_m * best.nthr_n;
            // Panels consider only the splits within the slack of the most
            // even one
            if (p.panels && even < most_even * (1 - balance_slack) - 1e-9)
                continue;
            bool better = best.nthr_m == 0;
            if (!better) {
                // Strided A splits M first, panels N, then the most even,
                // then the fewest threads
                const int s_dim = p.panels ? s.nthr_n : s.nthr_m;
                const int best_dim = p.panels ? best.nthr_n : best.nthr_m;
                better = s_dim != best_dim  ? s_dim > best_dim
                        : even != best_even ? even > best_even
                                            : used < best_used;
            }
            if (better) {
                best = s;
                best_even = even;
            }
        }
    best.pack = balanced_packs(p, best.NB);
    return best;
}

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
