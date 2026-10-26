/*******************************************************************************
* Copyright 2025 ZTE Corporation
* Copyright 2026 Intel Corporation
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

#ifndef CPU_RV64_JIT_RVV_1X1_CONVOLUTION_HPP
#define CPU_RV64_JIT_RVV_1X1_CONVOLUTION_HPP

#include <cstddef>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/platform.hpp"

#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/jit_primitive_conf.hpp"
#include "cpu/rv64/rvjit/rvjit_matmul.hpp"
#include "cpu/rv64/rvv_weights_layout.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// The blocking of a 1x1 convolution over nwc/nhwc/ndhwc data and OC-blocked
// weights, for `nthreads` threads: rvjit tunes the kernel `desc`, and its
// `blocks` set the OC block and the spatial chunk
status_t init_1x1_conv_conf(jit_1x1_conv_conf_t &jcp, rvjit_matmul_desc_t &desc,
        rvjit_matmul_blocks_t &blocks, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, int nthreads);

// The arguments of one call of the rvjit kernel: an OC block of the weights
// against a range of spatial positions, over the whole IC
struct rvjit_1x1_args_t {
    const void *wei;
    const void *src;
    void *dst;
    dim_t oc;
    dim_t os;
    const void *bias;
};

struct jit_rvv_1x1_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_1x1:", isa_, ""),
                jit_rvv_1x1_convolution_fwd_t);

        status_t init(const engine_t *engine) {
            using namespace utils;
            using namespace format_tag;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper weights_d(weights_md());
            const memory_desc_wrapper dst_d(dst_md());

            VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            // Accepted input dtype combos (bf16/f16 widen into f32
            // accumulators; dst is f32, or f16 narrowed from the f32
            // accumulators when Zvfh is available):
            //   f32 /f32  : plain f32.
            //   bf16/bf16 : symmetric bf16, widening FMA (Zvfbfwma).
            //   f16 /f16  : symmetric f16, widening FMA (Zvfh).
            //   f32 /bf16 : bf16 weight compression, weights widened to f32
            //               (Zvfbfwma); f32 src, f32 FMA.
            //   f32 /f16  : f16 weight compression, weights widened to f32
            //               (Zvfh); f32 src, f32 FMA.
            // The last two mirror x64 is_f32_bf16 / is_f32_f16.
            const auto src_dt = src_d.data_type();
            const auto wei_dt = weights_d.data_type();
            const auto dst_dt = dst_d.data_type();
            // Account for f16 destination narrowing; otherwise use src for
            // symmetric paths or weights for weight compression (f32 src).
            const auto name_dt = dst_dt == data_type::f16
                    ? data_type::f16
                    : (src_dt == data_type::f32 ? wei_dt : src_dt);
            isa_ = name_dt == data_type::bf16
                    ? zvfbfwma
                    : (name_dt == data_type::f16 ? zvfh : v);
            const bool all_f32
                    = src_dt == data_type::f32 && wei_dt == data_type::f32;
            const bool sym_lowp = wei_dt == src_dt
                    && ((src_dt == data_type::bf16 && mayiuse(zvfbfwma))
                            || (src_dt == data_type::f16 && mayiuse(zvfh)));
            const bool wei_decomp = src_dt == data_type::f32
                    && ((wei_dt == data_type::bf16 && mayiuse(zvfbfwma))
                            || (wei_dt == data_type::f16 && mayiuse(zvfh)));
            VDISPATCH_CONV((all_f32 || sym_lowp || wei_decomp)
                            && (dst_dt == data_type::f32
                                    || (dst_dt == data_type::f16
                                            && mayiuse(zvfh))),
                    VERBOSE_UNSUPPORTED_DT);
            // Bias is added into the f32 accumulators; a bf16/f16 bias (== src)
            // is widened to f32 in-kernel, matching x64/aarch64.
            VDISPATCH_CONV(IMPLICATION(with_bias(),
                                   utils::one_of(weights_md(1)->data_type,
                                           data_type::f32, src_dt)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(attr()->has_default_values(
                                   primitive_attr_t::skip_mask_t::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONV(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            // ISA check
            VDISPATCH_CONV(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);

            // 1x1 convolution check
            const int ndims = src_d.ndims();
            const int weights_ndims = weights_d.ndims();
            for (int i = 0; i < ndims - 2; ++i) {
                VDISPATCH_CONV(
                        weights_d.dims()[weights_ndims - (ndims - 2) + i] == 1,
                        VERBOSE_UNSUPPORTED_FEATURE,
                        "only 1x1 convolution is supported");
                VDISPATCH_CONV(desc()->strides[i] == 1,
                        VERBOSE_UNSUPPORTED_FEATURE,
                        "only stride 1 is supported");
                VDISPATCH_CONV(desc()->padding[0][i] == 0,
                        VERBOSE_UNSUPPORTED_FEATURE,
                        "padding is not supported");
            }

            // Only support data = nwc/nhwc/ndhwc
            const auto dat_tag = get_dat_tag();
            VDISPATCH_CONV(
                    IMPLICATION(src_d.matches_one_of_tag(dat_tag) != dat_tag,
                            src_d.format_kind() == format_kind::any),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(
                    IMPLICATION(dst_d.matches_one_of_tag(dat_tag) != dat_tag,
                            dst_d.format_kind() == format_kind::any),
                    VERBOSE_UNSUPPORTED_TAG);

            // The rvjit kernel and its blocks, decided once, before the
            // weights format they set
            VDISPATCH_CONV_SC(init_1x1_conv_conf(jcp_, rvjit_desc_,
                                      rvjit_blocks_, *desc(), src_d, weights_d,
                                      dst_d, dnnl_get_max_threads()),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "rvjit cannot tune the kernel");

            // Only support the weights laid out by OC block (init_wei_md)
            memory_desc_t want_wei_md = weights_md_;
            VDISPATCH_CONV_SC(
                    init_wei_md(want_wei_md), VERBOSE_UNSUPPORTED_TAG);
            if (weights_md_.format_kind == format_kind::any)
                weights_md_ = want_wei_md;
            VDISPATCH_CONV(weights_md_ == want_wei_md, VERBOSE_UNSUPPORTED_TAG);

            // Set default formats if format_kind == any
            VDISPATCH_CONV(set_default_formats_common(
                                   dat_tag, format_tag::undef, dat_tag),
                    VERBOSE_UNSUPPORTED_TAG);
            return status::success;
        }

        // ISA that drives the impl name: v (f32), zvfh (f16), or zvfbfwma
        // (bf16). Set in init() before any dtype/ISA rejection.
        cpu_isa_t isa_ = v;
        jit_1x1_conv_conf_t jcp_ = utils::zero<decltype(jcp_)>();
        rvjit_matmul_desc_t rvjit_desc_;
        rvjit_matmul_blocks_t rvjit_blocks_;

    protected:
        bool post_ops_ok() const {
            // TODO: Post-ops support is not implemented yet.
            return attr()->post_ops_.len() == 0;
        }

        inline format_tag_t get_dat_tag() const {
            using namespace format_tag;
            return utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
        }

        // The weights as g, OC blocks, IC and spatial, the OC lanes of a
        // block innermost (plain when OC is shorter than one block)
        status_t init_wei_md(memory_desc_t &md) const {
            const int oc_dim = with_groups() ? 1 : 0;
            int order[DNNL_MAX_NDIMS];
            for (int d = 0; d < md.ndims; ++d)
                order[d] = d;
            return init_m_blocked_md(
                    md, oc_dim, rvjit_blocks_.m_block(), order);
        }
    };

    jit_rvv_1x1_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        return rvjit_matmul_kernel_create(
                pd()->rvjit_desc_, pd()->rvjit_blocks_, kernel_);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_thr(const int ithr, const int nthr, const char *src,
            const char *weights, const char *bias, char *dst,
            const memory_tracking::grantor_t &scratchpad) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<rvjit_matmul_kernel_t> kernel_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
