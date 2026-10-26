/*******************************************************************************
* Copyright 2025 ZTE Corporation
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

#include <cstddef>
#include <type_traits>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/rv64/jit_rvv_1x1_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

namespace {

// The weights of an OC block are its IC rows of OC lanes (A), the source rows
// of a spatial position its IC (B), and the destination rows of a spatial
// position its OC (C), for at most `n_bound` positions a call. The register
// grouping is left to rvjit
void init_1x1_rvjit_desc(
        const jit_1x1_conv_conf_t &jcp, dim_t n_bound, rvjit_matmul_desc_t &d) {
    using a_t = rvjit_1x1_args_t;
    d = rvjit_matmul_desc_t();
    const bool decomp
            = jcp.src_dt == data_type::f32 && jcp.wei_dt != data_type::f32;
    d.a = rvjit_type_t(jcp.wei_dt, decomp ? data_type::f32 : data_type::undef);
    d.b = rvjit_type_t(jcp.src_dt);
    d.c = rvjit_type_t(jcp.dst_dt, data_type::f32);

    d.m = rvjit_value_t::at(offsetof(a_t, oc), jcp.oc_block);
    d.n = rvjit_value_t::at(offsetof(a_t, os), n_bound);
    d.k = rvjit_value_t::literal(jcp.ic_without_padding);
    d.lda = rvjit_value_t::literal(
            static_cast<dim_t>(jcp.oc_block) * jcp.typesize_wei);
    d.ldb = rvjit_value_t::literal(jcp.bcast_loop_bcast_step);
    d.ldc = rvjit_value_t::literal(jcp.bcast_loop_output_step);
    d.ptr_a = offsetof(a_t, wei);
    d.ptr_b = offsetof(a_t, src);
    d.ptr_c = offsetof(a_t, dst);
    if (jcp.with_bias) {
        d.features = rvjit_bias;
        d.bias = jcp.bia_dt;
        d.ptr_bias = offsetof(a_t, bias);
    }
}

} // namespace

status_t init_1x1_conv_conf(jit_1x1_conv_conf_t &jcp, rvjit_matmul_desc_t &desc,
        rvjit_matmul_blocks_t &blocks, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, int nthreads) {

    const int ndims = src_d.ndims();

    jcp.prop_kind = cd.prop_kind;
    jcp.nthr = nthreads;

    // src/weights dtype. f32 multiplies in place; a symmetric bf16/f16 src+wei
    // uses widening FMA (Zvfbfwma/Zvfh) into f32 accumulators. For weight
    // compression (f32 src, bf16/f16 wei) the weights are widened to f32 and a
    // plain f32 FMA is used; only the weight element size differs from f32.
    jcp.src_dt = src_d.data_type();
    jcp.wei_dt = weights_d.data_type();
    jcp.dst_dt = dst_d.data_type();

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;
    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;

    // Initialize dimensions
    jcp.mb = src_d.dims()[0];
    jcp.ngroups
            = weights_d.ndims() == src_d.ndims() + 1 ? weights_d.dims()[0] : 1;
    jcp.oc_without_padding = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = src_d.dims()[1] / jcp.ngroups;
    jcp.oc = jcp.oc_without_padding;
    jcp.ic = jcp.ic_without_padding;

    // 3D convolution support
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;

    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];

    // Spatial dimensions: D*H*W
    jcp.os = jcp.od * jcp.oh * jcp.ow;
    jcp.is = jcp.id * jcp.ih * jcp.iw;

    // Inputs (src/weights) may be 2-byte (bf16/f16); the destination
    // accumulates in f32 and stores f32, or f16 narrowed from the f32
    // accumulators (Zvfh).
    jcp.typesize_in = static_cast<int>(types::data_type_size(jcp.src_dt));
    // Weight element size. Equals typesize_in for f32/f32 and symmetric
    // bf16/f16; for weight compression the weights are 2 bytes while src is 4.
    jcp.typesize_wei = static_cast<int>(types::data_type_size(jcp.wei_dt));
    jcp.typesize_out = static_cast<int>(types::data_type_size(jcp.dst_dt));
    // bias element size (f32, or 2 bytes for a bf16/f16 bias widened to f32).
    jcp.typesize_bia = jcp.with_bias
            ? static_cast<int>(types::data_type_size(jcp.bia_dt))
            : 0;

    // Strides between spatial positions (NHWC)
    jcp.bcast_loop_bcast_step
            = jcp.ngroups * jcp.ic_without_padding * jcp.typesize_in;
    jcp.bcast_loop_output_step
            = jcp.ngroups * jcp.oc_without_padding * jcp.typesize_out;

    // rvjit groups the registers and sets the blocks, tuned on the whole OC
    // as one block: the OC block is its M block (the whole OC when shorter,
    // for plain weights) and the spatial chunk its N block
    jcp.oc_block = jcp.oc_without_padding;
    init_1x1_rvjit_desc(jcp, jcp.os, desc);
    CHECK(rvjit_matmul_tune(desc, blocks));

    jcp.oc_block = nstl::min(jcp.oc_without_padding, blocks.m_block());
    jcp.oc = rnd_up(jcp.oc_without_padding, jcp.oc_block);
    jcp.ic_block = jcp.ic_without_padding;
    jcp.ur = blocks.n_block();

    jcp.load_block = jcp.oc_block;
    jcp.reduce_block = jcp.ic_block;

    jcp.bcast_block = jcp.ur;
    jcp.load_dim = jcp.oc_without_padding;
    jcp.bcast_dim = jcp.os;
    jcp.reduce_dim = jcp.ic_without_padding;

    jcp.ur_tail = jcp.bcast_dim % jcp.ur;

    jcp.nb_bcast = div_up(jcp.os, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.oc_without_padding, jcp.load_block);
    jcp.nb_reduce = 1;
    jcp.load_grp_count = 1;

    // Each call covers the whole IC: a low-precision destination cannot hold
    // an intermediate f32 partial sum
    jcp.nb_reduce_blocking = jcp.nb_reduce;
    jcp.nb_load_blocking = jcp.nb_load;
    jcp.nb_load_blocking_max = jcp.nb_load;

    // Spatial dimension blocking (in ur units)
    const int target_bcast_blocking = 735;
    jcp.nb_bcast_blocking
            = nstl::min(jcp.nb_bcast, div_up(target_bcast_blocking, jcp.ur));
    if (jcp.nb_bcast_blocking == 0) jcp.nb_bcast_blocking = 1;
    jcp.nb_bcast_blocking_max = jcp.nb_bcast_blocking;

    // Strides between OC blocks of the weights
    jcp.reduce_loop_bcast_step = jcp.typesize_in;
    jcp.reduce_loop_load_step = jcp.oc_block * jcp.typesize_wei;
    jcp.load_loop_load_step
            = jcp.ic_without_padding * jcp.oc_block * jcp.typesize_wei;
    jcp.load_loop_iter_step = jcp.oc_block;

    // The kernel of one OC block over a spatial chunk, which must keep the
    // tuned blocks
    const rvjit_matmul_blocks_t tuned = blocks;
    init_1x1_rvjit_desc(jcp,
            static_cast<dim_t>(jcp.nb_bcast_blocking_max) * jcp.bcast_block,
            desc);
    CHECK(rvjit_matmul_tune(desc, blocks));
    if (blocks != tuned) return status::unimplemented;

    return status::success;
}

static_assert(std::is_standard_layout<rvjit_1x1_args_t>::value,
        "the kernel reads its arguments at offsetof() their fields");

void jit_rvv_1x1_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    // Address tensors as bytes because their element sizes are runtime
    // configuration parameters.
    auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const auto &scratchpad = ctx.get_scratchpad_grantor();

    parallel(pd()->jcp_.nthr, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src, weights, bias, dst, scratchpad);
    });
}

void jit_rvv_1x1_convolution_fwd_t::execute_forward_thr(const int ithr,
        const int nthr, const char *src, const char *weights, const char *bias,
        char *dst, const memory_tracking::grantor_t &scratchpad) const {

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    // RVV 1x1 convolution uses NHWC layout.
    // Spatial dimensions are collapsed into 'os'.
    // Threading is balanced over (MB * groups * nb_bcast) and (nb_load).

    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;
    int bcast_start {0}, bcast_end {0}, ocb_start {0}, ocb_end {0};

    balance2D(nthr, ithr, work_amount, bcast_start, bcast_end, jcp.nb_load,
            ocb_start, ocb_end, jcp.load_grp_count);

    if (bcast_start >= bcast_end || ocb_start >= ocb_end) return;

    // One call per OC block: consecutive blocks are IC rows apart in the
    // weights, so they do not form one matrix
    auto ker_1x1 = [&](int ocb, int load_step, int n, int g, int osb,
                           int bcast_step) {
        const int ic_off = g * jcp.ic_without_padding;
        const size_t src_off
                = (size_t)n * jcp.is * jcp.ngroups * jcp.ic_without_padding
                + (size_t)osb * jcp.bcast_block * jcp.ngroups
                        * jcp.ic_without_padding
                + ic_off;
        rvjit_1x1_args_t p;
        p.src = src + src_off * jcp.typesize_in;
        p.os = this_block_size(
                osb * jcp.bcast_block, jcp.os, bcast_step * jcp.bcast_block);

        for (int b = ocb; b < ocb + load_step && b < jcp.nb_load; ++b) {
            const int oc_off = g * jcp.oc_without_padding + b * jcp.oc_block;
            const size_t dst_off
                    = (size_t)n * jcp.os * jcp.ngroups * jcp.oc_without_padding
                    + (size_t)osb * jcp.bcast_block * jcp.ngroups
                            * jcp.oc_without_padding
                    + oc_off;
            const size_t wei_off = (size_t)g * jcp.oc * jcp.ic_without_padding
                    + (size_t)b * jcp.ic_without_padding * jcp.oc_block;
            p.wei = weights + wei_off * jcp.typesize_wei;
            p.dst = dst + dst_off * jcp.typesize_out;
            p.bias = bias ? bias + oc_off * jcp.typesize_bia : nullptr;
            p.oc = nstl::min<dim_t>(
                    jcp.oc_block, jcp.oc_without_padding - b * jcp.oc_block);
            (*kernel_)(&p);
        }
    };

    // Loop order: Load -> Bcast -> Reduce (LBR)
    // This order keeps weights in registers/L1 while iterating over spatial.
    for (int ocb = ocb_start; ocb < ocb_end;) {
        int load_step = step(
                jcp.nb_load_blocking, ocb_end - ocb, jcp.nb_load_blocking_max);
        int iwork = bcast_start;
        while (iwork < bcast_end) {
            int n {0}, g {0}, osb {0};
            nd_iterator_init(
                    iwork, n, jcp.mb, g, jcp.ngroups, osb, jcp.nb_bcast);

            int bcast_step = step(jcp.nb_bcast_blocking, bcast_end - iwork,
                    jcp.nb_bcast_blocking_max);
            bcast_step = nstl::min(bcast_step, jcp.nb_bcast - osb);

            ker_1x1(ocb, load_step, n, g, osb, bcast_step);
            iwork += bcast_step;
        }
        ocb += load_step;
    }
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
