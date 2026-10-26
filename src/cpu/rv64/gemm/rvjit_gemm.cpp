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

#include <cstddef>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
#include <type_traits>

#include "common/utils.hpp"

#include "cpu/rv64/gemm/rvjit_gemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

namespace {

static_assert(std::is_standard_layout<rvjit_gemm_args_t>::value,
        "the kernel reads its arguments at offsetof() their fields");

// The kernel of `conf` over strips of at most `m_bound` rows, any number when
// 0.
rvjit_matmul_desc_t make_desc(const rvjit_gemm_conf_t &conf, dim_t m_bound) {
    using a_t = rvjit_gemm_args_t;
    rvjit_matmul_desc_t d;
    d.a = conf.a;
    d.b = conf.b;
    d.c = conf.c;
    d.epilogue = conf.epilogue;
    d.bias = data_type::f32;
    d.bias_kind = conf.bias_kind;
    d.trans_a = conf.trans_a;
    d.trans_b = conf.trans_b;

    d.m = rvjit_value_t::at(offsetof(a_t, m), m_bound);
    d.n = rvjit_value_t::at(offsetof(a_t, n), conf.n_bound);
    d.k = rvjit_value_t::at(offsetof(a_t, k), conf.k_bound);
    d.lda = rvjit_value_t::at(offsetof(a_t, lda));
    d.ldb = rvjit_value_t::at(offsetof(a_t, ldb));
    d.ldc = rvjit_value_t::at(offsetof(a_t, ldc));
    d.ptr_a = offsetof(a_t, a);
    d.ptr_b = offsetof(a_t, b);
    d.ptr_c = offsetof(a_t, c);
    d.ptr_bias = offsetof(a_t, bias);

    // An integer epilogue adds C unscaled when its flag says so; a float one
    // reads both scalars
    d.features = rvjit_beta | rvjit_bias;
    if (conf.int_beta) {
        d.beta = 1.0;
    } else {
        d.features |= rvjit_alpha;
        d.alpha_arg = offsetof(a_t, alpha);
        d.beta_arg = offsetof(a_t, beta);
    }
    d.rt_flags = offsetof(a_t, rt_flags);
    return d;
}

data_type_t reg_of(const rvjit_type_t &t) {
    return t.reg == data_type::undef ? t.mem : t.reg;
}

using conf_key_t = std::tuple<int, int, int, int, int, int, int, bool, bool,
        int, bool, dim_t, dim_t>;

conf_key_t key_of(const rvjit_gemm_conf_t &c) {
    return conf_key_t(c.a.mem, reg_of(c.a), c.b.mem, reg_of(c.b), c.c.mem,
            reg_of(c.c), c.epilogue, c.trans_a, c.trans_b,
            static_cast<int>(c.bias_kind), c.int_beta, c.n_bound, c.k_bound);
}

uint64_t bits_of(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

} // namespace

void rvjit_gemm_t::operator()(
        rvjit_gemm_args_t &args, float alpha, float beta) const {
    args.alpha = bits_of(alpha);
    args.beta = bits_of(beta);
    args.rt_flags = rvjit_runtime_flags(
            alpha != 1.0f, beta != 0.0f, args.bias != nullptr);
    (*kernel)(&args);
}

rvjit_gemm_conf_t rvjit_gemm_f32_conf() {
    rvjit_gemm_conf_t conf;
    conf.a = conf.b = conf.c = rvjit_type_t(data_type::f32);
    return conf;
}

rvjit_gemm_conf_t rvjit_gemm_hp_conf(data_type_t dt, data_type_t dst_dt) {
    rvjit_gemm_conf_t conf;
    conf.a = conf.b = rvjit_type_t(dt);
    conf.c = rvjit_type_t(dst_dt, data_type::f32);
    return conf;
}

rvjit_gemm_conf_t rvjit_gemm_int8_conf(data_type_t a_dt, data_type_t b_dt,
        data_type_t dst_dt, bool bias_is_scalar) {
    rvjit_gemm_conf_t conf;
    conf.a = rvjit_type_t(a_dt, data_type::s32);
    conf.b = rvjit_type_t(b_dt, data_type::s32);
    conf.c = rvjit_type_t(dst_dt, data_type::s32);
    conf.int_beta = dst_dt == data_type::s32;
    conf.epilogue = conf.int_beta ? data_type::undef : data_type::f32;
    conf.bias_kind = bias_is_scalar ? rvjit_bias_kind_t::scalar
                                    : rvjit_bias_kind_t::per_m;
    return conf;
}

status_t rvjit_gemm_tune(
        const rvjit_gemm_conf_t &conf, rvjit_matmul_blocks_t &blocks) {
    return rvjit_matmul_tune(make_desc(conf, 0), blocks);
}

status_t rvjit_gemm_init(const rvjit_gemm_conf_t &conf, rvjit_gemm_t &g) {
    g.status = rvjit_gemm_tune(conf, g.blocks);
    if (g.status != status::success) return g.status;
    g.status = rvjit_matmul_kernel_create(
            make_desc(conf, g.m_block()), g.blocks, g.kernel);
    return g.status;
}

const rvjit_gemm_t &rvjit_gemm(const rvjit_gemm_conf_t &conf) {
    static std::mutex mutex;
    static std::map<conf_key_t, std::unique_ptr<rvjit_gemm_t>> cache;

    std::lock_guard<std::mutex> guard(mutex);
    std::unique_ptr<rvjit_gemm_t> &slot = cache[key_of(conf)];
    if (slot) return *slot;

    slot.reset(new rvjit_gemm_t);
    rvjit_gemm_init(conf, *slot);
    return *slot;
}

status_t rvjit_gemm_strips_t::init(
        rvjit_gemm_conf_t conf, bool trans_a, const rvjit_gemm_t *kernel) {
    pack_all = kernel && trans_a;
    if (kernel) {
        trans_a_ker = nontrans_a_ker = kernel;
    } else {
        conf.trans_a = false;
        nontrans_a_ker = &rvjit_gemm(conf);
        conf.trans_a = true;
        trans_a_ker = trans_a ? &rvjit_gemm(conf) : nontrans_a_ker;
    }
    CHECK(trans_a_ker->status);
    return nontrans_a_ker->status;
}

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
