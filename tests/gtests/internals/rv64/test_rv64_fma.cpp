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

#include <cstdint>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "common/c_types_map.hpp"
#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/rvjit/rvjit.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;
using namespace rvjit;

// Invalid combinations: all emit nothing, leaving only ret().

struct macc_vv_noop_probe_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(macc_vv_noop_probe_t)
    macc_vv_noop_probe_t() : jit_generator_t("rv64_macc_vv_noop_probe") {}

protected:
    void generate() override {
        using namespace data_type;

        rvjit_t m(*this);
        auto &mc = m.macc();

        const auto vd = v8;
        const auto vs1 = v16;
        const auto vs2 = v24;

        // None of these should emit anything
        mc.fmacc_int(vd, vs1, vs2, fma_t(s64, u8, s8)); // 8x widen
        mc.fmacc_int(vd, vs1, vs2, fma_t::widening(s8)); // widen from 8 to 16
        mc.fmacc_float(vd, vs1, vs2, fma_t::uniform(bf16)); // no uniform bf16
        mc.fmacc_float(vd, vs1, vs2, fma_t(f16, f32, f32)); // no narrowing

        ret();
    }
};

HANDLE_EXCEPTIONS_FOR_TEST(MacVv, InvalidParametersEmitNothing) {
    macc_vv_noop_probe_t k;
    ASSERT_EQ(k.create_kernel(), status::success);
    ASSERT_EQ(k.getSize(), sizeof(uint32_t));
}

// Valid combinations: probe and reference instructions interleaved.
// Each pair at (code[2i], code[2i+1]) must be identical.

struct macc_vv_valid_probe_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(macc_vv_valid_probe_t)
    macc_vv_valid_probe_t() : jit_generator_t("rv64_macc_vv_valid_probe") {}

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &mc = m.macc();
        // uniform int
        mc.fmacc_int(v8, v16, v24, fma_t::uniform(data_type::s8));
        vmacc_vv(v8, v16, v24);
        mc.fmacc_int(v8, v16, v24, fma_t::uniform(data_type::u8));
        vmacc_vv(v8, v16, v24);
        mc.fmacc_int(v8, v16, v24, fma_t::uniform(data_type::s32));
        vmacc_vv(v8, v16, v24);

        // widening int
        mc.fmacc_int(v8, v16, v24, fma_t::widening(data_type::s32));
        vwmacc_vv(v8, v16, v24);

        // uniform float
        mc.fmacc_float(v8, v16, v24, fma_t::uniform(data_type::f16));
        vfmacc_vv(v8, v16, v24);
        mc.fmacc_float(v8, v16, v24, fma_t::uniform(data_type::f32));
        vfmacc_vv(v8, v16, v24);
        mc.fmacc_float(v8, v16, v24, fma_t::uniform(data_type::f64));
        vfmacc_vv(v8, v16, v24);

        // widening float
        mc.fmacc_float(v8, v16, v24, fma_t::widening(data_type::f16));
        vfwmacc_vv(v8, v16, v24);
        mc.fmacc_float(v8, v16, v24, fma_t::widening(data_type::f32));
        vfwmacc_vv(v8, v16, v24);
        mc.fmacc_float(v8, v16, v24, fma_t::widening(data_type::bf16));
        vfwmaccbf16_vv(v8, v16, v24);

        ret();
    }
};

HANDLE_EXCEPTIONS_FOR_TEST(MacVv, ValidParametersEmitExpectedInstructions) {
    macc_vv_valid_probe_t k;
    ASSERT_EQ(k.create_kernel(), status::success);
    constexpr int n_pairs = 10;
    const auto *code = k.getCode<const uint32_t *>();
    ASSERT_GT(k.getSize(), n_pairs * 2 * sizeof(uint32_t));
    for (int i = 0; i < n_pairs; ++i)
        EXPECT_EQ(code[2 * i], code[2 * i + 1]) << "mismatch at pair " << i;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
