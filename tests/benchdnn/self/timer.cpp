/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "self/self.hpp"

namespace self {

static int check_timer_single_input() {
    timer::timer_t t;
    t.start();
    t.stop(2, 2);
    t.finalize_results();

    timer::timer_t tt;
    tt.start();
    tt.stop(1, 1);
    tt.start();
    tt.stop(1, 1);
    tt.finalize_results();

    for (auto m : {timer_mode_t::min, timer_mode_t::max, timer_mode_t::avg,
                 timer_mode_t::sum}) {
        SELF_CHECK_EQ(t.ms(m), tt.ms(m));
    }
    return OK;
}

static int check_timer_multiple_inputs() {
    timer::timer_t t;
    t.start();
    for (double ms : {1, 2, 4, 8}) {
        t.stop(1, ms);
    }
    t.finalize_results();
    SELF_CHECK_EQ(t.ms(timer_mode_t::min), 1);
    SELF_CHECK_EQ(t.ms(timer_mode_t::max), 8);
    SELF_CHECK_EQ(t.ms(timer_mode_t::sum), 15);
    SELF_CHECK_EQ(t.ms(timer_mode_t::avg), 3.75);

    return OK;
}

static int check_timer_outliers() {
    timer::timer_t t;
    t.start();
    t.stop(5, 5);
    t.stop(95, 95 * 2);
    t.finalize_results(0.05);
    SELF_CHECK_EQ(t.ms(timer_mode_t::min), 2);
    SELF_CHECK_EQ(t.ms(timer_mode_t::max), 2);
    SELF_CHECK_EQ(t.ms(timer_mode_t::sum), 95 * 2);
    SELF_CHECK_EQ(t.ms(timer_mode_t::avg), 2);

    return OK;
}

void timers() {
    RUN(check_timer_single_input());
    RUN(check_timer_multiple_inputs());
    RUN(check_timer_outliers());
}

} // namespace self
