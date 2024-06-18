/* Copyright 2022 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/conv_algorithm_picker.h"

#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/cudnn_fused_conv_rewriter.h"
#include "xla/service/gpu/gpu_conv_rewriter.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/service/platform_util.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

class GpuConvAlgorithmPickerTest : public HloTestBase {
 public:
  GpuConvAlgorithmPickerTest() { AutotunerUtil::ClearAutotuneResults(); }

  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
  stream_executor::dnn::VersionInfo GetDnnVersion() {
    return GetDnnVersionInfoOrDefault(backend().default_stream_executor());
  }
};

TEST_F(GpuConvAlgorithmPickerTest, SetAlgorithm) {
  constexpr absl::string_view kHlo = R"(
HloModule module

ENTRY main {
  %arg0 = f32[3,56,56,16]{2,1,0,3} parameter(0)
  %arg1 = f32[3,3,3,64]{2,1,0,3} parameter(1)
  ROOT %conv = f32[54,54,16,64]{1,0,3,2} convolution(%arg0, %arg1), window={size=3x3}, dim_labels=f01b_i01o->01bf
})";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHlo));

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                          PlatformUtil::GetStreamExecutors(platform));
  ASSERT_GT(executors.size(), 0);
  se::StreamExecutor* stream_exec = executors[0];

  const se::GpuComputeCapability& cc = backend()
                                           .default_stream_executor()
                                           ->GetDeviceDescription()
                                           .gpu_compute_capability();
  bool changed = false;
  TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(GpuConvRewriter(cc), m.get()));
  changed = false;
  DebugOptions opts = DefaultDebugOptionsIgnoringFlags();

  AutotuneConfig cfg{DeviceConfig{stream_exec, nullptr}, opts};
  TF_ASSERT_OK_AND_ASSIGN(changed,
                          RunHloPass(GpuConvAlgorithmPicker(cfg), m.get()));
  ASSERT_TRUE(changed);

  AutotuneResults results;
  TF_ASSERT_OK(AutotunerUtil::SerializeAutotuneResults(&results));
  ASSERT_EQ(results.results_size(), 1);
  auto& result = *results.mutable_results(0)->mutable_result();
  int64_t old_scratch_bytes = result.scratch_bytes();
  int64_t new_scratch_bytes = old_scratch_bytes + 1;
  result.set_scratch_bytes(new_scratch_bytes);

  AutotunerUtil::ClearAutotuneResults();
  TF_ASSERT_OK(AutotunerUtil::LoadAutotuneResults(results));

  // Now send the same module through GpuConvAlgorithmPicker again.  The conv
  // should have the new scratch bytes.
  TF_ASSERT_OK_AND_ASSIGN(m, ParseAndReturnVerifiedModule(kHlo));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(GpuConvRewriter(cc), m.get()));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(changed,
                          RunHloPass(GpuConvAlgorithmPicker(cfg), m.get()));
  ASSERT_TRUE(changed);

  // TupleSimplifier cleans this up a bit before we pattern-match
  TF_ASSERT_OK(RunHloPass(TupleSimplifier(), m.get()).status());

  SCOPED_TRACE(m->ToString());
  HloInstruction* conv;
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(m::CustomCall(&conv))));
  EXPECT_THAT(
      conv->shape(),
      GmockMatch(m::Shape().WithSubshape(
          {1}, m::Shape().WithElementType(U8).WithDims({new_scratch_bytes}))));
}

TEST_F(GpuConvAlgorithmPickerTest, SetAlgorithmGraphConvF8) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::HOPPER)) {
    GTEST_SKIP() << "FP8 convolutions require Hopper or newer architecture.";
  }
  constexpr absl::string_view kHlo = R"(
HloModule module

apply {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT c = f32[] maximum(a, b)
}

ENTRY main {
  input = f8e4m3fn[1,6,6,128] parameter(0)
  filter = f8e4m3fn[16,3,3,128] parameter(1)
  input_scale = f32[] parameter(2)
  input_scale_bcast = f32[1,6,6,128] broadcast(input_scale), dimensions={}
  filter_scale = f32[] parameter(3)
  filter_scale_bcast = f32[16,3,3,128] broadcast(filter_scale), dimensions={}
  input_f32 = f32[1,6,6,128] convert(input)
  input_unscaled = f32[1,6,6,128] multiply(input_f32, input_scale_bcast)
  filter_f32 = f32[16,3,3,128] convert(filter)
  filter_unscaled = f32[16,3,3,128] multiply(filter_f32, filter_scale_bcast)
  conv_a = f32[1,6,6,16] convolution(input_unscaled, filter_unscaled), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, feature_group_count=1
  z_scale = f32[] parameter(4)
  z_scale_bcast = f32[1,6,6,16] broadcast(z_scale), dimensions={}
  conv_a_scaled = f32[1,6,6,16] multiply(conv_a, z_scale_bcast)
  c1 = f32[] constant(-448.)
  c1_bcast = f32[1,6,6,16] broadcast(c1), dimensions={}
  c2 = f32[] constant(448.)
  c2_bcast = f32[1,6,6,16] broadcast(c2), dimensions={}
  conv_a_clamped = f32[1,6,6,16] clamp(c1_bcast, conv_a_scaled, c2_bcast)
  conv_a_clamped_f8 = f8e4m3fn[1,6,6,16] convert(conv_a_clamped)
  abs_conv_a = f32[1,6,6,16] abs(conv_a)
  c0 = f32[] constant(-inf)
  amax = f32[] reduce(abs_conv_a, c0), dimensions={0,1,2,3}, to_apply=apply
  ROOT conv_f8 = (f8e4m3fn[1,6,6,16], f32[]) tuple(conv_a_clamped_f8, amax)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHlo));

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                          PlatformUtil::GetStreamExecutors(platform));
  ASSERT_GT(executors.size(), 0);
  se::StreamExecutor* stream_exec = executors[0];

  const se::GpuComputeCapability& cc = GetCudaComputeCapability();
  bool changed = false;
  TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(GpuConvRewriter(cc), m.get()));
  TF_ASSERT_OK_AND_ASSIGN(
      changed, RunHloPass(CudnnFusedConvRewriter(GetCudaComputeCapability(),
                                                 GetDnnVersion(), CUDA_VERSION),
                          m.get()));

  changed = false;
  DebugOptions opts = DefaultDebugOptionsIgnoringFlags();

  AutotuneConfig cfg{DeviceConfig{stream_exec, nullptr}, opts};
  TF_ASSERT_OK_AND_ASSIGN(changed,
                          RunHloPass(GpuConvAlgorithmPicker(cfg), m.get()));
  ASSERT_TRUE(changed);

  AutotuneResults results;
  TF_ASSERT_OK(AutotunerUtil::SerializeAutotuneResults(&results));
  ASSERT_EQ(results.results_size(), 1);
  auto& result = *results.mutable_results(0)->mutable_result();
  int64_t old_scratch_bytes = result.scratch_bytes();
  int64_t new_scratch_bytes = old_scratch_bytes + 1;
  result.set_scratch_bytes(new_scratch_bytes);

  AutotunerUtil::ClearAutotuneResults();
  TF_ASSERT_OK(AutotunerUtil::LoadAutotuneResults(results));

  // Now send the same module through GpuConvAlgorithmPicker again.  The conv
  // should have the new scratch bytes.
  TF_ASSERT_OK_AND_ASSIGN(m, ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(GpuConvRewriter(cc), m.get()));
  TF_ASSERT_OK_AND_ASSIGN(
      changed, RunHloPass(CudnnFusedConvRewriter(GetCudaComputeCapability(),
                                                 GetDnnVersion(), CUDA_VERSION),
                          m.get()));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(changed,
                          RunHloPass(GpuConvAlgorithmPicker(cfg), m.get()));
  ASSERT_TRUE(changed);

  // TupleSimplifier cleans this up a bit before we pattern-match
  TF_ASSERT_OK(RunHloPass(TupleSimplifier(), m.get()).status());

  SCOPED_TRACE(m->ToString());
  HloInstruction* conv;
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::GetTupleElement(m::CustomCall(&conv), 0),
                                  m::GetTupleElement(m::CustomCall(), 1))));
  EXPECT_THAT(
      conv->shape(),
      GmockMatch(m::Shape().WithSubshape(
          {2}, m::Shape().WithElementType(U8).WithDims({new_scratch_bytes}))));
}

}  // namespace
}  // namespace xla::gpu
