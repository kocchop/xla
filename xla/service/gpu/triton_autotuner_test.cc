/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/triton_autotuner.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gemm_rewriter_triton.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/tests/verified_hlo_module.h"
#include "xla/xla.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/protobuf/autotuning.pb.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

using HloExtractionTest = HloTestBase;

TEST_F(HloExtractionTest, InstructionExtractionIsCorrect) {
  std::unique_ptr<VerifiedHloModule> module = ParseAndReturnVerifiedModule(R"(
HloModule module

triton_gemm_dot {
  p0 = s8[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  c0 = f32[10,10] convert(p0)
  ROOT dot.0 = f32[10,10] dot(c0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = s8[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  s = f32[10,10] sqrt(p1)
  d = f32[10,10] fusion(p0, p1),
    kind=kCustom, calls=triton_gemm_dot
  ROOT r = f32[10,10] add(d, s)
})")
                                                  .value();

  std::unique_ptr<HloModule> extracted_module = ExtractInstructionIntoNewModule(
      *module->entry_computation()->root_instruction()->operand(0));

  // Destroy the original module to be sure that the extracted one has no
  // dependency on it.
  module.release();

  EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
  EXPECT_EQ(extracted_module->entry_computation()->instruction_count(), 3);
  TF_EXPECT_OK(VerifyHloModule(extracted_module.get(),
                               /*layout_sensitive=*/true,
                               /*allow_mixed_precision=*/false));
}

TEST_F(HloExtractionTest, ComputationExtractionIsCorrect) {
  std::unique_ptr<VerifiedHloModule> module = ParseAndReturnVerifiedModule(R"(
HloModule module

triton_gemm_dot {
  p0 = s8[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  c0 = f32[10,10] convert(p0)
  ROOT dot.0 = f32[10,10] dot(c0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = s8[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  s = f32[10,10] sqrt(p1)
  d = f32[10,10] fusion(p0, p1),
    kind=kCustom, calls=triton_gemm_dot
  ROOT r = f32[10,10] add(d, s)
})")
                                                  .value();

  std::unique_ptr<HloModule> extracted_module =
      ExtractComputationIntoNewModule(*module->entry_computation()
                                           ->root_instruction()
                                           ->operand(0)
                                           ->fused_instructions_computation());

  // Destroy the original module to be sure that the extracted one has no
  // dependency on it.
  module.release();

  EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
              GmockMatch(m::Dot(m::Convert(m::Parameter()), m::Parameter())));
  EXPECT_EQ(extracted_module->entry_computation()->instruction_count(), 4);
  TF_EXPECT_OK(VerifyHloModule(extracted_module.get(),
                               /*layout_sensitive=*/true,
                               /*allow_mixed_precision=*/false));
}

class TritonAutotunerTest : public HloTestBase {
 public:
  TritonAutotunerTest()
      : HloTestBase(/*verifier_layout_sensitive=*/true,
                    /*allow_mixed_precision_in_hlo_verifier=*/false) {}

  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_triton_gemm(true);
    return debug_options;
  }

  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }

  void CheckTritonAutotuning(absl::string_view hlo,
                             absl::string_view expected) {
    HloPassPipeline pipeline("gemm_rewrite");
    pipeline.AddPass<GemmRewriterTriton>(backend()
                                             .default_stream_executor()
                                             ->GetDeviceDescription()
                                             .cuda_compute_capability());
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "",
                                        tsl::port::MaxParallelism());
    pipeline.AddPass<TritonAutotuner>(
        DeviceConfig{backend().default_stream_executor(),
                     backend().memory_allocator()},
        &thread_pool);

    RunAndFilecheckHloRewrite(
        hlo, std::move(pipeline), expected, [](const HloModule* m) {
          VLOG(5) << m->ToString();
          const HloInstruction* dot_fusion =
              m->entry_computation()->root_instruction();
          if (dot_fusion->opcode() == HloOpcode::kReduce) {
            dot_fusion = dot_fusion->operand(0);
          }
          CHECK_EQ(dot_fusion->opcode(), HloOpcode::kFusion);
          CHECK_GT(dot_fusion->backend_config<xla::gpu::FusionBackendConfig>()
                       .value()
                       .triton_gemm_config()
                       .block_m(),
                   0);
        });
  }
};

TEST_F(TritonAutotunerTest, VoltaUsesNoMoreThanTwoStages) {
  const se::CudaComputeCapability compute_capability{
      se::CudaComputeCapability::VOLTA, /*minor=*/0};
  const std::vector<tensorflow::AutotuneResult::TritonGemmKey> configs =
      GetPossibleMatmulAutotuneConfigs(compute_capability);
  EXPECT_FALSE(
      std::any_of(configs.begin(), configs.end(),
                  [](const tensorflow::AutotuneResult::TritonGemmKey& key) {
                    return key.num_stages() > 2;
                  }));
}

TEST_F(TritonAutotunerTest, AmpereUsesMoreThanTwoStages) {
  const se::CudaComputeCapability compute_capability{
      se::CudaComputeCapability::AMPERE, /*minor=*/0};
  const std::vector<tensorflow::AutotuneResult::TritonGemmKey> configs =
      GetPossibleMatmulAutotuneConfigs(compute_capability);
  EXPECT_TRUE(
      std::any_of(configs.begin(), configs.end(),
                  [](const tensorflow::AutotuneResult::TritonGemmKey& key) {
                    return key.num_stages() > 2;
                  }));
}

TEST_F(TritonAutotunerTest, Int8FusedGemm) {
  const std::string hlo = R"(
HloModule module

ENTRY e {
  x = s8[128,64] parameter(0)
  c = f16[128,64] convert(x)

  y = f16[64,6144] parameter(1)

  ROOT out = f16[128,6144] dot(c, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  CheckTritonAutotuning(hlo, R"(
// CHECK:   %triton_gemm_out_computation (
// CHECK:   ROOT %out.1 = f16[128,6144]{1,0} dot(%c.1, %parameter_1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
// CHECK:   ROOT %triton_gemm_out = f16[128,6144]{1,0} fusion(%x, %y), kind=kCustom, calls=%triton_gemm_out_computation
// CHECK-SAME: "block_m":
)");

  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{/*aabs=*/5e-3, /*arel=*/5e-3}));
}

TEST_F(TritonAutotunerTest, Int8FusedGemm256) {
  const std::string hlo = R"(
HloModule module

ENTRY e {
  x = s8[128,256] parameter(0)
  c = f16[128,256] convert(x)

  y = f16[256,6144] parameter(1)

  ROOT out = f16[128,6144] dot(c, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  CheckTritonAutotuning(hlo, R"(
// CHECK:   %triton_gemm_out_computation (
// CHECK:   ROOT %out.1 = f16[128,6144]{1,0} dot(%c.1, %parameter_1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
// CHECK:   ROOT %triton_gemm_out = f16[128,6144]{1,0} fusion(%x, %y), kind=kCustom, calls=%triton_gemm_out_computation
// CHECK-SAME: "block_m":
)");

  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(TritonAutotunerTest, SelectsSplitK) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "No BF16 before Ampere.";
  }
  // Shapes with K >> M, N have to force split-K configurations.
  const std::string kHloText = R"(
HloModule t

ENTRY e {
  p0 = s8[7,8192] parameter(0)
  p0c = bf16[7,8192] convert(p0)
  p1 = bf16[8192,18] parameter(1)
  ROOT dot.0 = bf16[7,18] dot(p0c, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: reduce
; CHECK: fusion(%p0, %p1), kind=kCustom
; CHECK: ROOT %fusion.1
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/0.5, /*arel=*/1e-1}));
}

// TODO(b/281489442): Write a testcase called
// `SkipConfigsProducingDeviantResults` or similar.

class TritonAutotunerLevelTest : public HloTestBase,
                                 public ::testing::WithParamInterface<int> {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_autotune_level(GetParam());
    return debug_options;
  }
};

TEST_P(TritonAutotunerLevelTest, AllAutotuningLevelsWorkCorrectly) {
  const std::string kHloText = R"(
HloModule m

ENTRY e {
  p0 = pred[64,10] parameter(0)
  p0c = f32[64,10] convert(p0)
  p1 = f32[10,128] parameter(1)
  ROOT r = f32[64,128] dot(p0c, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  TritonAutotuner::ClearAutotuneResults();

  if (GetDebugOptionsForTest().xla_gpu_autotune_level() == 0) {
    MatchOptimizedHlo(kHloText, R"(
; CHECK: kind=kCustom
; CHECK-NOT: block_m
      )");
  } else {
    MatchOptimizedHlo(kHloText, R"(
; CHECK: kind=kCustom
; CHECK-SAME: block_m
      )");
  }

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

INSTANTIATE_TEST_SUITE_P(TritonAutotunerLevelSweep, TritonAutotunerLevelTest,
                         ::testing::Range(0, 5));

class TritonAutotunerExhaustiveTest : public TritonAutotunerTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_triton_gemm(true);
    debug_options.set_xla_gpu_exhaustive_tiling_search(true);
    return debug_options;
  }
};

TEST_F(TritonAutotunerExhaustiveTest, CompileOnly) {
  const std::string hlo = R"(
HloModule module

ENTRY e {
  x = s8[16,16] parameter(0)
  c = f16[16,16] convert(x)
  y = f16[16,16] parameter(1)
  ROOT out = f16[16,16] dot(c, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  CheckTritonAutotuning(hlo, R"(
// CHECK:   %triton_gemm_out_computation (
// CHECK:   ROOT %out.1 = f16[16,16]{1,0} dot(%c.1, %parameter_1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
// CHECK:   ROOT %triton_gemm_out = f16[16,16]{1,0} fusion(%x, %y), kind=kCustom, calls=%triton_gemm_out_computation
// CHECK-SAME: "block_m":
)");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
