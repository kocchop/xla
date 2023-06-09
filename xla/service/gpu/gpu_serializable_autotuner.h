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
#ifndef XLA_SERVICE_GPU_GPU_SERIALIZABLE_AUTOTUNER_H_
#define XLA_SERVICE_GPU_GPU_SERIALIZABLE_AUTOTUNER_H_

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

#include "xla/autotune_results.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/stream_executor/stream_executor_pimpl.h"
#include "xla/types.h"
#include "tsl/protobuf/autotuning.pb.h"

namespace xla {
namespace gpu {

struct DeviceConfig {
  se::StreamExecutor* stream_exec;  // never null

  // If the `allocator` parameter is not null, we will use it to allocate temp
  // memory while timing the various convolution algorithms.  If it's null,
  // we'll use the default allocator on the StreamExecutor.
  se::DeviceMemoryAllocator* allocator;  // may be null
};

struct DevicelessConfig {
  // The human-readable description of the device.  It can be found by using
  // stream_exec->GetDeviceDescription().model_str() when the stream executor
  // is available.
  std::string model_str;

  // A field to determine the architecture of the device. We only pick an
  // algorithm for non-Ampere architectures.
  se::CudaComputeCapability cuda_compute_capability{0, 0};
};

struct AutotuningConfig : public std::variant<DeviceConfig, DevicelessConfig> {
  using std::variant<DeviceConfig, DevicelessConfig>::variant;
  bool is_offline() const {
    return std::holds_alternative<DevicelessConfig>(*this);
  }
  bool is_online() const { return std::holds_alternative<DeviceConfig>(*this); }
};

using AutotuneCacheKey =
    std::tuple<std::string /* stream_exec->GetDeviceDescription().model_str()*/,
               std::string /* instr->ToString(HloPrintOptions::Canonical()) */>;

using AutotuneCacheMap =
    absl::flat_hash_map<AutotuneCacheKey, tensorflow::AutotuneResult>;

inline AutotuneCacheKey AutotuneCacheKeyFromInstruction(
    const HloInstruction* instr, absl::string_view model_str) {
  auto options = HloPrintOptions::Canonical();
  options.set_print_backend_config(true);
  return std::make_tuple(std::string(model_str), instr->ToString(options));
}

Status SerializeAutotuneResults(const AutotuneCacheMap& autotune_cache,
                                AutotuneResults* results);

Status LoadAutotuneResults(AutotuneCacheMap& autotune_cache,
                           const AutotuneResults& results);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_SERIALIZABLE_AUTOTUNER_H_
