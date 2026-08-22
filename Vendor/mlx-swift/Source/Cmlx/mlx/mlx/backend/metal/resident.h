// Copyright © 2024 Apple Inc.

#pragma once

#include <cstdint>
#include <unordered_set>

#include <Metal/Metal.hpp>

namespace mlx::core::metal {

class ResidencySet {
 public:
  ResidencySet(MTL::Device* d);
  ~ResidencySet();

  ResidencySet(const ResidencySet&) = delete;
  ResidencySet& operator=(const ResidencySet&) = delete;

  const MTL::ResidencySet* mtl_residency_set() {
    return wired_set_.get();
  }

  void insert(MTL::Allocation* buf);
  void erase(MTL::Allocation* buf);

  void resize(size_t size);

 private:
  // E130 rung 10 admission probe. RESEARCH ONLY, REVERTED BEFORE SUBMISSION.
  // `resident.h` and `resident.cpp` are not in benchmark.json editablePaths,
  // so nothing here can reach a submitted archive, but the local worker binary
  // stops matching any submittable candidate while it is present. Never time
  // a leg against a build that carries it.
  //
  // Reports which allocations the greedy fill admitted and which it left in
  // `unwired_set_`, with page-rounded byte counts, so the page-rounding tax
  // is measured instead of bounded. Enabled by MLX_E130_ADMISSION_PROBE_PATH.
  void e130_dump(const char* phase);
  uint64_t e130_last_dump_ns_{0};

  NS::SharedPtr<MTL::ResidencySet> wired_set_;
  std::unordered_set<const MTL::Allocation*> unwired_set_;
  size_t capacity_{0};
};

} // namespace mlx::core::metal
