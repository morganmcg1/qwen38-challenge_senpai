// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/resident.h"
#include "mlx/backend/metal/device.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

namespace mlx::core::metal {

// E130 rung 10 admission probe. RESEARCH ONLY, REVERTED BEFORE SUBMISSION.
//
// Every call site below runs under `MetalAllocator::mutex_`, so reading
// `unwired_set_` and `wired_set_` here is race free without adding a lock to
// the allocation path.
namespace {

constexpr uint64_t kE130DumpIntervalNs = 2'000'000'000;
constexpr int kE130TopN = 16;

std::FILE* e130_sink() {
  static std::FILE* sink = [] () -> std::FILE* {
    const char* path = std::getenv("MLX_E130_ADMISSION_PROBE_PATH");
    if (path == nullptr || *path == '\0') {
      return nullptr;
    }
    return std::fopen(path, "a");
  }();
  return sink;
}

uint64_t e130_now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

// Log2 buckets, so "one large object" and "many small objects" are
// distinguishable without naming a single tensor.
void e130_write_sizes(std::FILE* sink, const char* label,
                      std::vector<size_t>& sizes) {
  std::sort(sizes.begin(), sizes.end(), std::greater<size_t>());
  std::fprintf(sink, " %s_top=", label);
  for (int i = 0; i < kE130TopN && i < static_cast<int>(sizes.size()); ++i) {
    std::fprintf(sink, "%s%zu", i == 0 ? "" : ",", sizes[i]);
  }
  size_t buckets[40] = {0};
  for (size_t s : sizes) {
    int b = 0;
    while (b < 39 && (static_cast<size_t>(1) << (b + 1)) <= s) {
      ++b;
    }
    buckets[b] += 1;
  }
  std::fprintf(sink, " %s_hist=", label);
  bool first = true;
  for (int b = 0; b < 40; ++b) {
    if (buckets[b] == 0) {
      continue;
    }
    std::fprintf(sink, "%s2^%d:%zu", first ? "" : ",", b, buckets[b]);
    first = false;
  }
}

} // namespace

void ResidencySet::e130_dump(const char* phase) {
  std::FILE* sink = e130_sink();
  if (sink == nullptr || !wired_set_) {
    return;
  }
  e130_last_dump_ns_ = e130_now_ns();

  std::vector<size_t> wired_sizes;
  auto allocations = wired_set_->allAllocations();
  auto wired_count = allocations == nullptr
      ? static_cast<NS::UInteger>(0)
      : wired_set_->allocationCount();
  wired_sizes.reserve(wired_count);
  for (NS::UInteger i = 0; i < wired_count; ++i) {
    auto buf = static_cast<const MTL::Allocation*>(allocations->object(i));
    wired_sizes.push_back(buf->allocatedSize());
  }
  size_t wired_sum = 0;
  for (size_t s : wired_sizes) {
    wired_sum += s;
  }

  std::vector<size_t> unwired_sizes;
  unwired_sizes.reserve(unwired_set_.size());
  size_t unwired_sum = 0;
  for (const MTL::Allocation* buf : unwired_set_) {
    size_t s = buf->allocatedSize();
    unwired_sizes.push_back(s);
    unwired_sum += s;
  }

  std::fprintf(sink,
               "e130-admission pid=%d phase=%s capacity=%zu"
               " wired_count=%zu wired_bytes_reported=%zu wired_bytes_sum=%zu"
               " unwired_count=%zu unwired_bytes=%zu total_bytes=%zu",
               getpid(), phase, capacity_,
               static_cast<size_t>(wired_count),
               static_cast<size_t>(wired_set_->allocatedSize()), wired_sum,
               unwired_set_.size(), unwired_sum, wired_sum + unwired_sum);
  e130_write_sizes(sink, "unwired", unwired_sizes);
  e130_write_sizes(sink, "wired", wired_sizes);
  std::fprintf(sink, "\n");
  std::fflush(sink);
}

ResidencySet::ResidencySet(MTL::Device* d) {
  if (!d->supportsFamily(MTL::GPUFamilyMetal3)) {
    return;
  } else if (__builtin_available(macOS 15, iOS 18, *)) {
    auto pool = new_scoped_memory_pool();
    auto desc = MTL::ResidencySetDescriptor::alloc()->init()->autorelease();
    NS::Error* error;
    wired_set_ = NS::TransferPtr(d->newResidencySet(desc, &error));
    if (!wired_set_) {
      std::ostringstream msg;
      msg << "[metal::Device] Unable to construct residency set.\n";
      if (error) {
        msg << error->localizedDescription()->utf8String() << "\n";
      }
      throw std::runtime_error(msg.str());
    }
    wired_set_->requestResidency();
  }
}

void ResidencySet::insert(MTL::Allocation* buf) {
  if (!wired_set_) {
    return;
  }
  if (wired_set_->allocatedSize() + buf->allocatedSize() <= capacity_) {
    wired_set_->addAllocation(buf);
    wired_set_->commit();
  } else {
    unwired_set_.insert(buf);
  }
  // E130 rung 10 admission probe, research only.
  if (capacity_ > 0 && e130_now_ns() - e130_last_dump_ns_ > kE130DumpIntervalNs) {
    e130_dump("steady");
  }
}

void ResidencySet::erase(MTL::Allocation* buf) {
  if (!wired_set_) {
    return;
  }
  if (auto it = unwired_set_.find(buf); it != unwired_set_.end()) {
    unwired_set_.erase(it);
  } else {
    wired_set_->removeAllocation(buf);
    wired_set_->commit();
  }
}

void ResidencySet::resize(size_t size) {
  if (!wired_set_) {
    return;
  }

  if (capacity_ == size) {
    return;
  }
  capacity_ = size;

  size_t current_size = wired_set_->allocatedSize();

  if (current_size < size) {
    auto pool = new_scoped_memory_pool();
    // Add unwired allocations to the set
    for (auto it = unwired_set_.begin(); it != unwired_set_.end();) {
      auto buf_size = (*it)->allocatedSize();
      if (current_size + buf_size > size) {
        it++;
      } else {
        current_size += buf_size;
        wired_set_->addAllocation(*it);
        unwired_set_.erase(it++);
      }
    }
    wired_set_->commit();
  } else if (current_size > size) {
    auto pool = new_scoped_memory_pool();
    // Remove wired allocations until under capacity
    auto allocations = wired_set_->allAllocations();
    auto num_allocations = wired_set_->allocationCount();
    for (int i = 0; i < num_allocations && current_size > size; ++i) {
      auto buf = static_cast<const MTL::Allocation*>(allocations->object(i));
      wired_set_->removeAllocation(buf);
      current_size -= buf->allocatedSize();
      unwired_set_.insert(buf);
    }
    wired_set_->commit();
  }

  // E130 rung 10 admission probe, research only. This is the greedy fill the
  // lottery hypothesis is about, so it is dumped unconditionally.
  e130_dump("resize");
}

ResidencySet::~ResidencySet() = default;

} // namespace mlx::core::metal
