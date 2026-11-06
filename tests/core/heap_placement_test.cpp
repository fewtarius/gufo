#include "src/core/hip/heap_placement.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

void Expect(bool condition, const char* message) {
  if (condition)
    return;
  std::cerr << message << '\n';
  std::abort();
}

}  // namespace

int main() {
  using gufo::hip::ChooseWeightPlacement;
  using gufo::hip::DispatchReserve;
  using gufo::hip::HeapBudget;
  using gufo::hip::HeapFitsAboveReserve;
  using gufo::hip::kDispatchReserve;
  using gufo::hip::Placement;
  constexpr std::uint64_t kGiB = 1ull << 30;

  Expect(kDispatchReserve == 4 * kGiB, "dispatch reserve is 4 GiB");

  // A heap with room keeps the flat reserve.
  Expect(DispatchReserve(16 * kGiB) == kDispatchReserve,
         "a heap with room keeps the flat reserve");
  Expect(HeapFitsAboveReserve(16 * kGiB, 12 * kGiB),
         "exactly the reserve remains acceptable");
  Expect(!HeapFitsAboveReserve(16 * kGiB, 12 * kGiB + 1),
         "one byte under the reserve is rejected");
  Expect(!HeapFitsAboveReserve(10, 11), "a request above free is rejected");

  // A heap too small for the flat reserve keeps a quarter of what it has.
  // The flat reserve alone rejected every size there, which is what made
  // the context clamp collapse to its floor on a 26 GiB APU.
  constexpr std::uint64_t kTight = 2 * kGiB;
  Expect(DispatchReserve(kTight) == kTight / 4,
         "a tight heap keeps a quarter of what is free");
  Expect(HeapBudget(kTight) == kTight * 3 / 4,
         "the budget is what is left after the reserve");
  Expect(HeapFitsAboveReserve(kTight, kTight * 3 / 4),
         "three quarters of a tight heap fits");
  Expect(!HeapFitsAboveReserve(kTight, kTight * 3 / 4 + 1),
         "one byte over the tight budget is rejected");

  // Measured heaps on a 128 GiB Strix Halo: the stock ttm.pages_limit of
  // half RAM, and ttm.pages_limit=32505856 (124 GiB).
  constexpr std::uint64_t kStockHeap = 67147403264ull;
  constexpr std::uint64_t kPagesLimitHeap = 133143986176ull;
  constexpr std::uint64_t kModel = 104 * kGiB;
  constexpr std::uint64_t kSmallModel = 30 * kGiB;

  Expect(ChooseWeightPlacement(kStockHeap, kModel) == Placement::kManaged,
         "a model larger than the stock heap is managed");
  Expect(ChooseWeightPlacement(kPagesLimitHeap, kModel) == Placement::kMalloc,
         "a model that fits the enlarged heap uses hipMalloc");
  Expect(ChooseWeightPlacement(kStockHeap, kSmallModel) == Placement::kMalloc,
         "a model that fits the stock heap uses hipMalloc");
  Expect(ChooseWeightPlacement(kModel + kDispatchReserve, kModel) ==
             Placement::kMalloc,
         "a model leaving exactly the reserve uses hipMalloc");
  Expect(ChooseWeightPlacement(kModel + kDispatchReserve - 1, kModel) ==
             Placement::kManaged,
         "a model eating into the reserve is managed");
  Expect(ChooseWeightPlacement(0, 0) == Placement::kManaged,
         "an empty heap is managed");

  // A 96 GiB BIOS carve-out. Flash-Next Q4 + MTP uploads about 80 GiB of
  // tensors; its shard files total about 106 GiB because the n-gram table
  // stays on disk. Sizing from the files would wrongly go managed.
  constexpr std::uint64_t kCarveOut = 96 * kGiB;
  constexpr std::uint64_t kUploaded = 85464330752ull;
  constexpr std::uint64_t kShardFiles = 114121223040ull;
  Expect(ChooseWeightPlacement(kCarveOut, kUploaded) == Placement::kMalloc,
         "the uploaded tensors fit a 96 GiB carve-out above the reserve");
  Expect(ChooseWeightPlacement(kCarveOut, kShardFiles) == Placement::kManaged,
         "the shard-file total would not fit that carve-out");

  std::cout << "Weight placement checks passed.\n";
}
