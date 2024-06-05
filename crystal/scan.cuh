#pragma once

#include "helpers.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cuda/atomic>

namespace cg = cooperative_groups;

namespace crystal
{

// Only for striped arrangements!
template <typename T, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
struct BlockScan
{
  static __device__ __forceinline__ void
  ExclusiveSum(T (&thread_input)[ITEMS_PER_THREAD], T (&thread_output)[ITEMS_PER_THREAD], T& result)
  {
    auto block = cg::tiled_partition<BLOCK_THREADS>(cg::this_thread_block());

    thread_output[0] = cg::exclusive_scan(block, thread_input[0]);

#pragma unroll
    for (int32_t ITEM = 1; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      thread_output[ITEM] = block.shfl(thread_output[ITEM - 1], block.num_threads() - 1) +
                            block.shfl(thread_input[ITEM - 1], block.num_threads() - 1) +
                            cg::exclusive_scan(block, thread_input[ITEM]);
    }

    result = block.shfl(thread_output[ITEMS_PER_THREAD - 1], block.num_threads() - 1) +
             block.shfl(thread_input[ITEMS_PER_THREAD - 1], block.num_threads() - 1);
  }
};

} // namespace crystal