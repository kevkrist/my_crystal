#pragma once

namespace crystal
{

template <int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
struct KernelConfig
{
  int32_t num_tile_items;
  int32_t block_offset;
  bool is_last_tile;

  __device__ __forceinline__ KernelConfig(int32_t BLOCK_ID, size_t num_in)
  {
    block_offset   = BLOCK_ID * BLOCK_THREADS * ITEMS_PER_THREAD;
    is_last_tile   = block_offset + BLOCK_THREADS * ITEMS_PER_THREAD >= num_in;
    num_tile_items = is_last_tile ? num_in - block_offset : BLOCK_THREADS * ITEMS_PER_THREAD;
  }
};
} // namespace crystal