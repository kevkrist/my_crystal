#pragma once

#include "../crystal.cuh"
#include "kernel_helpers.cuh"
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/iterator/iterator_traits.h>

namespace crystal
{

template <int32_t BLOCK_THREADS,
          int32_t ITEMS_PER_THREAD,
          typename CounterT,
          typename InputIt,
          typename StencilIt,
          typename Predicate,
          typename OutputIt>
__global__ void SelectKernel(cub::ScanTileState<CounterT> tile_state,
                             InputIt input,
                             StencilIt stencil,
                             Predicate predicate,
                             size_t num_in,
                             OutputIt output,
                             CounterT* num_out)
{
  // Typedefs
  typedef typename thrust::iterator_traits<InputIt>::value_type InputT;
  typedef typename thrust::iterator_traits<StencilIt>::value_type StencilT;
  typedef typename thrust::iterator_traits<OutputIt>::value_type OutputT;
  typedef cub::BlockLoad<StencilT, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>
    BlockLoadStencil;
  typedef cub::BlockScan<int32_t, BLOCK_THREADS> BlockScan;
  typedef cub::BlockStore<OutputT, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED>
    BlockStore;
  typedef BlockFlag<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD, DataArrangement::Blocked> BlockFlag;
  typedef BlockShuffle<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockShuffle;
  typedef BlockLoad<InputT, BLOCK_THREADS, ITEMS_PER_THREAD, DataArrangement::Striped>
    BlockLoadInput;
  typedef cub::Sum ScanOpT;
  typedef cub::ScanTileState<int32_t> ScanTileStateT;
  typedef cub::TilePrefixCallbackOp<int32_t, ScanOpT, ScanTileStateT> TilePrefixOpT;
  typedef KernelConfig<BLOCK_THREADS, ITEMS_PER_THREAD> KernelConfigT;

  // Shared memory
  __shared__ typename BlockLoadStencil::TempStorage temp_load_stencil_storage;
  __shared__ typename BlockScan::TempStorage temp_scan_storage;
  __shared__ typename BlockStore::TempStorage temp_store_storage;
  __shared__ typename TilePrefixOpT::TempStorage temp_prefix_storage;
  __shared__ typename BlockShuffle::TempStorage temp_shuffle_storage;
  __shared__ CounterT write_offset;

  // Thread memory
  StencilT stencil_items[ITEMS_PER_THREAD];
  int32_t flags[ITEMS_PER_THREAD];
  int32_t prefix_sums[ITEMS_PER_THREAD];
  InputT input_items[ITEMS_PER_THREAD];
  int32_t num_selected = 0;
  ScanOpT scan_op{};
  TilePrefixOpT prefix_op(tile_state, temp_prefix_storage, scan_op);
  int32_t block_id = prefix_op.GetTileIdx();
  KernelConfigT config(block_id, num_in);

  // Load stencil items and compute flags
  if (__builtin_expect(!config.is_last_tile, 1))
  {
    BlockLoadStencil(temp_load_stencil_storage).Load(stencil + config.block_offset, stencil_items);
    BlockFlag::SetFlags(stencil_items, predicate, flags);
  }
  else
  {
    BlockLoadStencil(temp_load_stencil_storage)
      .Load(stencil + config.block_offset, stencil_items, config.num_tile_items);
    BlockFlag::InitFlags(flags);
    BlockFlag::SetFlags(stencil_items, predicate, flags, config.num_tile_items);
  }

  // Scan flags
  BlockScan(temp_scan_storage).ExclusiveSum(flags, prefix_sums, num_selected);

  // Decoupled look-back
  if (block_id == 0)
  {
    if (threadIdx.x == 0)
    {
      tile_state.SetInclusive(block_id, num_selected);
      write_offset = 0;
    }
  }
  else
  {
    if ((threadIdx.x / CUB_PTX_WARP_THREADS) == 0)
    {
      // Collect exclusive aggregate from previous tiles and publish inclusive aggregate for
      // subsequent tiles.
      CounterT local_write_offset = prefix_op(num_selected);

      if(threadIdx.x == 0) {
        // Set shared write_offset
        write_offset = local_write_offset;

        // If last tile, set num_out
        if (config.is_last_tile && threadIdx.x == 0)
        {
          *num_out = scan_op(local_write_offset, num_selected);
        }
      }
    }
  }

  // Shuffle flagged row ids (implicit synchronization barrier for write_offset)
  BlockShuffle(temp_shuffle_storage)
    .Shuffle<DataArrangement::Blocked, DataArrangement::Striped, ShuffleOperator::SetBlockItems>(
      flags,
      flags,
      prefix_sums,
      num_selected);

  // Gathering load
  BlockLoadInput::Gather(input + config.block_offset, flags, input_items, num_selected);

  // Write compacted data
  BlockStore(temp_store_storage).Store(output + write_offset, input_items, num_selected);
}

template <typename CounterT>
__global__ void SelectInitKernel(cub::ScanTileState<CounterT> tile_state, int32_t num_tiles)
{
  tile_state.InitializeStatus(num_tiles);
}

struct DeviceSelect
{
  template <int32_t BLOCK_THREADS,
            int32_t ITEMS_PER_THREAD,
            typename InputIt,
            typename StencilIt,
            typename Predicate,
            typename OutputIt,
            typename CounterT>
  static void Select(uint8_t* temp_storage,
                     size_t& temp_storage_bytes,
                     InputIt input,
                     StencilIt stencil,
                     Predicate predicate,
                     size_t num_in,
                     OutputIt output,
                     CounterT& num_out)
  {
    // Typedefs
    typedef cub::ScanTileState<CounterT> ScanTileStateT;

    // Execution configuration
    int32_t num_tiles = cub::DivideAndRoundUp(num_in, BLOCK_THREADS * ITEMS_PER_THREAD);

    // Following cub API, if temp_storage is null, return with required allocation size
    if (temp_storage == nullptr)
    {
      ScanTileStateT::AllocationSize(num_tiles, temp_storage_bytes);
      return;
    }

    // Execution configuration for initialization
    int32_t num_tiles_init = cub::DivideAndRoundUp(num_tiles, BLOCK_THREADS);

    // Initialize temporary storage
    ScanTileStateT tile_state{};
    tile_state.Init(num_tiles, temp_storage, temp_storage_bytes);
    SelectInitKernel<<<num_tiles_init, BLOCK_THREADS>>>(tile_state, num_tiles);

    // Execute compaction kernel
    thrust::device_vector<CounterT> num_out_device(1);
    SelectKernel<BLOCK_THREADS, ITEMS_PER_THREAD>
      <<<num_tiles, BLOCK_THREADS>>>(tile_state,
                                     input,
                                     stencil,
                                     predicate,
                                     num_in,
                                     output,
                                     thrust::raw_pointer_cast(num_out_device.data()));

    // Copy num out
    num_out = num_out_device[0];
  }
};

} // namespace crystal