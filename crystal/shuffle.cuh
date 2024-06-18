#pragma once

#include "loop.cuh"
#include "store.cuh"

namespace crystal
{

template <typename FlagT,
          int32_t BLOCK_THREADS,
          int32_t ITEMS_PER_THREAD,
          DataArrangement ArrangementT>
struct BlockFlag
{

  static __device__ __forceinline__ void InitFlags(FlagT (&thread_flags)[ITEMS_PER_THREAD])
  {
#pragma unroll
    for (int32_t ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
      thread_flags[ITEM] = 0;
    }
  }

  template <typename T, typename FlagOp>
  static __device__ __forceinline__ void
  SetFlags(T (&input)[ITEMS_PER_THREAD], FlagOp& flag_op, FlagT (&flags)[ITEMS_PER_THREAD])
  {
    typedef LoopExecutor<ITEMS_PER_THREAD,
                         ThreadLoop<BLOCK_THREADS, ArrangementT>,
                         LoopType::Direct>
      Looper;

    Looper::Loop([&input, &flag_op, &flags] __device__(auto THREAD_ITEM) -> void {
      flags[THREAD_ITEM] = flag_op(input[THREAD_ITEM]);
    });
  }

  template <typename T, typename FlagOp>
  static __device__ __forceinline__ void SetFlags(T (&input)[ITEMS_PER_THREAD],
                                                  FlagOp& flag_op,
                                                  FlagT (&flags)[ITEMS_PER_THREAD],
                                                  int32_t num_items)
  {
    typedef LoopExecutor<ITEMS_PER_THREAD,
                         ThreadLoop<BLOCK_THREADS, ArrangementT>,
                         LoopType::Guarded>
      Looper;

    Looper::Loop([&input, &flag_op, &flags] __device__(
                   auto THREAD_ITEM) -> void { flags[THREAD_ITEM] = flag_op(input[THREAD_ITEM]); },
                 num_items);
  }
};

enum class ShuffleOperator
{
  SetBlockItems,
  ShuffleThreadItems,
};

template <typename T, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
struct BlockShuffle
{

  struct TempStorage
  {
    T temp_storage[BLOCK_THREADS * ITEMS_PER_THREAD];
  };

  T (&shared_items)[BLOCK_THREADS * ITEMS_PER_THREAD];

  __device__ explicit BlockShuffle(TempStorage& storage)
      : shared_items{storage.temp_storage}
  {}

  template <DataArrangement ArrangementT, typename FlagT, typename OffsetT>
  static __device__ __forceinline__ void
  ShuffleForward(T (&thread_items)[ITEMS_PER_THREAD],
                 FlagT (&thread_flags)[ITEMS_PER_THREAD],
                 OffsetT (&thread_offsets)[ITEMS_PER_THREAD],
                 T (&shared_items)[BLOCK_THREADS * ITEMS_PER_THREAD])
  {
    typedef LoopExecutor<ITEMS_PER_THREAD,
                         ThreadLoop<BLOCK_THREADS, ArrangementT>,
                         LoopType::Flagged>
      Looper;

    Looper::Loop(
      [&thread_items, &thread_offsets, &shared_items] __device__(auto THREAD_ITEM) -> void {
        shared_items[thread_offsets[THREAD_ITEM]] = thread_items[THREAD_ITEM];
      },
      thread_flags);
  }

  template <DataArrangement ArrangementT, typename FlagT, typename OffsetT>
  static __device__ __forceinline__ void
  ShuffleForward(FlagT (&thread_flags)[ITEMS_PER_THREAD],
                 OffsetT (&thread_offsets)[ITEMS_PER_THREAD],
                 T (&shared_items)[BLOCK_THREADS * ITEMS_PER_THREAD])
  {
    typedef LoopExecutor<ITEMS_PER_THREAD,
                         ThreadAndBlockLoop<BLOCK_THREADS, ArrangementT>,
                         LoopType::Flagged>
      Looper;

    Looper::Loop(
      [&thread_offsets, &shared_items] __device__(auto THREAD_ITEM, auto BLOCK_ITEM) -> void {
        shared_items[thread_offsets[THREAD_ITEM]] = BLOCK_ITEM;
      },
      thread_flags);
  }

  template <DataArrangement ArrangementT, typename FlagT, typename OffsetT>
  static __device__ __forceinline__ void
  ShuffleForward(T (&thread_items)[ITEMS_PER_THREAD],
                 FlagT (&thread_flags)[ITEMS_PER_THREAD],
                 OffsetT thread_offset,
                 T (&shared_items)[BLOCK_THREADS * ITEMS_PER_THREAD])
  {
    typedef LoopExecutor<ITEMS_PER_THREAD,
                         ThreadLoop<BLOCK_THREADS, ArrangementT>,
                         LoopType::Flagged>
      Looper;

    int32_t counter = 0;
    Looper::Loop(
      [&thread_items, thread_offset, &shared_items, &counter] __device__(auto THREAD_ITEM) -> void {
        shared_items[thread_offset + counter++] = thread_items[THREAD_ITEM];
      },
      thread_flags);
  }

  template <DataArrangement ArrangementT, typename FlagT, typename OffsetT>
  static __device__ __forceinline__ void
  ShuffleForward(FlagT (&thread_flags)[ITEMS_PER_THREAD],
                 OffsetT thread_offset,
                 T (&shared_items)[BLOCK_THREADS * ITEMS_PER_THREAD])
  {
    typedef LoopExecutor<ITEMS_PER_THREAD,
                         ThreadAndBlockLoop<BLOCK_THREADS, ArrangementT>,
                         LoopType::Flagged>
      Looper;

    int32_t counter = 0;
    Looper::Loop(
      [thread_offset, &shared_items, &counter] __device__(auto THREAD_ITEM, auto BLOCK_ITEM)
        -> void { shared_items[thread_offset + counter++] = BLOCK_ITEM; },
      thread_flags);
  }

  template <DataArrangement ArrangementT>
  static __device__ __forceinline__ void
  ShuffleBackward(T (&thread_items)[ITEMS_PER_THREAD],
                  T (&shared_items)[BLOCK_THREADS * ITEMS_PER_THREAD],
                  int32_t num_items)
  {
    typedef LoopExecutor<ITEMS_PER_THREAD,
                         ThreadAndBlockLoop<BLOCK_THREADS, ArrangementT>,
                         LoopType::Guarded>
      Looper;

    Looper::Loop(
      [&thread_items, &shared_items] __device__(auto THREAD_ITEM, auto BLOCK_ITEM) -> void {
        thread_items[THREAD_ITEM] = shared_items[BLOCK_ITEM];
      },
      num_items);
  }

  /**
   * The Shuffle operator compacts a tile of data in ForwardArrangementT, filtered by flags, into
   * a sequential tile of data in BackwardArrangmentT. Shared memory of size
   *    size(T) * ITEMS_PER_THREAD * BLOCK_THREADS
   * is required. This overload assumes the offsets are stored in an array corresponding to the
   * items to shuffle.
   * @tparam ForwardArrangementT The arrangement of the data before compaction
   * @tparam BackwardArrangementT The arrangement of the data after compaction
   * @tparam ShuffleOpT The shuffle operator type (among {SetBlockItems, ShuffleThreadItems}).
   * @tparam FlagT The type of the thread flags
   * @tparam OffsetT The type the offsets (likely prefix sums) at which to store items in shared
   * memory
   * @param thread_items The items to shuffle
   * @param thread_flags The selection flags
   * @param thread_offsets The offsets at which to store items in shared memory
   * @param num_items The number of valid tile items to shuffle (necessary for backward shuffle)
   */
  template <DataArrangement ForwardArrangementT,
            DataArrangement BackwardArrangementT,
            ShuffleOperator ShuffleOpT,
            typename FlagT,
            typename OffsetT>
  __device__ __forceinline__ void Shuffle(T (&thread_items)[ITEMS_PER_THREAD],
                                          FlagT (&thread_flags)[ITEMS_PER_THREAD],
                                          OffsetT (&thread_offsets)[ITEMS_PER_THREAD],
                                          int32_t num_items)
  {
    if constexpr (ShuffleOpT == ShuffleOperator::SetBlockItems)
    {
      ShuffleForward<ForwardArrangementT>(thread_flags, thread_offsets, shared_items);
    }
    else
    {
      ShuffleForward<ForwardArrangementT>(thread_items, thread_flags, thread_offsets, shared_items);
    }
    __syncthreads();
    ShuffleBackward<BackwardArrangementT>(thread_items, shared_items, num_items);
  }

  /**
   * The Shuffle operator compacts a tile of data in ForwardArrangementT, filtered by flags, into
   * a sequential tile of data in BackwardArrangmentT. Shared memory of size
   *    size(T) * ITEMS_PER_THREAD * BLOCK_THREADS
   * is required. This overload assumes the offset is simply the starting location in shared memory
   * at which each thread is to begin writing its local flagged items.
   * @tparam ForwardArrangementT The arrangement of the data before compaction
   * @tparam BackwardArrangementT The arrangement of the data after compaction
   * @tparam ShuffleOpT The shuffle operator type (among {SetBlockItems, ShuffleThreadItems}).
   * @tparam FlagT The type of the thread flags
   * @tparam OffsetT The type of the offset (likely a prefix sum) at which to begin storing items
   * in shared memory
   * @param thread_items The items to shuffle
   * @param thread_flags The selection flags
   * @param thread_offset The offset at which to begin storing items in shared memory
   * @param num_items The number of valid tile items to shuffle (necessary for backward shuffle)
   */
  template <DataArrangement ForwardArrangementT,
            DataArrangement BackwardArrangementT,
            ShuffleOperator ShuffleOpT,
            typename FlagT,
            typename OffsetT>
  __device__ __forceinline__ void Shuffle(T (&thread_items)[ITEMS_PER_THREAD],
                                          FlagT (&thread_flags)[ITEMS_PER_THREAD],
                                          OffsetT thread_offset,
                                          int32_t num_items)
  {
    if constexpr (ShuffleOpT == ShuffleOperator::SetBlockItems)
    {
      ShuffleForward<ForwardArrangementT>(thread_flags, thread_offset, shared_items);
    }
    else
    {
      ShuffleForward<ForwardArrangementT>(thread_items, thread_flags, thread_offset, shared_items);
    }
    __syncthreads();
    ShuffleBackward<BackwardArrangementT>(thread_items, shared_items, num_items);
  }

  /**
   * Shuffles items into shared memory and stores directly from shared memory, saving a
   * ShuffleBackward operation.
   * @tparam ArrangementT The arrangement of data in the tile
   * @tparam OutputIteratorT The type of the output iterator to which to store
   * @tparam FlagT The type of the flags used to indicate items to shuffle
   * @tparam OffsetT The type of the offset by which to increment the output iterator
   * @param block_itr The output iterator to which to store
   * @param thread_items The items to shuffle and store
   * @param thread_flags The flags indicating items to shuffle and store
   * @param thread_offsets The offsets at which to store for each item
   * @param num_items The number of items to store
   */
  template <DataArrangement ArrangementT,
            typename OutputIteratorT,
            typename FlagT,
            typename OffsetT>
  __device__ __forceinline__ void ShuffleStore(OutputIteratorT block_itr,
                                               T (&thread_items)[ITEMS_PER_THREAD],
                                               FlagT (&thread_flags)[ITEMS_PER_THREAD],
                                               OffsetT (&thread_offsets)[ITEMS_PER_THREAD],
                                               int32_t num_items)
  {
    typedef BlockStore<T, BLOCK_THREADS, ITEMS_PER_THREAD, ArrangementT> BlockStore;

    ShuffleForward<ArrangementT>(thread_items, thread_flags, thread_offsets, shared_items);
    __syncthreads();
    BlockStore::StoreShared(block_itr, shared_items, num_items);
  }

  /**
   * Shuffles (implicit) row ids into shared memory and stores directly from shared memory, saving a
   * ShuffleBackward operation.
   * @tparam ArrangementT The arrangement of data in the tile
   * @tparam OutputIteratorT The type of the output iterator to which to store.
   * @tparam FlagT The type of the flags used to indicate items to shuffle.
   * @tparam OffsetT The type of the offset by which to increment the output iterator.
   * @param block_itr The output iterator to which to store.
   * @param thread_flags The flags indicating items to shuffle and store.
   * @param thread_offsets The offsets at which to store for each item.
   * @param num_items The number of items to store.
   */
  template <DataArrangement ArrangementT,
            typename OutputIteratorT,
            typename FlagT,
            typename OffsetT>
  __device__ __forceinline__ void ShuffleStore(OutputIteratorT block_itr,
                                               FlagT (&thread_flags)[ITEMS_PER_THREAD],
                                               OffsetT (&thread_offsets)[ITEMS_PER_THREAD],
                                               int32_t num_items)
  {
    typedef BlockStore<T, BLOCK_THREADS, ITEMS_PER_THREAD, ArrangementT> BlockStore;

    ShuffleForward<ArrangementT>(thread_flags, thread_offsets, shared_items);
    __syncthreads();
    BlockStore::StoreShared(block_itr, shared_items, num_items);
  }
};

} // namespace crystal
