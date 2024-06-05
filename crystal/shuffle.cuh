#pragma once

#include "loop.cuh"

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

    Looper::Execute([&input, &flag_op, &flags] __device__(auto THREAD_ITEM) -> void {
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

    Looper::Execute(
      [&input, &flag_op, &flags] __device__(auto THREAD_ITEM) -> void {
        flags[THREAD_ITEM] = flag_op(input[THREAD_ITEM]);
      },
      num_items);
  }
};

template <typename T, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, DataArrangement ArrangementT>
struct BlockShuffle
{
  T* shared_items;

  struct TempStorage
  {
    T temp_storage[BLOCK_THREADS * ITEMS_PER_THREAD];
  };

  __device__ BlockShuffle(TempStorage& storage)
      : shared_items{storage.temp_storage}
  {}

  template <typename FlagT, typename OffsetT>
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

    Looper::Execute(
      [&thread_items, &thread_offsets, &shared_items] __device__(auto THREAD_ITEM) -> void {
        shared_items[thread_offsets[THREAD_ITEM]] = thread_items[THREAD_ITEM];
      },
      thread_flags);
  }

  template <typename FlagT, typename OffsetT>
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
    Looper::Execute(
      [&thread_items, thread_offset, &shared_items, &counter] __device__(auto THREAD_ITEM) -> void {
        shared_items[thread_offset + counter++] = thread_items[THREAD_ITEM];
      },
      thread_flags);
  }

  static __device__ __forceinline__ void
  ShuffleBackward(T (&thread_items)[ITEMS_PER_THREAD],
                  T (&shared_items)[BLOCK_THREADS * ITEMS_PER_THREAD],
                  int32_t num_items)
  {
    typedef LoopExecutor<ITEMS_PER_THREAD,
                         ThreadAndBlockLoop<BLOCK_THREADS, ArrangementT>,
                         LoopType::Guarded>
      Looper;

    Looper::Execute(
      [&thread_items, &shared_items] __device__(auto THREAD_ITEM, auto BLOCK_ITEM) -> void {
        thread_items[THREAD_ITEM] = shared_items[BLOCK_ITEM];
      },
      num_items);
  }

  template <typename FlagT, typename OffsetT>
  __device__ __forceinline__ void Shuffle(T (&thread_items)[ITEMS_PER_THREAD],
                                          FlagT (&thread_flags)[ITEMS_PER_THREAD],
                                          OffsetT (&thread_offsets)[ITEMS_PER_THREAD],
                                          int32_t num_items)
  {
    ShuffleForward(thread_items, thread_flags, thread_offsets, shared_items);
    __syncthreads();
    ShuffleBackward(thread_items, shared_items, num_items);
  }

  template <typename FlagT, typename OffsetT>
  __device__ __forceinline__ void Shuffle(T (&thread_items)[ITEMS_PER_THREAD],
                                          FlagT (&thread_flags)[ITEMS_PER_THREAD],
                                          OffsetT thread_offset,
                                          int32_t num_items)
  {
    ShuffleForward(thread_items, thread_flags, thread_offset, shared_items);
    __syncthreads();
    ShuffleBackward(thread_items, shared_items, num_items);
  }
};

} // namespace crystal
