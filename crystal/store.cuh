#pragma once

namespace crystal
{
template <typename T, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, DataArrangement ArrangementT>
struct BlockStore
{
  template <typename OutputIteratorT>
  static __device__ __forceinline__ void Store(OutputIteratorT block_itr,
                                               T (&thread_items)[ITEMS_PER_THREAD])
  {
    typedef LoopExecutor<ITEMS_PER_THREAD,
                         ThreadAndBlockLoop<BLOCK_THREADS, ArrangementT>,
                         LoopType::Direct>
      Looper;

    Looper::Loop([block_itr, &thread_items] __device__(auto THREAD_ITEM, auto BLOCK_ITEM) -> void {
      block_itr[BLOCK_ITEM] = thread_items[THREAD_ITEM];
    });
  }

  template <typename OutputIteratorT>
  static __device__ __forceinline__ void
  Store(OutputIteratorT block_itr, T (&thread_items)[ITEMS_PER_THREAD], int32_t num_items)
  {
    typedef LoopExecutor<ITEMS_PER_THREAD,
                         ThreadAndBlockLoop<BLOCK_THREADS, ArrangementT>,
                         LoopType::Guarded>
      Looper;

    Looper::Loop(
      [block_itr, &thread_items] __device__(auto THREAD_ITEM, auto BLOCK_ITEM) -> void {
        block_itr[BLOCK_ITEM] = thread_items[THREAD_ITEM];
      },
      num_items);
  }

  template <typename OutputIteratorT>
  static __device__ __forceinline__ void
  StoreShared(OutputIteratorT block_itr,
              T (&shared_items)[BLOCK_THREADS * ITEMS_PER_THREAD],
              int32_t num_items)
  {
    typedef LoopExecutor<ITEMS_PER_THREAD,
                         BlockLoop<BLOCK_THREADS, ArrangementT>,
                         LoopType::Guarded>
      Looper;

    Looper::Loop([block_itr, &shared_items] __device__(
                   auto BLOCK_ITEM) -> void { block_itr[BLOCK_ITEM] = shared_items[BLOCK_ITEM]; },
                 num_items);
  }
};
} // namespace crystal
