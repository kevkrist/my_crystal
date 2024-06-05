#pragma once

#include "helpers.cuh"
#include "loop.cuh"

namespace crystal
{

/**
 * The BlockLoad class provides static methods for loading data specialized to different data
 * arrangements and algorithms.
 * @tparam InputT The input type to load.
 * @tparam BLOCK_THREADS The number of threads in the CTA / thread block
 * @tparam ITEMS_PER_THREAD The coarsening factor, or the number of items for each thread.
 * @tparam ArrangementT The data arrangement (Striped or Blocked).
 */
template <typename InputT,
          int32_t BLOCK_THREADS,
          int32_t ITEMS_PER_THREAD,
          DataArrangement ArrangementT>
struct BlockLoad
{
  /**
   * Loads a tile of data.
   * @tparam InputIteratorT
   * @param block_itr
   * @param items
   */
  template <typename InputIteratorT>
  static __device__ __forceinline__ void Load(InputIteratorT block_itr,
                                              InputT (&items)[ITEMS_PER_THREAD])
  {
    typedef LoopExecutor<ITEMS_PER_THREAD,
                         ThreadAndBlockLoop<BLOCK_THREADS, ArrangementT>,
                         LoopType::Direct>
      Looper;

    Looper::Execute([block_itr, &items] __device__(auto THREAD_ITEM, auto BLOCK_ITEM) -> void {
      items[THREAD_ITEM] = block_itr[BLOCK_ITEM];
    });
  }

  template <typename InputIteratorT>
  static __device__ __forceinline__ void
  Load(InputIteratorT block_itr, InputT (&items)[ITEMS_PER_THREAD], int32_t num_items)
  {
    typedef LoopExecutor<ITEMS_PER_THREAD,
                         ThreadAndBlockLoop<BLOCK_THREADS, ArrangementT>,
                         LoopType::Guarded>
      Looper;

    Looper::Execute(
      [block_itr, &items] __device__(auto THREAD_ITEM, auto BLOCK_ITEM) -> void {
        items[THREAD_ITEM] = block_itr[BLOCK_ITEM];
      },
      num_items);
  }

  template <typename InputIteratorT, typename OffsetT>
  static __device__ __forceinline__ void Gather(InputIteratorT block_itr,
                                                OffsetT (&offsets)[ITEMS_PER_THREAD],
                                                InputT (&items)[ITEMS_PER_THREAD])
  {
    typedef LoopExecutor<ITEMS_PER_THREAD,
                         ThreadLoop<BLOCK_THREADS, ArrangementT>,
                         LoopType::Direct>
      Looper;

    Looper::Execute([block_itr, &offsets, &items] __device__(auto THREAD_ITEM) -> void {
      items[THREAD_ITEM] = block_itr[offsets[THREAD_ITEM]];
    });
  }

  template <typename InputIteratorT, typename OffsetT>
  static __device__ __forceinline__ void Gather(InputIteratorT block_itr,
                                                OffsetT (&offsets)[ITEMS_PER_THREAD],
                                                InputT (&items)[ITEMS_PER_THREAD],
                                                int32_t num_items)
  {
    typedef LoopExecutor<ITEMS_PER_THREAD,
                         ThreadLoop<BLOCK_THREADS, ArrangementT>,
                         LoopType::Guarded>
      Looper;

    Looper::Execute(
      [block_itr, &offsets, &items] __device__(auto THREAD_ITEM) -> void {
        items[THREAD_ITEM] = block_itr[offsets[THREAD_ITEM]];
      },
      num_items);
  }

  template <typename InputIteratorT, typename FlagT>
  static __device__ __forceinline__ void PredLoad(InputIteratorT block_itr,
                                                  InputT (&items)[ITEMS_PER_THREAD],
                                                  FlagT (&flags)[ITEMS_PER_THREAD])
  {
    typedef LoopExecutor<ITEMS_PER_THREAD,
                         ThreadAndBlockLoop<BLOCK_THREADS, ArrangementT>,
                         LoopType::Flagged>
      Looper;

    Looper::Execute(
      [block_itr, &items] __device__(auto THREAD_ITEM, auto BLOCK_ITEM) -> void {
        items[THREAD_ITEM] = block_itr[BLOCK_ITEM];
      },
      flags);
  }
};

} // namespace crystal