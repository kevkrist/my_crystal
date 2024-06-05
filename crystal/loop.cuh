#pragma once

#include "helpers.cuh"

namespace crystal
{
  /*
   * Loop policies.
   */
  template<int32_t BLOCK_THREADS, DataArrangement ArrangementT> struct ThreadLoop {
    template<int32_t ITEMS_PER_THREAD, typename OperatorT>
    static __device__ __forceinline__ void Execute(const OperatorT& op)
    {
#pragma unroll
      for (int32_t THREAD_ITEM = 0; THREAD_ITEM < ITEMS_PER_THREAD; ++THREAD_ITEM)
      {
        op(THREAD_ITEM);
      }
    }

    template<int32_t ITEMS_PER_THREAD, typename OperatorT>
    static __device__ __forceinline__ void ExecuteGuarded(const OperatorT& op, int32_t num_items)
    {
#pragma unroll
      for (int32_t THREAD_ITEM = 0; THREAD_ITEM < ITEMS_PER_THREAD; ++THREAD_ITEM)
      {
        if (ComputeBlockId<BLOCK_THREADS, ITEMS_PER_THREAD, ArrangementT>(THREAD_ITEM) < num_items)
        {
          op(THREAD_ITEM);
        }
      }
    }

    template<int32_t ITEMS_PER_THREAD, typename OperatorT, typename FlagT>
    static __device__ __forceinline__ void ExecuteFlagged(const OperatorT& op,
                                                          FlagT (&thread_flags)[ITEMS_PER_THREAD])
    {
#pragma unroll
      for (int32_t THREAD_ITEM = 0; THREAD_ITEM < ITEMS_PER_THREAD; ++THREAD_ITEM)
      {
        if (thread_flags[THREAD_ITEM]) { op(THREAD_ITEM); }
      }
    }
  };

  template<int32_t BLOCK_THREADS, DataArrangement ArrangementT> struct BlockLoop {
    template<int32_t ITEMS_PER_THREAD, typename OperatorT>
    static __device__ __forceinline__ void Execute(const OperatorT& op)
    {
#pragma unroll
      for (int32_t THREAD_ITEM = 0; THREAD_ITEM < ITEMS_PER_THREAD; ++THREAD_ITEM)
      {
        op(ComputeBlockId<BLOCK_THREADS, ITEMS_PER_THREAD, ArrangementT>(THREAD_ITEM));
      }
    }

    template<int32_t ITEMS_PER_THREAD, typename OperatorT>
    static __device__ __forceinline__ void ExecuteGuarded(const OperatorT& op, int32_t num_items)
    {
#pragma unroll
      for (int32_t THREAD_ITEM = 0; THREAD_ITEM < ITEMS_PER_THREAD; ++THREAD_ITEM)
      {
        int32_t BLOCK_ITEM =
          ComputeBlockId<BLOCK_THREADS, ITEMS_PER_THREAD, ArrangementT>(THREAD_ITEM);
        if (BLOCK_ITEM < num_items) { op(BLOCK_ITEM); }
      }
    }

    template<int32_t ITEMS_PER_THREAD, typename OperatorT, typename FlagT>
    static __device__ __forceinline__ void ExecuteFlagged(const OperatorT& op,
                                                          FlagT (&thread_flags)[ITEMS_PER_THREAD])
    {
#pragma unroll
      for (int32_t THREAD_ITEM = 0; THREAD_ITEM < ITEMS_PER_THREAD; ++THREAD_ITEM)
      {
        if (thread_flags[THREAD_ITEM])
        {
          op(ComputeBlockId<BLOCK_THREADS, ITEMS_PER_THREAD, ArrangementT>(THREAD_ITEM));
        }
      }
    }
  };

  template<int32_t BLOCK_THREADS, DataArrangement ArrangementT> struct ThreadAndBlockLoop {
    template<int32_t ITEMS_PER_THREAD, typename OperatorT>
    static __device__ __forceinline__ void Execute(const OperatorT& op)
    {
#pragma unroll
      for (int32_t THREAD_ITEM = 0; THREAD_ITEM < ITEMS_PER_THREAD; ++THREAD_ITEM)
      {
        op(THREAD_ITEM, ComputeBlockId<BLOCK_THREADS, ITEMS_PER_THREAD, ArrangementT>(THREAD_ITEM));
      }
    }

    template<int32_t ITEMS_PER_THREAD, typename OperatorT>
    static __device__ __forceinline__ void ExecuteGuarded(const OperatorT& op, int32_t num_items)
    {
#pragma unroll
      for (int32_t THREAD_ITEM = 0; THREAD_ITEM < ITEMS_PER_THREAD; ++THREAD_ITEM)
      {
        int32_t BLOCK_ITEM =
          ComputeBlockId<BLOCK_THREADS, ITEMS_PER_THREAD, ArrangementT>(THREAD_ITEM);
        if (BLOCK_ITEM < num_items) { op(THREAD_ITEM, BLOCK_ITEM); }
      }
    }

    template<int32_t ITEMS_PER_THREAD, typename OperatorT, typename FlagT>
    static __device__ __forceinline__ void ExecuteFlagged(const OperatorT& op,
                                                          FlagT (&thread_flags)[ITEMS_PER_THREAD])
    {
#pragma unroll
      for (int32_t THREAD_ITEM = 0; THREAD_ITEM < ITEMS_PER_THREAD; ++THREAD_ITEM)
      {
        if (thread_flags[THREAD_ITEM])
        {
          op(THREAD_ITEM,
             ComputeBlockId<BLOCK_THREADS, ITEMS_PER_THREAD, ArrangementT>(THREAD_ITEM));
        }
      }
    }
  };

  /*
   * Loop executor, following a given loop policy, specialized by loop type.
   */
  template<int32_t ITEMS_PER_THREAD, typename LoopPolicy, LoopType LoopT> struct LoopExecutor {
  };

  template<int32_t ITEMS_PER_THREAD, typename LoopPolicy>
  struct LoopExecutor<ITEMS_PER_THREAD, LoopPolicy, LoopType::Direct> {
    template<typename OperatorT> static __device__ __forceinline__ void Loop(const OperatorT& op)
    {
      LoopPolicy::template Execute<ITEMS_PER_THREAD>(op);
    }
  };

  template<int32_t ITEMS_PER_THREAD, typename LoopPolicy>
  struct LoopExecutor<ITEMS_PER_THREAD, LoopPolicy, LoopType::Guarded> {
    template<typename OperatorT>
    static __device__ __forceinline__ void Loop(const OperatorT& op, int32_t num_items)
    {
      LoopPolicy::template ExecuteGuarded<ITEMS_PER_THREAD>(op, num_items);
    }
  };

  template<int32_t ITEMS_PER_THREAD, typename LoopPolicy>
  struct LoopExecutor<ITEMS_PER_THREAD, LoopPolicy, LoopType::Flagged> {
    template<typename OperatorT, typename FlagT>
    static __device__ __forceinline__ void Loop(const OperatorT& op,
                                                FlagT (&thread_flags)[ITEMS_PER_THREAD])
    {
      LoopPolicy::template ExecuteFlagged<ITEMS_PER_THREAD>(op, thread_flags);
    }
  };

}// namespace crystal