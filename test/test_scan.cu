#include "crystal.cuh"
#include <cub/cub.cuh>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, typename IterT>
__global__ void TestScan(IterT in, size_t num_in, IterT out)
{
  typedef typename thrust::iterator_traits<IterT>::value_type T;
  typedef crystal::BlockScan<T, BLOCK_THREADS, ITEMS_PER_THREAD> BlockScan;
  typedef cub::BlockLoad<T, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> BlockLoad;
  typedef cub::BlockStore<T, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED> BlockStore;

  __shared__ typename BlockLoad::TempStorage load_temp_storage;
  __shared__ typename BlockStore::TempStorage store_temp_storage;

  T thread_items[ITEMS_PER_THREAD];
  T thread_scans[ITEMS_PER_THREAD];
  T scan_result = 0;

  BlockLoad(load_temp_storage).Load(in, thread_items);
  BlockScan::ExclusiveSum(thread_items, thread_scans, scan_result);
  BlockStore(store_temp_storage).Store(out, thread_scans);

  if (threadIdx.x == 0)
  {
    printf("Result: %d\n", INT32_C(scan_result));
  }
}

int main()
{
  constexpr int32_t block_threads    = 128;
  constexpr int32_t items_per_thread = 4;
  constexpr int32_t num_items        = block_threads * items_per_thread;
  thrust::device_vector<int32_t> input(num_items, 1);
  thrust::device_vector<int32_t> output(num_items);

  // Generate expected output
  thrust::device_vector<int32_t> expected_output(num_items);
  thrust::exclusive_scan(input.begin(), input.end(), expected_output.begin());

  // Run test kernel
  TestScan<block_threads, items_per_thread>
    <<<1, block_threads>>>(input.begin(), num_items, output.begin());

  // Compare
  thrust::host_vector<int32_t> expected_output_host = expected_output;
  thrust::host_vector<int32_t> output_host          = output;
  for (auto i = 0; i < num_items; ++i)
  {
    if (expected_output_host[i] != output_host[i])
    {
      std::cerr << "Index: " << i << "\n\tExpected: " << expected_output_host[i]
                << "\n\tActual: " << output_host[i] << "\n";
    }
  }

  return 0;
}
