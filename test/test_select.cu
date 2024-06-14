#include "crystal.cuh"
#include <iostream>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

constexpr int32_t block_threads       = 64;
constexpr int32_t items_per_thread    = 2;
constexpr int32_t num_tiles           = 3;
constexpr int32_t num_items           = num_tiles * block_threads * items_per_thread - 1;
constexpr int32_t inverse_selectivity = 2;
constexpr int32_t expected_num_out    = (num_items / 2) + 1;

struct SelectOp : thrust::unary_function<int32_t, bool>
{
  __host__ __device__ __forceinline__ bool operator()(const int32_t& value) const
  {
    return value % inverse_selectivity == 0;
  }
};

int main()
{
  // Initialize input data
  thrust::device_vector<int32_t> stencil(num_items);
  thrust::device_vector<int32_t> items(num_items);
  thrust::device_vector<int32_t> items_out(num_items);
  int32_t num_out = 0;
  thrust::sequence(stencil.begin(), stencil.end(), 0);
  thrust::sequence(items.begin(), items.end(), 1, 2);

  // Generate expected output
  thrust::host_vector<int32_t> expected_output(expected_num_out);
  thrust::sequence(expected_output.begin(), expected_output.end(), 1, 4);

  // Allocate temporary storage
  uint8_t* temp_storage     = nullptr;
  size_t temp_storage_bytes = 0;
  crystal::DeviceSelect::Select<block_threads, items_per_thread>(temp_storage,
                                                                 temp_storage_bytes,
                                                                 items.begin(),
                                                                 stencil.begin(),
                                                                 SelectOp{},
                                                                 num_items,
                                                                 items_out.begin(),
                                                                 num_out);
  CubDebugExit(cudaMalloc(&temp_storage, temp_storage_bytes));

  // Execute select kernel
  crystal::DeviceSelect::Select<block_threads, items_per_thread>(temp_storage,
                                                                 temp_storage_bytes,
                                                                 items.begin(),
                                                                 stencil.begin(),
                                                                 SelectOp{},
                                                                 num_items,
                                                                 items_out.begin(),
                                                                 num_out);

  // Check that everything went as expected.
  if (num_out != expected_num_out)
  {
    std::cerr << "Actual / expected num out: " << num_out << " / " << expected_num_out << "\n";
    exit(EXIT_FAILURE);
  }
  thrust::host_vector<int32_t> actual_output(num_out);
  thrust::copy(items_out.begin(), items_out.begin() + num_out, actual_output.begin());

  for (auto i = 0; i < expected_num_out; ++i)
  {
    if (actual_output[i] != expected_output[i])
    {
      std::cerr << "Index: " << i << "\n\tExpected: " << expected_output[i]
                << "\n\tActual: " << actual_output[i] << "\n";
    }
  }

  // Free explicitly allocated resources
  if (temp_storage)
  {
    CubDebugExit(cudaFree(temp_storage));
  }

  return 0;
}
