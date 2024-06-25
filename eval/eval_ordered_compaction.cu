#define CUB_STDERR

#include "crystal.cuh"
#include "test_util.h"
#include <algorithm>
#include <cub/cub.cuh>
#include <iostream>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/sequence.h>

constexpr int32_t block_threads                             = 128;
constexpr int32_t items_per_thread                          = 4;
constexpr int32_t seed                                      = 0;
thrust::host_vector<int32_t> possible_inverse_selectivities = {2, 4, 8, 12, 16, 24, 32, 64};

//--------------------------------------------------------------------------------------------------
// Functors
//--------------------------------------------------------------------------------------------------
struct SelectOp : thrust::unary_function<int32_t, bool>
{
  int32_t mod;

  __host__ __device__ explicit SelectOp(int32_t mod)
      : mod{mod}
  {}

  __host__ __device__ __forceinline__ bool operator()(const int32_t& value) const
  {
    return (value % mod) == 0;
  }
};

//--------------------------------------------------------------------------------------------------
// Kernels
//--------------------------------------------------------------------------------------------------
template <typename InputIt, typename StencilIt, typename Predicate, typename OutputIt>
__global__ void UnorderedSelectionKernel(InputIt input,
                                         StencilIt stencil,
                                         Predicate predicate,
                                         int32_t num_in,
                                         OutputIt output,
                                         int32_t* num_out)
{
  // Typedefs
  typedef typename thrust::iterator_traits<InputIt>::value_type InputT;
  typedef typename thrust::iterator_traits<StencilIt>::value_type StencilT;
  typedef cub::BlockLoad<InputT, block_threads, items_per_thread, cub::BLOCK_LOAD_STRIPED>
    BlockLoadInput;
  typedef cub::BlockLoad<StencilT, block_threads, items_per_thread, cub::BLOCK_LOAD_STRIPED>
    BlockLoadStencil;
  typedef cub::BlockScan<int32_t, block_threads> BlockScan;
  typedef crystal::
    BlockFlag<int32_t, block_threads, items_per_thread, crystal::DataArrangement::Striped>
      BlockFlag;
  typedef crystal::BlockShuffle<InputT, block_threads, items_per_thread> BlockShuffle;
  typedef crystal::KernelConfig<block_threads, items_per_thread> KernelConfig;

  // Shared memory
  __shared__ typename BlockLoadInput::TempStorage temp_load_input_storage;
  __shared__ typename BlockLoadStencil::TempStorage temp_load_stencil_storage;
  __shared__ typename BlockScan::TempStorage temp_scan_storage;
  __shared__ typename BlockShuffle::TempStorage temp_shuffle_storage;
  __shared__ int32_t write_offset;

  // Thread memory
  InputT thread_input[items_per_thread];
  StencilT thread_stencil[items_per_thread];
  int32_t thread_flags[items_per_thread];
  int32_t prefix_sums[items_per_thread];
  int32_t num_selected = 0;
  KernelConfig kernel_config(blockIdx.x, num_in);

  // Do stuff
  if (!kernel_config.is_last_tile)
  {
    BlockLoadInput(temp_load_input_storage).Load(input + kernel_config.block_offset, thread_input);
    BlockLoadStencil(temp_load_stencil_storage)
      .Load(stencil + kernel_config.block_offset, thread_stencil);
    BlockFlag::SetFlags(thread_stencil, predicate, thread_flags);
  }
  else
  {
    BlockLoadInput(temp_load_input_storage)
      .Load(input + kernel_config.block_offset, thread_input, kernel_config.num_tile_items);
    BlockLoadStencil(temp_load_stencil_storage)
      .Load(stencil + kernel_config.block_offset, thread_stencil, kernel_config.num_tile_items);
    BlockFlag::InitFlags(thread_flags);
    BlockFlag::SetFlags(thread_stencil, predicate, thread_flags, kernel_config.num_tile_items);
  }
  BlockScan(temp_scan_storage).ExclusiveSum(thread_flags, prefix_sums, num_selected);
  if (threadIdx.x == 0)
  {
    write_offset = atomicAdd(num_out, num_selected);
  }
  __syncthreads(); // For write_offset
  BlockShuffle(temp_shuffle_storage)
    .ShuffleStore<crystal::DataArrangement::Striped>(output + write_offset,
                                                     thread_input,
                                                     thread_flags,
                                                     prefix_sums,
                                                     num_selected);
}

template <typename InputIt, typename OffsetIt, typename OutputIt>
__global__ void GatherKernel(InputIt input, OffsetIt offsets, OutputIt output, size_t num_in)
{
  typedef typename thrust::iterator_traits<InputIt>::value_type InputT;
  typedef typename thrust::iterator_traits<OutputIt>::value_type OffsetT;
  typedef crystal::
    BlockLoad<OffsetT, block_threads, items_per_thread, crystal::DataArrangement::Striped>
      BlockLoadOffsets;
  typedef crystal::
    BlockLoad<InputT, block_threads, items_per_thread, crystal::DataArrangement::Striped>
      BlockLoadInputs;
  typedef crystal::
    BlockStore<InputT, block_threads, items_per_thread, crystal::DataArrangement::Striped>
      BlockStore;
  typedef crystal::KernelConfig<block_threads, items_per_thread> KernelConfig;

  // Thread memory
  InputT thread_input[items_per_thread];
  OffsetT thread_offsets[items_per_thread];
  KernelConfig kernel_config(blockIdx.x, num_in);

  // Do stuff
  if (!kernel_config.is_last_tile)
  {
    BlockLoadOffsets::Load(offsets + kernel_config.block_offset, thread_offsets);
    BlockLoadInputs::Gather(input, thread_offsets, thread_input);
    BlockStore::Store(output + kernel_config.block_offset, thread_input);
  }
  else
  {
    BlockLoadOffsets::Load(offsets + kernel_config.block_offset,
                           thread_offsets,
                           kernel_config.num_tile_items);
    BlockLoadInputs::Gather(input, thread_offsets, thread_input, kernel_config.num_tile_items);
    BlockStore::Store(output + kernel_config.block_offset,
                      thread_input,
                      kernel_config.num_tile_items);
  }
}

//--------------------------------------------------------------------------------------------------
// Sweep functions
//--------------------------------------------------------------------------------------------------
void Sweep(int32_t max_inverse_selectivity,
           int32_t max_input_size,
           bool fix_inverse_selectivity,
           bool fix_output_size)
{
  // ?
  int32_t num_input = max_input_size;

  // Initialize output buffers
  thrust::device_vector<int32_t> row_ids_out_sorted_1(max_input_size);
  thrust::device_vector<int32_t> stencil_out_sorted_1(max_input_size);
  thrust::device_vector<int32_t> row_ids_out_sorted_2(max_input_size);
  thrust::device_vector<int32_t> row_ids_out_unsorted_1(max_input_size);
  thrust::device_vector<int32_t> stencil_out_unsorted_1(max_input_size);
  thrust::device_vector<int32_t> row_ids_out_unsorted_2(max_input_size);

  // Initialize row ids, stencil
  thrust::device_vector<int32_t> row_ids_in(max_input_size);
  thrust::sequence(row_ids_in.begin(), row_ids_in.end(), 0);
  thrust::device_vector<int32_t> stencil_in(max_input_size);
  thrust::host_vector<int32_t> stencil_in_host(max_input_size);
  thrust::sequence(stencil_in_host.begin(), stencil_in_host.end(), 0);
  std::shuffle(stencil_in_host.begin(), stencil_in_host.end(), std::default_random_engine(seed));
  thrust::copy(stencil_in_host.begin(), stencil_in_host.end(), stencil_in.begin());

  // Sweep selectivities
  for (auto inverse_selectivity : possible_inverse_selectivities)
  {
    // Check that we have not exceeded maximum specified inverse selectivity
    if(inverse_selectivity > max_inverse_selectivity) {
      return;
    }
    std::cout << "SELECTIVITY: " << (1.0f / static_cast<float>(inverse_selectivity)) << "\n";

    // Initialize counters
    thrust::device_vector<int32_t> num_out_sorted_1(1);
    thrust::device_vector<int32_t> num_out_unsorted_1(1);
    thrust::device_vector<int32_t> num_out_sorted_2(1);
    thrust::device_vector<int32_t> num_out_unsorted_2(1);

    /* 1 ordered compaction + gather */
    uint8_t* temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::FlaggedIf(temp_storage,
                                 temp_storage_bytes,
                                 row_ids_in.begin(),
                                 stencil_in.begin(),
                                 row_ids_out_sorted_1.begin(),
                                 num_out_sorted_1.begin(),
                                 num_input,
                                 SelectOp{inverse_selectivity});
    CubDebugExit(cudaMalloc(&temp_storage, temp_storage_bytes));
    cub::DeviceSelect::FlaggedIf(temp_storage,
                                 temp_storage_bytes,
                                 row_ids_in.begin(),
                                 stencil_in.begin(),
                                 row_ids_out_sorted_1.begin(),
                                 num_out_sorted_1.begin(),
                                 num_input,
                                 SelectOp{inverse_selectivity});
    CubDebugExit(cudaDeviceSynchronize());
    int32_t num_out_sorted_1_host = num_out_sorted_1[0];
    thrust::device_vector<int32_t> gathered_data(num_out_sorted_1_host);
    int32_t blocks_in_gather_grid =
      cub::DivideAndRoundUp(num_out_sorted_1_host, block_threads * items_per_thread);
    GatherKernel<<<blocks_in_gather_grid, block_threads>>>(stencil_in.begin(),
                                                           row_ids_out_sorted_1.begin(),
                                                           gathered_data.begin(),
                                                           num_out_sorted_1_host);
    CubDebugExit(cudaDeviceSynchronize());

    /* 1 unordered compaction + gather */
    int32_t blocks_in_compaction_grid =
      cub::DivideAndRoundUp(num_input, block_threads * items_per_thread);
    UnorderedSelectionKernel<<<blocks_in_compaction_grid, block_threads>>>(
      row_ids_in.begin(),
      stencil_in.begin(),
      SelectOp{inverse_selectivity},
      num_input,
      row_ids_out_unsorted_1.begin(),
      thrust::raw_pointer_cast(num_out_unsorted_1.data()));
    CubDebugExit(cudaDeviceSynchronize());
    GatherKernel<<<blocks_in_gather_grid, block_threads>>>(stencil_in.begin(),
                                                           row_ids_out_unsorted_1.begin(),
                                                           gathered_data.begin(),
                                                           num_out_sorted_1_host);
    CubDebugExit(cudaDeviceSynchronize());

    /* 2 ordered compactions and gathers */
    uint8_t* temp_storage_2 = nullptr;
    temp_storage_bytes      = 0;
    auto stencil_gather =
      thrust::make_permutation_iterator(stencil_in.begin(), row_ids_out_sorted_1.begin());
    cub::DeviceSelect::FlaggedIf(temp_storage_2,
                                 temp_storage_bytes,
                                 row_ids_out_sorted_1.begin(),
                                 stencil_gather,
                                 row_ids_out_sorted_2.begin(),
                                 num_out_sorted_2.begin(),
                                 num_out_sorted_1_host,
                                 SelectOp{inverse_selectivity / 2});
    CubDebugExit(cudaMalloc(&temp_storage_2, temp_storage_bytes));
    cub::DeviceSelect::FlaggedIf(temp_storage_2,
                                 temp_storage_bytes,
                                 row_ids_out_sorted_1.begin(),
                                 stencil_gather,
                                 row_ids_out_sorted_2.begin(),
                                 num_out_sorted_2.begin(),
                                 num_out_sorted_1_host,
                                 SelectOp{inverse_selectivity / 2});
    CubDebugExit(cudaDeviceSynchronize());
    int32_t num_out_sorted_2_host = num_out_sorted_2[0];
    thrust::device_vector<int32_t> gathered_data_2(num_out_sorted_2_host);
    int32_t blocks_in_gather_grid_2 =
      cub::DivideAndRoundUp(num_out_sorted_2_host, block_threads * items_per_thread);
    GatherKernel<<<blocks_in_gather_grid_2, block_threads>>>(stencil_in.begin(),
                                                             row_ids_out_sorted_2.begin(),
                                                             gathered_data_2.begin(),
                                                             num_out_sorted_2_host);
    CubDebugExit(cudaDeviceSynchronize());

    /* 2 unordered compactions + gathers */
    int32_t blocks_in_compaction_grid_2 =
      cub::DivideAndRoundUp(num_out_sorted_1_host, block_threads * items_per_thread);
    auto stencil_gather_unsorted =
      thrust::make_permutation_iterator(stencil_in.begin(), row_ids_out_sorted_1.begin());
    UnorderedSelectionKernel<<<blocks_in_compaction_grid_2, block_threads>>>(
      row_ids_out_unsorted_1.begin(),
      stencil_gather_unsorted,
      SelectOp{inverse_selectivity / 2},
      num_out_sorted_1_host,
      row_ids_out_unsorted_2.begin(),
      thrust::raw_pointer_cast(num_out_unsorted_2.data()));
    CubDebugExit(cudaDeviceSynchronize());
    GatherKernel<<<blocks_in_gather_grid, block_threads>>>(stencil_in.begin(),
                                                           row_ids_out_unsorted_2.begin(),
                                                           gathered_data_2.begin(),
                                                           num_out_sorted_2_host);
    CubDebugExit(cudaDeviceSynchronize());

    // Free explicitly allocated resources
    if (temp_storage)
    {
      CubDebugExit(cudaFree(temp_storage));
    }
    if (temp_storage_2)
    {
      CubDebugExit(cudaFree(temp_storage_2));
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Main
//--------------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
  // Gather command-line args
  int32_t max_inverse_selectivity = 64;
  int32_t max_input_size          = 1 << 22;
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("mis", max_inverse_selectivity);
  args.GetCmdLineArgument("mos", max_input_size);
  bool fix_inverse_selectivity = args.CheckCmdLineFlag("fis");
  bool fix_output_size         = args.CheckCmdLineFlag("fos");

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    std::cout << argv[0]
              << "\n\t[--mis=<max inverse selectivity, i.e. sweep selectivities 1/2...1/ms>]"
              << "\n\t[--os=<output size for compaction>]\n";
    return 0;
  }

  // Sweep selectivities
  Sweep(max_inverse_selectivity, max_input_size, fix_inverse_selectivity, fix_output_size);

  return 0;
}