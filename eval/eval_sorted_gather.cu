#define CUB_STDERR

#include "crystal.cuh"
#include "test_util.h"
#include <algorithm>
#include <cub/cub.cuh>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

constexpr int32_t block_threads                             = 128;
constexpr int32_t items_per_thread                          = 6;
constexpr int32_t seed                                      = 0;
constexpr size_t min_output_size                            = 1 << 10;
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
                                         size_t num_in,
                                         OutputIt output,
                                         size_t* num_out)
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

template <typename InputIt,
          typename StencilIt,
          typename PredicateA,
          typename PredicateB,
          typename OutputItA,
          typename OutputItB>
__global__ void DoubleUnorderedSelectionKernel(InputIt input,
                                               StencilIt stencil,
                                               PredicateA predicateA,
                                               PredicateB predicateB,
                                               size_t num_in,
                                               OutputItA output,
                                               OutputItB stencil_output,
                                               size_t* num_out)
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
  typedef crystal::BlockShuffle<InputT, block_threads, items_per_thread> BlockShuffleInput;
  typedef crystal::BlockShuffle<StencilT, block_threads, items_per_thread> BlockShuffleStencil;
  typedef crystal::KernelConfig<block_threads, items_per_thread> KernelConfig;

  // Shared memory
  __shared__ typename BlockLoadInput::TempStorage temp_load_input_storage;
  __shared__ typename BlockLoadStencil::TempStorage temp_load_stencil_storage;
  __shared__ typename BlockScan::TempStorage temp_scan_storage;
  __shared__ union
  {
    typename BlockShuffleInput::TempStorage input;
    typename BlockShuffleStencil::TempStorage stencil;
  } temp_shuffle_storage;
  __shared__ int32_t write_offset;

  // Thread memory
  InputT thread_input[items_per_thread];
  StencilT thread_stencil[items_per_thread];
  int32_t thread_flags[items_per_thread];
  int32_t prefix_sums[items_per_thread];
  int32_t num_selected = 0;
  int32_t num_final    = 0;
  KernelConfig kernel_config(blockIdx.x, num_in);

  // Do stuff
  if (!kernel_config.is_last_tile)
  {
    BlockLoadInput(temp_load_input_storage).Load(input + kernel_config.block_offset, thread_input);
    BlockLoadStencil(temp_load_stencil_storage)
      .Load(stencil + kernel_config.block_offset, thread_stencil);
    BlockFlag::SetFlags(thread_stencil, predicateA, thread_flags);
  }
  else
  {
    BlockLoadInput(temp_load_input_storage)
      .Load(input + kernel_config.block_offset, thread_input, kernel_config.num_tile_items);
    BlockLoadStencil(temp_load_stencil_storage)
      .Load(stencil + kernel_config.block_offset, thread_stencil, kernel_config.num_tile_items);
    BlockFlag::InitFlags(thread_flags);
    BlockFlag::SetFlags(thread_stencil, predicateA, thread_flags, kernel_config.num_tile_items);
  }
  BlockScan(temp_scan_storage).ExclusiveSum(thread_flags, prefix_sums, num_selected);
  BlockShuffleInput(temp_shuffle_storage.input)
    .Shuffle<crystal::DataArrangement::Striped,
             crystal::DataArrangement::Striped,
             crystal::ShuffleOperator::ShuffleThreadItems>(thread_input,
                                                           thread_flags,
                                                           prefix_sums,
                                                           num_selected);
  __syncthreads(); // For temp_shuffle_storage
  BlockShuffleStencil(temp_shuffle_storage.stencil)
    .Shuffle<crystal::DataArrangement::Striped,
             crystal::DataArrangement::Striped,
             crystal::ShuffleOperator::ShuffleThreadItems>(thread_stencil,
                                                           thread_flags,
                                                           prefix_sums,
                                                           num_selected);
  BlockFlag::InitFlags(thread_flags);
  BlockFlag::SetFlags(thread_stencil, predicateB, thread_flags, num_selected);
  BlockScan(temp_scan_storage).ExclusiveSum(thread_flags, prefix_sums, num_final);
  if (threadIdx.x == 0)
  {
    write_offset = atomicAdd(num_out, num_final);
  }
  __syncthreads(); // For write_offset, temp_shuffle_storage
  BlockShuffleInput(temp_shuffle_storage.input)
    .ShuffleStore<crystal::DataArrangement::Striped>(output + write_offset,
                                                     thread_input,
                                                     thread_flags,
                                                     prefix_sums,
                                                     num_final);
  __syncthreads(); // For temp_shuffle_storage
  BlockShuffleStencil(temp_shuffle_storage.stencil)
    .ShuffleStore<crystal::DataArrangement::Striped>(stencil_output + write_offset,
                                                     thread_stencil,
                                                     thread_flags,
                                                     prefix_sums,
                                                     num_final);
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
void SweepSelectivity(int32_t max_inverse_selectivity,
                      int32_t max_output_size,
                      bool fix_inverse_selectivity,
                      bool fix_output_size)
{
  // Initialize starting output size, and sweep over output size
  size_t start_output_size = fix_output_size ? max_output_size : min_output_size;
  for (size_t output_size = start_output_size; output_size <= max_output_size; output_size <<= 1)
  {
    std::cout << "Output size: " << output_size << "\n";
    // Initialize output buffers
    thrust::device_vector<int32_t> row_ids_out_sorted(output_size);
    thrust::device_vector<int32_t> row_ids_out_unsorted(output_size);
    thrust::device_vector<int32_t> row_ids_out_random(output_size);
    thrust::device_vector<int32_t> row_ids_out_double_1(2 * output_size);
    thrust::device_vector<int32_t> row_ids_out_double_2(output_size);
    thrust::device_vector<int32_t> row_ids_out_triple_1(2 * output_size);
    thrust::device_vector<int32_t> row_ids_out_triple_2(output_size);
    thrust::host_vector<int32_t> row_ids_out_random_host(output_size);
    thrust::device_vector<int32_t> stencil_buffer_double(2 * output_size);
    thrust::device_vector<int32_t> stencil_buffer_triple(2 * output_size);
    thrust::device_vector<int32_t> gathered_data_sorted(output_size);
    thrust::device_vector<int32_t> gathered_data_unsorted(output_size);
    thrust::device_vector<int32_t> gathered_data_double(output_size);
    thrust::device_vector<int32_t> gathered_data_triple(output_size);
    thrust::device_vector<int32_t> gathered_data_random(output_size);

    // Initialize row ids
    thrust::device_vector<int32_t> row_ids_in(output_size * max_inverse_selectivity);
    thrust::sequence(row_ids_in.begin(), row_ids_in.end(), 0);

    // Initialize/seed random generator
    std::default_random_engine rng(seed);

    // Initialize starting selectivity
    auto max_possible_inverse_selectivity = std::max_element(possible_inverse_selectivities.begin(),
                                                             possible_inverse_selectivities.end());
    auto inverse_selectivity_iter         = possible_inverse_selectivities.begin();
    if (fix_inverse_selectivity)
    {
      while (*inverse_selectivity_iter != max_inverse_selectivity &&
             inverse_selectivity_iter < possible_inverse_selectivities.end())
      {
        ++inverse_selectivity_iter;
      }
    }
    if (inverse_selectivity_iter == possible_inverse_selectivities.end())
    {
      std::cerr << "Invalid inverse selectivity value.";
      exit(EXIT_FAILURE);
    }

    // Sweep over selectivities
    while (inverse_selectivity_iter < max_possible_inverse_selectivity &&
           *inverse_selectivity_iter <= max_inverse_selectivity)
    {
      int32_t inverse_selectivity = *inverse_selectivity_iter;
      std::cout << "Selectivity: " << (1 / static_cast<float>(inverse_selectivity)) << "\n";

      // Initialize and populate stencil
      size_t num_input = inverse_selectivity * output_size;
      thrust::host_vector<int32_t> stencil_host(num_input);
      thrust::device_vector<int32_t> stencil(num_input);
      thrust::sequence(stencil_host.begin(), stencil_host.end(), 0);
      std::shuffle(stencil_host.begin(), stencil_host.end(), rng);
      thrust::copy(stencil_host.begin(), stencil_host.end(), stencil.begin());
      CubDebugExit(cudaDeviceSynchronize());

      // Initialize counters
      thrust::device_vector<size_t> num_out_unsorted(1);
      thrust::device_vector<size_t> num_out_sorted(1);
      thrust::device_vector<size_t> num_out_double_1(1);
      thrust::device_vector<size_t> num_out_double_2(1);
      thrust::device_vector<size_t> num_out_triple_1(1);
      thrust::device_vector<size_t> num_out_triple_2(1);

      // Cub order-preserving compaction
      uint8_t* temp_storage     = nullptr;
      size_t temp_storage_bytes = 0;
      cub::DeviceSelect::FlaggedIf(temp_storage,
                                   temp_storage_bytes,
                                   row_ids_in.begin(),
                                   stencil.begin(),
                                   row_ids_out_sorted.begin(),
                                   num_out_sorted.begin(),
                                   num_input,
                                   SelectOp{inverse_selectivity});
      CubDebugExit(cudaMalloc(&temp_storage, temp_storage_bytes));
      cub::DeviceSelect::FlaggedIf(temp_storage,
                                   temp_storage_bytes,
                                   row_ids_in.begin(),
                                   stencil.begin(),
                                   row_ids_out_sorted.begin(),
                                   num_out_sorted.begin(),
                                   num_input,
                                   SelectOp{inverse_selectivity});
      CubDebugExit(cudaDeviceSynchronize());

      // Crystal non-order-preserving compaction
      int32_t blocks_in_grid = cub::DivideAndRoundUp(num_input, block_threads * items_per_thread);
      UnorderedSelectionKernel<<<blocks_in_grid, block_threads>>>(
        row_ids_in.begin(),
        stencil.begin(),
        SelectOp{inverse_selectivity},
        num_input,
        row_ids_out_unsorted.begin(),
        thrust::raw_pointer_cast(num_out_unsorted.data()));
      CubDebugExit(cudaDeviceSynchronize());

      // Crystal non-order-preserving double compaction (NOTE: we are not evaluating the
      // performance of this kernel)
      auto input_iter_double = thrust::make_zip_iterator(row_ids_in.begin(), stencil.begin());
      auto output_iter_double =
        thrust::make_zip_iterator(row_ids_out_double_1.begin(), stencil_buffer_double.begin());
      UnorderedSelectionKernel<<<blocks_in_grid, block_threads>>>( // First compaction
        input_iter_double,
        stencil.begin(),
        SelectOp{inverse_selectivity / 2},
        num_input,
        output_iter_double,
        thrust::raw_pointer_cast(num_out_double_1.data()));
      CubDebugExit(cudaDeviceSynchronize());
      size_t num_out_double_1_host = num_out_double_1[0]; // Copy to host for DivideAndRoundUp
      int32_t blocks_in_second_grid =
        cub::DivideAndRoundUp(num_out_double_1_host, block_threads * items_per_thread);
      UnorderedSelectionKernel<<<blocks_in_second_grid, block_threads>>>( // Second compaction
        row_ids_out_double_1.begin(),
        stencil_buffer_double.begin(),
        SelectOp{inverse_selectivity},
        num_out_double_1_host,
        row_ids_out_double_2.begin(),
        thrust::raw_pointer_cast(num_out_double_2.data()));
      CubDebugExit(cudaDeviceSynchronize());

      // Crystal non-order-preserving double compaction with initial shuffle from filter in 1st
      // kernel
      DoubleUnorderedSelectionKernel<<<blocks_in_grid, block_threads>>>( // First compaction
        row_ids_in.begin(),
        stencil.begin(),
        SelectOp{2},
        SelectOp{inverse_selectivity / 2},
        num_input,
        row_ids_out_triple_1.begin(),
        stencil_buffer_triple.begin(),
        thrust::raw_pointer_cast(num_out_triple_1.data()));
      CubDebugExit(cudaDeviceSynchronize());
      size_t num_out_triple_1_host = num_out_triple_1[0]; // Copy to host for DivideAndRoundUp
      blocks_in_second_grid =
        cub::DivideAndRoundUp(num_out_triple_1_host, block_threads * items_per_thread);
      UnorderedSelectionKernel<<<blocks_in_second_grid, block_threads>>>( // Second compaction
        row_ids_out_triple_1.begin(),
        stencil_buffer_triple.begin(),
        SelectOp{inverse_selectivity},
        num_out_triple_1_host,
        row_ids_out_triple_2.begin(),
        thrust::raw_pointer_cast(num_out_triple_2.data()));
      CubDebugExit(cudaDeviceSynchronize());

      // Generate a completely random sequence of row ids
      thrust::copy(row_ids_out_sorted.begin(),
                   row_ids_out_sorted.end(),
                   row_ids_out_random_host.begin());
      std::shuffle(row_ids_out_random_host.begin(), row_ids_out_random_host.end(), rng);
      thrust::copy(row_ids_out_random_host.begin(),
                   row_ids_out_random_host.end(),
                   row_ids_out_random.begin());

      // Sanity check that the outputs are the same size
      if (num_out_sorted[0] != num_out_unsorted[0] || num_out_sorted[0] != num_out_double_2[0] ||
          num_out_sorted[0] != num_out_triple_2[0] || num_out_sorted[0] != output_size)
      {
        std::cerr << "Kernels produced inconsistent output sizes (cub / crystal / crystal double / "
                     "crystal triple / expected): "
                  << num_out_sorted[0] << " / " << num_out_unsorted[0] << " / "
                  << num_out_double_2[0] << " / " << num_out_triple_2[0] << " / " << output_size
                  << "\n";
        exit(EXIT_FAILURE);
      }

      // Custom gathers
      int32_t blocks_in_gather_grid =
        cub::DivideAndRoundUp(output_size, block_threads * items_per_thread);
      GatherKernel<<<blocks_in_gather_grid, block_threads>>>(stencil.begin(),
                                                             row_ids_out_sorted.begin(),
                                                             gathered_data_sorted.begin(),
                                                             output_size);
      CubDebugExit(cudaDeviceSynchronize());
      GatherKernel<<<blocks_in_gather_grid, block_threads>>>(stencil.begin(),
                                                             row_ids_out_unsorted.begin(),
                                                             gathered_data_unsorted.begin(),
                                                             output_size);
      CubDebugExit(cudaDeviceSynchronize());
      GatherKernel<<<blocks_in_gather_grid, block_threads>>>(stencil.begin(),
                                                             row_ids_out_double_2.begin(),
                                                             gathered_data_double.begin(),
                                                             output_size);
      CubDebugExit(cudaDeviceSynchronize());
      GatherKernel<<<blocks_in_gather_grid, block_threads>>>(stencil.begin(),
                                                             row_ids_out_triple_2.begin(),
                                                             gathered_data_triple.begin(),
                                                             output_size);
      CubDebugExit(cudaDeviceSynchronize());
      GatherKernel<<<blocks_in_gather_grid, block_threads>>>(stencil.begin(),
                                                             row_ids_out_random.begin(),
                                                             gathered_data_random.begin(),
                                                             output_size);
      CubDebugExit(cudaDeviceSynchronize());

      // Free explicitly allocated resources
      if (temp_storage)
      {
        CubDebugExit(cudaFree(temp_storage));
      }

      // Increment iterator
      ++inverse_selectivity_iter;
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Main
//--------------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
  // Gather command-line args
  int32_t max_inverse_selectivity = 0;
  int32_t max_output_size         = 1 << 22;
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("mis", max_inverse_selectivity);
  args.GetCmdLineArgument("mos", max_output_size);
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
  if (max_inverse_selectivity > 0)
  {
    SweepSelectivity(max_inverse_selectivity,
                     max_output_size,
                     fix_inverse_selectivity,
                     fix_output_size);
  }

  return 0;
}