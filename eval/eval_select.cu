#define CUB_STDERR

#include "crystal.cuh"
#include "test_util.h"
#include <cub/cub.cuh>
#include <iostream>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

constexpr int32_t seed                                      = 0;
constexpr int32_t default_num_items                         = 1 << 26;
constexpr int32_t block_threads                             = 128;
constexpr int32_t items_per_thread                          = 4;
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
template <int32_t BLOCK_THREADS,
          int32_t ITEMS_PER_THREAD,
          typename CounterT,
          typename InputIt,
          typename StencilIt,
          typename Predicate,
          typename OutputIt>
__global__ void UnorderedSelectKernel(InputIt input,
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
  typedef crystal::
    BlockFlag<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD, crystal::DataArrangement::Blocked>
      BlockFlag;
  typedef crystal::BlockShuffle<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockShuffle;
  typedef crystal::
    BlockLoad<InputT, BLOCK_THREADS, ITEMS_PER_THREAD, crystal::DataArrangement::Striped>
      BlockLoadInput;
  typedef crystal::KernelConfig<BLOCK_THREADS, ITEMS_PER_THREAD> KernelConfigT;

  // Shared memory
  __shared__ typename BlockLoadStencil::TempStorage temp_load_stencil_storage;
  __shared__ typename BlockScan::TempStorage temp_scan_storage;
  __shared__ typename BlockStore::TempStorage temp_store_storage;
  __shared__ typename BlockShuffle::TempStorage temp_shuffle_storage;
  __shared__ CounterT write_offset;

  // Thread memory
  StencilT stencil_items[ITEMS_PER_THREAD];
  int32_t flags[ITEMS_PER_THREAD];
  int32_t prefix_sums[ITEMS_PER_THREAD];
  InputT input_items[ITEMS_PER_THREAD];
  int32_t num_selected = 0;
  KernelConfigT config(blockIdx.x, num_in);

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
  if (threadIdx.x == 0)
  {
    write_offset = atomicAdd(num_out, num_selected);
  }
  cub::CTA_SYNC();

  // Shuffle
  BlockShuffle(temp_shuffle_storage)
    .Shuffle<crystal::DataArrangement::Blocked,
             crystal::DataArrangement::Striped,
             crystal::ShuffleOperator::SetBlockItems>(flags, flags, prefix_sums, num_selected);

  // Gathering load
  BlockLoadInput::Gather(input + config.block_offset, flags, input_items, num_selected);

  // Write compacted data
  BlockStore(temp_store_storage).Store(output + write_offset, input_items, num_selected);
}

//--------------------------------------------------------------------------------------------------
// Sweep
//--------------------------------------------------------------------------------------------------
void SweepSelectivity()
{
  // Initialize default input vector
  std::default_random_engine rng(seed);
  thrust::host_vector<int32_t> input_host(default_num_items);
  thrust::sequence(input_host.begin(), input_host.end(), 0, 2);
  std::shuffle(input_host.begin(), input_host.end(), rng);
  thrust::device_vector<int32_t> input(default_num_items);
  thrust::copy(input_host.begin(), input_host.end(), input.begin());

  // Initialize stencil vector
  thrust::host_vector<int32_t> stencil_host(default_num_items);
  thrust::sequence(stencil_host.begin(), stencil_host.end(), 0);
  std::shuffle(stencil_host.begin(), stencil_host.end(), rng);
  thrust::device_vector<int32_t> stencil(default_num_items);
  thrust::copy(stencil_host.begin(), stencil_host.end(), stencil.begin());

  // Initialize output buffers
  thrust::device_vector<int32_t> output(default_num_items);

  for (auto inverse_selectivity : possible_inverse_selectivities)
  {
    std::cout << "Selectivity: " << (1.0f / static_cast<float>(inverse_selectivity)) << "\n";

    // Cub ordered compaction
    thrust::device_vector<int32_t> num_out_cub_vector(1);
    uint8_t* temp_storage_cub = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::FlaggedIf(temp_storage_cub,
                                 temp_storage_bytes,
                                 input.begin(),
                                 stencil.begin(),
                                 output.begin(),
                                 num_out_cub_vector.begin(),
                                 default_num_items,
                                 SelectOp{inverse_selectivity});
    CubDebugExit(cudaMalloc(&temp_storage_cub, temp_storage_bytes));
    cub::DeviceSelect::FlaggedIf(temp_storage_cub,
                                 temp_storage_bytes,
                                 input.begin(),
                                 stencil.begin(),
                                 output.begin(),
                                 num_out_cub_vector.begin(),
                                 default_num_items,
                                 SelectOp{inverse_selectivity});
    CubDebugExit(cudaDeviceSynchronize());

    // Crystal ordered compaction
    uint8_t* temp_storage_crystal = nullptr;
    temp_storage_bytes            = 0;
    int32_t num_out               = 0;
    crystal::DeviceSelect::Select<block_threads, items_per_thread>(temp_storage_crystal,
                                                                   temp_storage_bytes,
                                                                   input.begin(),
                                                                   stencil.begin(),
                                                                   SelectOp{inverse_selectivity},
                                                                   default_num_items,
                                                                   output.begin(),
                                                                   num_out);
    CubDebugExit(cudaMalloc(&temp_storage_crystal, temp_storage_bytes));
    crystal::DeviceSelect::Select<block_threads, items_per_thread>(temp_storage_crystal,
                                                                   temp_storage_bytes,
                                                                   input.begin(),
                                                                   stencil.begin(),
                                                                   SelectOp{inverse_selectivity},
                                                                   default_num_items,
                                                                   output.begin(),
                                                                   num_out);
    CubDebugExit(cudaDeviceSynchronize());

    // Crystal unordered compaction
    thrust::device_vector<int32_t> num_out_crystal_vector(1);
    int32_t blocks_in_grid =
      cub::DivideAndRoundUp(default_num_items, block_threads * items_per_thread);
    UnorderedSelectKernel<block_threads, items_per_thread>
      <<<blocks_in_grid, block_threads>>>(input.begin(),
                                          stencil.begin(),
                                          SelectOp{inverse_selectivity},
                                          default_num_items,
                                          output.begin(),
                                          thrust::raw_pointer_cast(num_out_crystal_vector.data()));
    CubDebugExit(cudaDeviceSynchronize());

    // Free explicitly allocated resources
    if (temp_storage_cub)
    {
      CubDebugExit(cudaFree(temp_storage_cub));
    }
    if (temp_storage_crystal)
    {
      CubDebugExit(cudaFree(temp_storage_crystal));
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Main
//--------------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
  // Gather command-line args
  int32_t num_selectivities = 0;
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("s", num_selectivities);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    std::cout << argv[0] << "\n\t[--s=<sweep selectivities 1...1/s>]"
              << "\n\t[--n=<sweep input sizes 2^n...2^28>]\n";
    return 0;
  }

  // Sweep selectivity
  SweepSelectivity();

  return 0;
}
