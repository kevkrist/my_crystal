#define CUB_STDERR

#include "crystal.cuh"
#include "test_util.h"
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

constexpr int32_t seed = 0;
constexpr int32_t default_num_items = 1 << 28;
constexpr int32_t default_random_bound = 100;
constexpr int32_t block_threads = 128;
constexpr int32_t items_per_thread = 4;

//--------------------------------------------------------------------------------------------------
// Functors
//--------------------------------------------------------------------------------------------------
struct SelectOp : thrust::unary_function<int32_t, bool>
{
  __host__ __device__ __forceinline__ bool operator()(const int32_t &value) const
  {
    return value == 0;
  }
};

// Random functor
struct RandUniformOp
{
  int32_t min, max;
  __host__ __device__ RandUniformOp(int32_t min, int32_t max)
      : min{min}, max{max}
  {
  }

  __host__ __device__ int32_t operator()(uint32_t index) const
  {
    thrust::default_random_engine rng(seed);
    rng.discard(index);
    thrust::uniform_int_distribution<int32_t> dist(min, max);
    return dist(rng);
  }
};

//--------------------------------------------------------------------------------------------------
// Sweep functions
//--------------------------------------------------------------------------------------------------
void SweepSelectivity(int32_t num_selectivities)
{
  // Sweep selectivities 1, 1/2, ..., 1/num_selectivities
  thrust::host_vector<int32_t> inverse_selectivities(num_selectivities);
  thrust::sequence(inverse_selectivities.begin(), inverse_selectivities.end(), 1);

  // Initialize default input vector
  thrust::device_vector<int32_t> input(default_num_items);
  thrust::counting_iterator<uint32_t> index_begin(0);
  thrust::transform(index_begin,
                    index_begin + default_num_items,
                    input.begin(),
                    RandUniformOp{-default_random_bound, default_random_bound + 1});

  // Initialize stencil vector
  thrust::device_vector<int32_t> stencil(default_num_items);

  // Initialize output buffers
  thrust::device_vector<int32_t> output(default_num_items);

  // Allocate cub resources
  thrust::device_vector<int32_t> num_out_cub_vector(1);
  uint8_t *temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::FlaggedIf(temp_storage,
                               temp_storage_bytes,
                               input.begin(),
                               stencil.begin(),
                               output.begin(),
                               num_out_cub_vector.begin(),
                               default_num_items,
                               SelectOp{});
  CubDebugExit(cudaMalloc(&temp_storage, temp_storage_bytes));

  // Allocate crystal resources
  uint8_t *temp_storage_crystal = nullptr;
  size_t temp_storage_crystal_bytes = 0;
  int32_t num_out = 0;
  crystal::DeviceSelect::Select<block_threads, items_per_thread>(temp_storage_crystal,
                                                                 temp_storage_crystal_bytes,
                                                                 input.begin(),
                                                                 stencil.begin(),
                                                                 SelectOp{},
                                                                 default_num_items,
                                                                 output.begin(),
                                                                 num_out);
  CubDebugExit(cudaMalloc(&temp_storage_crystal, temp_storage_crystal_bytes));

  for (auto inverse_selectivity : inverse_selectivities)
  {
    std::cout << "Selectivity: " << (1 / static_cast<float>(inverse_selectivity)) << "\n";

    // Populate stencil
    thrust::counting_iterator<uint32_t> stencil_index_begin(0);
    thrust::transform(stencil_index_begin,
                      stencil_index_begin + default_num_items,
                      stencil.begin(),
                      RandUniformOp{0, inverse_selectivity});

    // Crystal
    crystal::DeviceSelect::Select<block_threads, items_per_thread>(temp_storage_crystal,
                                                                   temp_storage_crystal_bytes,
                                                                   input.begin(),
                                                                   stencil.begin(),
                                                                   SelectOp{},
                                                                   default_num_items,
                                                                   output.begin(),
                                                                   num_out);

    CubDebugExit(cudaDeviceSynchronize());

    std::cout << "Finished crystal.\n";

    // Cub
    cub::DeviceSelect::FlaggedIf(temp_storage,
                                 temp_storage_bytes,
                                 input.begin(),
                                 stencil.begin(),
                                 output.begin(),
                                 num_out_cub_vector.begin(),
                                 default_num_items,
                                 SelectOp{});

    CubDebugExit(cudaDeviceSynchronize());

    std::cout << "Finished cub.\n";
  }

  // Free cub temp storage
  if (temp_storage)
  {
    CubDebugExit(cudaFree(temp_storage));
    temp_storage = nullptr;
  }
}

//--------------------------------------------------------------------------------------------------
// Main
//--------------------------------------------------------------------------------------------------
int main(int argc, char **argv)
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
  if (num_selectivities > 0)
  {
    SweepSelectivity(num_selectivities);
  }

  return 0;
}
