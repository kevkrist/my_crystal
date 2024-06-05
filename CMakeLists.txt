cmake_minimum_required(VERSION 3.26)
project(my_crystal CUDA)

### CONFIG ###
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_STANDARD 20)
### CONFIG ###

# Download CPM and get latest version of CCCL
set(CPM_DOWNLOAD_VERSION 0.34.0)
if (CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif (DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else ()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif ()
if (NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
            https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
            ${CPM_DOWNLOAD_LOCATION}
    )
endif ()
include(${CPM_DOWNLOAD_LOCATION})

# Add CCCL package
CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        GIT_TAG main
)

# Set global compiler flags
set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr -lineinfo"
)

# Variable for path to CUB's test utils
SET(CUB_TEST_UTILS "${CMAKE_BINARY_DIR}/_deps/cccl-src/cub/test"
        CACHE PATH "Path to CUB's test utils")

add_subdirectory(test)
