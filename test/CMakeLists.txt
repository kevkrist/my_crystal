# Executable name / sourcefile pairs
list(APPEND EXECUTABLES test_scan)
list(APPEND SOURCE_FILES test_scan.cu)

# Loop over name / sourcefile pairs
foreach(exe_name source_file IN ZIP_LISTS EXECUTABLES SOURCE_FILES)
    # Add executable
    add_executable(${exe_name} ${source_file})

    # Set target properties common to all executables
    set_target_properties(${exe_name} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)

    # Link libraries
    target_link_libraries(${exe_name} PRIVATE CCCL::CCCL)

    # Include directories
    target_include_directories(${exe_name} PRIVATE ${CMAKE_SOURCE_DIR}/crystal)
    target_include_directories(${exe_name} PRIVATE ${CUB_TEST_UTILS})
endforeach()
