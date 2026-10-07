cmake_minimum_required(VERSION 3.24)

project(attnrank LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ATTNRANK_BUILD_PYTHON "Build the attnrank._core Python extension" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(attnrank_lib STATIC
  attnrank/src/core.cpp
  attnrank/src/checkpoint.cpp
  attnrank/src/tokenizer.cpp
  attnrank/src/prompt.cpp
  attnrank/src/kernels.cu
  attnrank/src/model.cpp
  attnrank/src/model.cu
  attnrank/src/attnrank.cpp
  attnrank/src/engine.cpp
)

set_target_properties(attnrank_lib PROPERTIES
  OUTPUT_NAME attnrank
  CUDA_SEPARABLE_COMPILATION OFF
)

target_include_directories(attnrank_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(attnrank_lib PUBLIC
  CUDA::cudart
  CUDA::cublas
  Threads::Threads
)

set(ATTNRANK_CXX_WARNINGS
  -Wall
  -Wextra
  -Wshadow
  -Wpedantic
  -Wconversion
  -Wsign-conversion
  -Wold-style-cast
  -Wnon-virtual-dtor
  -Wuseless-cast
  -Wdouble-promotion
  -Wformat=2
)

target_compile_options(attnrank_lib PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${ATTNRANK_CXX_WARNINGS}>
)

add_executable(attnrank attnrank/src/cli.cpp)
target_link_libraries(attnrank PRIVATE attnrank_lib)
target_compile_options(attnrank PRIVATE ${ATTNRANK_CXX_WARNINGS})

if(SKBUILD)
  install(TARGETS attnrank RUNTIME DESTINATION ${SKBUILD_SCRIPTS_DIR})
else()
  install(TARGETS attnrank RUNTIME DESTINATION bin)
endif()

if(ATTNRANK_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  if(SKBUILD)
    find_package(pybind11 CONFIG REQUIRED)
  else()
    find_package(pybind11 CONFIG)
  endif()
  if(pybind11_FOUND)
    pybind11_add_module(_core NO_EXTRAS attnrank/src/bindings.cpp)
    target_link_libraries(_core PRIVATE attnrank_lib)
    set_target_properties(_core PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    )
    if(SKBUILD)
      install(TARGETS _core LIBRARY DESTINATION attnrank)
    endif()
  else()
    message(WARNING "pybind11 not found; skipping the attnrank._core Python extension")
  endif()
endif()
