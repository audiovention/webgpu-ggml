cmake_minimum_required(VERSION 3.15)
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

project(myggml)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

set(GGML_PERF ON CACHE BOOL "enable X functionality" FORCE)

add_subdirectory("ggml")


file(GLOB my_sources
    "src/*.h"
    "src/*.hpp"
    "src/*.cpp"
    "src/*.c"
)


add_executable(my_ggml_exe ${my_sources})
target_include_directories(my_ggml_exe PRIVATE
    "src"
)

target_link_libraries(my_ggml_exe PRIVATE
    ggml
)

