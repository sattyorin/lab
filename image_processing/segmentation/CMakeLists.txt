cmake_minimum_required(VERSION 3.10)
project(segmentation)

set(CMAKE_CXX_COMPILER
    "/usr/bin/g++"
    CACHE STRING "g++ compiler" FORCE)
set(CMAKE_C_COMPILER
    "/usr/bin/gcc"
    CACHE STRING "gcc compiler" FORCE)
add_compile_options(-g -O0 -fmax-errors=5 -Wfatal-errors -fPIC)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(segmentation segmentation.cpp)
target_include_directories(segmentation PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(segmentation ${OpenCV_LIBRARIES})
