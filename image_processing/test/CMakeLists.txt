cmake_minimum_required(VERSION 3.10)
project(test)

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
find_package(realsense2 REQUIRED)

add_executable(rs-imshow rs-imshow.cpp)
target_include_directories(rs-imshow PRIVATE ${OpenCV_INCLUDE_DIRS}
                                             ${realsense2_INCLUDE_DIRS})
target_link_libraries(rs-imshow ${OpenCV_LIBRARIES} ${realsense2_LIBRARY})

add_executable(test test.cpp)
target_include_directories(test PRIVATE ${OpenCV_INCLUDE_DIRS}
                                        ${realsense2_INCLUDE_DIRS})
target_link_libraries(test ${OpenCV_LIBRARIES} ${realsense2_LIBRARY})
