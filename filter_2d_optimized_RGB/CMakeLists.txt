cmake_minimum_required(VERSION 3.21)
project(filter_2d_optimized_RGB)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCL REQUIRED )
find_package( OpenCV REQUIRED )

include_directories( ${OpenCL_INCLUDE_DIR} ${OpenCV_INCLUDE_DIR})

add_executable(filter_2d_optimized_RGB main.cpp)
target_link_libraries(filter_2d_optimized_RGB ${OpenCL_LIBRARIES} ${OpenCV_LIBS})
