cmake_minimum_required(VERSION 3.21)
project(opencl_c)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCL REQUIRED )
find_package( OpenCV REQUIRED )

include_directories( ${OpenCL_INCLUDE_DIR} ${OpenCV_INCLUDE_DIR})

add_executable(opencl_c main.cpp)
target_link_libraries(opencl_c ${OpenCL_LIBRARIES} ${OpenCV_LIBS})
