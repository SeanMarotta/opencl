cmake_minimum_required(VERSION 3.21)
project(vadd)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCL REQUIRED )
find_package( OpenCV REQUIRED )

include_directories( ${OpenCL_INCLUDE_DIR} ${OpenCV_INCLUDE_DIR})

add_executable(vadd main.cpp)
target_link_libraries(vadd ${OpenCL_LIBRARIES} ${OpenCV_LIBS})
