cmake_minimum_required(VERSION 2.6)

project(retinaface)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

# cuda
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()

# tensorrt + cudnn
include_directories(
    /usr/include/x86_64-linux-gnu/
#    /workspace/install_package/nvidia-pkg/TensorRT-8.2.5.1/include
    ${PROJECT_SOURCE_DIR}/opencv3410/include
)
link_directories(
    /usr/lib/x86_64-linux-gnu/
#    /workspace/install_package/nvidia-pkg/TensorRT-8.2.5.1/lib
    ${PROJECT_SOURCE_DIR}/opencv3410/lib
    ${PROJECT_SOURCE_DIR}/opencv3410/jasper
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(decodeplugin SHARED ${PROJECT_SOURCE_DIR}/decode.cu)
target_link_libraries(decodeplugin nvinfer cudart)

#find_package(OpenCV)
#include_directories(${OpenCV_INCLUDE_DIRS})
set(OPENCV_LIBS opencv_core opencv_imgproc opencv_dnn opencv_imgcodecs)
message(${OPENCV_LIBS})

add_executable(retina_r50 ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/retina_r50.cpp)
target_link_libraries(retina_r50 nvinfer)
target_link_libraries(retina_r50 cudart)
target_link_libraries(retina_r50 decodeplugin)
target_link_libraries(retina_r50 opencv_highgui ${OPENCV_LIBS})    #${OpenCV_LIBRARIES}

add_executable(retina_mnet ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/retina_mnet.cpp)
target_link_libraries(retina_mnet nvinfer)
target_link_libraries(retina_mnet cudart)
target_link_libraries(retina_mnet decodeplugin)
target_link_libraries(retina_mnet opencv_highgui ${OPENCV_LIBS})   #${OpenCV_LIBRARIES}

add_definitions(-O2 -pthread)

