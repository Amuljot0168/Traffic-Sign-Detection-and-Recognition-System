# ==============================================================================
#                       Project Build Configuration
# ==============================================================================
#
# This project expects OpenCV to be installed on your system.
#
# To simplify setup, edit `setup_env.cmake` (or `setup_env_example.cmake`) to
# specify your local OpenCV installation path. This file is included here and
# sets necessary variables like OpenCV_DIR automatically.
#
# ==============================================================================

cmake_minimum_required(VERSION 3.10)
project(TSDR)

set(CMAKE_CXX_STANDARD 17)

# OpenCV setup
if(EXISTS "${CMAKE_SOURCE_DIR}/setup_env.cmake")
    include(${CMAKE_SOURCE_DIR}/setup_env.cmake)
endif()
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")


# ============= Include Directories ==================
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# ================ Main Executable ===================
add_executable(main 

        # Entry point
        main.cpp

        # Pipeline
        src/pipeline/detection.cpp
        src/pipeline/recognition.cpp
        src/pipeline/runner.cpp

        # Utils
        src/utils/draw.cpp 
        src/utils/convert_cnn_labels.cpp

)
target_link_libraries(main ${OpenCV_LIBS})

# ============ YOLO Test Executable ==================
add_executable(yolo_test test/yolo_test/yolo_test.cpp 
        test/yolo_test/yolo_utils.h
        test/yolo_test/yolo_utils.cpp
        test/yolo_test/detection.h
        test/yolo_test/detection.cpp
)
target_link_libraries(yolo_test ${OpenCV_LIBS})

# ============= CNN Test Executable ==================

add_executable(cnn_test 
            test/cnn_test/cnn_test.cpp
)
target_link_libraries(cnn_test ${OpenCV_LIBS})