cmake_minimum_required(VERSION 3.10)
project(opencvtest)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "C:/OpenCV-MinGW-Build-OpenCV-4.5.5-x64/x64/mingw/lib")
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

# include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS} include src/test)

# add_executable(opencvtest src/main.cpp)

# ====================================================
#                   Main Pipeline
# ====================================================
add_executable(main src/main.cpp

        # Detection
            src/pipeline/detection.h 
            src/pipeline/detection.cpp

        # Recognition
            src/pipeline/recognition.cpp
            src/pipeline/recognition.h

        # Runner
            src/pipeline/runner.h
            src/pipeline/runner.cpp

        # Utils
            src/utils/draw.h
            src/utils/draw.cpp 
            src/utils/convert_cnn_labels.h
            src/utils/convert_cnn_labels.cpp

        # Metrics
            src/metrics/cnn_metrics.h
            src/metrics/cnn_metrics.cpp
            src/metrics/yolo_metrics.h
            src/metrics/yolo_metrics.cpp
            src/metrics/metrics.h
            src/metrics/metrics.cpp
)
target_link_libraries(main ${OpenCV_LIBS})



# ====================================================
#                YOLO Test Executable
# ====================================================
add_executable(yolo_test test/yolo_test/yolo_test.cpp 
            test/yolo_test/yolo_utils.h
            test/yolo_test/yolo_utils.cpp
            test/yolo_test/detection.h
            test/yolo_test/detection.cpp)
target_link_libraries(yolo_test ${OpenCV_LIBS})

# ====================================================
#                 CNN Test Executable
# ====================================================

add_executable(cnn_test test/cnn_test/cnn_test.cpp)
target_link_libraries(cnn_test ${OpenCV_LIBS})

# target_link_libraries(opencvtest ${OpenCV_LIBS})
