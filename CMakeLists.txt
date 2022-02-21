cmake_minimum_required(VERSION 3.0)
project(def_det)

set(CMAKE_CXX_STANDARD 11)

add_executable(detect main.cpp Macro.h fuzzy_matching.cpp clustering.cpp miscellaneous.cpp adaptive_threshold.cpp)
set(OpenCV_DIR D:/opencv/build1)
FIND_PACKAGE(OpenCV REQUIRED)
IF (OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(detect ${OpenCV_LIBS})
ELSE (OpenCV_FOUND)
    MESSAGE(FATAL_ERROR "OpenCV library not found")
ENDIF (OpenCV_FOUND)