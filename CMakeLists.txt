cmake_minimum_required(VERSION 3.10)
project(FF)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Tesseract Include directories
include_directories("C:/Users/kirit/.vcpkg-clion/vcpkg/installed/x64-windows/include")

# Tesseract Library directories
link_directories("C:/Users/kirit/.vcpkg-clion/vcpkg/installed/x64-windows/lib")

#set(TESSDATA_PREFIX "C:/Program Files/Tesseract-OCR/")


# Add source files
add_executable(FF main.cpp)

# Link libraries (OpenCV, Tesseract, Leptonica)
target_link_libraries(FF ${OpenCV_LIBS} tesseract54 leptonica-1.84.1)
