cmake_minimum_required(VERSION 3.16)
project(MiniTensorImageLibrary LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(image_filters main.cpp Image.cpp)
target_compile_options(image_filters PRIVATE -Wall -Wextra -Wpedantic)
