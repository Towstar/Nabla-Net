cmake_minimum_required(VERSION 3.21)

project(NablaNetTestPackage LANGUAGES CXX)

find_package(NablaNet CONFIG REQUIRED)

add_executable(nablanet_test_package src/example.cpp)
target_compile_features(nablanet_test_package PRIVATE cxx_std_20)
target_link_libraries(nablanet_test_package PRIVATE NablaNet::NablaNet)
