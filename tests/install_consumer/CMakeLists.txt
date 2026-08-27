cmake_minimum_required(VERSION 3.21)

project(NablaNetInstallConsumer LANGUAGES CXX)

find_package(NablaNet CONFIG REQUIRED)

add_executable(nablanet_install_consumer main.cpp)
target_link_libraries(nablanet_install_consumer PRIVATE NablaNet::NablaNet)
