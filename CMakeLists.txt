cmake_minimum_required(VERSION 3.21)

project(NeuralNetworkCPP VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CTest)
include(CMakePackageConfigHelpers)

option(NNCPP_BUILD_EXAMPLES "Build the small library usage example" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(NNCPP_CORE_SOURCES
    src/common.cpp
    src/mlp.cpp
    src/objective_functions.cpp
    src/optimizers.cpp
    src/train.cpp
    src/wolfe_analysis.cpp
)

add_library(nncpp STATIC ${NNCPP_CORE_SOURCES})
add_library(nncpp::nncpp ALIAS nncpp)

target_include_directories(nncpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/NNCPPP-Practice>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(MSVC)
    target_compile_options(nncpp PRIVATE /W4 /permissive-)
else()
    target_compile_options(nncpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(NNCPP_BUILD_EXAMPLES)
    add_executable(nncpp_example examples/nncpp_library_example.cpp)
    target_link_libraries(nncpp_example PRIVATE nncpp::nncpp)
endif()

if(BUILD_TESTING)
    function(add_nncpp_test test_name source label)
        add_executable(${test_name} ${source})
        target_link_libraries(${test_name} PRIVATE nncpp::nncpp)
        add_test(NAME ${test_name} COMMAND ${test_name})
        set_tests_properties(${test_name} PROPERTIES LABELS "${label}")
        if(MINGW)
            get_filename_component(NNCPP_COMPILER_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)
            set_tests_properties(${test_name} PROPERTIES
                ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${NNCPP_COMPILER_BIN}"
            )
        endif()
    endfunction()

    add_nncpp_test(nncpp_test_numerics tests/test_numerics.cpp Numerics)
    add_nncpp_test(nncpp_test_initializers tests/test_initializers.cpp Initialization)
    add_nncpp_test(nncpp_test_activations tests/test_activations.cpp Activations)
    add_nncpp_test(nncpp_test_adamw tests/test_adamw.cpp Optimizers)
    add_nncpp_test(nncpp_test_cadamw tests/test_cadamw.cpp Optimizers)
    add_nncpp_test(nncpp_test_proximal tests/test_proximal.cpp Regularization)
    add_nncpp_test(nncpp_test_regularization tests/test_regularization.cpp Regularization)

    add_executable(nncpp_legacy_self_checks
        tests/legacy_main.cpp
        NNCPPP-Practice/self_checks.cpp
    )
    target_link_libraries(nncpp_legacy_self_checks PRIVATE nncpp::nncpp)
    add_test(NAME nncpp_legacy_self_checks COMMAND nncpp_legacy_self_checks)
    set_tests_properties(nncpp_legacy_self_checks PROPERTIES LABELS "Legacy;Regression")
    if(MINGW)
        get_filename_component(NNCPP_COMPILER_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)
        set_tests_properties(nncpp_legacy_self_checks PROPERTIES
            ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${NNCPP_COMPILER_BIN}"
        )
    endif()
endif()

install(TARGETS nncpp
    EXPORT NeuralNetworkCPPTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/nncpp/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nncpp
    FILES_MATCHING PATTERN "*.hpp"
)
install(FILES include/nncpp/nncpp.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nncpp
)

install(EXPORT NeuralNetworkCPPTargets
    FILE NeuralNetworkCPPTargets.cmake
    NAMESPACE nncpp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuralNetworkCPP
)

configure_package_config_file(
    cmake/NeuralNetworkCPPConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/NeuralNetworkCPPConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuralNetworkCPP
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/NeuralNetworkCPPConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/NeuralNetworkCPPConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NeuralNetworkCPPConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuralNetworkCPP
)
