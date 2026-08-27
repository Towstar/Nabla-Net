cmake_minimum_required(VERSION 3.21)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" NABLANET_VERSION)
string(STRIP "${NABLANET_VERSION}" NABLANET_VERSION)
project(NablaNet VERSION ${NABLANET_VERSION} LANGUAGES CXX)

include(GNUInstallDirs)
include(CTest)
include(CMakePackageConfigHelpers)

option(NABLANET_BUILD_EXAMPLES "Build NablaNet usage examples" OFF)

set(NABLANET_CORE_SOURCES
    src/common.cpp
    src/mlp.cpp
    src/objective_functions.cpp
    src/optimizers.cpp
    src/train.cpp
    src/wolfe_analysis.cpp
)

add_library(NablaNet STATIC ${NABLANET_CORE_SOURCES})
add_library(NablaNet::NablaNet ALIAS NablaNet)

target_compile_features(NablaNet PUBLIC cxx_std_20)
set_target_properties(NablaNet PROPERTIES CXX_EXTENSIONS OFF)

target_include_directories(NablaNet
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(MSVC)
    target_compile_options(NablaNet PRIVATE /W4 /permissive-)
else()
    target_compile_options(NablaNet PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(NABLANET_BUILD_EXAMPLES)
    add_executable(nablanet_example examples/nablanet_library_example.cpp)
    target_link_libraries(nablanet_example PRIVATE NablaNet::NablaNet)
endif()

if(BUILD_TESTING)
    function(add_nablanet_test test_name source label)
        add_executable(${test_name} ${source})
        target_link_libraries(${test_name} PRIVATE NablaNet::NablaNet)
        add_test(NAME ${test_name} COMMAND ${test_name})
        set_tests_properties(${test_name} PROPERTIES LABELS "${label}")
        if(MINGW)
            get_filename_component(NABLANET_COMPILER_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)
            set_tests_properties(${test_name} PROPERTIES
                ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${NABLANET_COMPILER_BIN}"
            )
        endif()
    endfunction()

    add_nablanet_test(nablanet_test_numerics tests/test_numerics.cpp Numerics)
    add_nablanet_test(nablanet_test_initializers tests/test_initializers.cpp Initialization)
    add_nablanet_test(nablanet_test_activations tests/test_activations.cpp Activations)
    add_nablanet_test(nablanet_test_adamw tests/test_adamw.cpp Optimizers)
    add_nablanet_test(nablanet_test_cadamw tests/test_cadamw.cpp Optimizers)
    add_nablanet_test(nablanet_test_proximal tests/test_proximal.cpp Regularization)
    add_nablanet_test(nablanet_test_regularization tests/test_regularization.cpp Regularization)
    add_nablanet_test(nablanet_test_autodiff tests/test_autodiff.cpp Autodiff)
    add_nablanet_test(nablanet_test_objective_hvps tests/test_objective_hvps.cpp Objectives)
    add_nablanet_test(nablanet_test_classification_objectives tests/test_classification_objectives.cpp Objectives)
    add_nablanet_test(nablanet_test_newton_cg tests/test_newton_cg.cpp Optimizers)
    add_nablanet_test(nablanet_test_lbfgs tests/test_lbfgs.cpp Optimizers)
    add_nablanet_test(nablanet_test_regression tests/test_regression.cpp Regression)
endif()

install(TARGETS NablaNet
    EXPORT NablaNetTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/nablanet
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(FILES LICENSE.txt
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/NablaNet
)

install(EXPORT NablaNetTargets
    FILE NablaNetTargets.cmake
    NAMESPACE NablaNet::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NablaNet
)

configure_package_config_file(
    cmake/NablaNetConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/NablaNetConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NablaNet
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/NablaNetConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/NablaNetConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NablaNetConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NablaNet
)
