import os

from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import load


class NablaNetConan(ConanFile):
    name = "nablanet"
    license = "MIT"
    homepage = "https://github.com/Towstar/Neural-Network-Implementation-In-CPP"
    url = "https://github.com/Towstar/Neural-Network-Implementation-In-CPP"
    description = "A dependency-free C++20 multilayer-perceptron learning library."
    topics = ("machine-learning", "neural-network", "cpp20", "education")
    package_type = "static-library"
    required_conan_version = ">=2.0"

    settings = "os", "compiler", "build_type", "arch"

    exports_sources = (
        "CMakeLists.txt",
        "VERSION",
        "LICENSE.txt",
        "README.md",
        "cmake/*",
        "include/*",
        "src/*",
        "src/detail/*",
    )

    def set_version(self):
        self.version = load(self, os.path.join(self.recipe_folder, "VERSION")).strip()

    def layout(self):
        cmake_layout(self)

    def validate(self):
        check_min_cppstd(self, "20")

    def generate(self):
        dependencies = CMakeDeps(self)
        dependencies.generate()

        toolchain = CMakeToolchain(self)
        toolchain.variables["BUILD_TESTING"] = False
        toolchain.variables["NABLANET_BUILD_EXAMPLES"] = False
        toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["NablaNet"]
        self.cpp_info.set_property("cmake_file_name", "NablaNet")
        self.cpp_info.set_property("cmake_target_name", "NablaNet::NablaNet")
