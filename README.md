# NablaNet

NablaNet is a dependency-free C++20 multilayer-perceptron library written
from first principles. It is an MIT-licensed educational project: the public
API and test suite are designed to make model construction, objectives,
optimization, and curvature methods inspectable.

The initial `0.x` series is static-library only. It provides dense MLPs,
configurable activations and initializers, binary/softmax objectives, MSE,
regularization, SGD through AdamW/CAdamW, deterministic training, exact
Hessian-vector products, Newton-CG, and L-BFGS.

## Requirements

- CMake 3.21 or newer (CTest is distributed with CMake).
- A C++20 compiler. On Windows, Visual Studio 2026 with the Desktop
  development with C++ workload is supported.
- No third-party C++ dependencies.

## Build and test

Run these commands from the repository root.

### Windows and Visual Studio 2026

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 `
  -DBUILD_TESTING=ON `
  -DNABLANET_BUILD_EXAMPLES=ON

cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

### Linux, macOS, or another single-configuration generator

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DNABLANET_BUILD_EXAMPLES=ON

cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Focused objective checks are available with:

```sh
ctest --test-dir build -L Objectives --output-on-failure
```

## Use from CMake

Install the static library to a prefix:

```powershell
cmake --install build --config Release --prefix "$PWD/install"
```

Then consume the exported package from another CMake project:

```cmake
find_package(NablaNet CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE NablaNet::NablaNet)
```

The public umbrella header is `<nablanet/nablanet.hpp>`, and all public C++
symbols live in `namespace nablanet`:

```cpp
#include <nablanet/nablanet.hpp>

const nablanet::MLP network = nablanet::make_mlp({2, 3, 1}, 7);
```

## Conan 2

The repository contains a Conan 2 recipe for the static package
`nablanet/0.1.0`. Validate it locally with Conan 2:

```sh
conan profile detect --force
conan create . -s build_type=Release -s compiler.cppstd=20
conan create . -s build_type=Debug -s compiler.cppstd=20
```

The recipe uses the CMake install rules and its `test_package` builds and runs
an independent consumer linked through `NablaNet::NablaNet`. The project is
Conan-ready; publishing to a remote is an explicit later release action.

## Compatibility and releases

NablaNet follows semantic versioning. Before `1.0.0`, source and API changes
may occur in minor releases; patch releases are reserved for compatible fixes.
Each release must update `VERSION` and `CHANGELOG.md`, pass the CTest suite,
verify the installed CMake consumer, and pass `conan create .` for the
supported configurations.

See [LICENSE.txt](LICENSE.txt) for the MIT license.
