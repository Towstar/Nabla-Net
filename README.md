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

## Standalone norms

`<nablanet/norms.hpp>` (also included by the umbrella header) provides read-only
norm diagnostics independently of objectives and regularizers:

```cpp
const auto l2 = nablanet::make_lp_norm(2.0);
const double length = nablanet::vector_norm({3.0, 4.0}, l2); // 5, not 25
const double parameters = nablanet::network_norm(network, l2); // with biases
const double weights = nablanet::network_norm(network, l2, false);
const auto gradients = nablanet::make_zero_gradients_like(network);
const double gradient_length = nablanet::network_norm(gradients, l2);
```

The factory accepts any finite real `p >= 1`, including fractional values, or
`std::numeric_limits<double>::infinity()` for the maximum absolute component.
It computes `(sum(abs(x)^p))^(1/p)` using maximum-magnitude scaling. Empty and
zero vectors return zero. An L2 norm is the square root of a sum of squares;
the existing L2 regularizer uses a squared penalty. Norms do not add losses,
gradients, proximal updates, or change optimizer/stopping behavior.

`NormFunction` holds a display `name` and an `evaluate(const Values&)` callback.
Use the helpers to validate custom evaluations. They require a nonempty name,
a callback, and finite inputs. Negative/NaN results throw `std::invalid_argument`;
positive infinite results throw `std::overflow_error`. Factory input errors
also throw `std::invalid_argument`; unrepresentable norms throw
`std::overflow_error`. Other callback exceptions propagate. Empty input returns
zero without invoking the callback; custom callbacks must obey the norm axioms
on nonempty inputs, which runtime validation cannot prove.

For example, a caller can supply a weighted L1 norm:

```cpp
const nablanet::NormFunction weighted_l1{
    "weighted L1", [](const nablanet::Values& values) {
        double result = 0.0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            result += static_cast<double>(i + 1) * std::abs(values[i]);
        }
        return result;
    }
}; // include <cmath> for std::abs
```

`network_norm` is overloaded for `MLP` parameters and `NetworkGradients`.
Both overloads evaluate one global vector in layer order:
weights, then selected biases, in their stored order. Biases are included by
default. Parameters require a valid network (including finite excluded biases);
the gradient overload requires only finite selected components, not a reference
network. These are entrywise vector norms, not induced matrix/operator norms.
The existing `gradient_l2_norm` and `maximum_absolute_gradient` APIs are unchanged.

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
