# NeuralNetworkCPP

NeuralNetworkCPP is a dependency-free C++20 scientific-ML learning library
written from first principles. Its current core is a configurable multilayer
perceptron with manual forward propagation, backpropagation, objective
callbacks, regularization, optimizer factories, deterministic batching, and a
report-returning training facade.

The repository is organized as a reusable library, but remains explicit and
small enough to study. The canonical build is CMake-based. The older Visual
Studio project is retained as a compatibility and learning path.

## Get the project locally

From PowerShell, Git Bash, or a terminal:

```text
git clone <repository-url>
cd NeuralNetworkCPP
```

If you received the project as a folder or archive, open that folder instead.
The directory containing this README and `CMakeLists.txt` is the project root.

You need:

- CMake 3.21 or newer.
- A C++20 compiler.
- On Windows, Visual Studio 2022 with the Desktop development with C++
  workload, or a working MinGW/GCC installation.
- No third-party C++ libraries are required.

## Build and test with CMake

Run these commands from the project root.

### Windows with Visual Studio 2022

```powershell
cmake -S . -B build -A x64 `
  -DBUILD_TESTING=ON `
  -DNNCPP_BUILD_EXAMPLES=ON

cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

Run only optimizer tests:

```powershell
ctest --test-dir build -C Release -L Optimizers --output-on-failure
```

### MinGW or another single-configuration generator

```powershell
cmake -S . -B build-mingw `
  -G "MinGW Makefiles" `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_TESTING=ON `
  -DNNCPP_BUILD_EXAMPLES=ON

cmake --build build-mingw --parallel
ctest --test-dir build-mingw --output-on-failure
```

`build/` and `build-mingw/` are generated directories. They are intentionally
ignored by Git and can be deleted and regenerated at any time.

## Using Visual Studio or VS Code

### Visual Studio 2022 CMake mode

Use `File → Open → Folder` and select the directory containing this README and
`CMakeLists.txt`. Visual Studio will detect the CMake project and expose the
library, example, and test targets.

You can also generate a Visual Studio solution from a terminal:

```powershell
cmake -S . -B build-vs -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTING=ON
```

Open the generated solution under `build-vs/`. Do not edit generated project
files; edit `CMakeLists.txt` and regenerate instead.

The older solution remains at:

```text
NNCPPP-Practice/NNCPPP-Practice.sln
```

It is preserved for compatibility with the original Visual Studio workflow.

### VS Code

Open the project root in VS Code:

```powershell
code .
```

The Microsoft C/C++ and CMake Tools extensions are convenient but not required
when using the terminal commands above. The important rule is to edit the
canonical files under `src/`, `include/nncpp/`, `tests/`, and `examples/`.

## Repository structure

```text
include/nncpp/       Public headers installed for library users.
src/                 Library implementations.
tests/               Focused test executables registered with CTest.
examples/            Small consumer-facing examples.
cmake/               Installed-package configuration templates.
CMakeLists.txt       Library, example, test, install, and package definitions.
.github/workflows/   Windows CMake and CTest continuous integration.
NNCPPP-Practice/     Legacy Visual Studio project and compatibility checks.
```

The public umbrella include is:

```cpp
#include <nncpp/nncpp.hpp>
```

The CMake library target is `nncpp`, with the consumer-facing alias
`nncpp::nncpp`.

## Current functionality

- Dense MLP construction and validation.
- Deterministic Xavier, Kaiming-He, LeCun, and zero initialization.
- One selectable activation per dense layer.
- Stable sigmoid, normal PDF/CDF helpers, forward caches, manual backward
  propagation, batch gradients, and finite-difference checks.
- Binary cross-entropy, softmax cross-entropy, activated-output MSE, objective
  composition, L2, L1 subgradient, and proximal-L1 regularization.
- SGD, Momentum SGD, AdaGrad, RMSProp, Adam, AdamW, and cautious AdamW.
- Full-batch, mini-batch, and stochastic training with deterministic shuffling.
- Training stopping conditions and `TrainingReport` streaming.
- End-to-end XOR convergence through the public `train()` API.
- Wolfe and line-search groundwork for a future L-BFGS implementation.

## Testing

CTest runs the focused tests individually and supports labels:

```powershell
ctest --test-dir build -C Release --output-on-failure
ctest --test-dir build -C Release -L Activations --output-on-failure
ctest --test-dir build -C Release -L Initialization --output-on-failure
ctest --test-dir build -C Release -L Optimizers --output-on-failure
ctest --test-dir build -C Release -L Regularization --output-on-failure
```

The legacy `self_checks.cpp` executable remains a broad regression harness.
New behavior should normally receive a focused test under `tests/` and a CTest
label.

## Installation for another CMake project

After building, install the library to a local prefix:

```powershell
cmake --install build --config Release --prefix "$PWD/install"
```

An external CMake project can then use the exported target:

```cmake
find_package(NeuralNetworkCPP CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE nncpp::nncpp)
```

## Development notes

Generated build directories, compiler output, Codex scratch files, Visual
Studio caches, object files, and LaTeX auxiliary files are excluded by
`.gitignore`. Keep source code, tests, examples, public documentation, and
reproducible build configuration under version control.

Future performance work should document deterministic multithreaded CPU
execution for ordinary batches and PINN collocation points before parallel
execution is enabled. Determinism is a correctness requirement, not merely a
benchmark option.
