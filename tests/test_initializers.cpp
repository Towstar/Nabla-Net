#include <nablanet/mlp.hpp>

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

using namespace nablanet;

namespace {
void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}
}

int main()
{
    try {
        for (const InitializationType type : {
                 InitializationType::KaimingHe,
                 InitializationType::LeCunn,
                 InitializationType::LeCun,
                 InitializationType::Zero
             }) {
            const MLP first = make_mlp({ 3, 5, 2 }, 17, type);
            const MLP second = make_mlp({ 3, 5, 2 }, 17, type);
            validate_network(first);
            require(first.layers.size() == 2, "initializer must create every layer");
            for (std::size_t layer_index = 0; layer_index < first.layers.size(); ++layer_index) {
                require(first.layers[layer_index].weights == second.layers[layer_index].weights,
                    "initializer must be deterministic for a fixed seed");
                require(first.layers[layer_index].biases == second.layers[layer_index].biases,
                    "initializer biases must be deterministic");
                for (const double value : first.layers[layer_index].weights) {
                    require(std::isfinite(value), "initializer weights must be finite");
                }
            }
            if (type == InitializationType::Zero) {
                for (const DenseLayer& layer : first.layers) {
                    for (const double value : layer.weights) {
                        require(value == 0.0, "zero initializer must zero weights");
                    }
                    for (const double value : layer.biases) {
                        require(value == 0.0, "zero initializer must zero biases");
                    }
                }
            }
        }

        std::cout << "[PASS] Kaiming, LeCun, and zero initializers\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << '\n';
        return 1;
    }
}
