#include <nablanet/mlp.hpp>
#include <nablanet/objective_functions.hpp>

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

using namespace nablanet;

namespace {
void require_near(double actual, double expected, double tolerance, const char* message)
{
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}
}

int main()
{
    try {
        MLP network = make_zero_network({ 1, 1 });
        network.layer_activations = { Activations::Sigmoid };
        network.layers[0].weights[0] = 0.7;
        network.layers[0].biases[0] = -0.2;
        const Dataset batch{ { { 0.4 }, { 0.8 } } };
        const ObjectiveFunctions objective = make_mean_squared_error_objective();
        const ForwardCache cache = forward_pass(network, batch.front().input);
        const NetworkGradients gradients = backward(
            network,
            cache,
            batch.front().target,
            objective
        );

        constexpr double finite_difference_step = 1e-6;
        const double original = network.layers[0].weights[0];
        network.layers[0].weights[0] = original + finite_difference_step;
        const double plus = batch_loss(network, batch, objective);
        network.layers[0].weights[0] = original - finite_difference_step;
        const double minus = batch_loss(network, batch, objective);
        network.layers[0].weights[0] = original;

        const double numerical = (plus - minus) / (2.0 * finite_difference_step);
        require_near(
            gradients.layers[0].weights[0],
            numerical,
            1e-7,
            "MSE must pull the gradient through the configured output activation"
        );

        require_near(
            Activations::GELU.forward({ 0.0 })[0],
            0.0,
            1e-15,
            "GELU(0) must equal zero"
        );
        require_near(
            Activations::GELU.backward({ 0.0 }, { 1.0 })[0],
            0.5,
            1e-12,
            "GELU derivative at zero must equal one half"
        );

        std::cout << "[PASS] per-layer activation selection and output pullback\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << '\n';
        return 1;
    }
}
