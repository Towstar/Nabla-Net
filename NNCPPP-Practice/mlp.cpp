#include "mlp.hpp"
#include "objective_functions.hpp"
#include "common.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <format>
#include <stdexcept>
#include <utility>

#pragma region Dense Layer Accessors

double& DenseLayer::weight(const std::size_t output_index, const std::size_t input_index)
{
    return weights[output_index * input_size + input_index];
}

const double& DenseLayer::weight(const std::size_t output_index, const std::size_t input_index) const
{
    return weights[output_index * input_size + input_index];
}

#pragma endregion

#pragma region Activation Function Scaffolding

ActivationFunction make_elementwise_activation(
    std::string name,
    ScalarActivationForward forward,
    ScalarActivationDerivative derivative,
    const ActivationParameter parameter
)
{
    if (name.empty()) {
        throw std::invalid_argument(
            "Activation name must not be empty."
        );
    }

    if (!forward || !derivative) {
        throw std::invalid_argument(
            "Activation must provide forward and derivative callbacks."
        );
    }

    if (parameter.has_value() && !std::isfinite(*parameter)) {
        throw std::invalid_argument(
            "Activation parameter must be finite when provided."
        );
    }

    ActivationFunction activation;
    activation.name = std::move(name);

    activation.forward = [forward, parameter](
        const Values& pre_activations
    ) {
        if (pre_activations.empty()) {
            throw std::invalid_argument(
                "Activation input must not be empty."
            );
        }

        Values result;
        result.reserve(pre_activations.size());

        for (const double input : pre_activations) {
            if (!std::isfinite(input)) {
                throw std::invalid_argument(
                    "Activation inputs must be finite."
                );
            }

            const double output = forward(input, parameter);
            if (!std::isfinite(output)) {
                throw std::runtime_error(
                    "Activation forward callback produced a non-finite value."
                );
            }

            result.push_back(output);
        }

        return result;
    };

    activation.backward = [derivative, parameter](
        const Values& pre_activations,
        const Values& upstream_gradient
    ) {
        if (pre_activations.empty()) {
            throw std::invalid_argument(
                "Activation input must not be empty."
            );
        }

        if (pre_activations.size() != upstream_gradient.size()) {
            throw std::invalid_argument(
                "Activation input and upstream-gradient sizes must match."
            );
        }

        Values result(pre_activations.size(), 0.0);

        for (std::size_t i = 0; i < pre_activations.size(); ++i) {
            const double input = pre_activations[i];
            const double upstream = upstream_gradient[i];

            if (!std::isfinite(input) || !std::isfinite(upstream)) {
                throw std::invalid_argument(
                    "Activation backward inputs must be finite."
                );
            }

            const double local_derivative = derivative(input, parameter);
            if (!std::isfinite(local_derivative)) {
                throw std::runtime_error(
                    "Activation derivative callback produced a non-finite value."
                );
            }

            result[i] = local_derivative * upstream;
            if (!std::isfinite(result[i])) {
                throw std::overflow_error(
                    "Activation backward result is not finite."
                );
            }
        }

        return result;
    };

    return activation;
}

namespace Activations {
    const ActivationFunction Tanh = make_elementwise_activation(
        "tanh",
        [](double z, ActivationParameter) { return std::tanh(z); },
        [](double z, ActivationParameter) {
            const double value = std::tanh(z);
            return 1.0 - value * value;
        }
    );

    const ActivationFunction Linear = make_elementwise_activation(
        "linear",
        [](double z, ActivationParameter) { return z; },
        [](double, ActivationParameter) { return 1.0; }
    );

    /// <summary>
    /// Rectified Linear Unit Hidden Activation Function.
    /// Implemented such that the subgradient value at z = 0 is 0
    /// since the derivative is undefined at z = 0
    /// (LHS limit and RHS limit are not equal.)
    /// </summary>
    const ActivationFunction ReLU = make_elementwise_activation(
        "relu",
        [](double z, ActivationParameter) { return std::max(0.0, z); },
        [](double z, ActivationParameter) {
            return z > 0.0 ? 1.0 : 0.0;
        }
    );

    const ActivationFunction LeakyReLU = make_elementwise_activation(
        "leaky_relu",
        [](double z, ActivationParameter negative_slope) {
            const double slope = negative_slope.value_or(0.01);
            return z > 0.0 ? z : slope * z;
        },
        [](double z, ActivationParameter negative_slope) {
            const double slope = negative_slope.value_or(0.01);
            return z > 0.0 ? 1.0 : slope;
        }
    );

    const ActivationFunction ELU = make_elementwise_activation(
        "elu",
        [](double z, ActivationParameter negative_coefficient) {
            const double coefficient = negative_coefficient.value_or(1.0);
            return z > 0.0
                ? z
                : coefficient * (std::exp(z) - 1.0);
        },
        [](double z, ActivationParameter negative_coefficient) {
            const double coefficient = negative_coefficient.value_or(1.0);
            return z > 0.0
                ? 1.0
                : coefficient * std::exp(z);
        }
    );

    const ActivationFunction GELU = make_elementwise_activation(
        "gelu",
        [](double z, ActivationParameter) { return z * normalcdf(z); },
        [](double z, ActivationParameter) {
            return normalcdf(z) + z * normalpdf(z);
        }
    );

    /// <summary>
    /// Equivalent to swish(z) where beta (trainable parameter) = 1
    /// </summary>
    const ActivationFunction SiLU = make_elementwise_activation(
        "silu",
        [](double z, ActivationParameter) { return z * stable_sigmoid(z); },
        [](double z, ActivationParameter) {
            const double sigmoid = stable_sigmoid(z);
            return sigmoid + z * sigmoid * (1.0 - sigmoid);
        }
    );

    const ActivationFunction Mish = make_elementwise_activation(
        "mish",
        [](double z, ActivationParameter) {
            return z * std::tanh(stable_softplus(z));
        },
        [](double z, ActivationParameter) {
            const double exp_2z = std::exp(2.0 * z);
            const double exp_z = std::exp(z);
            const double delta =
                4.0 * (z + 1.0) +
                4.0 * exp_2z +
                std::exp(3.0 * z) +
                exp_z * (4.0 * z + 6.0);
            const double omega = 2.0 * exp_z + exp_2z + 2.0;
            return (exp_z * omega) / (delta * delta);
        }
    );

    const ActivationFunction Sigmoid = make_elementwise_activation(
        "sigmoid",
        [](double z, ActivationParameter) { return stable_sigmoid(z); },
        [](double z, ActivationParameter) {
            const double sigmoid = stable_sigmoid(z);
            return (1.0 - sigmoid) * sigmoid;
        }
    );
}

#pragma endregion

#pragma region Network Validation and Construction

namespace {
    void validate_activation_descriptor(
        const ActivationFunction& activation
    )
    {
        if (activation.name.empty()) {
            throw std::invalid_argument(
                "Activation name must not be empty."
            );
        }

        if (!activation.forward) {
            throw std::invalid_argument(
                "Activation must provide a forward callback."
            );
        }

        if (!activation.backward) {
            throw std::invalid_argument(
                "Activation must provide a backward callback."
            );
        }
    }

    const ActivationFunction& effective_activation(
        const MLP& network,
        std::size_t layer_index
    )
    {
        if (layer_index >= network.layers.size()) {
            throw std::out_of_range("Invalid layer index.");
        }
        if (!network.layer_activations.empty()) {
            return network.layer_activations[layer_index];
        }

        return layer_index + 1 == network.layers.size()
            ? Activations::Sigmoid
            : Activations::Tanh;
    }
}

void validate_network_architecture(const std::vector<std::size_t>& layer_sizes)
{
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("Network architecture must have at least two layers.");
    }

    for (const std::size_t layer_size : layer_sizes) {
        if (layer_size == 0) {
            throw std::invalid_argument("Layer sizes must be greater than zero.");
        }
    }
}

void validate_network(const MLP& network)
{
    validate_network_architecture(network.layer_sizes);

    if (!network.layer_activations.empty()) {
        if (network.layer_activations.size() != network.layers.size()) {
            throw std::invalid_argument(
                "Activation count must match the dense-layer count."
            );
        }

        for (const ActivationFunction& activation :
            network.layer_activations) {
            validate_activation_descriptor(activation);
        }
    }

    if (network.layers.size() + 1 != network.layer_sizes.size()) {
        throw std::invalid_argument("Number of layers does not match network architecture.");
    }

    for (std::size_t k = 0; k < network.layers.size(); ++k) {
        const DenseLayer& layer = network.layers[k];

        if (layer.input_size != network.layer_sizes[k]) {
            throw std::invalid_argument("Layer input size does not match network architecture.");
        }

        if (layer.output_size != network.layer_sizes[k + 1]) {
            throw std::invalid_argument("Layer output size does not match network architecture.");
        }

        if (layer.weights.size() != layer.input_size * layer.output_size) {
            throw std::invalid_argument("Layer weight count does not match layer shape.");
        }

        if (layer.biases.size() != layer.output_size) {
            throw std::invalid_argument("Layer bias count does not match layer shape.");
        }
    }
}

std::size_t parameter_count(const MLP& network)
{
    validate_network(network);

    std::size_t total = 0;
    for (const DenseLayer& layer : network.layers) {
        total += layer.output_size * layer.input_size;
        total += layer.output_size;
    }

    return total;
}

MLP make_zero_network(std::vector<std::size_t> layer_sizes)
{
    validate_network_architecture(layer_sizes);

    MLP network;
    network.layer_sizes = layer_sizes;

    for (std::size_t k = 0; k + 1 < layer_sizes.size(); ++k) {
        DenseLayer layer;
        layer.input_size = layer_sizes[k];
        layer.output_size = layer_sizes[k + 1];
        layer.weights.resize(layer.input_size * layer.output_size, 0.0);
        layer.biases.resize(layer.output_size, 0.0);

        network.layers.push_back(layer);
    }

    return network;
}

namespace {
    void initialize_network_glorot(
        MLP& network,
        const std::vector<std::size_t>& layer_sizes,
        std::uint32_t seed
    );

    void initialize_network_kaiming(
        MLP& network,
        const std::vector<std::size_t>& layer_sizes,
        std::uint32_t seed
    );

    void initialize_network_lecunn(
        MLP& network,
        const std::vector<std::size_t>& layer_sizes,
        std::uint32_t seed
    );

    void initialize_network_zero(
        MLP& network,
        const std::vector<std::size_t>& layer_sizes,
        std::uint32_t seed
    );
}

void initialize_network(
    MLP& network,
    const InitializationType initialization_type,
    const std::vector<std::size_t>& layer_sizes,
    const std::uint32_t seed
)
{
    switch (initialization_type) {
    case InitializationType::XavierGlorot:
        initialize_network_glorot(network, layer_sizes, seed);
        break;
    case InitializationType::KaimingHe:
        initialize_network_kaiming(network, layer_sizes, seed);
        break;
    case InitializationType::LeCunn:
        initialize_network_lecunn(network, layer_sizes, seed);
        break;
    case InitializationType::Zero:
        initialize_network_zero(network, layer_sizes, seed);
        break;
    default:
        throw std::invalid_argument("Unknown initialization type.");
    }
}

namespace {
    void initialize_network_glorot(MLP& network, const std::vector<std::size_t>& layer_sizes, const std::uint32_t seed) {
        std::mt19937 rng(seed);

        for (std::size_t i = 0; i + 1 < layer_sizes.size(); ++i) {
            DenseLayer layer;
            layer.input_size = layer_sizes[i];
            layer.output_size = layer_sizes[i + 1];
            const double limit = std::sqrt(
                6.0 / static_cast<double>(layer.input_size + layer.output_size)
            );

            std::uniform_real_distribution<double> distribution(-limit, limit);

            layer.weights.resize(layer.input_size * layer.output_size);

            for (double& weight : layer.weights) {
                weight = distribution(rng);
            }

            layer.biases.resize(layer.output_size, 0.0);

            network.layers.push_back(layer);
        }

    }
    void initialize_network_kaiming(MLP& network, const std::vector<std::size_t>& layer_sizes, const std::uint32_t seed) {

    }
    void initialize_network_lecunn(MLP& network, const std::vector<std::size_t>& layer_sizes, const std::uint32_t seed) {

    }
    void initialize_network_zero(MLP& network, const std::vector<std::size_t>& layer_sizes, const std::uint32_t seed) {

    }
}

MLP make_mlp(const std::vector<std::size_t>& layer_sizes, const std::uint32_t seed, InitializationType initialization_type)
{
    validate_network_architecture(layer_sizes);

    std::mt19937 rng(seed);

    MLP network;
    network.layer_sizes = layer_sizes;

    initialize_network(network, initialization_type, layer_sizes, seed);

    validate_network(network);
    return network;
}

MLP make_mlp(const NetworkSpec& specification) {
    MLP network = make_mlp(specification.layer_sizes, specification.seed, specification.initialization_type);
    network.layer_activations = specification.layer_activations;
    validate_network(network);
    return network;
}

#pragma endregion

#pragma region Forward Propagation

double stable_sigmoid(double x)
{
    if (x < 0.0) {
        const double e = std::exp(x);
        return e / (1.0 + e);
    }

    const double e = std::exp(-x);
    return 1.0 / (1.0 + e);
}

ForwardCache forward_pass(const MLP& network, const Values& input)
{
    validate_network(network);

    if (input.size() != network.layer_sizes.front()) {
        throw std::invalid_argument("Input size does not match network input layer size.");
    }

    for (const double value : input) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Input values must be finite.");
        }
    }

    ForwardCache cache;
    cache.activations.push_back(input);

    for (std::size_t layer_index = 0; layer_index < network.layers.size(); ++layer_index) {
        const DenseLayer& layer = network.layers[layer_index];
        const Values& previous_activation = cache.activations[layer_index];

        Values pre_activation(layer.output_size, 0.0);

        for (std::size_t output_index = 0; output_index < layer.output_size; output_index++) {
                double z = layer.biases[output_index];

                for (std::size_t input_index = 0; input_index < layer.input_size; ++input_index) {
                    z += layer.weight(output_index, input_index) * previous_activation[input_index];
                }

                if (!std::isfinite(z)) {
                    throw std::runtime_error("Forward pass produced a non-finite pre-activation.");
                }

                pre_activation[output_index] = z;
        }
        cache.pre_activations.push_back(pre_activation);

        const ActivationFunction& activation_function = effective_activation(network, layer_index);
        Values activation = activation_function.forward(pre_activation);
        if (activation.size() != layer.output_size) {
            throw std::invalid_argument(
                "Activation result size does not match layer output size."
            );
        }

        for (const double value : activation) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Activation result must be finite."
                );
            }
        }
        cache.activations.push_back(activation);
    }

    return cache;
}

#pragma endregion

#pragma region Gradient Storage and Input Validation

NetworkGradients make_zero_gradients_like(const MLP& network) {
	validate_network(network);
	NetworkGradients gradients;
    for (const DenseLayer& layer : network.layers) {
        if (layer.weights.size() != layer.input_size * layer.output_size) {
            throw std::invalid_argument("Layer weight count does not match layer shape.");
        }
        if (layer.biases.size() != layer.output_size) {
            throw std::invalid_argument("Layer bias count does not match layer shape.");
        }
		LayerGradients layer_gradients;
		layer_gradients.weights.resize(layer.weights.size(), 0.0);
		layer_gradients.biases.resize(layer.biases.size(), 0.0);
		gradients.layers.push_back(layer_gradients);
	}
    return gradients;
}

void validate_backward_inputs(const MLP& network, const ForwardCache& cache, const Values& target)
{
    validate_network(network);

    if (cache.activations.size() != network.layer_sizes.size()) {
        throw std::invalid_argument(
            "Forward cache activation count does not match network architecture."
        );
    }

    if (cache.pre_activations.size() != network.layers.size()) {
        throw std::invalid_argument(
            "Forward cache pre-activation count does not match network layers."
        );
    }

    for (std::size_t layer_index = 0; layer_index < cache.activations.size(); ++layer_index) {
        const Values& activation = cache.activations[layer_index];

        if (activation.size() != network.layer_sizes[layer_index]) {
            throw std::invalid_argument(
                "Forward cache activation size does not match network architecture."
            );
        }

        for (const double value : activation) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Forward cache activations must be finite.");
            }
        }
    }

    for (std::size_t layer_index = 0;
         layer_index < cache.pre_activations.size();
         ++layer_index) {
        const Values& pre_activation = cache.pre_activations[layer_index];

        if (pre_activation.size() != network.layers[layer_index].output_size) {
            throw std::invalid_argument(
                "Forward cache pre-activation size does not match layer output size."
            );
        }

        for (const double value : pre_activation) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Forward cache pre-activations must be finite.");
            }
        }
    }

    if (target.size() != network.layer_sizes.back()) {
        throw std::invalid_argument("Target size does not match network output layer size.");
    }

    for (const double value : target) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Target values must be finite.");
        }

        if (value < 0.0 || value > 1.0) {
            throw std::invalid_argument("Target values must be in the range [0,1].");
        }
    }
}

void validate_backward_inputs(const MLP& network, const ForwardCache& cache, const Values& target, const ObjectiveFunctions& objective)
{
    validate_objective_functions(objective);
    validate_network(network);

    if (cache.activations.size() != network.layer_sizes.size()) {
        throw std::invalid_argument(
            "Forward cache activation count does not match network architecture."
        );
    }

    if (cache.pre_activations.size() != network.layers.size()) {
        throw std::invalid_argument(
            "Forward cache pre-activation count does not match network layers."
        );
    }

    for (std::size_t layer_index = 0; layer_index < cache.activations.size(); ++layer_index) {
        const Values& activation = cache.activations[layer_index];

        if (activation.size() != network.layer_sizes[layer_index]) {
            throw std::invalid_argument(
                "Forward cache activation size does not match network architecture."
            );
        }

        for (const double value : activation) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Forward cache activations must be finite.");
            }
        }
    }

    for (std::size_t layer_index = 0;
        layer_index < cache.pre_activations.size();
        ++layer_index) {
        const Values& pre_activation = cache.pre_activations[layer_index];

        if (pre_activation.size() != network.layers[layer_index].output_size) {
            throw std::invalid_argument(
                "Forward cache pre-activation size does not match layer output size."
            );
        }

        for (const double value : pre_activation) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Forward cache pre-activations must be finite.");
            }
        }
    }

    if (target.size() != network.layer_sizes.back()) {
        throw std::invalid_argument("Target size does not match network output layer size.");
    }

    for (const double value : target) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Target values must be finite.");
        }
    }
}

#pragma endregion

#pragma region Backward Propagation

NetworkGradients backward(const MLP& network, const ForwardCache& cache, const Values& target) {
        return backward(
            network,
            cache,
            target,
            make_binary_cross_entropy_objective()
        );
}

NetworkGradients backward(const MLP& network, const ForwardCache& cache, const Values& target, const ObjectiveFunctions& objective) {
    validate_backward_inputs(network, cache, target, objective);

    NetworkGradients gradients = make_zero_gradients_like(network);

    std::vector<Values> deltas;
    deltas.reserve(network.layers.size());
    for (const DenseLayer& layer : network.layers) {
        deltas.emplace_back(layer.output_size, 0.0);
    }

    const std::size_t output_layer_index = network.layers.size() - 1;
    const std::size_t output_width = network.layers[output_layer_index].output_size;
    const Values& output_logits = cache.pre_activations.back();
    
    if (!(output_logits.size() == output_width)) {
        throw std::invalid_argument("Output logits size is not equal to output width");
    }

    const Values output_sample_loss_grad =
        objective.sample_loss_gradient(output_logits, target);

    if (output_sample_loss_grad.size() != output_width) {
        throw std::invalid_argument(
            "Objective gradient size does not match the network output width."
        );
    }
    
    for (const double component : output_sample_loss_grad) {
        if (!std::isfinite(component)) {
            throw std::invalid_argument(
                "Objective gradient values must be finite."
            );
        }
    }
    
    deltas[output_layer_index] = output_sample_loss_grad;

    for (std::size_t next_layer_index = output_layer_index;
        next_layer_index > 0;
        --next_layer_index) {
        const std::size_t layer_index = next_layer_index - 1;
        const DenseLayer& layer = network.layers[layer_index];
        const DenseLayer& next_layer = network.layers[next_layer_index];

        Values transported_grad(layer.output_size, 0.0);

        for (std::size_t j = 0; j < layer.output_size; ++j) {
            for (std::size_t r = 0; r < next_layer.output_size; ++r) {
                transported_grad[j] += next_layer.weight(r, j) * deltas[next_layer_index][r];
            }
        }
        for (const double value : transported_grad) {
            if (!std::isfinite(value)) {
                throw std::overflow_error(
                    "Transported gradient is not finite."
                );
            }
        }
        const ActivationFunction& activation_function = effective_activation(network, layer_index);
        Values hidden_delta =
            activation_function.backward(
                cache.pre_activations[layer_index],
                transported_grad
            );

        if (hidden_delta.size() != layer.output_size) {
            throw std::invalid_argument(
                "Activation backward result size does not match layer output size."
            );
        }

        for (const double value : hidden_delta) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Activation backward result must be finite."
                );
            }
        }

        deltas[layer_index] = std::move(hidden_delta);
    }

    for (std::size_t layer_index = 0; layer_index < network.layers.size(); ++layer_index) {
        const DenseLayer& layer = network.layers[layer_index];
        const Values& previous_activation = cache.activations[layer_index];
        LayerGradients& layer_gradients = gradients.layers[layer_index];

        for (std::size_t j = 0; j < layer.output_size; ++j) {
            layer_gradients.biases[j] = deltas[layer_index][j];

            for (std::size_t i = 0; i < layer.input_size; ++i) {
                layer_gradients.weights[j * layer.input_size + i] =
                    deltas[layer_index][j] * previous_activation[i];
            }
        }
    }

    return gradients;
}

NetworkGradients batch_gradients(const MLP& network, const Dataset& batch) {
    return batch_gradients(
        network,
        batch,
        make_binary_cross_entropy_objective()
    );
}

NetworkGradients batch_gradients(const MLP& network, const Dataset& batch, const ObjectiveFunctions& objective) {
    if (batch.empty()) {
        throw std::invalid_argument("Batch must not be empty.");
    }
    validate_network(network);
	validate_objective_functions(objective);

    NetworkGradients averaged = make_zero_gradients_like(network);

    for (const Sample& sample : batch) {
        const ForwardCache cache = forward_pass(network, sample.input);
        const NetworkGradients sample_gradients = backward(network, cache, sample.target, objective);

        for (std::size_t layer_index = 0; layer_index < averaged.layers.size(); layer_index++) {
            for (std::size_t k = 0; k < averaged.layers[layer_index].weights.size(); ++k) {
                averaged.layers[layer_index].weights[k] += sample_gradients.layers[layer_index].weights[k];
            }
            for (std::size_t j = 0; j < averaged.layers[layer_index].biases.size(); ++j) {
                averaged.layers[layer_index].biases[j] += sample_gradients.layers[layer_index].biases[j];
            }
        }
    }

    for (std::size_t layer_index = 0; layer_index < averaged.layers.size(); layer_index++) {
        for (std::size_t j = 0; j < averaged.layers[layer_index].biases.size(); j++) {
            averaged.layers[layer_index].biases[j] /= static_cast<double>(batch.size());
        }
        for (std::size_t k = 0; k < averaged.layers[layer_index].weights.size(); k++) {
            averaged.layers[layer_index].weights[k] /= static_cast<double>(batch.size());
        }
    }

    return averaged;
}

#pragma endregion

#pragma region Gradient Diagnostics and Updates

double gradient_l2_norm(const NetworkGradients& gradients) {
    double sum_of_squares = 0.0;

    for (const LayerGradients& layer_gradients : gradients.layers) {
        for (const double gradient : layer_gradients.weights) {
            if (!std::isfinite(gradient)) {
                throw std::invalid_argument(
                    "Gradient values must be finite."
                );
            }

            sum_of_squares += gradient * gradient;
            if (!std::isfinite(sum_of_squares)) {
                throw std::overflow_error(
                    "Gradient L2 norm overflowed."
                );
            }
        }

        for (const double gradient : layer_gradients.biases) {
            if (!std::isfinite(gradient)) {
                throw std::invalid_argument(
                    "Gradient values must be finite."
                );
            }

            sum_of_squares += gradient * gradient;
            if (!std::isfinite(sum_of_squares)) {
                throw std::overflow_error(
                    "Gradient L2 norm overflowed."
                );
            }
        }
	}

    return std::sqrt(sum_of_squares);
}

double maximum_absolute_gradient(const NetworkGradients& gradients) {
    double maximum = 0.0;

    for (const LayerGradients& layer_gradients : gradients.layers) {
        for (const double gradient : layer_gradients.weights) {
            if (!std::isfinite(gradient)) {
                throw std::invalid_argument(
                    "Gradient values must be finite."
                );
            }

            maximum = std::max(maximum, std::abs(gradient));
        }

        for (const double gradient : layer_gradients.biases) {
            if (!std::isfinite(gradient)) {
                throw std::invalid_argument(
                    "Gradient values must be finite."
                );
            }

            maximum = std::max(maximum, std::abs(gradient));
        }
    }

    return maximum;
}

bool learning_rate_is_valid(const double learning_rate) noexcept
{
    return std::isfinite(learning_rate) && learning_rate > 0.0;
}

void validate_gradients_like(const MLP& network, const NetworkGradients& gradients) {
    validate_network(network);
    if (gradients.layers.size() != network.layers.size()) {
        throw std::invalid_argument(
            "Gradient layer count does not match network layer count."
        );
    }
    for (std::size_t i = 0; i < network.layers.size(); ++i) {
        for (std::size_t j = 0; j < gradients.layers[i].weights.size(); ++j) {
            if (!std::isfinite(gradients.layers[i].weights[j]))
                throw std::invalid_argument("Weight gradient is not finite");
        }
        for (std::size_t j = 0; j < gradients.layers[i].biases.size(); ++j) {
            if (!std::isfinite(gradients.layers[i].biases[j]))
                throw std::invalid_argument("Bias gradient is not finite");
        }
        if (gradients.layers[i].weights.size() != network.layers[i].weights.size()) {
            throw std::invalid_argument(
                "Gradient weight count does not match network layer weight count."
            );
        }
        if (gradients.layers[i].biases.size() != network.layers[i].biases.size()) {
            throw std::invalid_argument(
                "Gradient bias count does not match network layer bias count."
            );
        }
    }
}

void apply_gradient(MLP& network, const NetworkGradients& gradients, const double learning_rate)
{
    if (!learning_rate_is_valid(learning_rate)) {
        throw std::invalid_argument("Learning rate must be positive and finite.");
    }

	validate_gradients_like(network, gradients);
	MLP candidate = network;

    for (std::size_t layer_index = 0;
        layer_index < network.layers.size();
        ++layer_index) {
        for (std::size_t parameter_index = 0;
             parameter_index < network.layers[layer_index].weights.size(); parameter_index++) {
             const double updated_parameter = network.layers[layer_index].weights[parameter_index] -
                learning_rate * gradients.layers[layer_index].weights[parameter_index];

            if (!std::isfinite(updated_parameter))
                throw std::overflow_error("Updated weight is not finite.");

            candidate.layers[layer_index].weights[parameter_index] =
                updated_parameter;
        }

        for (std::size_t parameter_index = 0; parameter_index < network.layers[layer_index].biases.size(); parameter_index++) {
            const double updated_parameter = network.layers[layer_index].biases[parameter_index] -
                learning_rate * gradients.layers[layer_index].biases[parameter_index];

            if (!std::isfinite(updated_parameter))
                throw std::overflow_error("Updated bias is not finite.");

            candidate.layers[layer_index].biases[parameter_index] =
                updated_parameter;
        }
    }

	network = candidate;
}

#pragma endregion
