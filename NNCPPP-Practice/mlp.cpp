#include "mlp.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <format>
#include <stdexcept>

double& DenseLayer::weight(const std::size_t output_index, const std::size_t input_index)
{
    return weights[output_index * input_size + input_index];
}

const double& DenseLayer::weight(const std::size_t output_index, const std::size_t input_index) const
{
    return weights[output_index * input_size + input_index];
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

MLP make_mlp(const std::vector<std::size_t>& layer_sizes, const std::uint32_t seed)
{
    validate_network_architecture(layer_sizes);

    std::mt19937 rng(seed);

    MLP network;
    network.layer_sizes = layer_sizes;

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

    validate_network(network);
    return network;
}

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
        const bool is_output_layer = layer_index + 1 == network.layers.size();

        Values pre_activation(layer.output_size, 0.0);
        Values activation(layer.output_size, 0.0);

        for (std::size_t output_index = 0; output_index < layer.output_size; ++output_index) {
            double z = layer.biases[output_index];

            for (std::size_t input_index = 0; input_index < layer.input_size; ++input_index) {
                z += layer.weight(output_index, input_index) * previous_activation[input_index];
            }

            if (!std::isfinite(z)) {
                throw std::runtime_error("Forward pass produced a non-finite pre-activation.");
            }

            pre_activation[output_index] = z;

            const double a = is_output_layer ? stable_sigmoid(z) : std::tanh(z);

            if (!std::isfinite(a)) {
                throw std::runtime_error("Forward pass produced a non-finite activation.");
            }

            activation[output_index] = a;
        }

        cache.pre_activations.push_back(pre_activation);
        cache.activations.push_back(activation);
    }

    return cache;
}

/// <summary>
/// Cost Function: Binary Cross-Entropy (BCE) from Logits, modified to be more numerically stable.
/// </summary>
/// <param name="logit"></param>
/// <param name="target"></param>
/// <returns></returns>
double binary_cross_entropy_from_logit(double logit, double target) {
    if (!std::isfinite(logit))
        throw std::invalid_argument("Logit value must be finite.");
    if (!std::isfinite(target))
        throw std::invalid_argument("Target value must be finite.");
    if (target < 0.0 || target > 1.0)
        throw std::invalid_argument("Target value must be in the range [0, 1].");

    return (std::max(logit, 0.0) - logit * target + log1p(exp(-abs(logit))));
}

ObjectiveFunctions make_binary_cross_entropy_objective()
{
    ObjectiveFunctions objective;

    objective.sample_loss = [](const Values& logit, const Values& target) 
    {
        double average = 0.0;
        if (logit.empty() || target.empty())
            throw std::invalid_argument("Logit size and Target Size must be greater than zero.");
        if (logit.size() != target.size())
            throw std::invalid_argument("Logit Size Must be Same Size as Target Size");
            
        for (int i = 0; i < logit.size(); i++) {
            if (!std::isfinite(target[i]) ||
                target[i] < 0.0 ||
                target[i] > 1.0) {
                throw std::invalid_argument(
                    "Targets must be finite and in the range [0,1]."
                );
            }
            double acx_term = (binary_cross_entropy_from_logit(logit[i], target[i])) / logit.size();
            if (!std::isfinite(acx_term))
                throw std::invalid_argument("Argument must be finite");
            double sum = average + acx_term;
            if (!std::isfinite(sum))
                throw std::invalid_argument("Argument must be finite");
            average = sum;
        }
        return average;
    };

    objective.sample_loss_gradient = [](const Values& logits, const Values& targets) -> Values{
        if (logits.empty() || targets.empty()) {
            throw std::invalid_argument(
                "Logits and targets must contain at least one value."
            );
        }

        if (logits.size() != targets.size()) {
            throw std::invalid_argument(
                "Logits and targets must have matching sizes."
            );
        }

        Values gradient(logits.size(), 0.0);
        const double output_width = static_cast<double>(logits.size());

        for (std::size_t i = 0; i < logits.size(); ++i) {
            if (!std::isfinite(logits[i])) {
                throw std::invalid_argument("Logits must be finite.");
            }

            if (!std::isfinite(targets[i]) ||
                targets[i] < 0.0 ||
                targets[i] > 1.0) {
                throw std::invalid_argument(
                    "Targets must be finite and in the range [0,1]."
                );
            }

            gradient[i] =
                (stable_sigmoid(logits[i]) - targets[i]) / output_width;

            if (!std::isfinite(gradient[i])) {
                throw std::runtime_error(
                    "BCE loss gradient produced a non-finite value."
                );
            }
        }

        return gradient;
    };

    return objective;
}

void validate_objective_functions(const ObjectiveFunctions& objective)
{
    // TODO for Ethan: keep this validation focused on callback presence.
    // Callback result size and finite-value checks require a network output
    // width, so sample_cost/backward should perform those checks at use time.
    if (!objective.sample_loss || !objective.sample_loss_gradient) {
        throw std::invalid_argument(
            "Objective must provide both sample loss callbacks."
        );
    }
}

/// <summary>
/// Defaults to BCE
/// </summary>
/// <param name="network"></param>
/// <param name="cache"></param>
/// <param name="target"></param>
/// <returns></returns>
double sample_cost(const MLP& network, const ForwardCache& cache, const Values& target)
{
    return sample_cost(
        network,
        cache,
        target,
        make_binary_cross_entropy_objective()
    );
}

/// <summary>
/// Computes a sample cost
/// </summary>
/// <param name="network"></param>
/// <param name="cache"></param>
/// <param name="target"></param>
/// <param name="objective"></param>
/// <returns></returns>
double sample_cost(const MLP& network, const ForwardCache& cache, const Values& target, const ObjectiveFunctions& objective){
    validate_objective_functions(objective);
    validate_network(network);

    if (cache.pre_activations.size() != network.layers.size()) {
        throw std::invalid_argument("Forward cache pre-activation count does not match network layers.");
    }

    const Values& output_logits = cache.pre_activations.back();

    if (output_logits.size() != network.layer_sizes.back()) {
        throw std::invalid_argument("Forward cache output size does not match network output layer size.");
    }

    if (target.size() != output_logits.size()) {
        throw std::invalid_argument("Target size does not match network output layer size.");
    }

    for (const double logit : output_logits) {
        if (!std::isfinite(logit)) {
            throw std::invalid_argument("Output logits must be finite.");
        }
    }

    for (const double target_value : target) {
        if (!std::isfinite(target_value)) {
            throw std::invalid_argument("Target values must be finite.");
        }
    }
    const double value = objective.sample_loss(output_logits, target);
    if (!std::isfinite(value)) {
        throw std::invalid_argument("Objective sample loss must be finite.");
    }

    return value;
}

double batch_loss(const MLP& network, const Dataset& batch) {
    if (batch.empty()) {
        throw std::invalid_argument("Batch must not be empty.");
	}
    double sum = std::accumulate(batch.begin(), batch.end(), 0.0, 
        [&network](double accumulated_cost, const Sample& sample) {
		    const ForwardCache cache = forward_pass(network, sample.input);
		    const double cost = sample_cost(network, cache, sample.target);
		    return accumulated_cost + cost;
        });
	double average = sum / static_cast<double>(batch.size());
    return average;
}

double batch_loss(const MLP& network, const Dataset& batch, const ObjectiveFunctions& objective)
{
    validate_objective_functions(objective);

    if (batch.empty()) {
        throw std::invalid_argument("Batch must not be empty.");
    }
    double sum = std::accumulate(batch.begin(), batch.end(), 0.0,
        [&network, objective](double accumulated_cost, const Sample& sample) {
            const ForwardCache cache = forward_pass(network, sample.input);
            const double cost = sample_cost(network, cache, sample.target, objective);
            return accumulated_cost + cost;
        });

    if (!std::isfinite(sum)) {
        throw std::overflow_error(
            "Objective batch loss accumulation is not finite."
        );
    }

    const double average = sum / static_cast<double>(batch.size());
    if (!std::isfinite(average)) {
        throw std::overflow_error(
            "Objective batch loss average is not finite."
        );
    }

    return average;
}

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

NetworkGradients backward(const MLP& network, const ForwardCache& cache, const Values& target) {
    validate_backward_inputs(network, cache, target);
    NetworkGradients gradients = make_zero_gradients_like(network);

    std::vector<Values> deltas;
    deltas.reserve(network.layers.size());
    for (const DenseLayer& layer : network.layers) {
        deltas.emplace_back(layer.output_size, 0.0);
    }

    const std::size_t output_layer_index = network.layers.size() - 1;
    const std::size_t output_width = network.layers[output_layer_index].output_size;
    const Values& output_activation = cache.activations[output_layer_index + 1];

    for (std::size_t j = 0; j < output_width; ++j) {
        deltas[output_layer_index][j] =
            (output_activation[j] - target[j]) / static_cast<double>(output_width);
    }

    for (std::size_t next_layer_index = output_layer_index;
         next_layer_index > 0;
         --next_layer_index) {
        const std::size_t layer_index = next_layer_index - 1;
        const DenseLayer& layer = network.layers[layer_index];
        const DenseLayer& next_layer = network.layers[next_layer_index];
        const Values& activation = cache.activations[layer_index + 1];

        for (std::size_t j = 0; j < layer.output_size; ++j) {
            double transported_delta = 0.0;

            for (std::size_t r = 0; r < next_layer.output_size; ++r) {
                transported_delta +=
                    next_layer.weight(r, j) * deltas[next_layer_index][r];
            }

            const double tanh_derivative = 1.0 - activation[j] * activation[j];
            deltas[layer_index][j] = transported_delta * tanh_derivative;
        }
    }

    for (std::size_t layer_index = 0;
         layer_index < network.layers.size();
         ++layer_index) {
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
        const Values& activation = cache.activations[layer_index + 1];

        for (std::size_t j = 0; j < layer.output_size; ++j) {
            double transported_delta = 0.0;
            for (std::size_t r = 0; r < next_layer.output_size; ++r) {
                transported_delta += next_layer.weight(r,j) * deltas[next_layer_index][r];
            }

            const double tanh_derivative = 1.0 - activation[j] * activation[j];
            deltas[layer_index][j] = transported_delta * tanh_derivative;
        }
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
    if (batch.empty()) {
        throw std::invalid_argument("Batch must not be empty.");
	}
    validate_network(network);

    NetworkGradients averaged = make_zero_gradients_like(network);
    
    for (const Sample& sample : batch) {
        const ForwardCache cache = forward_pass(network, sample.input);
        const NetworkGradients sample_gradients = backward(network, cache, sample.target);

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

double dot_product(const Values& left, const Values& right)
{
    if (left.size() != right.size()) {
        throw std::invalid_argument("Vector sizes must match for dot product.");
    }
    double sum = 0.0;
    for (size_t i = 0; i < left.size(); i++) {
        sum += left[i] * right[i];
    }
    return sum;
}

double network_vector_dot(const NetworkGradients& left, const NetworkGradients& right)
{
    if (left.layers.size() != right.layers.size()) {
        throw std::invalid_argument("Gradient layer counts must match for dot product.");
	}
    double sum = 0.0;
    for (size_t i = 0; i < left.layers.size(); i++) {
        if (left.layers[i].weights.size() != right.layers[i].weights.size())
			throw std::invalid_argument("Gradient weight counts must match for dot product.");
		if (left.layers[i].biases.size() != right.layers[i].biases.size())
			throw std::invalid_argument("Gradient bias counts must match for dot product.");
        for (size_t j = 0; j < left.layers[i].weights.size(); j++) {
            if (!std::isfinite(left.layers[i].weights[j]))
                throw std::invalid_argument("Weight gradient is not finite");
            if (!std::isfinite(right.layers[i].weights[j]))
                throw std::invalid_argument("Right Weight gradient is not finite");
        }
        for (size_t j = 0; j < left.layers[i].biases.size(); j++) {
            if (!std::isfinite(left.layers[i].biases[j]))
                throw std::invalid_argument("Bias gradient is not finite");
            if (!std::isfinite(right.layers[i].biases[j]))
                throw std::invalid_argument("Right Bias gradient is not finite");
        }
		double weight_dot = dot_product(left.layers[i].weights, right.layers[i].weights);
		double bias_dot = dot_product(left.layers[i].biases, right.layers[i].biases);
		
        sum += std::isfinite(weight_dot) ?
            weight_dot : throw std::invalid_argument("Weight Dot Product Is Not Finite");
        if (!std::isfinite(sum))
            throw std::overflow_error("Sum overflow in network_vector_dot weight computation");
		sum += std::isfinite(bias_dot) ?
            bias_dot : throw std::invalid_argument("Bias Dot Product Is Not Finite");
        if (!std::isfinite(sum))
            throw std::overflow_error("Sum overflow in network_vector_dot bias computation");

    }
    return sum;
}

NetworkDirection make_negative_gradient_direction(const NetworkGradients& gradient)
{
    for (size_t i = 0; i < gradient.layers.size(); i++) {
        if (gradient.layers[i].biases.size() != gradient.layers[i].biases.size())
            throw std::invalid_argument("Gradient bias counts must match for dot product.");
        for (size_t j = 0; j < gradient.layers[i].weights.size(); j++) {
            if (!std::isfinite(gradient.layers[i].weights[j]))
                throw std::invalid_argument("Weight gradient is not finite");
        }
        for (size_t j = 0; j < gradient.layers[i].biases.size(); j++)
            if (!std::isfinite(gradient.layers[i].biases[j]))
                throw std::invalid_argument("Bias gradient is not finite");
    }

    NetworkGradients neg_grad = gradient;
	for (LayerGradients& layer : neg_grad.layers) {
        for (double& weight : layer.weights) {
            weight = -weight;
        }
        for (double& bias : layer.biases) {
            bias = -bias;
        }
    }
	return neg_grad;
}

double downhill_cosine(const NetworkGradients& gradient, const NetworkDirection& direction){
	double numerator = -1 * network_vector_dot(gradient, direction);
    double grad_norm = gradient_l2_norm(gradient);
    double direction_norm = gradient_l2_norm(direction);
	if (grad_norm <= 0 || direction_norm <= 0)
		throw std::invalid_argument("Gradient or direction L2 norm is zero or negative, cannot compute downhill cosine");
    double denominator = grad_norm * direction_norm;
	double fraction = std::isfinite(denominator) ? numerator / denominator : 
        throw std::invalid_argument("||gradient|| * ||denominator|| is not finite");
	fraction = std::isfinite(fraction) ? fraction : throw std::invalid_argument("Downhill cosine is not finite");
	fraction = std::clamp(fraction, -1.0, 1.0);
    return fraction;
}

bool is_downhill_direction(const NetworkGradients& gradient, const NetworkDirection& direction, const double minimum_cosine){
    if (gradient_l2_norm(gradient) == 0.0 || gradient_l2_norm(direction) == 0.0) {
        return false; 
    }
    double downhill_cos = downhill_cosine(gradient, direction);
    return downhill_cos > minimum_cosine;
}

bool wolfe_parameters_are_valid(const WolfeParameters& params) noexcept {
	double sufficient_decrease = params.sufficient_decrease;
	double shrink_factor = params.shrink_factor;
	double maximum_step = params.maximum_step;
	size_t maximum_trials = params.maximum_trials;
	double downhill_cosine_threshold = params.minimum_downhill_cosine;
	double curvature = params.curvature;


    if (!std::isfinite(sufficient_decrease) || sufficient_decrease <= 0.0 || sufficient_decrease >= 1.0) {
        return false;
	}
    if (!std::isfinite(shrink_factor) || 0 >= shrink_factor || shrink_factor >= 1) {
        return false;
    }
    if (!std::isfinite(maximum_step) || maximum_step <= 0) {
        return false;
    }
    if (!std::isfinite(maximum_trials) || maximum_trials <= 0) {
        return false;
	}
    if (!std::isfinite(downhill_cosine_threshold) || downhill_cosine_threshold < -1.0 || downhill_cosine_threshold > 1.0) {
        return false;
    }
    if (!std::isfinite(curvature) || curvature <= 0.0 || curvature >= 1.0) {
        return false;
	}
    if (sufficient_decrease >= curvature) {
        return false;
	}
	return true;
}

void validate_wolfe_parameters(const WolfeParameters& params) {
    if (!wolfe_parameters_are_valid(params))
		throw std::invalid_argument("Wolfe parameters are invalid.");
}

void apply_direction(MLP& network, const NetworkDirection& direction, const double scale) {
    if (!std::isfinite(scale)) {
        throw std::invalid_argument("Direction scale must be finite.");
    }

    validate_network(network);
    validate_gradients_like(network, direction);
    MLP candidate = network;

    for (std::size_t layer_index = 0; layer_index < network.layers.size(); ++layer_index) {
        for (std::size_t parameter_index = 0; parameter_index < network.layers[layer_index].weights.size(); parameter_index++) {
            const double updated_parameter = network.layers[layer_index].weights[parameter_index] +
                scale * direction.layers[layer_index].weights[parameter_index];

            if (!std::isfinite(updated_parameter))
                throw std::overflow_error("Updated weight is not finite.");

            candidate.layers[layer_index].weights[parameter_index] =
                updated_parameter;
        }

        for (std::size_t parameter_index = 0; parameter_index < network.layers[layer_index].biases.size(); parameter_index++) {
            const double updated_parameter = network.layers[layer_index].biases[parameter_index] +
                scale * direction.layers[layer_index].biases[parameter_index];

            if (!std::isfinite(updated_parameter))
                throw std::overflow_error("Updated bias is not finite.");

            candidate.layers[layer_index].biases[parameter_index] =
                updated_parameter;
        }
    }

    network = candidate;
}

MLP make_candidate_network(const MLP& current_network, const NetworkDirection& direction, const double step_size)
{
    if (!learning_rate_is_valid(step_size)) {
        throw std::invalid_argument("Candidate step size must be positive and finite.");
    }

    MLP candidate_network = current_network;
    apply_direction(candidate_network, direction, step_size);
    return candidate_network;
}

bool satisfies_sufficient_decrease(
    const double candidate_loss,
    const double current_loss,
    const double step_size,
    const double initial_directional_derivative,
    const double sufficient_decrease_constant
)
{
    if (!(std::isfinite(candidate_loss)))
		throw std::invalid_argument("Candidate loss must be finite.");
	if (!(std::isfinite(current_loss)))
		throw std::invalid_argument("Current loss must be finite.");
	if (!(std::isfinite(step_size)) || step_size <= 0.0)
		throw std::invalid_argument("Step size must be positive and finite.");
	if (!(std::isfinite(initial_directional_derivative)))
		throw std::invalid_argument("Initial directional derivative must be finite.");
	if (!(std::isfinite(sufficient_decrease_constant)) || sufficient_decrease_constant <= 0.0 || sufficient_decrease_constant >= 1.0)
		throw std::invalid_argument("Sufficient decrease constant must be in the range (0,1).");
    if (step_size <= 0)
		throw std::invalid_argument("Step size must be positive.");
	if (sufficient_decrease_constant <= 0 || sufficient_decrease_constant >= 1)
		throw std::invalid_argument("Sufficient decrease constant must be in the range (0,1).");
	if (initial_directional_derivative >= 0)
		throw std::invalid_argument("Initial directional derivative must be negative for a downhill direction.");

    bool satisfies = 
        (candidate_loss <= current_loss + (sufficient_decrease_constant * step_size * initial_directional_derivative)) 
        ? true 
        : false;
    return satisfies;
}

bool satisfies_strong_curvature(const double candidate_directional_derivative, const double initial_directional_derivative, const double curvature_constant)
{
	if (!(std::isfinite(candidate_directional_derivative)))
		throw std::invalid_argument("Candidate directional derivative must be finite.");
	if (!(std::isfinite(initial_directional_derivative)))
		throw std::invalid_argument("Initial directional derivative must be finite.");
	if (!(std::isfinite(curvature_constant)) || curvature_constant <= 0.0 || curvature_constant >= 1.0)
		throw std::invalid_argument("Curvature constant must be in the range (0,1).");
	if (initial_directional_derivative >= 0)
		throw std::invalid_argument("Initial directional derivative must be negative for a downhill direction.");

    bool satisfies = 
        (std::abs(candidate_directional_derivative) <= curvature_constant * std::abs(initial_directional_derivative)) 
        ? true 
		: false;
    return satisfies;
}

bool satisfies_strong_wolfe(const WolfeEvaluation& evaluation) noexcept
{
	return evaluation.sufficient_decrease && evaluation.strong_curvature;
}

WolfeEvaluation evaluate_wolfe_candidate(
    const MLP& current_network,
    const Dataset& batch,
    const double current_loss,
    const NetworkGradients& current_gradient,
    const NetworkDirection& direction,
    const double step_size,
    const WolfeParameters& parameters,
	const ObjectiveFunctions& objective
)
{
	validate_objective_functions(objective);
	validate_wolfe_parameters(parameters);
    validate_network(current_network);
    validate_gradients_like(current_network, current_gradient);
    validate_gradients_like(current_network, direction);
    if (!is_downhill_direction(
        current_gradient,
        direction,
        parameters.minimum_downhill_cosine)) {
        throw std::invalid_argument(
            "Wolfe line search requires a downhill direction."
        );
    }

    auto candidate_network = make_candidate_network(current_network, direction, step_size);
    auto candidate_loss = batch_loss(candidate_network, batch, objective);
    bool satisfies_suff_decrease = satisfies_sufficient_decrease(
        candidate_loss, 
        current_loss, 
        step_size, 
        network_vector_dot(current_gradient, direction), 
        parameters.sufficient_decrease
	);
    auto candidate_gradient = batch_gradients(candidate_network, batch, objective);
    bool satisfies_strong_curv = satisfies_strong_curvature(
        network_vector_dot(candidate_gradient, direction),
        network_vector_dot(current_gradient, direction),
        parameters.curvature);
	auto initial_directional_derivative = network_vector_dot(current_gradient, direction);
	auto candidate_directional_derivative = network_vector_dot(candidate_gradient, direction);

    WolfeEvaluation evaluation
    {
        step_size,
        candidate_loss,
        initial_directional_derivative,
        candidate_directional_derivative,
        satisfies_suff_decrease,
        satisfies_strong_curv,
        candidate_network,
        candidate_gradient,
        objective
    };

    return evaluation;
}

LineSearchResult backtracking_wolfe_stepsize(
    const MLP& current_network,
    const Dataset& batch,
    const double current_loss,
    const NetworkGradients& current_gradient,
    const NetworkDirection& direction,
    const WolfeParameters& parameters,
    const ObjectiveFunctions& objective
)
{
    validate_network(current_network);
    validate_gradients_like(current_network, current_gradient);
    validate_gradients_like(current_network, direction);
    validate_objective_functions(objective);
    validate_wolfe_parameters(parameters);

    if (!std::isfinite(current_loss)) {
        throw std::invalid_argument("Current loss must be finite.");
    }

    if (!is_downhill_direction(
            current_gradient,
            direction,
            parameters.minimum_downhill_cosine)) {
        throw std::invalid_argument("Wolfe line search requires a downhill direction.");
    }

    const double initial_slope =
        network_vector_dot(current_gradient, direction);
    double step_size = parameters.maximum_step;

    LineSearchResult result;
    result.status = LineSearchStatus::Failed;
    result.selected_loss = current_loss;
    result.initial_directional_derivative = initial_slope;
    result.selected_directional_derivative = initial_slope;
    result.selected_network = current_network;
    result.selected_gradient = current_gradient;

    bool have_fallback = false;
    WolfeEvaluation fallback;

    for (std::size_t trial = 0;
         trial < parameters.maximum_trials;
         ++trial) {
        const WolfeEvaluation evaluation =
            evaluate_wolfe_candidate(
                current_network,
                batch,
                current_loss,
                current_gradient,
                direction,
                step_size,
                parameters,
                objective
            );

        result.history.push_back({
            evaluation.step_size,
            evaluation.candidate_loss,
            evaluation.candidate_directional_derivative,
            evaluation.sufficient_decrease,
            evaluation.strong_curvature
        });

        if (satisfies_strong_wolfe(evaluation)) {
            result.status = LineSearchStatus::StrongWolfeSatisfied;
            result.step_size = evaluation.step_size;
            result.selected_loss = evaluation.candidate_loss;
            result.initial_directional_derivative =
                evaluation.initial_directional_derivative;
            result.selected_directional_derivative =
                evaluation.candidate_directional_derivative;
            result.selected_network = evaluation.candidate_network;
            result.selected_gradient = evaluation.candidate_gradient;
            return result;
        }

        if (evaluation.sufficient_decrease && !have_fallback) {
            fallback = evaluation;
            have_fallback = true;
        }

        if (trial + 1 < parameters.maximum_trials) {
            step_size *= parameters.shrink_factor;
            if (!std::isfinite(step_size) || step_size <= 0.0) {
                break;
            }
        }
    }

    if (have_fallback) {
        result.status = LineSearchStatus::SufficientDecreaseFallback;
        result.step_size = fallback.step_size;
        result.selected_loss = fallback.candidate_loss;
        result.initial_directional_derivative =
            fallback.initial_directional_derivative;
        result.selected_directional_derivative =
            fallback.candidate_directional_derivative;
        result.selected_network = fallback.candidate_network;
        result.selected_gradient = fallback.candidate_gradient;
    }

    return result;
}

TrainingStepResult take_wolfe_gradient_step(
    MLP& network,
    const Dataset& batch,
    const WolfeParameters& parameters,
    const double gradient_tolerance,
	const ObjectiveFunctions& objective = make_binary_cross_entropy_objective()
)
{
	validate_network(network);
    validate_wolfe_parameters(parameters);
    if (!std::isfinite(gradient_tolerance) || gradient_tolerance <= 0.0)
		throw std::invalid_argument("Gradient tolerance must be positive and finite.");
	auto result = TrainingStepResult{};
	result.previous_loss = batch_loss(network, batch, objective);
	auto current_gradient = batch_gradients(network, batch, objective);
	result.gradient_norm = gradient_l2_norm(current_gradient);
    if (result.gradient_norm <= gradient_tolerance) {
        result.new_loss = result.previous_loss;
		result.updated = false;
        return result;
	}
	auto direction = make_negative_gradient_direction(current_gradient);
    result.line_search = backtracking_wolfe_stepsize(
        network,
        batch,
        result.previous_loss,
        current_gradient,
        direction,
        parameters,
        objective
	);
    if (result.line_search.status == LineSearchStatus::Failed) {
		result.new_loss = result.previous_loss;
        result.updated = false;
		return result;
    }
	network = result.line_search.selected_network;
	result.new_loss = result.line_search.selected_loss;
    result.updated = true;
    return result;
}

const char* line_search_status_name(const LineSearchStatus status) noexcept
{
    switch (status) {
        case LineSearchStatus::StrongWolfeSatisfied:
            return "StrongWolfeSatisfied";
        case LineSearchStatus::SufficientDecreaseFallback:
            return "SufficientDecreaseFallback";
        case LineSearchStatus::Failed:
            return "Failed";
	}
    return "Unknown";
}
