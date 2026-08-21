#include "../include/nncpp/objective_functions.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

/*
This file contains objectives, loss functions, and regularization terms for training MLPs.
Each objective is represented by a pair of functions.  The input domain is
explicit: logits/pre-activations for logit objectives, or activated outputs
for objectives such as ordinary mean-squared error.
*/

#pragma region Binary Cross-Entropy Objective
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

    return (std::max(logit, 0.0) - logit * target +
        std::log1p(std::exp(-std::abs(logit))));
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

            for (std::size_t i = 0; i < logit.size(); i++) {
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

    objective.sample_loss_gradient = [](const Values& logits, const Values& targets) -> Values {
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
#pragma endregion

#pragma region Softmax Cross-Entropy Objective
Values stable_softmax(const Values& logits) {
    if (logits.empty()) {
        throw std::invalid_argument("Logits vector must not be empty.");
	}
    for (const double logit : logits) {
        if (!std::isfinite(logit)) {
            throw std::invalid_argument("Logits must be finite.");
        }
	}
    const double max_logit = *std::max_element(logits.begin(), logits.end());
	double sum = 0.0;
    for (auto logit : logits) {
		auto shifted = logit - max_logit;
		double exp_shifted = std::exp(shifted);
        sum += exp_shifted;
    }
    if (!std::isfinite(sum) || sum <= 0.0) {
        throw std::runtime_error("Softmax computation produced a non-finite or non-positive sum.");
	}
	auto probabilities = Values(logits.size(), 0.0);
    for (std::size_t i = 0; i < logits.size(); ++i) {
        double shifted = logits[i] - max_logit;
        double exp_shifted = std::exp(shifted);
        probabilities[i] = exp_shifted / sum;
        if (!std::isfinite(probabilities[i])) {
            throw std::runtime_error("Softmax computation produced a non-finite probability.");
		}
    }
	return probabilities;
}

double cross_entropy_from_logits(
    const Values& logits,
    const Values& targets
)
{
    if (logits.empty() || targets.empty()) {
        throw std::invalid_argument(
            "Logits and targets must not be empty."
        );
    }

    if (logits.size() != targets.size()) {
        throw std::invalid_argument(
            "Logits and targets must have the same size."
        );
    }

    double target_sum = 0.0;

    for (const double logit : logits) {
        if (!std::isfinite(logit)) {
            throw std::invalid_argument("Logits must be finite.");
        }
    }

    for (const double target : targets) {
        if (!std::isfinite(target) ||
            target < 0.0 ||
            target > 1.0) {
            throw std::invalid_argument(
                "Targets must be finite and in [0,1]."
            );
        }

        target_sum += target;
    }

    constexpr double target_tolerance = 1e-12;

    if (!std::isfinite(target_sum) ||
        std::abs(target_sum - 1.0) > target_tolerance) {
        throw std::invalid_argument(
            "Categorical targets must sum to 1."
        );
    }

    const double maximum =
        *std::max_element(logits.begin(), logits.end());

    double sum_exp = 0.0;
    double weighted_logit_sum = 0.0;

    for (std::size_t i = 0; i < logits.size(); ++i) {
        sum_exp += std::exp(logits[i] - maximum);
        weighted_logit_sum += targets[i] * logits[i];
    }

    if (!std::isfinite(sum_exp) || sum_exp <= 0.0) {
        throw std::runtime_error(
            "Log-sum-exp calculation failed."
        );
    }

    const double log_sum_exp =
        maximum + std::log(sum_exp);

    const double loss =
        log_sum_exp - weighted_logit_sum;

    if (!std::isfinite(loss)) {
        throw std::runtime_error(
            "Cross-entropy loss is not finite."
        );
    }

    return loss;
}

ObjectiveFunctions make_softmax_cross_entropy_objective() {
    ObjectiveFunctions objective;

    objective.sample_loss =
        [](const Values& logits, const Values& targets) {
        return cross_entropy_from_logits(logits, targets);
        };

    objective.sample_loss_gradient = [](const Values& logits, const Values& targets) -> Values {
        static_cast<void>(cross_entropy_from_logits(logits, targets));
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

        Values probabilities = stable_softmax(logits);
        Values gradient(logits.size(), 0.0);
        double target_sum = 0.0;

        for (std::size_t i = 0; i < logits.size(); ++i) {
            if (!std::isfinite(targets[i]) ||
                targets[i] < 0.0 ||
                targets[i] > 1.0) {
                throw std::invalid_argument(
                    "Targets must be finite and in [0,1]."
                );
            }

            target_sum += targets[i];
            gradient[i] = probabilities[i] - targets[i];

            if (!std::isfinite(gradient[i])) {
                throw std::runtime_error(
                    "Softmax cross-entropy gradient is not finite."
                );
            }
        }

        constexpr double target_tolerance = 1e-12;
        if (!std::isfinite(target_sum) ||
            std::abs(target_sum - 1.0) > target_tolerance) {
            throw std::invalid_argument(
                "Categorical targets must sum to 1."
            );
        }

        return gradient;
    };

    return objective;
}
#pragma endregion

#pragma region Mean Squared Error Objective
double mean_squared_error_from_logits(const Values& logits, const Values& targets) {
    if (logits.empty()) {
        throw std::invalid_argument("Logits vector must not be empty.");
    }
    for (const double logit : logits) {
        if (!std::isfinite(logit)) {
            throw std::invalid_argument("Logits must be finite.");
        }
    }

    if (targets.size() != logits.size()) {
        throw std::invalid_argument(
            "Logits and targets must have the same size."
        );
	}

    return std::inner_product(
        logits.begin(), logits.end(),
        targets.begin(), 0.0,
        std::plus<double>(),
        [](double logit, double target) {
            if (!std::isfinite(target)) {
                throw std::invalid_argument("Targets must be finite.");
            }
            double diff = logit - target;
            return diff * diff;
        }
	) / static_cast<double>(logits.size());
}

ObjectiveFunctions make_mean_squared_error_objective() {
    ObjectiveFunctions objective;
    objective.input_domain = ObjectiveInputDomain::ActivatedOutput;

    objective.sample_loss =
        [](const Values& logits, const Values& targets) {
        return mean_squared_error_from_logits(logits, targets);
        };
    objective.sample_loss_gradient = [](const Values& logits, const Values& targets) -> Values {
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
            if (!std::isfinite(targets[i])) {
                throw std::invalid_argument("Targets must be finite.");
            }
            gradient[i] = (2.0 * (logits[i] - targets[i])) / output_width;
            if (!std::isfinite(gradient[i])) {
                throw std::runtime_error(
                    "MSE loss gradient produced a non-finite value."
                );
            }
        }
        return gradient;
		};

    return objective;
}
#pragma endregion

#pragma region Exponential Objective
ObjectiveFunctions make_exponential_objective() {
    throw std::logic_error("Not Implemented");
}
#pragma endregion

#pragma region Hinge Objective
ObjectiveFunctions make_hinge_objective() {
    throw std::logic_error("Not Implemented");
}
#pragma endregion

#pragma region Common Evaluation Objective Infrastructure
void validate_objective_functions(const ObjectiveFunctions& objective)
{
    if (!objective.sample_loss || !objective.sample_loss_gradient) {
        throw std::invalid_argument(
            "Objective must provide both sample loss callbacks."
        );
    }
    switch (objective.input_domain) {
    case ObjectiveInputDomain::PreActivation:
    case ObjectiveInputDomain::ActivatedOutput:
        break;
    default:
        throw std::invalid_argument("Objective has an invalid input domain.");
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
    if (cache.activations.size() != network.layer_sizes.size()) {
        throw std::invalid_argument("Forward cache activation count does not match network layers.");
    }

    const Values& output_logits = cache.pre_activations.back();
    const Values& objective_input = objective.input_domain ==
        ObjectiveInputDomain::PreActivation
        ? cache.pre_activations.back()
        : cache.activations.back();

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
    const double value = objective.sample_loss(objective_input, target);
    if (!std::isfinite(value)) {
        throw std::invalid_argument("Objective sample loss must be finite.");
    }

    return value;
}

// BCE
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


// Objective composition and regularization
ObjectiveConfig make_objective_config(
    const ObjectiveFunctions& data_objective,
    std::vector<RegularizationTerm> regularization_terms
)
{
	ObjectiveConfig config{
        data_objective,
        std::move(regularization_terms)
	};

    validate_objective_config(config);

    return config;
}

ObjectiveConfig make_objective_config(const ObjectiveFunctions& data_objective, RegularizationTerm regularizer) {
    return make_objective_config(data_objective, std::vector<RegularizationTerm>{std::move(regularizer)});
}

double regularization_loss(
    const MLP& network,
    const ObjectiveConfig& config
)
{
    validate_objective_config(config);
    double total_loss = 0.0;

    for (const RegularizationTerm& term : config.regularizers) {
        double term_loss = term.value(network);
        if (!std::isfinite(term_loss)) {
            throw std::invalid_argument(
                "Regularization term loss must be finite."
            );
        }
        total_loss += term.coefficient * term_loss;
        if (!std::isfinite(total_loss)) {
            throw std::overflow_error(
                "Total regularization loss overflowed."
            );
        }
    }
    return total_loss;
}

void validate_objective_config(const ObjectiveConfig& config)
{
    validate_objective_functions(config.data_objective);

    for (const RegularizationTerm& term : config.regularizers) {
        if (term.name.empty()) {
            throw std::invalid_argument(
                "Regularization term name must not be empty."
            );
        }
        if (!std::isfinite(term.coefficient)) {
            throw std::invalid_argument(
                "Regularization term must be finite"
            );
        }
        if (term.coefficient < 0.0) {
            throw std::invalid_argument(
                "Regularization term coefficient must be non-negative."
            );
        }
        if (!term.value) {
            throw std::invalid_argument(
                "Regularization term must provide value callback."
            );
        }
        if (term.proximal) {
            if (!term.proximal_update) {
                throw std::invalid_argument(
                    "Proximal regularization term must provide proximal callback."
                );
            }
            if (term.add_gradient) {
                throw std::invalid_argument(
                    "Proximal regularization terms must not also provide a gradient callback."
                );
            }
        }
        else if (!term.add_gradient) {
            throw std::invalid_argument(
                "Regularization term must provide gradient callback."
            );
        }
    }
}

void add_regularization_gradients(
    const MLP& network,
    const ObjectiveConfig& config,
    NetworkGradients& gradients
)
{
    validate_objective_config(config);
    validate_network(network);
    validate_gradients_like(network, gradients);

    for (const RegularizationTerm& term : config.regularizers) {
        if (term.proximal) {
            continue;
        }
        // Regularizer callbacks produce unscaled contributions. Apply the
        // coefficient here so data gradients are never scaled and every
        // coefficient is applied exactly once.
        NetworkGradients regularizer_gradients =
            make_zero_gradients_like(network);
        term.add_gradient(network, regularizer_gradients);
        validate_gradients_like(network, regularizer_gradients);

        NetworkGradients candidate = gradients;

        for (std::size_t layer_index = 0;
             layer_index < candidate.layers.size();
             ++layer_index) {
            LayerGradients& candidate_layer = candidate.layers[layer_index];
            const LayerGradients& regularizer_layer =
                regularizer_gradients.layers[layer_index];

            for (std::size_t i = 0;
                 i < candidate_layer.weights.size();
                 ++i) {
                const double contribution =
                    term.coefficient * regularizer_layer.weights[i];

                if (!std::isfinite(contribution)) {
                    throw std::overflow_error(
                        "Regularization gradient contribution overflowed."
                    );
                }

                candidate_layer.weights[i] += contribution;
            }

            for (std::size_t i = 0;
                 i < candidate_layer.biases.size();
                 ++i) {
                const double contribution =
                    term.coefficient * regularizer_layer.biases[i];

                if (!std::isfinite(contribution)) {
                    throw std::overflow_error(
                        "Regularization gradient contribution overflowed."
                    );
                }

                candidate_layer.biases[i] += contribution;
            }
        }

        validate_gradients_like(network, candidate);
        gradients = std::move(candidate);
    }
}

void apply_proximal_updates(
    MLP& network,
    const ObjectiveConfig& config,
    const double effective_step_size
)
{
    validate_objective_config(config);
    validate_network(network);
    if (!std::isfinite(effective_step_size) || effective_step_size <= 0.0) {
        throw std::invalid_argument(
            "A proximal update requires a positive finite step size."
        );
    }

    MLP candidate = network;
    for (const RegularizationTerm& term : config.regularizers) {
        if (term.proximal) {
            term.proximal_update(candidate, effective_step_size);
            validate_network(candidate);
        }
    }
    network = std::move(candidate);
}

double objective_loss(
    const MLP& network,
    const Dataset& dataset,
    const ObjectiveConfig& config
)
{
    validate_objective_config(config);

    double loss_no_reg_terms = batch_loss(network, dataset, config.data_objective);
    double reg_terms = regularization_loss(network, config);
    double total_loss = loss_no_reg_terms + reg_terms;
    if (!std::isfinite(total_loss)) {
        throw std::overflow_error(
            "Total objective loss overflowed."
        );
    }
    return total_loss;
}

NetworkGradients objective_gradients(
    const MLP& network,
    const Dataset& dataset,
    const ObjectiveConfig& config
)
{
    auto gradients = batch_gradients(network, dataset, config.data_objective);
    add_regularization_gradients(network, config, gradients);
    return gradients;
}

#pragma endregion

#pragma region L2 Regularization

/// <summary>
/// Ridge Regularization (L2) regularization term with the specified strength.
/// L2 regularization is the sum of the squares of the weights (and optionally biases) multiplied by a scalar.
/// It encourages smaller weights and can help prevent overfitting.
/// Mathematically it is the L2 norm of the weights (and optionally biases) squared, scaled by the given scalar.
/// </summary>
/// <param name="scalar">The regularization strength (scale factor) applied to the L2 penalty - typically a non-negative value.</param>
/// <param name="include_biases">If true, include bias parameter in the regularization; otherwise only apply to weights. Default is false</param>
/// <returns>A RegularizationTerm configured to apply an L2 penalty with the given scalar. If include_biases is true, the penalty will also be applied to bias parameters.</returns>
RegularizationTerm make_l2_regularization(
    const double scalar,
    const bool include_biases
)
{
    if (!std::isfinite(scalar) || scalar < 0.0) {
        throw std::invalid_argument(
            "L2 regularization scalar must be finite and non-negative."
        );
    }

    RegularizationTerm term;
    term.name = "L2 Regularization";
    term.value = [include_biases](const MLP& network) -> double {
		validate_network(network);

        double l2_sum = 0.0;
        for (const auto& layer : network.layers) {
            for (const auto& weight : layer.weights) {
                l2_sum += weight*weight;
            }
            if (!std::isfinite(l2_sum)) {
                throw std::overflow_error(
                    "L2 regularization value overflowed."
                );
            }
            if (include_biases) {
                for (double bias : layer.biases) {
                    l2_sum += bias*bias;
                }
                if (!std::isfinite(l2_sum)) {
                    throw std::overflow_error(
                        "L2 regularization value overflowed."
                    );
                }
            }
        }
        return l2_sum;
    };
    term.add_gradient = [include_biases](const MLP& network, NetworkGradients& gradients) {
		validate_network(network);
		validate_gradients_like(network, gradients);

        for (std::size_t layer_idx = 0; layer_idx < network.layers.size(); ++layer_idx) {
            const auto& layer = network.layers[layer_idx];
            auto& grad_layer = gradients.layers[layer_idx];
            for (std::size_t i = 0; i < layer.weights.size(); i++) {
                grad_layer.weights[i] += 2.0 * layer.weights[i];
            }
            if (include_biases) {
                for (std::size_t j = 0; j < layer.biases.size(); ++j) {
                    grad_layer.biases[j] += 2.0 * layer.biases[j];
                }
            }
        }
        validate_gradients_like(network, gradients);
    };
    term.smooth = true;
    term.includes_biases = include_biases;
    term.coefficient = scalar;
    return term;
}

#pragma endregion

#pragma region L1 Regularization

RegularizationTerm make_l1_regularization(
    const double scalar,
    const bool include_biases,
    const L1Method method
)
{
    switch (method) {
        case L1Method::Subgradient:
            return make_l1_regularization_subgradient(scalar, include_biases);
        case L1Method::Proximal:
            return make_l1_regularization_Proximal(scalar, include_biases);
        case L1Method::Eps_SmoothL1Regularizer:
        case L1Method::Log_SmoothL1Regularizer:
            throw std::logic_error(
                "Phase 1 exercise stub: implement smooth L1 regularization."
            );
        default:
            throw std::invalid_argument(
                "Invalid L1 regularization method."
            );
    }
}

constexpr double l1_subgradient(const double value) noexcept {
    if (value > 0.0) {
        return 1.0;
    }
    else if (value < 0.0) {
        return -1.0;
    }
    else {
        return 0.0; // Subgradient at zero can be any value in [-1, 1], but we choose 0 for simplicity
    }
}

//Smooth L1 Regularizer using epsilon smoothing ((theta^2 + eps^2)^(1/2) in place of absolute value)
RegularizationTerm eps_make_l1_regularization_smooth(
    const double scalar,
    const bool include_biases
)
{
    if (!std::isfinite(scalar) || scalar < 0.0) {
        throw std::invalid_argument(
            "L1 regularization scalar must be finite and non-negative."
        );
    }



    RegularizationTerm term;
    term.name = "L1 Regularization";
    term.value = [scalar, include_biases](const MLP& network) -> double {
        double l1_sum = 0.0;
        for (const auto& layer : network.layers) {
            for (const auto& weight : layer.weights) {
                l1_sum += std::abs(weight);
            }
            if (include_biases) {
                for (double bias : layer.biases) {
                    l1_sum += std::abs(bias);
                }
            }
            if (l1_sum < 0.0 || !std::isfinite(l1_sum)) {
                throw std::overflow_error(
                    "L1 regularization sum overflowed."
                );
            }
        }
        return scalar * l1_sum;
    };
    term.add_gradient = [scalar, include_biases](const MLP& network, NetworkGradients& gradients) {
        for (std::size_t layer_idx = 0; layer_idx < network.layers.size(); ++layer_idx) {
            const auto& layer = network.layers[layer_idx];
            auto& grad_layer = gradients.layers[layer_idx];
            for (std::size_t i = 0; i < layer.weights.size(); i++) {
                grad_layer.weights[i] += scalar * l1_subgradient(layer.weights[i]);
            }
            if (include_biases) {
                for (std::size_t j = 0; j < layer.biases.size(); ++j) {
                    grad_layer.biases[j] += scalar * l1_subgradient(layer.biases[j]);
                }
            }
        }
    };
    term.smooth = false;
    term.includes_biases = include_biases;
    term.coefficient = scalar;
    return term;
}

//Smooth L1 Regularizer using log smoothing T(log(1 + exp(theta/T)) + T(log(-theta/T)) in place of absolute value)
//where T is a temperature parameter that controls the smoothness of the approximation. As T approaches 0,
//the approximation approaches the true L1 norm. As T increases, the approximation becomes smoother and less sensitive to small changes in theta.
// (needs to be sufficiently small)
RegularizationTerm log_make_l1_regularization_smooth(
    const double scalar,
    const bool include_biases
)
{
    if (!std::isfinite(scalar) || scalar < 0.0) {
        throw std::invalid_argument(
            "L1 regularization scalar must be finite and non-negative."
        );
    }



    RegularizationTerm term;
    term.name = "L1 Regularization";
    term.value = [scalar, include_biases](const MLP& network) -> double {
        double l1_sum = 0.0;
        for (const auto& layer : network.layers) {
            for (const auto& weight : layer.weights) {
                l1_sum += std::abs(weight);
            }
            if (include_biases) {
                for (double bias : layer.biases) {
                    l1_sum += std::abs(bias);
                }
            }
            if (l1_sum < 0.0 || !std::isfinite(l1_sum)) {
                throw std::overflow_error(
                    "L1 regularization sum overflowed."
                );
            }
        }
        return scalar * l1_sum;
        };
    term.add_gradient = [scalar, include_biases](const MLP& network, NetworkGradients& gradients) {
        for (std::size_t layer_idx = 0; layer_idx < network.layers.size(); ++layer_idx) {
            const auto& layer = network.layers[layer_idx];
            auto& grad_layer = gradients.layers[layer_idx];
            for (std::size_t i = 0; i < layer.weights.size(); i++) {
                grad_layer.weights[i] += scalar * l1_subgradient(layer.weights[i]);
            }
            if (include_biases) {
                for (std::size_t j = 0; j < layer.biases.size(); ++j) {
                    grad_layer.biases[j] += scalar * l1_subgradient(layer.biases[j]);
                }
            }
        }
        };
    term.smooth = false;
    term.includes_biases = include_biases;
    term.coefficient = scalar;
    return term;
}

//Proximal Code
RegularizationTerm make_l1_regularization_Proximal(
    const double scalar,
    const bool include_biases
)
{
    if (!std::isfinite(scalar) || scalar < 0.0) {
        throw std::invalid_argument(
            "L1 regularization scalar must be finite and non-negative."
        );
    }



    RegularizationTerm term;
    term.name = "L1 Regularization";
    term.value = [include_biases](const MLP& network) -> double {
        double l1_sum = 0.0;
        validate_network(network);
        for (const auto& layer : network.layers) {
            for (const auto& weight : layer.weights) {
                l1_sum += std::abs(weight);
            }
            if (include_biases) {
                for (double bias : layer.biases) {
                    l1_sum += std::abs(bias);
                }
            }
        }
        if (!std::isfinite(l1_sum)) {
            throw std::overflow_error(
                "L1 regularization sum overflowed."
            );
        }
        return l1_sum;
    };
    term.proximal = true;
    term.proximal_update = [scalar, include_biases](MLP& network, const double step_size) {
        validate_network(network);
        if (!std::isfinite(step_size) || step_size <= 0.0) {
            throw std::invalid_argument(
                "Proximal L1 requires a positive finite step size."
            );
        }
        const double threshold = step_size * scalar;
        if (!std::isfinite(threshold)) {
            throw std::overflow_error(
                "Proximal L1 threshold is not finite."
            );
        }
        const auto shrink = [threshold](double value) {
            const double magnitude = std::max(0.0, std::abs(value) - threshold);
            return std::copysign(magnitude, value);
        };
        for (DenseLayer& layer : network.layers) {
            for (double& weight : layer.weights) {
                weight = shrink(weight);
            }
            if (include_biases) {
                for (double& bias : layer.biases) {
                    bias = shrink(bias);
                }
            }
        }
        validate_network(network);
    };
    term.smooth = false;
    term.includes_biases = include_biases;
    term.coefficient = scalar;
    return term;
}

/// <summary>
/// Creates an L1 (Lasso) regularization term with the specified strength.
/// </summary>
/// <param name="scalar">The regularization strength (scale factor) applied to the L1 penalty — typically a non-negative value.</param>
/// <param name="include_biases">If true, include bias parameters in the regularization; otherwise only apply to weights. Default is false.</param>
/// <returns>A RegularizationTerm configured to apply an L1 penalty with the given scalar. If include_biases is true, the penalty will also be applied to bias parameters.</returns>
RegularizationTerm make_l1_regularization_subgradient(
    const double scalar,
    const bool include_biases
)
{
    if (!std::isfinite(scalar) || scalar < 0.0) {
        throw std::invalid_argument(
            "L1 regularization scalar must be finite and non-negative."
        );
	}



	RegularizationTerm term;
    term.name = "L1 Regularization";
    term.value = [include_biases](const MLP& network) -> double {
        double l1_sum = 0.0;
        for (const auto& layer : network.layers) {
            for (const auto& weight : layer.weights) {
                    l1_sum += std::abs(weight);
                    if (!std::isfinite(l1_sum)) {
                        throw std::overflow_error(
                            "L1 regularization sum overflowed."
                        );
                    }
            }
            if (include_biases) {
                for (double bias : layer.biases) {
                    l1_sum += std::abs(bias);
                }
                if (!std::isfinite(l1_sum)) {
                    throw std::overflow_error(
                        "L1 regularization sum overflowed."
                    );
                }
            }
        }
        return l1_sum;
    };
    term.add_gradient = [include_biases](const MLP& network, NetworkGradients& gradients) {
        validate_network(network);
        validate_gradients_like(network, gradients);
        for (std::size_t layer_idx = 0; layer_idx < network.layers.size(); ++layer_idx) {
            const auto& layer = network.layers[layer_idx];
            auto& grad_layer = gradients.layers[layer_idx];
            for (std::size_t i = 0; i < layer.weights.size(); i++) {
				grad_layer.weights[i] += l1_subgradient(layer.weights[i]);
            }
            if (include_biases) {
                for (std::size_t j = 0; j < layer.biases.size(); ++j) {
                    grad_layer.biases[j] += l1_subgradient(layer.biases[j]);
                }
            }
        }
        validate_gradients_like(network, gradients);
		};
	term.smooth = false;
	term.includes_biases = include_biases;
    term.coefficient = scalar;
    return term;
}

#pragma endregion

#pragma region Elastic Net Regularization

/// <summary>
/// Creates an Elastic Net regularization term that combines L1 and L2 penalties with the specified strengths.
/// </summary>
/// <param name="L1_scalar">The value specifying the strength of the L1 regularization term</param>
/// <param name="L2_scalar">The value specifying the strength of the L2 regularization term</param>
/// <param name="include_biases">If true, include bias parameters in the regularization; otherwise only apply to weights. Default is false.</param>
/// <returns></returns>
RegularizationTerm make_elastic_net_regularization(
    const double L1_scalar,
    const double L2_scalar,
	const bool include_biases
)
{
    static_cast<void>(L1_scalar);
    static_cast<void>(L2_scalar);
    static_cast<void>(include_biases);
    throw std::logic_error(
        "Phase 1 exercise stub: implement make_elastic_net_regularization."
    );
}

#pragma endregion
