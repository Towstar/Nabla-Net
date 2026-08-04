#include "objective_functions.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

// Binary cross-entropy objective
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


// Softmax and categorical cross-entropy objective
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


// Mean squared error objective
// (Not Complete Yet)
ObjectiveFunctions make_mean_squared_error_objective() {
    throw std::logic_error("Not Implemented");
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

// Mean squared error objective
// (Not Complete Yet)
ObjectiveFunctions make_exponential_objective() {
    throw std::logic_error("Not Implemented");
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


// Common objective evaluation infrastructure
void validate_objective_functions(const ObjectiveFunctions& objective)
{
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

RegularizationTerm make_l2_regularization(
    const double,
    const bool
)
{
    throw std::logic_error(
        "Phase 1 exercise stub: implement make_l2_regularization."
    );
}

RegularizationTerm make_l1_regularization(
    const double,
    const bool
)
{
    throw std::logic_error(
        "Phase 1 exercise stub: implement make_l1_regularization."
    );
}

RegularizationTerm make_elastic_net_regularization(
    const double,
    const double,
    const bool
)
{
    throw std::logic_error(
        "Phase 1 exercise stub: implement make_elastic_net_regularization."
    );
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
        if (!term.add_gradient) {
            throw std::invalid_argument(
                "Regularization term must provide gradient callback."
            );
        }
	}
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

void add_regularization_gradients(
    const MLP& network,
    const ObjectiveConfig& config,
    NetworkGradients& gradients
)
{
    validate_objective_config(config);
	validate_gradients_like(network, gradients);
    for (const RegularizationTerm& term : config.regularizers) {
        NetworkGradients cand = gradients;
		term.add_gradient(network, cand);
		validate_gradients_like(network, cand);
		gradients = std::move(cand);
	}
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
