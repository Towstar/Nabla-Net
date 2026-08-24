#pragma once

#include "mlp.hpp"

#include <vector>

// Objective-function primitives and objective factories.
/// <summary>
/// Converts logits to normalized probabilities using a stable softmax.
/// </summary>
Values stable_softmax(const Values& logits);

/// <summary>
/// Evaluates binary cross-entropy directly from one logit and target.
/// </summary>
double binary_cross_entropy_from_logit(
    double logit,
    double target
);

/// <summary>
/// Evaluates multiclass cross-entropy from logits and target probabilities.
/// </summary>
double cross_entropy_from_logits(
    const Values& logits,
    const Values& targets
);

/// <summary>
/// Creates the binary-cross-entropy-from-logits objective.
/// </summary>
ObjectiveFunctions make_binary_cross_entropy_objective();

/// <summary>
/// Creates the softmax cross-entropy objective.
/// </summary>
ObjectiveFunctions make_softmax_cross_entropy_objective();

/// <summary>
/// Creates the mean-squared-error objective.
/// </summary>
ObjectiveFunctions make_mean_squared_error_objective();

/// <summary>
/// Creates the exponential-loss objective.
/// </summary>
ObjectiveFunctions make_exponential_objective();

/// <summary>
/// Creates the hinge-loss objective.
/// </summary>
ObjectiveFunctions make_hinge_objective();

// Common objective evaluation and composition.
/// <summary>
/// Combines a data objective with zero or more regularization terms.
/// </summary>
ObjectiveConfig make_objective_config(
    const ObjectiveFunctions& data_objective =
        make_binary_cross_entropy_objective(),
    std::vector<RegularizationTerm> regularizers = {}
);

/// <summary>
/// Combines a data objective with one regularization term.
/// </summary>
ObjectiveConfig make_objective_config(
    const ObjectiveFunctions& data_objective,
    RegularizationTerm regularizer
);

/// <summary>
/// Validates that an objective supplies compatible loss and gradient callbacks.
/// </summary>
void validate_objective_functions(
    const ObjectiveFunctions& objective
);

/// <summary>
/// Evaluates default-objective loss for one cached network sample.
/// </summary>
double sample_cost(
    const MLP& network,
    const ForwardCache& cache,
    const Values& target
);

/// <summary>
/// Evaluates a specified objective's loss for one cached network sample.
/// </summary>
double sample_cost(
    const MLP& network,
    const ForwardCache& cache,
    const Values& target,
    const ObjectiveFunctions& objective
);

/// <summary>
/// Averages default-objective loss across a non-empty batch.
/// </summary>
double batch_loss(
    const MLP& network,
    const Dataset& batch
);

/// <summary>
/// Averages a specified objective's loss across a non-empty batch.
/// </summary>
double batch_loss(
    const MLP& network,
    const Dataset& batch,
    const ObjectiveFunctions& objective
);

/// <summary>
/// Creates an L2 regularization term with the requested coefficient.
/// </summary>
RegularizationTerm make_l2_regularization(
    double coefficient,
    bool include_biases = false
);

enum class L1Method {
    Subgradient,
    Proximal,
    EpsilonSmooth,
    LogCoshSmooth,
    // Compatibility aliases for the original public spellings.
    Eps_SmoothL1Regularizer = EpsilonSmooth,
    Log_SmoothL1Regularizer = LogCoshSmooth
};

/// Options shared by L1 and elastic-net regularization.
/// epsilon is used only by EpsilonSmooth; temperature is used only by
/// LogCoshSmooth. Both smooth penalties approach |x| as their parameter
/// approaches zero from above.
struct L1RegularizationOptions {
    bool include_biases{ false };
    L1Method method{ L1Method::Subgradient };
    double epsilon{ 1e-6 };
    double temperature{ 1e-2 };
};

/// Elastic net combines the selected L1 implementation with an L2 penalty.
/// Select L1Method::Proximal to use a smooth L2 gradient plus a post-step L1
/// soft-thresholding operation.
struct ElasticNetOptions {
    double l1_coefficient{};
    double l2_coefficient{};
    L1RegularizationOptions l1{};
};

// Options-based overload for callers who need smoothing or proximal choices.
/// <summary>
/// Creates an L1 regularization term using the supplied implementation options.
/// </summary>
RegularizationTerm make_l1_regularization(
    double coefficient,
    L1RegularizationOptions options
);

// Convenience overload retained for the original common call form.
/// <summary>
/// Creates an L1 regularization term with the selected implementation.
/// </summary>
RegularizationTerm make_l1_regularization(
    double coefficient,
    bool include_biases = false,
    L1Method method = L1Method::Subgradient
);

/// <summary>
/// Creates an L1 term that contributes a subgradient during backpropagation.
/// </summary>
RegularizationTerm make_l1_regularization_subgradient(
    double coefficient,
    bool include_biases = false
);

/// <summary>
/// Creates a differentiable epsilon-smoothed L1 regularization term.
/// </summary>
RegularizationTerm eps_make_l1_regularization_smooth(
    double coefficient,
    bool include_biases = false,
    double epsilon = 1e-6
);

/// <summary>
/// Creates a differentiable log-cosh-smoothed L1 regularization term.
/// </summary>
RegularizationTerm log_make_l1_regularization_smooth(
    double coefficient,
    bool include_biases = false,
    double temperature = 1e-2
);

/// <summary>
/// Creates an L1 term applied through a post-step proximal update.
/// </summary>
RegularizationTerm make_l1_regularization_Proximal(
    double coefficient,
    bool include_biases = false
);

/// <summary>
/// Creates an elastic-net term from explicit L1 and L2 options.
/// </summary>
RegularizationTerm make_elastic_net_regularization(
    ElasticNetOptions options
);

// Convenience overload for the original elastic-net call form. It selects
// the L1 subgradient implementation.
/// <summary>
/// Creates an elastic-net term using the L1 subgradient implementation.
/// </summary>
RegularizationTerm make_elastic_net_regularization(
    double l1_coefficient,
    double l2_coefficient,
    bool include_biases = false
);

/// <summary>
/// Validates a composed data objective and regularization configuration.
/// </summary>
void validate_objective_config(
    const ObjectiveConfig& objective
);

/// <summary>
/// Computes the total regularization penalty for a network.
/// </summary>
double regularization_loss(
    const MLP& network,
    const ObjectiveConfig& objective
);

/// <summary>
/// Adds smooth regularization gradients to an existing gradient vector.
/// </summary>
void add_regularization_gradients(
    const MLP& network,
    const ObjectiveConfig& objective,
    NetworkGradients& gradients
);

/// <summary>
/// Applies all configured proximal regularization updates in place.
/// </summary>
void apply_proximal_updates(
    MLP& network,
    const ObjectiveConfig& objective,
    double effective_step_size
);

/// <summary>
/// Computes data loss plus regularization loss for a batch.
/// </summary>
double objective_loss(
    const MLP& network,
    const Dataset& batch,
    const ObjectiveConfig& objective
);

/// <summary>
/// Computes data and regularization gradients for a batch.
/// </summary>
NetworkGradients objective_gradients(
    const MLP& network,
    const Dataset& batch,
    const ObjectiveConfig& objective
);

/// <summary>
/// Computes the Hessian-vector product of the batch data objective plus all
/// differentiable regularizers. Non-smooth or proximal regularizers, and
/// regularizers lacking an add_gradient_jvp callback, are rejected.
/// </summary>
NetworkGradients objective_hessian_vector_product(
    const MLP& network,
    const Dataset& batch,
    const NetworkTangent& parameter_tangent,
    const ObjectiveConfig& objective
);
