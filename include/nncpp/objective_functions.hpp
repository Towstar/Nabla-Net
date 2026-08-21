#pragma once

#include "mlp.hpp"

#include <vector>

// Objective-function primitives and objective factories.
Values stable_softmax(const Values& logits);

double binary_cross_entropy_from_logit(
    double logit,
    double target
);

double cross_entropy_from_logits(
    const Values& logits,
    const Values& targets
);

ObjectiveFunctions make_binary_cross_entropy_objective();
ObjectiveFunctions make_softmax_cross_entropy_objective();
ObjectiveFunctions make_mean_squared_error_objective();
ObjectiveFunctions make_exponential_objective();
ObjectiveFunctions make_hinge_objective();

// Common objective evaluation and composition.
ObjectiveConfig make_objective_config(
    const ObjectiveFunctions& data_objective =
        make_binary_cross_entropy_objective(),
    std::vector<RegularizationTerm> regularizers = {}
);

ObjectiveConfig make_objective_config(
    const ObjectiveFunctions& data_objective,
    RegularizationTerm regularizer
);

void validate_objective_functions(
    const ObjectiveFunctions& objective
);

double sample_cost(
    const MLP& network,
    const ForwardCache& cache,
    const Values& target
);

double sample_cost(
    const MLP& network,
    const ForwardCache& cache,
    const Values& target,
    const ObjectiveFunctions& objective
);

double batch_loss(
    const MLP& network,
    const Dataset& batch
);

double batch_loss(
    const MLP& network,
    const Dataset& batch,
    const ObjectiveFunctions& objective
);

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
RegularizationTerm make_l1_regularization(
    double coefficient,
    L1RegularizationOptions options
);

// Convenience overload retained for the original common call form.
RegularizationTerm make_l1_regularization(
    double coefficient,
    bool include_biases = false,
    L1Method method = L1Method::Subgradient
);

RegularizationTerm make_l1_regularization_subgradient(
    double coefficient,
    bool include_biases = false
);

RegularizationTerm eps_make_l1_regularization_smooth(
    double coefficient,
    bool include_biases = false,
    double epsilon = 1e-6
);

RegularizationTerm log_make_l1_regularization_smooth(
    double coefficient,
    bool include_biases = false,
    double temperature = 1e-2
);

RegularizationTerm make_l1_regularization_Proximal(
    double coefficient,
    bool include_biases = false
);

RegularizationTerm make_elastic_net_regularization(
    ElasticNetOptions options
);

// Convenience overload for the original elastic-net call form. It selects
// the L1 subgradient implementation.
RegularizationTerm make_elastic_net_regularization(
    double l1_coefficient,
    double l2_coefficient,
    bool include_biases = false
);

void validate_objective_config(
    const ObjectiveConfig& objective
);

double regularization_loss(
    const MLP& network,
    const ObjectiveConfig& objective
);

void add_regularization_gradients(
    const MLP& network,
    const ObjectiveConfig& objective,
    NetworkGradients& gradients
);

void apply_proximal_updates(
    MLP& network,
    const ObjectiveConfig& objective,
    double effective_step_size
);

double objective_loss(
    const MLP& network,
    const Dataset& batch,
    const ObjectiveConfig& objective
);

NetworkGradients objective_gradients(
    const MLP& network,
    const Dataset& batch,
    const ObjectiveConfig& objective
);
