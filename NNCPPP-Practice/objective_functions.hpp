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
    Eps_SmoothL1Regularizer,
    Log_SmoothL1Regularizer
};

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
    bool include_biases = false
);

RegularizationTerm log_make_l1_regularization_smooth(
    double coefficient,
    bool include_biases = false
);

RegularizationTerm make_l1_regularization_Proximal(
    double coefficient,
    bool include_biases = false
);

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
