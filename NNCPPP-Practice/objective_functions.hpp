#pragma once

#include "mlp.hpp"

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

// Common objective evaluation and composition.
ObjectiveConfig make_objective_config(
    const ObjectiveFunctions& data_objective =
        make_binary_cross_entropy_objective(),
    std::vector<RegularizationTerm> regularizers = {}
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

RegularizationTerm make_l1_regularization(
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
