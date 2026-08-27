#pragma once

#include "../../include/nablanet/objective_functions.hpp"

namespace nablanet {

// Internal line-search implementation contract. Public callers configure the
// line search through WolfeParameters in optimizer options; they do not need
// these individual search helpers.
bool wolfe_parameters_are_valid(
    const WolfeParameters& parameters
) noexcept;

void validate_wolfe_parameters(
    const WolfeParameters& parameters
);

double network_vector_dot(
    const NetworkGradients& left,
    const NetworkGradients& right
);

NetworkDirection make_negative_gradient_direction(
    const NetworkGradients& gradient
);

double downhill_cosine(
    const NetworkGradients& gradient,
    const NetworkDirection& direction
);

bool is_downhill_direction(
    const NetworkGradients& gradient,
    const NetworkDirection& direction,
    double minimum_cosine = 0.0
);

void apply_direction(
    MLP& network,
    const NetworkDirection& direction,
    double scale
);

MLP make_candidate_network(
    const MLP& current_network,
    const NetworkDirection& direction,
    double step_size
);

bool satisfies_sufficient_decrease(
    double candidate_loss,
    double current_loss,
    double step_size,
    double initial_directional_derivative,
    double sufficient_decrease_constant
);

bool satisfies_strong_curvature(
    double candidate_directional_derivative,
    double initial_directional_derivative,
    double curvature_constant
);

bool satisfies_strong_wolfe(
    const WolfeEvaluation& evaluation
) noexcept;

WolfeEvaluation evaluate_wolfe_candidate(
    const MLP& current_network,
    const Dataset& batch,
    double current_loss,
    const NetworkGradients& current_gradient,
    const NetworkDirection& direction,
    double step_size,
    const WolfeParameters& parameters,
    const ObjectiveFunctions& objective =
        make_binary_cross_entropy_objective()
);

WolfeEvaluation evaluate_wolfe_candidate(
    const MLP& current_network,
    const Dataset& batch,
    double current_loss,
    const NetworkGradients& current_gradient,
    const NetworkDirection& direction,
    double step_size,
    const WolfeParameters& parameters,
    const ObjectiveConfig& objective
);

LineSearchResult backtracking_wolfe_stepsize(
    const MLP& current_network,
    const Dataset& batch,
    double current_loss,
    const NetworkGradients& current_gradient,
    const NetworkDirection& direction,
    const WolfeParameters& parameters,
    const ObjectiveFunctions& objective =
        make_binary_cross_entropy_objective()
);

LineSearchResult backtracking_wolfe_stepsize(
    const MLP& current_network,
    const Dataset& batch,
    double current_loss,
    const NetworkGradients& current_gradient,
    const NetworkDirection& direction,
    const WolfeParameters& parameters,
    const ObjectiveConfig& objective
);

TrainingStepResult take_wolfe_gradient_step(
    MLP& network,
    const Dataset& batch,
    const WolfeParameters& parameters,
    double gradient_tolerance,
    const ObjectiveFunctions& objective =
        make_binary_cross_entropy_objective()
);

const char* line_search_status_name(
    LineSearchStatus status
) noexcept;

} // namespace nablanet
