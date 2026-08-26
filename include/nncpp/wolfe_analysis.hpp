#pragma once

#include "objective_functions.hpp"

/// <summary>
/// Determines whether Wolfe line-search parameters satisfy their required bounds.
/// </summary>
bool wolfe_parameters_are_valid(
    const WolfeParameters& parameters
) noexcept;

/// <summary>
/// Validates Wolfe line-search parameters and throws for invalid values.
/// </summary>
void validate_wolfe_parameters(
    const WolfeParameters& parameters
);

/// <summary>
/// Computes the dot product of two network-shaped gradient vectors.
/// </summary>
double network_vector_dot(
    const NetworkGradients& left,
    const NetworkGradients& right
);

/// <summary>
/// Creates the steepest-descent direction for a gradient.
/// </summary>
NetworkDirection make_negative_gradient_direction(
    const NetworkGradients& gradient
);

/// <summary>
/// Computes the cosine between a gradient and a proposed search direction.
/// </summary>
double downhill_cosine(
    const NetworkGradients& gradient,
    const NetworkDirection& direction
);

/// <summary>
/// Determines whether a direction meets the required descent cosine.
/// </summary>
bool is_downhill_direction(
    const NetworkGradients& gradient,
    const NetworkDirection& direction,
    double minimum_cosine = 0.0
);

/// <summary>
/// Adds a scaled network direction to a network in place.
/// </summary>
void apply_direction(
    MLP& network,
    const NetworkDirection& direction,
    double scale
);

/// <summary>
/// Returns a copy of a network displaced along a search direction.
/// </summary>
MLP make_candidate_network(
    const MLP& current_network,
    const NetworkDirection& direction,
    double step_size
);

/// <summary>
/// Tests the Armijo sufficient-decrease condition for a trial step.
/// </summary>
bool satisfies_sufficient_decrease(
    double candidate_loss,
    double current_loss,
    double step_size,
    double initial_directional_derivative,
    double sufficient_decrease_constant
);

/// <summary>
/// Tests the strong-curvature condition for a trial step.
/// </summary>
bool satisfies_strong_curvature(
    double candidate_directional_derivative,
    double initial_directional_derivative,
    double curvature_constant
);

/// <summary>
/// Determines whether an evaluation satisfies both strong Wolfe conditions.
/// </summary>
bool satisfies_strong_wolfe(
    const WolfeEvaluation& evaluation
) noexcept;

/// <summary>
/// Evaluates a candidate step against the configured Wolfe conditions.
/// </summary>
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

/// <summary>
/// Evaluates a candidate against the complete data-plus-regularization objective.
/// </summary>
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

/// <summary>
/// Searches for a valid step size by backtracking under Wolfe conditions.
/// </summary>
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

/// <summary>
/// Searches using the complete data-plus-regularization objective.
/// </summary>
LineSearchResult backtracking_wolfe_stepsize(
    const MLP& current_network,
    const Dataset& batch,
    double current_loss,
    const NetworkGradients& current_gradient,
    const NetworkDirection& direction,
    const WolfeParameters& parameters,
    const ObjectiveConfig& objective
);

/// <summary>
/// Attempts one gradient step selected by the Wolfe line search.
/// </summary>
TrainingStepResult take_wolfe_gradient_step(
    MLP& network,
    const Dataset& batch,
    const WolfeParameters& parameters,
    double gradient_tolerance,
    const ObjectiveFunctions& objective =
        make_binary_cross_entropy_objective()
);

/// <summary>
/// Returns the display name for a line-search status.
/// </summary>
const char* line_search_status_name(
    LineSearchStatus status
) noexcept;
