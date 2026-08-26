// Wolfe line-search and direction-analysis implementation.
#include "../include/nncpp/wolfe_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

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
	return evaluate_wolfe_candidate(
		current_network,
		batch,
		current_loss,
		current_gradient,
		direction,
		step_size,
		parameters,
		make_objective_config(objective)
	);
}

WolfeEvaluation evaluate_wolfe_candidate(
    const MLP& current_network,
    const Dataset& batch,
    const double current_loss,
    const NetworkGradients& current_gradient,
    const NetworkDirection& direction,
    const double step_size,
    const WolfeParameters& parameters,
    const ObjectiveConfig& objective
)
{
	validate_objective_config(objective);
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
	auto candidate_loss = objective_loss(candidate_network, batch, objective);
    bool satisfies_suff_decrease = satisfies_sufficient_decrease(
        candidate_loss,
        current_loss,
        step_size,
        network_vector_dot(current_gradient, direction),
        parameters.sufficient_decrease
	);
	auto candidate_gradient = objective_gradients(candidate_network, batch, objective);
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
        objective.data_objective
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
	return backtracking_wolfe_stepsize(
		current_network,
		batch,
		current_loss,
		current_gradient,
		direction,
		parameters,
		make_objective_config(objective)
	);
}

LineSearchResult backtracking_wolfe_stepsize(
    const MLP& current_network,
    const Dataset& batch,
    const double current_loss,
    const NetworkGradients& current_gradient,
    const NetworkDirection& direction,
    const WolfeParameters& parameters,
    const ObjectiveConfig& objective
)
{
    validate_network(current_network);
    validate_gradients_like(current_network, current_gradient);
    validate_gradients_like(current_network, direction);
    validate_objective_config(objective);
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
	const ObjectiveFunctions& objective
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
