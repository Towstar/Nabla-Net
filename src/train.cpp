#include "detail/train.hpp"

#include "detail/objective_functions.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace nablanet {

namespace {
bool has_proximal_regularizer(const ObjectiveConfig& objective)
{
    return std::any_of(
        objective.regularizers.begin(),
        objective.regularizers.end(),
        [](const RegularizationTerm& term) { return term.proximal; }
    );
}
}

TrainingReport train(
    MLP& network,
    const Dataset& dataset,
    const OptimizerSpec& optimizer,
    const TrainingConfig& config,
    const ObjectiveConfig& objective
) {
    validate_training_inputs(network, dataset, optimizer, config, objective);
    std::unique_ptr<Optimizer> optimizer_instance = optimizer.make();
    if (!optimizer_instance)
        throw std::runtime_error("Optimizer factory returned a null optimizer.");
    optimizer_instance->reset(network);

    TrainingReport report;
    report.optimizer_name = optimizer.name;

    std::mt19937 random_engine(config.shuffle_seed);

    const std::size_t batch_size = get_effective_batch_size(config, dataset.size());

    const auto maximum_epochs = config.max_epochs;

    const auto maximum_iterations =
        config.max_iterations;

    std::size_t epoch = 0;
    while (!maximum_epochs.has_value() || epoch < *maximum_epochs) {
        const std::vector<std::size_t> order = make_epoch_order(dataset.size(), config, random_engine);
        for (std::size_t begin = 0; begin < dataset.size(); begin += batch_size) {
            if (maximum_iterations.has_value() && report.steps >= *maximum_iterations) {
                report.completed = true;
                report.stop_reason = TrainingStopReason::MaxIterations;
                return report;
            }
            const std::size_t remaining = dataset.size() - begin;
            const std::size_t count = std::min(batch_size, remaining);
            const std::size_t end = begin + count;
            const Dataset batch = make_batch(dataset, order, begin, end);
            const double current_loss = objective_loss(network, batch, objective);
            NetworkGradients current_gradient = objective_gradients(network, batch, objective);
            const double current_gradient_norm = gradient_l2_norm(current_gradient);
            if (report.steps == 0)
                report.initial_loss = current_loss;
            if (config.gradient_tolerance.has_value() && current_gradient_norm <= *config.gradient_tolerance) {
                report.final_loss = current_loss;
                report.final_gradient_norm = current_gradient_norm;
                report.final_parameter_change = 0.0;
                report.completed = true;
                report.stop_reason = TrainingStopReason::GradientTolerance;
                return report;
            }
            const MLP before_step = network;
            OptimizerContext context{
                network,
                batch,
                objective,
                current_loss,
                std::move(current_gradient)
            };

            // THE DESCENT STEP
            TrainingStepResult step_result = optimizer_instance->step(context);

            if (!step_result.updated) {
                report.final_loss = step_result.new_loss;
                report.final_gradient_norm = step_result.gradient_norm;
                report.final_parameter_change = parameter_change(before_step, network);
                report.completed = true;
                report.stop_reason = TrainingStopReason::OptimizerStopped;
                return report;
            }

            if (has_proximal_regularizer(objective)) {
                if (!step_result.effective_step_size.has_value()) {
                    throw std::runtime_error(
                        "Optimizer did not report an effective step size required by a proximal regularizer."
                    );
                }
                apply_proximal_updates(
                    network,
                    objective,
                    *step_result.effective_step_size
                );
                step_result.new_loss = objective_loss(network, batch, objective);
            }
            validate_network(network);
            const double change = parameter_change(before_step, network);

            const bool completed_epoch = (end == dataset.size());

            report.steps++;

            report.final_loss = step_result.new_loss;
            report.final_gradient_norm = step_result.gradient_norm;
            report.final_parameter_change = change;

            if (config.record_history) {
                report.losses.push_back(step_result.new_loss);
                report.gradient_norms.push_back(step_result.gradient_norm);
                report.parameter_changes.push_back(change);
            }

            if (completed_epoch) {
                epoch++;
                report.epochs_completed = epoch;
            }

            if (config.parameter_change_tolerance.has_value() && change <= *config.parameter_change_tolerance) {
                report.completed = true;
                report.stop_reason = TrainingStopReason::ParameterChangeTolerance;
                return report;
            }
            if (maximum_iterations.has_value() && report.steps >= *maximum_iterations) {
                report.completed = true;
                report.stop_reason = TrainingStopReason::MaxIterations;
                return report;
            }
            if (completed_epoch &&
                maximum_epochs.has_value() &&
                epoch >= *maximum_epochs) {
                report.completed = true;
                report.stop_reason = TrainingStopReason::MaxEpochs;
                return report;
            }
        }
    }
    throw std::logic_error(
        "Training loop exited without a stop reason."
    );
}

void validate_training_inputs(
    const MLP& network,
    const Dataset& dataset,
    const OptimizerSpec& optimizer,
    const TrainingConfig& config,
    const ObjectiveConfig& objective)
{
    validate_network(network);
    validate_optimizer_spec(optimizer);
    validate_training_config(config);
    validate_objective_config(objective);

    if (dataset.empty())
        throw std::invalid_argument("Dataset cannot be empty.");

    if (has_proximal_regularizer(objective) && !optimizer.supports_proximal) {
        throw std::invalid_argument(
            "The selected optimizer does not support proximal regularization."
        );
    }

    if ((optimizer.requirement == OptimizerRequirement::DeterministicFullBatch)
        && (config.batch_mode != BatchMode::FullBatch || config.shuffle))
        throw std::invalid_argument("This optimizer requires deterministic full batch training.");
}

std::size_t get_effective_batch_size(
    const TrainingConfig& config,
    const std::size_t dataset_size
) {
    switch (config.batch_mode) {
        case BatchMode::FullBatch:
            return dataset_size;
        case BatchMode::MiniBatch:
            return config.batch_size;
        case BatchMode::Stochastic:
            return 1;
        default:
            throw std::invalid_argument("Invalid batch mode.");
    }
}

std::vector<std::size_t> make_epoch_order(
    const std::size_t dataset_size,
    const TrainingConfig& config,
    std::mt19937& random_engine
) {
    std::vector<std::size_t> order(dataset_size);

    for (std::size_t idx = 0; idx < dataset_size; idx++)
        order[idx] = idx;
    if (config.shuffle)
        std::shuffle(order.begin(), order.end(), random_engine);
    return order;
}

Dataset make_batch(
    const Dataset& dataset,
    const std::vector<std::size_t>& order,
    const std::size_t begin,
    const std::size_t end
) {
    Dataset batch;
    batch.reserve(end - begin);

    for (std::size_t pos = begin; pos < end; pos++)
        batch.push_back(dataset[order[pos]]);
    return batch;
}

/// <summary>
/// Uses infinity norm to asses change in parameters
/// </summary>
/// <param name="before">: Network parameters (weights and biases) from before any parameter changes</param>
/// <param name="after">: Network parameters (weights and biases) after any parameter changes</param>
/// <returns></returns>
double parameter_change(const MLP& before, const MLP& after) {
    validate_network(before);
    validate_network(after);

    if (before.layers.size() != after.layers.size())
        throw std::invalid_argument("networks must have matching layer counts.");

    double maximum_change = 0.0;

    for (std::size_t layer_idx = 0; layer_idx < before.layers.size(); layer_idx++) {
        const DenseLayer& before_layer = before.layers[layer_idx];
        const DenseLayer& after_layer = after.layers[layer_idx];

        if (before_layer.weights.size() != after_layer.weights.size() ||
            before_layer.biases.size() != after_layer.biases.size())
            throw std::invalid_argument("Networks must have matching parameter shapes.");

        for (std::size_t idx = 0; idx < before_layer.weights.size(); idx++)
            maximum_change =
                std::max(maximum_change, std::abs(after_layer.weights[idx] - before_layer.weights[idx]));
        for (std::size_t idx = 0; idx < before_layer.biases.size(); idx++)
            maximum_change =
                std::max(maximum_change, std::abs(after_layer.biases[idx] - before_layer.biases[idx]));
        if (!std::isfinite(maximum_change))
            throw std::overflow_error("Parameter-change norm is not finite.");
    }
    return maximum_change;
}

} // namespace nablanet
