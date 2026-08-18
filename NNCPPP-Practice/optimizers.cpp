#include "optimizers.hpp"

#include <cmath>
#include <cctype>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <iostream>
#include <ranges>
#include <algorithm>

#pragma region Optimizer Implementations and Factories

namespace
{
    class SGDOptimizer final : public Optimizer
    {
    public:
        explicit SGDOptimizer(LearningRateSchedule schedule)
            : learning_rate_schedule_(std::move(schedule))
        {
            if (!learning_rate_schedule_) {
                throw std::invalid_argument(
                    "SGD learning-rate schedule cannot be empty."
                );
            }
        }

        const char* name() const noexcept override
        {
            return "SGD";
        }

        void reset(const MLP& network) override
        {
            validate_network(network);
            update_index_ = 0;
        }

        TrainingStepResult step(OptimizerContext& context) override
        {
            validate_network(context.network);
            validate_objective_config(context.objective);

            if (context.batch.empty()) {
                throw std::invalid_argument(
                    "SGD batch cannot be empty."
                );
            }

            validate_gradients_like(
                context.network,
                context.current_gradient
            );

            if (!std::isfinite(context.current_loss)) {
                throw std::invalid_argument(
                    "Current loss must be finite."
                );
            }

            return _step(context);
        }

    private:
        LearningRateSchedule learning_rate_schedule_;
        std::size_t update_index_{};

        TrainingStepResult _step(OptimizerContext& context)
        {
            const double learning_rate =
                learning_rate_schedule_(update_index_);

            if (!learning_rate_is_valid(learning_rate)) {
                throw std::invalid_argument(
                    "Learning-rate schedule produced an invalid rate."
                );
            }

            TrainingStepResult result{};
            result.previous_loss = context.current_loss;
            result.gradient_norm =
                gradient_l2_norm(context.current_gradient);

            MLP candidate = context.network;

            apply_gradient(
                candidate,
                context.current_gradient,
                learning_rate
            );

            result.new_loss = objective_loss(
                candidate,
                context.batch,
                context.objective
            );

            context.network = std::move(candidate);
            ++update_index_;

            result.updated = true;
            return result;
        }
    };

    class MomentumSGDOptimizer final : public Optimizer
    {
    public:
        explicit MomentumSGDOptimizer(LearningRateSchedule schedule, double momentum_coefficient=0.5)
            : learning_rate_schedule_(std::move(schedule))
        {
            if (!learning_rate_schedule_) {
                throw std::invalid_argument(
                    "Momentum SGD learning-rate schedule cannot be empty."
                );
            }
            if (momentum_coefficient < 0 || momentum_coefficient >= 1)
                throw std::invalid_argument("Momentum coefficient must be in the range [0,1).");
            if (!std::isfinite(momentum_coefficient))
                throw std::invalid_argument("Momentum coefficient must be finite.");
            _momentum_coefficient = momentum_coefficient;
        }

        const char* name() const noexcept override
        {
            return "MomentumSGD";
        }

        void reset(const MLP& network) override
        {
            validate_network(network);
            _velocity = make_zero_gradients_like(network);
            update_index_ = 0;
        }

        TrainingStepResult step(OptimizerContext& context) override
        {
            validate_network(context.network);
            validate_objective_config(context.objective);
            validate_gradients_like(context.network, _velocity);

            if (context.batch.empty()) {
                throw std::invalid_argument(
                    "Momentum SGD batch cannot be empty."
                );
            }

            validate_gradients_like(
                context.network,
                context.current_gradient
            );

            if (!std::isfinite(context.current_loss)) {
                throw std::invalid_argument(
                    "Current loss must be finite."
                );
            }

            return _step(context);
        }

    private:
        NetworkGradients _velocity;
        double _momentum_coefficient{};

        LearningRateSchedule learning_rate_schedule_;
        std::size_t update_index_{};

        TrainingStepResult _step(OptimizerContext& context)
        {
            const double learning_rate =
                learning_rate_schedule_(update_index_);

            if (!learning_rate_is_valid(learning_rate)) {
                throw std::invalid_argument(
                    "Learning-rate schedule produced an invalid rate."
                );
            }

            TrainingStepResult result{};
            result.previous_loss = context.current_loss;
            result.gradient_norm =
                gradient_l2_norm(context.current_gradient);

            MLP candidate = context.network;

            NetworkGradients next_velocity = _velocity;

            std::size_t layer_idx = 0;
            
            for (LayerGradients& next_layer : next_velocity.layers) {
                const LayerGradients& old_layer = _velocity.layers[layer_idx];
                const LayerGradients& gradient_layer = context.current_gradient.layers[layer_idx];
                
                for (std::size_t idx = 0; idx < next_layer.weights.size(); idx++)
                    next_layer.weights[idx] = _momentum_coefficient * old_layer.weights[idx] + gradient_layer.weights[idx];
            
                for (std::size_t idx = 0; idx < next_layer.biases.size(); idx++)
                    next_layer.biases[idx] = _momentum_coefficient * old_layer.biases[idx] + gradient_layer.biases[idx];
                layer_idx++;
            }
            
            apply_gradient(candidate, next_velocity, learning_rate);
            const double new_loss = objective_loss(candidate, context.batch, context.objective);
            result.new_loss = new_loss;

            context.network = std::move(candidate);
            _velocity = std::move(next_velocity);

            update_index_++;

            result.updated = true;
            return result;
        }
    };

    class AdaGradOptimizer final : public Optimizer
    {
    public:
        explicit AdaGradOptimizer(AdaGradOptions options) {
            if (!options.learning_rate_schedule) {
                throw std::invalid_argument(
                    "AdaGrad learning-rate schedule cannot be empty."
                );
            }
            if (options.epsilon < 0.0 || !std::isfinite(options.epsilon))
                throw std::invalid_argument(
                    "Epsilon cannot be negative"
                );
            schedule = options.learning_rate_schedule;
            epsilon = options.epsilon;
        }
        const char* name() const noexcept override {
            return "AdaGrad";
        }
        void reset(const MLP& network) override {
            validate_network(network);
            accumulated_squared_gradient = make_zero_gradients_like(network);
            update_index = 0;
        }
        TrainingStepResult step(OptimizerContext& context) override {

            validate_network(context.network);
            validate_objective_config(context.objective);
            validate_gradients_like(context.network, context.current_gradient);

            if (context.batch.empty()) {
                throw std::invalid_argument(
                    "Momentum SGD batch cannot be empty."
                );
            }
            if (!std::isfinite(context.current_loss)) {
                throw std::invalid_argument(
                    "Current loss must be finite."
                );
            }

            return _step(context);
        }
    private:
        size_t update_index{ 0 };
        NetworkGradients accumulated_squared_gradient;
        double epsilon{ 0.01 };
        LearningRateSchedule schedule;

        TrainingStepResult _step(OptimizerContext& context) {

            MLP candidate = context.network;
            const NetworkGradients& gradient = context.current_gradient;

            NetworkGradients next_accumulated = accumulated_squared_gradient;
            
            const double learning_rate = schedule(update_index);
            
            if (!learning_rate_is_valid(learning_rate)) {
                throw std::invalid_argument(
                    "Learning-rate schedule produced an invalid rate."
                );
            }

            for (std::size_t layer_index = 0; layer_index < gradient.layers.size(); layer_index++) {
                const LayerGradients& grad_layer = gradient.layers[layer_index];
                LayerGradients& acx_layer = next_accumulated.layers[layer_index];
                DenseLayer& candidate_layer = candidate.layers[layer_index];

                for (std::size_t i = 0; i < grad_layer.weights.size(); i++) {
                    const double g = grad_layer.weights[i];
                    acx_layer.weights[i] += g * g;

                    if (!std::isfinite(acx_layer.weights[i])) {
                        throw std::overflow_error(
                            "AdaGrad squared-gradient accumulator overflowed."
                        );
                    }

                    const double update = learning_rate * g / (std::sqrt(acx_layer.weights[i]) + epsilon);
                    candidate_layer.weights[i] -= update;
                }

                for (std::size_t i = 0; i < grad_layer.biases.size(); i++) {
                    const double g = grad_layer.biases[i];
                    acx_layer.biases[i] += g * g;

                    if (!std::isfinite(acx_layer.biases[i])) {
                        throw std::overflow_error(
                            "AdaGrad squared-gradient accumulator overflowed."
                        );
                    }

                    const double update = learning_rate * g / (std::sqrt(acx_layer.biases[i]) + epsilon);
                    candidate_layer.biases[i] -= update;
                }
            }

            validate_network(candidate);
            const double new_loss = objective_loss(candidate, context.batch, context.objective);

            if (!std::isfinite(new_loss))
                throw std::runtime_error("AdaGrad produced a non-finite loss.");

            context.network = std::move(candidate);
            accumulated_squared_gradient = std::move(next_accumulated);

            update_index++;

            TrainingStepResult result{};
            result.updated = true;
            result.previous_loss = context.current_loss;
            result.new_loss = new_loss;
            result.gradient_norm =
                gradient_l2_norm(context.current_gradient);

            return result;
        }
    };

    class RMSPropOptimizer final : public Optimizer
    {
    public:
        const char* name() const noexcept override;
        void reset(const MLP& network) override;
        TrainingStepResult step(OptimizerContext& context) override;
    };

    class AdamOptimizer final : public Optimizer
    {
    public:
        const char* name() const noexcept override;
        void reset(const MLP& network) override;
        TrainingStepResult step(OptimizerContext& context) override;
    };

    class AdamWOptimizer final : public Optimizer
    {
    public:
        const char* name() const noexcept override;
        void reset(const MLP& network) override;
        TrainingStepResult step(OptimizerContext& context) override;
    };

    class LBFGSOptimizer final : public Optimizer
    {
    public:
        const char* name() const noexcept override;
        void reset(const MLP& network) override;
        TrainingStepResult step(OptimizerContext& context) override;
    };

    std::unique_ptr<Optimizer> make_sgd_instance(
        LearningRateSchedule schedule
    )
    {
        return std::make_unique<SGDOptimizer>(
            std::move(schedule)
        );
    }

    std::unique_ptr<Optimizer> make_momentum_sgd_instance(
        LearningRateSchedule schedule,
        double momentum_coefficient
    )
    {
        return std::make_unique<MomentumSGDOptimizer>(
            std::move(schedule),
            momentum_coefficient
        );
    }

    std::unique_ptr<Optimizer> make_adagrad_instance(AdaGradOptions options) {
        return std::make_unique<AdaGradOptimizer>(
            std::move(options)
        );
    }
    
    std::unique_ptr<Optimizer> make_rmsprop_optimizer();
    std::unique_ptr<Optimizer> make_adam_optimizer();
    std::unique_ptr<Optimizer> make_adamw_optimizer();
    std::unique_ptr<Optimizer> make_lbfgs_optimizer();
}

OptimizerSpec make_sgd(SGDOptions options)
{
    if (!options.learning_rate_schedule) {
        throw std::invalid_argument(
            "SGD learning-rate schedule cannot be empty."
        );
    }

    const LearningRateSchedule schedule =
        options.learning_rate_schedule;

    return make_custom_optimizer(
        "SGD",
        OptimizerRequirement::MiniBatchCompatible,
        [schedule] {
            return make_sgd_instance(schedule);
        }
    );
}

OptimizerSpec make_momentum_sgd(MomentumSGDOptions options) {
    if (!options.learning_rate_schedule) {
        throw std::invalid_argument(
            "Momentum SGD learning-rate schedule cannot be empty."
        );
    }

    const LearningRateSchedule schedule =
        options.learning_rate_schedule;
    const double momentum =
        options.momentum;

    return make_custom_optimizer(
        "MomentumSGD",
        OptimizerRequirement::MiniBatchCompatible,
        [schedule, momentum] {
            return make_momentum_sgd_instance(schedule, momentum);
        }
    );
}

OptimizerSpec make_adagrad(AdaGradOptions options) {
    if (!options.learning_rate_schedule) {
        throw std::invalid_argument(
            "Adagrad learning-rate schedule cannot be empty."
        );
    }
    if (!options.epsilon) {
        throw std::invalid_argument(
            "Adagrad epsilon must be defined."
        );
    }
    if (!std::isfinite(options.epsilon) || options.epsilon < 0.0) {
        throw std::invalid_argument(
            "epsilon must be finite and positive."
        );
    }
    const LearningRateSchedule schedule = options.learning_rate_schedule;
    const double epsilon = options.epsilon;
    return make_custom_optimizer("AdaGrad", OptimizerRequirement::MiniBatchCompatible, [schedule, epsilon]() {return make_adagrad_instance(AdaGradOptions{ schedule, epsilon }); });
}

#pragma endregion

#pragma region Built-in Optimizer Specifications

namespace Optimizers {
    const OptimizerSpec SGD = make_sgd();

    const OptimizerSpec MomentumSGD = make_momentum_sgd();

    const OptimizerSpec AdaGrad = make_adagrad();

    const OptimizerSpec RMSProp{
        "RMSProp",
        OptimizerRequirement::MiniBatchCompatible,
        {}
    };

    const OptimizerSpec Adam{
        "Adam",
        OptimizerRequirement::MiniBatchCompatible,
        {}
    };

    const OptimizerSpec AdamW{
        "AdamW",
        OptimizerRequirement::MiniBatchCompatible,
        {}
    };

    const OptimizerSpec LBFGS{
        "LBFGS",
        OptimizerRequirement::DeterministicFullBatch,
        {}
    };
}

#pragma endregion

#pragma region Optimizer Validation Helpers

namespace
{
    bool is_valid_optimizer_requirement(
        const OptimizerRequirement requirement
    ) noexcept
    {
        switch (requirement)
        {
        case OptimizerRequirement::MiniBatchCompatible:
        case OptimizerRequirement::DeterministicFullBatch:
            return true;

        default:
            return false;
        }
    }
}

#pragma endregion

#pragma region Custom Optimizer API

OptimizerSpec make_custom_optimizer(
    std::string name,
    const OptimizerRequirement requirement,
    OptimizerFactory factory
)
{
    OptimizerSpec spec{
        std::move(name),
        requirement,
        std::move(factory)
    };

    validate_optimizer_spec(spec);
    return spec;
}

void validate_optimizer_spec(const OptimizerSpec& optimizer)
{
    if (optimizer.name.empty())
    {
        throw std::invalid_argument(
            "Optimizer name cannot be empty."
        );
    }

    if (!is_valid_optimizer_requirement(optimizer.requirement))
    {
        throw std::invalid_argument(
            "Optimizer has an invalid requirement."
        );
    }

    if (!optimizer.make)
    {
        throw std::invalid_argument(
            "Optimizer factory cannot be empty."
        );
    }
}

#pragma endregion

#pragma region Training Configuration & Report Formatting

void validate_training_config(const TrainingConfig& config)
{
    switch (config.batch_mode) {
    case BatchMode::FullBatch:
        if (config.batch_size != 0) {
            throw std::invalid_argument(
                "Full-batch training must use batch_size == 0."
            );
        }
        break;

    case BatchMode::MiniBatch:
        if (config.batch_size == 0) {
            throw std::invalid_argument(
                "Mini-batch training requires a positive batch size."
            );
        }
        break;

    case BatchMode::Stochastic:
        if (config.batch_size > 1) {
            throw std::invalid_argument(
                "Stochastic training uses batch size one."
            );
        }
        break;

    default:
        throw std::invalid_argument(
            "Training configuration has an invalid batch mode."
        );
    }

    if (!config.max_iterations.has_value() &&
        !config.max_epochs.has_value()) {
        throw std::invalid_argument(
            "Training must specify a maximum iteration or epoch count."
        );
    }

    if (config.max_iterations.has_value() &&
        *config.max_iterations == 0) {
        throw std::invalid_argument(
            "Maximum training iterations must be positive."
        );
    }

    if (config.max_epochs.has_value() &&
        *config.max_epochs == 0) {
        throw std::invalid_argument(
            "Maximum training epochs must be positive."
        );
    }

    const auto validate_tolerance = [](
        const std::optional<double>& tolerance,
        const char* name
    ) {
        if (tolerance.has_value() &&
            (!std::isfinite(*tolerance) || *tolerance < 0.0)) {
            throw std::invalid_argument(
                std::string(name) +
                " must be non-negative and finite."
            );
        }
    };

    validate_tolerance(config.gradient_tolerance, "Gradient tolerance");
    validate_tolerance(
        config.parameter_change_tolerance,
        "Parameter-change tolerance"
    );
}

std::ostream& operator<<(std::ostream& output, const TrainingReport& report)
{
    std::string stop_reason =
        trainingStopReasonToString(report.stop_reason);

    std::transform(stop_reason.begin(), stop_reason.end(), stop_reason.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::toupper(character));
        }
    );

    output << "OPTIMIZER: " << report.optimizer_name << std::endl
           << "COMPLETED: " << (report.completed ? "TRUE" : "FALSE") << std::endl
           << "STOP REASON: " << stop_reason << std::endl 
           << "EPOCHS COMPLETED: " << report.epochs_completed << std::endl
           << "STEPS: " << report.steps << std::endl
           << "INITIAL LOSS: " << report.initial_loss << std::endl
           << "FINAL LOSS: " << report.final_loss << std::endl
           << "FINAL GRADIENT NORM: " << report.final_gradient_norm << std::endl
           << "FINAL PARAMETER CHANGE: " << report.final_parameter_change << std::endl;

    return output;
}

std::string trainingStopReasonToString(TrainingStopReason reason) {
    switch (reason) {
    case TrainingStopReason::NotStarted:
        return "Not Started";
    case TrainingStopReason::MaxIterations:
        return "Max Iterations";
    case TrainingStopReason::MaxEpochs:
        return "Max Epochs";
    case TrainingStopReason::GradientTolerance:
        return "Gradient Tolerance";
    case TrainingStopReason::ParameterChangeTolerance:
        return "Parameter Change Tolerance";
    case TrainingStopReason::OptimizerStopped:
        return "Optimizer Stopped";
    default:
        throw std::invalid_argument("TrainingStopReason Provided does not have a valid string overload");
    }
}

#pragma endregion
