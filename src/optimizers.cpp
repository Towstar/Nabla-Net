#include "../include/nncpp/optimizers.hpp"

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
            result.effective_step_size = learning_rate;

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
            result.effective_step_size = learning_rate;

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
            result.effective_step_size = learning_rate;

            return result;
        }
    };

    class RMSPropOptimizer final : public Optimizer
    {
    public:
        RMSPropOptimizer(RMSPropOptions options) {
            if (!options.learning_rate_schedule) {
                throw std::invalid_argument("Learning Rate Schedule must be defined.");
            }
            if (options.epsilon < 0.0 || !std::isfinite(options.epsilon)) {
                throw std::invalid_argument("Epsilon must be positive and finite");
            }
            if (!std::isfinite(options.decay) || options.decay <= 0 || options.decay >= 1) {
                throw std::invalid_argument("Decay must be in the range (0, 1).");
            }
            epsilon = options.epsilon;
            decay = options.decay;
            schedule = options.learning_rate_schedule;
        }
        const char* name() const noexcept override {
            return "RMSProp";
        }
        void reset(const MLP& network) override {
            squared_gradient_exponential_moving_average =
                make_zero_gradients_like(network);

            update_index = 0;
        }
        TrainingStepResult step(OptimizerContext& context) override {
            validate_network(context.network);
            validate_objective_config(context.objective);
            validate_gradients_like(context.network, context.current_gradient);

            if (context.batch.empty()) {
                throw std::invalid_argument(
                    "RMSProp batch cannot be empty."
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
        std::size_t update_index{};
        NetworkGradients squared_gradient_exponential_moving_average;
        double decay;
        double epsilon;
        LearningRateSchedule schedule;

        TrainingStepResult _step(OptimizerContext& context) {
            const NetworkGradients& gradient = context.current_gradient;
            MLP candidate = context.network;
            NetworkGradients next_average = squared_gradient_exponential_moving_average;
            const double learning_rate = schedule(update_index);

            if (!learning_rate_is_valid(learning_rate)) {
                throw std::invalid_argument(
                    "RMSProp schedule produced an invalid learning rate."
                );
            }

            for (std::size_t layer_index = 0; layer_index < gradient.layers.size(); layer_index++) {
                const LayerGradients& grad_layer = gradient.layers[layer_index];
                LayerGradients& average_layer = next_average.layers[layer_index];
                DenseLayer& candidate_layer = candidate.layers[layer_index];

                for (std::size_t i = 0; i < grad_layer.weights.size(); i++) {
                    const double grad = grad_layer.weights[i];
                    const double current_average = decay * average_layer.weights[i] + (1.0 - decay) * (grad * grad);

                    if (!std::isfinite(current_average)) {
                        throw std::overflow_error(
                            "RMSProp squared-gradient average overflowed."
                        );
                    }

                    const double update = learning_rate * grad / (std::sqrt(current_average) + epsilon);

                    if (!std::isfinite(update)) {
                        throw std::overflow_error(
                            "RMSProp update overflowed."
                        );
                    }

                    const double new_weight = candidate_layer.weights[i] - update;

                    average_layer.weights[i] = current_average;
                    candidate_layer.weights[i] = new_weight;
                }
                for (std::size_t i = 0; i < grad_layer.biases.size(); i++) {
                    const double grad = grad_layer.biases[i];
                    const double current_average = decay * average_layer.biases[i] + (1.0 - decay) * (grad * grad);
                    const double update = learning_rate * grad / (std::sqrt(current_average) + epsilon);
                    const double new_bias = candidate_layer.biases[i] - update;

                    if (!std::isfinite(current_average)) {
                        throw std::overflow_error(
                            "RMSProp squared-gradient average overflowed."
                        );
                    }

                    average_layer.biases[i] = current_average;
                    candidate_layer.biases[i] = new_bias;
                }
            }
            validate_network(candidate);

            const double new_loss = objective_loss(candidate, context.batch, context.objective);
            if (!std::isfinite(new_loss)) {
                throw std::runtime_error("RMSProp produced a non-finite loss.");
            }
            context.network = std::move(candidate);
            squared_gradient_exponential_moving_average = std::move(next_average);
            update_index++;
            TrainingStepResult result{};
            result.updated = true;
            result.previous_loss = context.current_loss;
            result.new_loss = new_loss;
            result.gradient_norm =
                gradient_l2_norm(context.current_gradient);
            result.effective_step_size = learning_rate;

            return result;
        }
    };

    /// <summary>
    /// Designed with the intentions of combining the strengths of AdaGrad and RMSProp, Adam only requires first order gradients.
    /// </summary>
    class AdamOptimizer final : public Optimizer
    {
    public:
        AdamOptimizer(AdamOptions options) {
            if (!std::isfinite(options.beta1) || options.beta1 >= 1.0 || options.beta1 <= 0.0) {
                throw std::invalid_argument("beta1 must be finite and in the range (0,1).");
            }
            if (!std::isfinite(options.beta2) || options.beta2 >= 1.0 || options.beta2 <= 0.0) {
                throw std::invalid_argument("beta2 must be finite and in the range (0,1).");
            }
            if (!std::isfinite(options.epsilon) || options.epsilon <= 0.0) {
                throw std::invalid_argument("Epsilon must be positive and finite.");
            }
            if (!options.learning_rate_schedule) {
                throw std::invalid_argument("Learning rate schedule must be defined.");
            }
            beta1 = options.beta1;
            beta2 = options.beta2;
            epsilon = options.epsilon;
            schedule = options.learning_rate_schedule;
        }
        const char* name() const noexcept override {
            return "Adam";
        }
        void reset(const MLP& network) override {
            validate_network(network);
            first_moment = make_zero_gradients_like(network);
            second_moment = make_zero_gradients_like(network);
            update_index = 0;

        }
        TrainingStepResult step(OptimizerContext& context) override {
            validate_network(context.network);
            validate_objective_config(context.objective);
            validate_gradients_like(context.network, first_moment);
            validate_gradients_like(context.network, second_moment);
            validate_gradients_like(context.network, context.current_gradient);
            if (context.batch.empty())
                throw std::invalid_argument("Batch size must be a positive integer.");
            if (!std::isfinite(context.current_loss))
                throw std::invalid_argument("Current loss must be finite.");
            return _step(context);
        }
    private:
        std::size_t update_index{};

        // Roughly E[g_t]
        NetworkGradients first_moment;

        // Roughly E[(g_t)^2]
        NetworkGradients second_moment;

        double beta1;
        double beta2;
        double epsilon;
        LearningRateSchedule schedule;
        TrainingStepResult _step(OptimizerContext& context) {
            size_t t = update_index + 1;
            double learning_rate = schedule(update_index);

            if (!std::isfinite(learning_rate) || learning_rate <= 0.0)
                throw std::invalid_argument("Invalid learning rate. Learning rate must be positive and finite.");

            // remove the pull towards zero on start
            double bias1_correction = 1 - std::pow(beta1, t);
            double bias2_correction = 1 - std::pow(beta2, t);

            const auto& grad = context.current_gradient;
            MLP candidate = context.network;

            auto next_first_moment = first_moment;
            auto next_second_moment = second_moment;

            for (std::size_t layer_index = 0; layer_index < grad.layers.size(); layer_index++) {
                const LayerGradients& grad_layer = grad.layers[layer_index];
                DenseLayer& candidate_layer = candidate.layers[layer_index];

                for (std::size_t i = 0; i < grad_layer.weights.size(); i++) {
                    const double weight_grad_i = grad_layer.weights[i];
                    next_first_moment.layers[layer_index].weights[i] = beta1 * first_moment.layers[layer_index].weights[i] + (1 - beta1) * weight_grad_i;
                    next_second_moment.layers[layer_index].weights[i] = beta2 * second_moment.layers[layer_index].weights[i] + (1 - beta2) * (weight_grad_i * weight_grad_i);
                    double bias_corrected_first_moment = next_first_moment.layers[layer_index].weights[i] / bias1_correction;
                    double bias_corrected_second_moment = next_second_moment.layers[layer_index].weights[i] / bias2_correction;
                    double update = learning_rate * (bias_corrected_first_moment) / (std::sqrt(bias_corrected_second_moment) + epsilon);
                    candidate_layer.weights[i] -= update;
                }
                for (std::size_t i = 0; i < grad_layer.biases.size(); i++) {
                    const double bias_grad_i = grad_layer.biases[i];
                    next_first_moment.layers[layer_index].biases[i] = beta1 * first_moment.layers[layer_index].biases[i] + (1 - beta1) * bias_grad_i;
                    next_second_moment.layers[layer_index].biases[i] = beta2 * second_moment.layers[layer_index].biases[i] + (1 - beta2) * (bias_grad_i * bias_grad_i);
                    double bias_corrected_first_moment = next_first_moment.layers[layer_index].biases[i] / bias1_correction;
                    double bias_corrected_second_moment = next_second_moment.layers[layer_index].biases[i] / bias2_correction;
                    double update = learning_rate * (bias_corrected_first_moment) / (std::sqrt(bias_corrected_second_moment) + epsilon);
                    candidate_layer.biases[i] -= update;
                }
            }
            validate_network(candidate);
            const double new_loss = objective_loss(candidate, context.batch, context.objective);
            if (!std::isfinite(new_loss)) {
                throw std::runtime_error("Adam produced a non-finite loss.");
            }
            validate_gradients_like(context.network, next_first_moment);
            validate_gradients_like(context.network, next_second_moment);
            context.network = std::move(candidate);
            first_moment = next_first_moment;
            second_moment = next_second_moment;
            update_index++;
            TrainingStepResult result{};
            result.updated = true;
            result.previous_loss = context.current_loss;
            result.new_loss = new_loss;
            result.gradient_norm =
                gradient_l2_norm(context.current_gradient);
            result.effective_step_size = learning_rate;

            return result;
        }
    };

    class AdamWOptimizer final : public Optimizer
    {
    public:
        explicit AdamWOptimizer(AdamWOptions options) :
            beta1(options.beta1),
            beta2(options.beta2),
            epsilon(options.epsilon),
            weight_decay(options.weight_decay),
            decay_biases(options.decay_biases),
            schedule(std::move(options.learning_rate_schedule)) {
            if (!schedule)
                throw std::invalid_argument("AdamW learning-rate schedule cannot be empty.");
            if (!std::isfinite(beta1) ||
                beta1 <= 0.0 ||
                beta1 >= 1.0) {
                throw std::invalid_argument("AdamW beta1 must be in (0,1).");
            }
            if (!std::isfinite(beta2) || beta2 <= 0.0 || beta2 >= 1.0) {
                throw std::invalid_argument("AdamW beta2 must be in (0,1).");
            }
            if (!std::isfinite(epsilon) || epsilon <= 0.0) {
                throw std::invalid_argument(
                    "AdamW epsilon must be positive and finite."
                );
            }
            if (!std::isfinite(weight_decay) || weight_decay < 0.0) {
                throw std::invalid_argument("AdamW weight decay must be finite and non-negative.");
            }
        }
        const char* name() const noexcept override {
            return "AdamW";
        }
        void reset(const MLP& network) override {
            validate_network(network);
            first_moment = make_zero_gradients_like(network);
            second_moment = make_zero_gradients_like(network);
            update_index = 0;
        }
        TrainingStepResult step(OptimizerContext& context) override {
            validate_network(context.network);
            validate_objective_config(context.objective);
            validate_gradients_like(context.network, first_moment);
            validate_gradients_like(context.network, second_moment);
            validate_gradients_like(context.network, context.current_gradient);
            if (context.batch.empty())
                throw std::invalid_argument("Batch size must be a positive integer.");
            if (!std::isfinite(context.current_loss))
                throw std::invalid_argument("Current loss must be finite.");
            return _step(context);
        }
    private:
        std::size_t update_index{};

        // Roughly E[g_t]
        NetworkGradients first_moment;

        // Roughly E[(g_t)^2]
        NetworkGradients second_moment;

        double beta1;
        double beta2;
        double epsilon;
        double weight_decay;
        bool decay_biases;
        LearningRateSchedule schedule;

        TrainingStepResult _step(OptimizerContext& context) {
            size_t t = update_index + 1;
            double learning_rate = schedule(update_index);

            if (!std::isfinite(learning_rate) || learning_rate <= 0.0)
                throw std::invalid_argument("Invalid learning rate. Learning rate must be positive and finite.");

            // remove the pull towards zero on start
            double bias1_correction = 1 - std::pow(beta1, t);
            double bias2_correction = 1 - std::pow(beta2, t);

            const auto& grad = context.current_gradient;
            MLP candidate = context.network;

            auto next_first_moment = first_moment;
            auto next_second_moment = second_moment;

            for (std::size_t layer_index = 0; layer_index < grad.layers.size(); layer_index++) {
                const LayerGradients& grad_layer = grad.layers[layer_index];
                DenseLayer& candidate_layer = candidate.layers[layer_index];

                for (std::size_t i = 0; i < grad_layer.weights.size(); i++) {
                    const double weight_grad_i = grad_layer.weights[i];
                    next_first_moment.layers[layer_index].weights[i] = beta1 * first_moment.layers[layer_index].weights[i] + (1 - beta1) * weight_grad_i;
                    next_second_moment.layers[layer_index].weights[i] = beta2 * second_moment.layers[layer_index].weights[i] + (1 - beta2) * (weight_grad_i * weight_grad_i);
                    double bias_corrected_first_moment = next_first_moment.layers[layer_index].weights[i] / bias1_correction;
                    double bias_corrected_second_moment = next_second_moment.layers[layer_index].weights[i] / bias2_correction;
                    const double adam_update = learning_rate * bias_corrected_first_moment / (std::sqrt(bias_corrected_second_moment) + epsilon);
                    candidate_layer.weights[i] -= adam_update;
                    candidate_layer.weights[i] -=
                        learning_rate * weight_decay * candidate_layer.weights[i];
                }
                for (std::size_t i = 0; i < grad_layer.biases.size(); i++) {
                    const double bias_grad_i = grad_layer.biases[i];
                    next_first_moment.layers[layer_index].biases[i] = beta1 * first_moment.layers[layer_index].biases[i] + (1 - beta1) * bias_grad_i;
                    next_second_moment.layers[layer_index].biases[i] = beta2 * second_moment.layers[layer_index].biases[i] + (1 - beta2) * (bias_grad_i * bias_grad_i);
                    double bias_corrected_first_moment = next_first_moment.layers[layer_index].biases[i] / bias1_correction;
                    double bias_corrected_second_moment = next_second_moment.layers[layer_index].biases[i] / bias2_correction;
                    const double adam_update = learning_rate * bias_corrected_first_moment / (std::sqrt(bias_corrected_second_moment) + epsilon);
                    candidate_layer.biases[i] -= adam_update;
                    if (decay_biases) {
                        candidate_layer.biases[i] -=
                            learning_rate * weight_decay * candidate_layer.biases[i];
                    }
                }
            }
            validate_network(candidate);
            const double new_loss = objective_loss(candidate, context.batch, context.objective);
            if (!std::isfinite(new_loss)) {
                throw std::runtime_error("AdamW produced a non-finite loss.");
            }
            validate_gradients_like(context.network, next_first_moment);
            validate_gradients_like(context.network, next_second_moment);
            context.network = std::move(candidate);
            first_moment = next_first_moment;
            second_moment = next_second_moment;
            update_index++;
            TrainingStepResult result{};
            result.updated = true;
            result.previous_loss = context.current_loss;
            result.new_loss = new_loss;
            result.gradient_norm =
                gradient_l2_norm(context.current_gradient);
            result.effective_step_size = learning_rate;

            return result;
        }
    };

    // Cautious AdamW
    class CAdamWOptimizer final : public Optimizer
    {
    public:
        explicit CAdamWOptimizer(CAdamWOptions options) :
            beta1(options.beta1),
            beta2(options.beta2),
            epsilon(options.epsilon),
            weight_decay(options.weight_decay),
            decay_biases(options.decay_biases),
            cautious_epsilon(options.cautious_epsilon),
            schedule(std::move(options.learning_rate_schedule)) {
            if (!schedule)
                throw std::invalid_argument("CAdamW learning-rate schedule cannot be empty.");
            if (!std::isfinite(beta1) ||
                beta1 <= 0.0 ||
                beta1 >= 1.0) {
                throw std::invalid_argument("CAdamW beta1 must be in (0,1).");
            }
            if (!std::isfinite(beta2) || beta2 <= 0.0 || beta2 >= 1.0) {
                throw std::invalid_argument("CAdamW beta2 must be in (0,1).");
            }
            if (!std::isfinite(epsilon) || epsilon <= 0.0) {
                throw std::invalid_argument(
                    "CAdamW epsilon must be positive and finite."
                );
            }
            if (!std::isfinite(weight_decay) || weight_decay < 0.0) {
                throw std::invalid_argument("CAdamW weight decay must be finite and non-negative.");
            }
            if (!std::isfinite(cautious_epsilon) || cautious_epsilon <= 0.0) {
                throw std::invalid_argument("CAdamW cautious epsilon must be positive and finite.");
            }
        }
        const char* name() const noexcept override {
            return "CAdamW";
        }
        void reset(const MLP& network) override {
            validate_network(network);
            first_moment = make_zero_gradients_like(network);
            second_moment = make_zero_gradients_like(network);
            update_index = 0;
        }
        TrainingStepResult step(OptimizerContext& context) override {
            validate_network(context.network);
            validate_objective_config(context.objective);
            validate_gradients_like(context.network, first_moment);
            validate_gradients_like(context.network, second_moment);
            validate_gradients_like(context.network, context.current_gradient);
            if (context.batch.empty())
                throw std::invalid_argument("Batch size must be a positive integer.");
            if (!std::isfinite(context.current_loss))
                throw std::invalid_argument("Current loss must be finite.");
            return _step(context);
        }
    private:
        std::size_t update_index{};

        // Roughly E[g_t]
        NetworkGradients first_moment;

        // Roughly E[(g_t)^2]
        NetworkGradients second_moment;

        double beta1;
        double beta2;
        double epsilon;
        double weight_decay;
        bool decay_biases;
        double cautious_epsilon;
        LearningRateSchedule schedule;

        TrainingStepResult _step(OptimizerContext& context) {
            size_t t = update_index + 1;
            double learning_rate = schedule(update_index);

            if (!std::isfinite(learning_rate) || learning_rate <= 0.0)
                throw std::invalid_argument("Invalid learning rate. Learning rate must be positive and finite.");

            // remove the pull towards zero on start
            double bias1_correction = 1 - std::pow(beta1, t);
            double bias2_correction = 1 - std::pow(beta2, t);

            const auto& grad = context.current_gradient;
            MLP candidate = context.network;

            auto next_first_moment = first_moment;
            auto next_second_moment = second_moment;

            for (std::size_t layer_index = 0; layer_index < grad.layers.size(); ++layer_index) {
                const LayerGradients& grad_layer = grad.layers[layer_index];
                LayerGradients& next_layer = next_first_moment.layers[layer_index];
                LayerGradients& next_square_layer = next_second_moment.layers[layer_index];
                const LayerGradients& old_layer = first_moment.layers[layer_index];
                const LayerGradients& old_square_layer = second_moment.layers[layer_index];

                for (std::size_t i = 0; i < grad_layer.weights.size(); ++i) {
                    const double g = grad_layer.weights[i];
                    next_layer.weights[i] = beta1 * old_layer.weights[i] + (1.0 - beta1) * g;
                    next_square_layer.weights[i] = beta2 * old_square_layer.weights[i] + (1.0 - beta2) * g * g;
                }
                for (std::size_t i = 0; i < grad_layer.biases.size(); ++i) {
                    const double g = grad_layer.biases[i];
                    next_layer.biases[i] = beta1 * old_layer.biases[i] + (1.0 - beta1) * g;
                    next_square_layer.biases[i] = beta2 * old_square_layer.biases[i] + (1.0 - beta2) * g * g;
                }
            }

            std::size_t total_coordinates = 0;
            std::size_t active_coordinates = 0;
            for (std::size_t layer_index = 0; layer_index < grad.layers.size(); ++layer_index) {
                const LayerGradients& grad_layer = grad.layers[layer_index];
                const LayerGradients& next_layer = next_first_moment.layers[layer_index];
                const LayerGradients& next_square_layer = next_second_moment.layers[layer_index];
                total_coordinates += grad_layer.weights.size() + grad_layer.biases.size();

                for (std::size_t i = 0; i < grad_layer.weights.size(); ++i) {
                    const double direction =
                        (next_layer.weights[i] / bias1_correction) /
                        (std::sqrt(next_square_layer.weights[i] / bias2_correction) + epsilon);
                    if (direction * grad_layer.weights[i] > 0.0) {
                        ++active_coordinates;
                    }
                }
                for (std::size_t i = 0; i < grad_layer.biases.size(); ++i) {
                    const double direction =
                        (next_layer.biases[i] / bias1_correction) /
                        (std::sqrt(next_square_layer.biases[i] / bias2_correction) + epsilon);
                    if (direction * grad_layer.biases[i] > 0.0) {
                        ++active_coordinates;
                    }
                }
            }

            const double mask_scale =
                static_cast<double>(total_coordinates) /
                (static_cast<double>(active_coordinates) + cautious_epsilon);
            if (!std::isfinite(mask_scale)) {
                throw std::overflow_error("CAdamW mask scale is not finite.");
            }

            for (std::size_t layer_index = 0; layer_index < grad.layers.size(); ++layer_index) {
                const LayerGradients& grad_layer = grad.layers[layer_index];
                const LayerGradients& next_layer = next_first_moment.layers[layer_index];
                const LayerGradients& next_square_layer = next_second_moment.layers[layer_index];
                DenseLayer& candidate_layer = candidate.layers[layer_index];

                for (std::size_t i = 0; i < grad_layer.weights.size(); ++i) {
                    const double direction =
                        (next_layer.weights[i] / bias1_correction) /
                        (std::sqrt(next_square_layer.weights[i] / bias2_correction) + epsilon);
                    if (direction * grad_layer.weights[i] > 0.0) {
                        candidate_layer.weights[i] -= learning_rate * mask_scale * direction;
                    }
                    candidate_layer.weights[i] -=
                        learning_rate * weight_decay * candidate_layer.weights[i];
                }
                for (std::size_t i = 0; i < grad_layer.biases.size(); ++i) {
                    const double direction =
                        (next_layer.biases[i] / bias1_correction) /
                        (std::sqrt(next_square_layer.biases[i] / bias2_correction) + epsilon);
                    if (direction * grad_layer.biases[i] > 0.0) {
                        candidate_layer.biases[i] -= learning_rate * mask_scale * direction;
                    }
                    if (decay_biases) {
                        candidate_layer.biases[i] -=
                            learning_rate * weight_decay * candidate_layer.biases[i];
                    }
                }
            }
            validate_network(candidate);
            const double new_loss = objective_loss(candidate, context.batch, context.objective);
            if (!std::isfinite(new_loss)) {
                throw std::runtime_error("AdamW produced a non-finite loss.");
            }
            validate_gradients_like(context.network, next_first_moment);
            validate_gradients_like(context.network, next_second_moment);
            context.network = std::move(candidate);
            first_moment = next_first_moment;
            second_moment = next_second_moment;
            update_index++;
            TrainingStepResult result{};
            result.updated = true;
            result.previous_loss = context.current_loss;
            result.new_loss = new_loss;
            result.gradient_norm =
                gradient_l2_norm(context.current_gradient);
            result.effective_step_size = learning_rate;

            return result;
        }
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

    std::unique_ptr<Optimizer> make_rmsprop_instance(RMSPropOptions options) {
        return std::make_unique<RMSPropOptimizer>(std::move(options));
    }
    std::unique_ptr<Optimizer> make_adam_instance(AdamOptions options) {
        return std::make_unique<AdamOptimizer>(std::move(options));
    }
    std::unique_ptr<Optimizer> make_adamw_instance(AdamWOptions options) {
        return std::make_unique<AdamWOptimizer>(std::move(options));
    }
    std::unique_ptr<Optimizer> make_cadamw_instance(CAdamWOptions options) {
        return std::make_unique<CAdamWOptimizer>(std::move(options));
    }
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
        },
        true
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
        },
        true
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
    return make_custom_optimizer("AdaGrad", OptimizerRequirement::MiniBatchCompatible, [schedule, epsilon]() {return make_adagrad_instance(AdaGradOptions{ schedule, epsilon }); }, true);
}

OptimizerSpec make_rmsprop(RMSPropOptions options) {
    if (!options.learning_rate_schedule) {
        throw std::invalid_argument(
            "RMSProp learning-rate schedule cannot be empty."
        );
    }
    if (!options.epsilon) {
        throw std::invalid_argument(
            "RMPSProp epsilon must be defined."
        );
    }
    if (!std::isfinite(options.epsilon) || options.epsilon < 0.0) {
        throw std::invalid_argument(
            "epsilon must be finite and positive."
        );
    }
    if (!std::isfinite(options.decay) || options.decay <= 0 || options.decay >= 1) {
        throw std::invalid_argument("Decay must be in the range (0, 1).");
    }
    const LearningRateSchedule schedule = options.learning_rate_schedule;
    const double epsilon = options.epsilon;
    const double decay = options.decay;
    return make_custom_optimizer("RMSProp", OptimizerRequirement::MiniBatchCompatible, [schedule, epsilon, decay]() {return make_rmsprop_instance(RMSPropOptions{ schedule, decay, epsilon }); }, true);
}

OptimizerSpec make_adam(AdamOptions options) {
    if (!options.learning_rate_schedule)
        throw std::invalid_argument("Adam's learning-rate schedule cannot be empty");
    if (!std::isfinite(options.beta1) || options.beta1 <= 0.0 || options.beta1 >= 1.0)
        throw std::invalid_argument("Adam's Beta 1 must be in the range (0,1).");
    if (!std::isfinite(options.beta2) || options.beta2 <= 0.0 || options.beta2 >= 1.0)
        throw std::invalid_argument("Adam's Beta 2 must be in the range (0,1).");
    if (!std::isfinite(options.epsilon))
        throw std::invalid_argument("Adam's epsilon must be finite.");
    if (options.epsilon <= 0.0)
        throw std::invalid_argument("Adam's epsilon must be greater than 0.");
    const LearningRateSchedule schedule = options.learning_rate_schedule;
    double epsilon = options.epsilon;
    double beta2 = options.beta2;
    double beta1 = options.beta1;
    return make_custom_optimizer("Adam", OptimizerRequirement::MiniBatchCompatible,
        [schedule, epsilon, beta2, beta1]() {return make_adam_instance(AdamOptions{ schedule, beta1, beta2, epsilon }); },
        true
    );
}

OptimizerSpec make_adamw(AdamWOptions options) {
    if (!options.learning_rate_schedule)
        throw std::invalid_argument("AdamW's learning-rate schedule cannot be empty");
    if (!std::isfinite(options.beta1) || options.beta1 <= 0.0 || options.beta1 >= 1.0)
        throw std::invalid_argument("AdamW's Beta 1 must be in the range (0,1).");
    if (!std::isfinite(options.beta2) || options.beta2 <= 0.0 || options.beta2 >= 1.0)
        throw std::invalid_argument("AdamW's Beta 2 must be in the range (0,1).");
    if (!std::isfinite(options.epsilon))
        throw std::invalid_argument("AdamW's epsilon must be finite.");
    if (options.epsilon <= 0.0)
        throw std::invalid_argument("AdamW's epsilon must be greater than 0.");
    if (!std::isfinite(options.weight_decay) || options.weight_decay < 0.0)
        throw std::invalid_argument("AdamW's weight decay must be finite and non-negative.");
    const AdamWOptions configured = options;
    return make_custom_optimizer("AdamW", OptimizerRequirement::MiniBatchCompatible,
        [configured]() { return make_adamw_instance(configured); }, true);
}

OptimizerSpec make_cadamw(CAdamWOptions options) {
    if (!options.learning_rate_schedule)
        throw std::invalid_argument("CAdamW's learning-rate schedule cannot be empty.");
    if (!std::isfinite(options.beta1) || options.beta1 <= 0.0 || options.beta1 >= 1.0)
        throw std::invalid_argument("CAdamW's beta1 must be in the range (0,1).");
    if (!std::isfinite(options.beta2) || options.beta2 <= 0.0 || options.beta2 >= 1.0)
        throw std::invalid_argument("CAdamW's beta2 must be in the range (0,1).");
    if (!std::isfinite(options.epsilon) || options.epsilon <= 0.0)
        throw std::invalid_argument("CAdamW's epsilon must be positive and finite.");
    if (!std::isfinite(options.weight_decay) || options.weight_decay < 0.0)
        throw std::invalid_argument("CAdamW's weight decay must be finite and non-negative.");
    if (!std::isfinite(options.cautious_epsilon) || options.cautious_epsilon <= 0.0)
        throw std::invalid_argument("CAdamW's cautious epsilon must be positive and finite.");
    const CAdamWOptions configured = options;
    return make_custom_optimizer(
        "CAdamW",
        OptimizerRequirement::MiniBatchCompatible,
        [configured]() { return make_cadamw_instance(configured); },
        true
    );
}

OptimizerSpec make_lbfgs(LBFGSOptions options) {
    static_cast<void>(options);
    return OptimizerSpec{
        "LBFGS",
        OptimizerRequirement::DeterministicFullBatch,
        {}
    };
}
#pragma endregion

#pragma region Built-in Optimizer Specifications

namespace Optimizers {
    const OptimizerSpec SGD = make_sgd();

    const OptimizerSpec MomentumSGD = make_momentum_sgd();

    const OptimizerSpec AdaGrad = make_adagrad();

    const OptimizerSpec RMSProp = make_rmsprop();

    const OptimizerSpec Adam = make_adam();

    const OptimizerSpec AdamW = make_adamw();

    const OptimizerSpec CAdamW = make_cadamw();

    const OptimizerSpec LBFGS = make_lbfgs();
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
    OptimizerFactory factory,
    const bool supports_proximal
)
{
    OptimizerSpec spec{
        std::move(name),
        requirement,
        std::move(factory),
        supports_proximal
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
