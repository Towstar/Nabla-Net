#include "mlp.hpp"
#include "objective_functions.hpp"
#include "optimizers.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(double actual, double expected, double tolerance, const char* message)
{
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}
}

int main()
{
    try {
        MLP network = make_zero_network({ 1, 1 });
        network.layer_activations = { Activations::Linear };
        network.layers[0].weights[0] = 2.0;
        network.layers[0].biases[0] = 3.0;
        const Dataset batch{ { { 0.0 }, { 0.0 } } };
        const ObjectiveConfig objective = make_objective_config();

        AdamWOptions options;
        options.learning_rate_schedule = [](std::size_t) { return 0.1; };
        options.beta1 = 0.9;
        options.beta2 = 0.999;
        options.epsilon = 1e-8;
        options.weight_decay = 0.1;
        const OptimizerSpec spec = make_adamw(options);
        require(spec.supports_proximal, "AdamW should advertise proximal support");
        std::unique_ptr<Optimizer> optimizer = spec.make();
        optimizer->reset(network);

        NetworkGradients gradient = make_zero_gradients_like(network);
        gradient.layers[0].weights[0] = 0.5;
        gradient.layers[0].biases[0] = 0.25;
        OptimizerContext context{
            network,
            batch,
            objective,
            objective_loss(network, batch, objective),
            gradient
        };
        const TrainingStepResult result = optimizer->step(context);
        const double normalized_weight_direction =
            0.5 / (std::sqrt(0.25) + options.epsilon);
        const double normalized_bias_direction =
            0.25 / (std::sqrt(0.0625) + options.epsilon);
        const double expected_weight =
            (2.0 - 0.1 * normalized_weight_direction) * (1.0 - 0.1 * 0.1);
        const double expected_bias = 3.0 - 0.1 * normalized_bias_direction;

        require(result.updated, "AdamW must update successfully");
        require(result.effective_step_size.has_value(), "AdamW must report its step size");
        require_near(network.layers[0].weights[0], expected_weight, 1e-10,
            "AdamW must combine bias-corrected moments with decoupled weight decay");
        require_near(network.layers[0].biases[0], expected_bias, 1e-10,
            "AdamW must not decay biases by default");

        options.decay_biases = true;
        MLP bias_decay_network = make_zero_network({ 1, 1 });
        bias_decay_network.layer_activations = { Activations::Linear };
        bias_decay_network.layers[0].weights[0] = 2.0;
        bias_decay_network.layers[0].biases[0] = 3.0;
        std::unique_ptr<Optimizer> bias_decay_optimizer = make_adamw(options).make();
        bias_decay_optimizer->reset(bias_decay_network);
        OptimizerContext bias_decay_context{
            bias_decay_network,
            batch,
            objective,
            objective_loss(bias_decay_network, batch, objective),
            gradient
        };
        static_cast<void>(bias_decay_optimizer->step(bias_decay_context));
        require_near(
            bias_decay_network.layers[0].biases[0],
            expected_bias * (1.0 - 0.1 * 0.1),
            1e-10,
            "AdamW must optionally decay biases"
        );

        std::cout << "[PASS] AdamW reference update and decay policy\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << '\n';
        return 1;
    }
}
