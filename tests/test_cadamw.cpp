#include <nncpp/mlp.hpp>
#include <nncpp/objective_functions.hpp>
#include <nncpp/optimizers.hpp>

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
        MLP network = make_zero_network({ 2, 1 });
        network.layer_activations = { Activations::Linear };
        network.layers[0].weights = { 0.5, 0.5 };
        network.layers[0].biases[0] = 0.5;
        const Dataset batch{ { { 0.0, 0.0 }, { 0.0 } } };
        const ObjectiveConfig objective = make_objective_config();

        CAdamWOptions options;
        options.learning_rate_schedule = [](std::size_t) { return 0.1; };
        options.beta1 = 0.9;
        options.beta2 = 0.999;
        options.epsilon = 1e-8;
        options.weight_decay = 0.0;
        options.cautious_epsilon = 1.0;
        const OptimizerSpec spec = make_cadamw(options);
        require(spec.supports_proximal, "CAdamW should advertise proximal support");
        std::unique_ptr<Optimizer> optimizer = spec.make();
        optimizer->reset(network);

        NetworkGradients first_gradient = make_zero_gradients_like(network);
        first_gradient.layers[0].weights = { 1.0, 1.0 };
        first_gradient.layers[0].biases[0] = 1.0;
        OptimizerContext first_context{
            network,
            batch,
            objective,
            objective_loss(network, batch, objective),
            first_gradient
        };
        static_cast<void>(optimizer->step(first_context));

        const MLP after_first_step = network;
        NetworkGradients second_gradient = make_zero_gradients_like(network);
        second_gradient.layers[0].weights = { -0.5, 1.0 };
        second_gradient.layers[0].biases[0] = 0.0;
        OptimizerContext second_context{
            network,
            batch,
            objective,
            objective_loss(network, batch, objective),
            second_gradient
        };
        const TrainingStepResult second_result = optimizer->step(second_context);
        require(second_result.updated, "CAdamW must update with a partially active mask");

        // On step two the first coordinate keeps a positive moment direction
        // while its gradient is negative, so it is masked.  The second
        // coordinate is the sole active coordinate and receives 3/(1+1)=1.5
        // times the ordinary Adam direction.
        const double active_direction = 1.0;
        require_near(
            network.layers[0].weights[0],
            after_first_step.layers[0].weights[0],
            1e-12,
            "CAdamW must mask a contrary weight coordinate"
        );
        require_near(
            network.layers[0].weights[1],
            after_first_step.layers[0].weights[1] - 0.1 * 1.5 * active_direction,
            1e-8,
            "CAdamW must apply global active-coordinate scaling"
        );
        require_near(
            network.layers[0].biases[0],
            after_first_step.layers[0].biases[0],
            1e-12,
            "CAdamW must treat direction*gradient == 0 as masked"
        );

        NetworkGradients all_masked_gradient = make_zero_gradients_like(network);
        all_masked_gradient.layers[0].weights = { -0.5, -0.5 };
        all_masked_gradient.layers[0].biases[0] = -0.5;
        OptimizerContext all_masked_context{
            network,
            batch,
            objective,
            objective_loss(network, batch, objective),
            all_masked_gradient
        };
        const TrainingStepResult all_masked_result = optimizer->step(all_masked_context);
        require(all_masked_result.updated, "CAdamW must remain finite when all coordinates are masked");
        for (const DenseLayer& layer : network.layers) {
            for (const double value : layer.weights) {
                require(std::isfinite(value), "CAdamW all-masked weights must remain finite");
            }
            for (const double value : layer.biases) {
                require(std::isfinite(value), "CAdamW all-masked biases must remain finite");
            }
        }

        CAdamWOptions invalid = options;
        invalid.cautious_epsilon = 0.0;
        bool rejected = false;
        try {
            static_cast<void>(make_cadamw(invalid));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "CAdamW must validate cautious_epsilon");

        std::cout << "[PASS] CAdamW mask, scaling, finite all-masked path, and validation\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << '\n';
        return 1;
    }
}
