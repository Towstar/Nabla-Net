#include "mlp.hpp"
#include "objective_functions.hpp"
#include "optimizers.hpp"
#include "train.hpp"

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
        network.layers[0].weights[0] = 0.15;
        network.layers[0].biases[0] = 0.8;
        const RegularizationTerm proximal =
            make_l1_regularization(0.5, false, L1Method::Proximal);
        const ObjectiveConfig objective = make_objective_config(
            make_mean_squared_error_objective(),
            proximal
        );
        require(proximal.proximal, "proximal L1 must be marked proximal");
        require(!proximal.add_gradient, "proximal L1 must not add a subgradient");
        require_near(proximal.value(network), 0.15, 1e-12,
            "proximal L1 value callback must be unscaled");
        require_near(regularization_loss(network, objective), 0.075, 1e-12,
            "proximal L1 coefficient must be applied exactly once");

        apply_proximal_updates(network, objective, 0.2);
        require_near(network.layers[0].weights[0], 0.05, 1e-12,
            "proximal L1 must soft-threshold weights");
        require_near(network.layers[0].biases[0], 0.8, 1e-12,
            "proximal L1 must honor the bias inclusion policy");

        MLP trainer_network = make_zero_network({ 1, 1 });
        trainer_network.layer_activations = { Activations::Linear };
        trainer_network.layers[0].weights[0] = 0.15;
        trainer_network.layers[0].biases[0] = 0.8;
        const Dataset batch{ { { 0.0 }, { 0.0 } } };
        TrainingConfig config;
        config.max_iterations = 1;
        config.max_epochs.reset();
        SGDOptions sgd_options;
        sgd_options.learning_rate_schedule = [](std::size_t) { return 0.2; };
        const TrainingReport report = train(
            trainer_network,
            batch,
            make_sgd(sgd_options),
            config,
            objective
        );
        require(report.completed, "trainer must complete a proximal step");
        require_near(trainer_network.layers[0].weights[0], 0.05, 1e-12,
            "trainer must apply proximal thresholding after the optimizer step");
        require_near(trainer_network.layers[0].biases[0], 0.48, 1e-12,
            "trainer must apply the data gradient before proximal thresholding");

        std::cout << "[PASS] proximal L1 value, thresholding, and trainer contract\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << '\n';
        return 1;
    }
}
