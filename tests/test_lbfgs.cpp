#include "mlp.hpp"
#include "objective_functions.hpp"
#include "optimizers.hpp"
#include "train.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

void require(const bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(const double actual, const double expected, const double tolerance, const char* message)
{
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

template <typename Action>
void require_invalid_argument(Action&& action, const char* message)
{
    try {
        action();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

MLP make_linear_network()
{
    MLP network = make_zero_network({ 1, 1 });
    network.layer_activations = { Activations::Linear };
    return network;
}

LBFGSOptions reference_options(const LBFGSLineSearchPolicy policy)
{
    LBFGSOptions options;
    options.history_size = 3;
    options.curvature_tolerance = 1e-12;
    options.line_search_policy = policy;
    options.line_search.maximum_step = 1.0;
    options.line_search.maximum_trials = 12;
    return options;
}

ObjectiveFunctions make_linear_objective()
{
    ObjectiveFunctions objective;
    objective.input_domain = ObjectiveInputDomain::ActivatedOutput;
    objective.sample_loss = [](const Values& output, const Values&) {
        return output.at(0);
    };
    objective.sample_loss_gradient = [](const Values&, const Values&) {
        return Values{ 1.0 };
    };
    return objective;
}

void test_strong_wolfe_quadratic_and_reset()
{
    const Dataset batch{ { { 1.0 }, { 1.0 } } };
    const ObjectiveConfig objective = make_objective_config(
        make_mean_squared_error_objective()
    );
    MLP network = make_linear_network();
    const OptimizerSpec spec = make_lbfgs(
        reference_options(LBFGSLineSearchPolicy::StrongWolfeRequired)
    );
    require(spec.requirement == OptimizerRequirement::DeterministicFullBatch,
        "L-BFGS must require deterministic full-batch training");
    require(!spec.supports_proximal, "L-BFGS must reject proximal regularization");

    std::unique_ptr<Optimizer> optimizer = spec.make();
    optimizer->reset(network);
    OptimizerContext context{
        network,
        batch,
        objective,
        objective_loss(network, batch, objective),
        objective_gradients(network, batch, objective)
    };
    const TrainingStepResult result = optimizer->step(context);
    require(result.updated, "L-BFGS must accept a strong-Wolfe quadratic step");
    require(result.line_search.status == LineSearchStatus::StrongWolfeSatisfied,
        "L-BFGS must report an accepted strong-Wolfe step");
    require(result.new_loss < result.previous_loss,
        "L-BFGS must decrease the complete objective");

    optimizer->reset(network);
    OptimizerContext second_context{
        network,
        batch,
        objective,
        objective_loss(network, batch, objective),
        objective_gradients(network, batch, objective)
    };
    static_cast<void>(optimizer->step(second_context));
}

void test_configurable_fallback_policy()
{
    const Dataset batch{ { { 1.0 }, { 0.0 } } };
    const ObjectiveConfig objective = make_objective_config(make_linear_objective());

    MLP strict_network = make_linear_network();
    std::unique_ptr<Optimizer> strict = make_lbfgs(
        reference_options(LBFGSLineSearchPolicy::StrongWolfeRequired)
    ).make();
    strict->reset(strict_network);
    OptimizerContext strict_context{
        strict_network, batch, objective,
        objective_loss(strict_network, batch, objective),
        objective_gradients(strict_network, batch, objective)
    };
    const TrainingStepResult strict_result = strict->step(strict_context);
    require(!strict_result.updated,
        "Strict L-BFGS must refuse an Armijo-only line-search result");
    require_near(strict_network.layers[0].weights[0], 0.0, 0.0,
        "Strict L-BFGS failure must leave weights unchanged");

    MLP fallback_network = make_linear_network();
    std::unique_ptr<Optimizer> fallback = make_lbfgs(
        reference_options(LBFGSLineSearchPolicy::ArmijoFallbackWithoutHistory)
    ).make();
    fallback->reset(fallback_network);
    OptimizerContext fallback_context{
        fallback_network, batch, objective,
        objective_loss(fallback_network, batch, objective),
        objective_gradients(fallback_network, batch, objective)
    };
    const TrainingStepResult fallback_result = fallback->step(fallback_context);
    require(fallback_result.updated,
        "Configured L-BFGS fallback must accept Armijo sufficient decrease");
    require(fallback_result.line_search.status == LineSearchStatus::SufficientDecreaseFallback,
        "Fallback L-BFGS must report Armijo-only acceptance");
    require(fallback_network.layers[0].weights[0] < 0.0,
        "Armijo fallback must take the downhill L-BFGS direction");
}

void test_validation_and_training_rejections()
{
    LBFGSOptions invalid = reference_options(LBFGSLineSearchPolicy::StrongWolfeRequired);
    invalid.history_size = 0;
    require_invalid_argument(
        [&invalid] { static_cast<void>(make_lbfgs(invalid)); },
        "L-BFGS must require a positive history size"
    );
    invalid = reference_options(LBFGSLineSearchPolicy::StrongWolfeRequired);
    invalid.curvature_tolerance = 0.0;
    require_invalid_argument(
        [&invalid] { static_cast<void>(make_lbfgs(invalid)); },
        "L-BFGS must require a positive curvature tolerance"
    );

    MLP network = make_linear_network();
    const Dataset batch{ { { 1.0 }, { 1.0 } } };
    TrainingConfig mini_batch;
    mini_batch.batch_mode = BatchMode::MiniBatch;
    mini_batch.batch_size = 1;
    mini_batch.max_iterations = 1;
    mini_batch.max_epochs.reset();
    require_invalid_argument(
        [&] { static_cast<void>(train(network, batch, make_lbfgs(), mini_batch, make_objective_config())); },
        "L-BFGS must reject mini-batch training"
    );

    TrainingConfig full_batch;
    full_batch.max_iterations = 1;
    full_batch.max_epochs.reset();
    const ObjectiveConfig proximal = make_objective_config(
        make_mean_squared_error_objective(),
        make_l1_regularization_Proximal(0.1)
    );
    require_invalid_argument(
        [&] { static_cast<void>(train(network, batch, make_lbfgs(), full_batch, proximal)); },
        "L-BFGS must reject proximal objectives before updating"
    );
}

void test_deterministic_comparison_with_newton_cg()
{
    const Dataset batch{
        { { 0.0 }, { 0.0 } }, { { 1.0 }, { 1.0 } }, { { 2.0 }, { 2.0 } }
    };
    const ObjectiveConfig objective = make_objective_config(
        make_mean_squared_error_objective()
    );
    MLP newton_network = make_linear_network();
    MLP lbfgs_network = newton_network;
    TrainingConfig config;
    config.max_iterations = 4;
    config.max_epochs.reset();
    const double initial_loss = objective_loss(newton_network, batch, objective);

    NewtonCGOptions newton_options;
    newton_options.damping = 0.1;
    newton_options.maximum_cg_iterations = 8;
    newton_options.line_search.maximum_trials = 12;
    const TrainingReport newton_report = train(
        newton_network, batch, make_newton_cg(newton_options), config, objective
    );
    const TrainingReport lbfgs_report = train(
        lbfgs_network,
        batch,
        make_lbfgs(reference_options(LBFGSLineSearchPolicy::ArmijoFallbackWithoutHistory)),
        config,
        objective
    );
    require(std::isfinite(newton_report.final_loss) && std::isfinite(lbfgs_report.final_loss),
        "Newton-CG and L-BFGS comparison runs must remain finite");
    require(newton_report.final_loss < initial_loss && lbfgs_report.final_loss < initial_loss,
        "Newton-CG and L-BFGS must both reduce the shared complete objective");
}

} // namespace

int main()
{
    try {
        test_strong_wolfe_quadratic_and_reset();
        test_configurable_fallback_policy();
        test_validation_and_training_rejections();
        test_deterministic_comparison_with_newton_cg();
        std::cout << "[PASS] L-BFGS line-search policies, validation, and Newton-CG comparison\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << '\n';
        return 1;
    }
}
