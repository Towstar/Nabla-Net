#include <nablanet/mlp.hpp>
#include <nablanet/objective_functions.hpp>
#include <nablanet/optimizers.hpp>
#include <nablanet/train.hpp>

#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace nablanet;

namespace {

void require(const bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(
    const double actual,
    const double expected,
    const double tolerance,
    const char* message
)
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

MLP make_linear_network(const double weight, const double bias)
{
    MLP network = make_zero_network({ 1, 1 });
    network.layer_activations = { Activations::Linear };
    network.layers[0].weights[0] = weight;
    network.layers[0].biases[0] = bias;
    return network;
}

ObjectiveFunctions make_counted_squared_objective(std::size_t& hvp_calls)
{
    ObjectiveFunctions objective;
    objective.input_domain = ObjectiveInputDomain::ActivatedOutput;
    objective.sample_loss = [](const Values& outputs, const Values& targets) {
        const double difference = outputs.at(0) - targets.at(0);
        return 0.5 * difference * difference;
    };
    objective.sample_loss_gradient = [](const Values& outputs, const Values& targets) {
        return Values{ outputs.at(0) - targets.at(0) };
    };
    objective.sample_loss_hessian_vector_product = [&hvp_calls](
        const Values& outputs,
        const Values& targets,
        const Values& tangent
    ) {
        static_cast<void>(outputs);
        static_cast<void>(targets);
        ++hvp_calls;
        return tangent;
    };
    return objective;
}

NewtonCGOptions reference_options()
{
    NewtonCGOptions options;
    options.damping = 1.0;
    options.maximum_cg_iterations = 8;
    options.absolute_residual_tolerance = 1e-12;
    options.relative_residual_tolerance = 1e-12;
    options.line_search.maximum_step = 1.0;
    options.line_search.maximum_trials = 12;
    return options;
}

void test_exact_damped_step_and_hvp_use()
{
    std::size_t hvp_calls = 0;
    const ObjectiveConfig objective = make_objective_config(
        make_counted_squared_objective(hvp_calls)
    );
    const Dataset batch{ { { 1.0 }, { 1.0 } } };
    MLP network = make_linear_network(0.0, 0.0);
    const OptimizerSpec spec = make_newton_cg(reference_options());
    require(spec.requirement == OptimizerRequirement::DeterministicFullBatch,
        "Newton-CG must require deterministic full-batch training");
    require(!spec.supports_proximal,
        "Newton-CG must reject proximal regularization");

    std::unique_ptr<Optimizer> optimizer = spec.make();
    optimizer->reset(network);
    const NetworkGradients gradient = objective_gradients(network, batch, objective);
    OptimizerContext context{
        network,
        batch,
        objective,
        objective_loss(network, batch, objective),
        gradient
    };
    const TrainingStepResult result = optimizer->step(context);

    // For L = 1/2(w + b - 1)^2 and damping lambda = 1,
    // (H + lambda I)p = -g gives p = (1/3, 1/3).
    require(result.updated, "Damped Newton-CG must accept the quadratic step");
    require(hvp_calls > 0, "Newton-CG must obtain curvature through the HVP callback");
    require_near(network.layers[0].weights[0], 1.0 / 3.0, 1e-10,
        "Newton-CG must solve the damped quadratic weight step");
    require_near(network.layers[0].biases[0], 1.0 / 3.0, 1e-10,
        "Newton-CG must solve the damped quadratic bias step");
}

void test_negative_curvature_and_transactional_failure()
{
    ObjectiveFunctions concave;
    concave.input_domain = ObjectiveInputDomain::ActivatedOutput;
    concave.sample_loss = [](const Values& outputs, const Values&) {
        return -0.5 * outputs.at(0) * outputs.at(0);
    };
    concave.sample_loss_gradient = [](const Values& outputs, const Values&) {
        return Values{ -outputs.at(0) };
    };
    concave.sample_loss_hessian_vector_product = [](const Values&, const Values&, const Values& tangent) {
        return Values{ -tangent.at(0) };
    };
    const ObjectiveConfig concave_objective = make_objective_config(concave);
    const Dataset batch{ { { 1.0 }, { 0.0 } } };
    MLP network = make_linear_network(1.0, 0.0);
    std::unique_ptr<Optimizer> optimizer = make_newton_cg(reference_options()).make();
    optimizer->reset(network);
    OptimizerContext context{
        network,
        batch,
        concave_objective,
        objective_loss(network, batch, concave_objective),
        objective_gradients(network, batch, concave_objective)
    };
    const TrainingStepResult result = optimizer->step(context);
    require(result.updated, "Newton-CG must fall back to a downhill direction on negative curvature");
    require(network.layers[0].weights[0] > 1.0,
        "Negative-curvature fallback must take the negative-gradient direction");

    ObjectiveFunctions inconsistent;
    inconsistent.input_domain = ObjectiveInputDomain::ActivatedOutput;
    inconsistent.sample_loss = [](const Values&, const Values&) { return 0.0; };
    inconsistent.sample_loss_gradient = [](const Values&, const Values&) { return Values{ 1.0 }; };
    inconsistent.sample_loss_hessian_vector_product = [](const Values&, const Values&, const Values&) {
        return Values{ 0.0 };
    };
    const ObjectiveConfig failed_objective = make_objective_config(inconsistent);
    MLP unchanged = make_linear_network(0.0, 0.0);
    optimizer->reset(unchanged);
    OptimizerContext failed_context{
        unchanged,
        batch,
        failed_objective,
        objective_loss(unchanged, batch, failed_objective),
        objective_gradients(unchanged, batch, failed_objective)
    };
    const TrainingStepResult failed = optimizer->step(failed_context);
    require(!failed.updated, "Newton-CG must report a failed Armijo search");
    require_near(unchanged.layers[0].weights[0], 0.0, 0.0,
        "Failed Newton-CG search must not mutate weights");
    require_near(unchanged.layers[0].biases[0], 0.0, 0.0,
        "Failed Newton-CG search must not mutate biases");
}

void test_rejections_and_determinism()
{
    NewtonCGOptions invalid = reference_options();
    invalid.damping = 0.0;
    require_invalid_argument(
        [&invalid] { static_cast<void>(make_newton_cg(invalid)); },
        "Newton-CG must reject non-positive damping"
    );
    invalid = reference_options();
    invalid.maximum_cg_iterations = 0;
    require_invalid_argument(
        [&invalid] { static_cast<void>(make_newton_cg(invalid)); },
        "Newton-CG must require a positive CG iteration limit"
    );

    ObjectiveFunctions first_order_only = make_mean_squared_error_objective();
    first_order_only.sample_loss_hessian_vector_product = {};
    const ObjectiveConfig no_hvp = make_objective_config(first_order_only);
    const Dataset regression{ { { 1.0 }, { 1.0 } } };
    MLP network = make_linear_network(0.0, 0.0);
    const MLP before = network;
    std::unique_ptr<Optimizer> optimizer = make_newton_cg(reference_options()).make();
    optimizer->reset(network);
    OptimizerContext context{
        network,
        regression,
        no_hvp,
        objective_loss(network, regression, no_hvp),
        objective_gradients(network, regression, no_hvp)
    };
    require_invalid_argument(
        [&] { static_cast<void>(optimizer->step(context)); },
        "Newton-CG must reject objectives without an HVP callback"
    );
    require_near(network.layers[0].weights[0], before.layers[0].weights[0], 0.0,
        "HVP rejection must not mutate the network");

    MLP first = make_mlp({ 2, 3, 1 }, 7);
    MLP second = first;
    const Dataset xor_data{
        { { 0.0, 0.0 }, { 0.0 } }, { { 0.0, 1.0 }, { 1.0 } },
        { { 1.0, 0.0 }, { 1.0 } }, { { 1.0, 1.0 }, { 0.0 } }
    };
    const ObjectiveConfig objective = make_objective_config();
    const double initial_loss = objective_loss(first, xor_data, objective);
    TrainingConfig config;
    config.max_iterations = 20;
    config.max_epochs.reset();
    NewtonCGOptions options = reference_options();
    options.damping = 0.1;
    options.maximum_cg_iterations = 20;
    const TrainingReport first_report = train(first, xor_data, make_newton_cg(options), config, objective);
    const TrainingReport second_report = train(second, xor_data, make_newton_cg(options), config, objective);
    require(first_report.final_loss < initial_loss,
        "Newton-CG must reduce deterministic full-batch XOR loss");
    require_near(first_report.final_loss, second_report.final_loss, 1e-12,
        "Newton-CG runs with identical inputs must be deterministic");
}

} // namespace

int main()
{
    try {
        test_exact_damped_step_and_hvp_use();
        test_negative_curvature_and_transactional_failure();
        test_rejections_and_determinism();
        std::cout << "[PASS] Newton-CG HVP, curvature fallback, transactions, and determinism\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << '\n';
        return 1;
    }
}
