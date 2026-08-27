#include <nablanet/mlp.hpp>
#include <nablanet/objective_functions.hpp>
#include <nablanet/optimizers.hpp>
#include <nablanet/train.hpp>

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
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

void require_values_near(
    const Values& actual,
    const Values& expected,
    const double tolerance,
    const char* message
)
{
    require(actual.size() == expected.size(),
        "Objective vectors must have matching sizes.");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require_near(actual[index], expected[index], tolerance, message);
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

template <typename Action>
void require_overflow_error(Action&& action, const char* message)
{
    try {
        action();
    } catch (const std::overflow_error&) {
        return;
    }
    throw std::runtime_error(message);
}

Values central_difference_gradient(
    const ObjectiveFunctions& objective,
    const Values& logits,
    const Values& targets
)
{
    constexpr double step = 1e-6;
    Values numerical_gradient(logits.size(), 0.0);

    for (std::size_t index = 0; index < logits.size(); ++index) {
        Values plus = logits;
        Values minus = logits;
        plus[index] += step;
        minus[index] -= step;
        numerical_gradient[index] =
            (objective.sample_loss(plus, targets) -
             objective.sample_loss(minus, targets)) /
            (2.0 * step);
    }

    return numerical_gradient;
}

void test_exponential_objective_contract()
{
    const ObjectiveFunctions objective = make_exponential_objective();
    validate_objective_functions(objective);

    require(objective.input_domain == ObjectiveInputDomain::PreActivation,
        "Exponential loss must consume logits.");
    require(static_cast<bool>(objective.sample_loss_hessian_vector_product),
        "Exponential loss must provide an exact loss HVP callback.");

    const Values logits{ 0.5, -0.25 };
    const Values targets{ 1.0, -1.0 };
    const Values tangent{ 0.4, -0.6 };
    const double first_term = std::exp(-0.5);
    const double second_term = std::exp(-0.25);

    // L = (exp(-y_0 z_0) + exp(-y_1 z_1)) / 2.
    require_near(
        objective.sample_loss(logits, targets),
        0.5 * (first_term + second_term),
        1e-12,
        "Exponential loss must average exp(-y * z) across output coordinates"
    );

    // dL/dz_i = -y_i exp(-y_i z_i) / output_width.
    require_values_near(
        objective.sample_loss_gradient(logits, targets),
        Values{ -0.5 * first_term, 0.5 * second_term },
        1e-12,
        "Exponential loss gradient must match -y * exp(-y * z)"
    );
    require_values_near(
        objective.sample_loss_gradient(logits, targets),
        central_difference_gradient(objective, logits, targets),
        1e-7,
        "Exponential loss gradient must match a finite difference of its loss"
    );

    // H_L v has coordinates y_i^2 exp(-y_i z_i) v_i / output_width.
    require_values_near(
        objective.sample_loss_hessian_vector_product(logits, targets, tangent),
        Values{ 0.5 * first_term * tangent[0],
                0.5 * second_term * tangent[1] },
        1e-12,
        "Exponential loss HVP must match the diagonal exact curvature"
    );

    require_invalid_argument(
        [&] { static_cast<void>(objective.sample_loss({ 0.0 }, { 0.0 })); },
        "Exponential loss must reject targets outside {-1,+1}"
    );
    require_invalid_argument(
        [&] { static_cast<void>(objective.sample_loss({ 0.0 }, { 1.0, -1.0 })); },
        "Exponential loss must reject mismatched logits and targets"
    );
    require_invalid_argument(
        [&] {
            static_cast<void>(objective.sample_loss_hessian_vector_product(
                logits, targets,
                Values{ std::numeric_limits<double>::quiet_NaN(), 0.1 }
            ));
        },
        "Exponential loss HVP must reject non-finite tangents"
    );
    require_overflow_error(
        [&] { static_cast<void>(objective.sample_loss({ -1000.0 }, { 1.0 })); },
        "Exponential loss must report values beyond double precision as overflow"
    );
}

void test_hinge_objective_contract()
{
    const ObjectiveFunctions objective = make_hinge_objective();
    validate_objective_functions(objective);

    require(objective.input_domain == ObjectiveInputDomain::PreActivation,
        "Hinge loss must consume logits.");
    require(!objective.sample_loss_hessian_vector_product,
        "Non-smooth hinge loss must not advertise an HVP callback.");

    const Values logits{ 0.25, 2.0, 0.5, -1.0 };
    const Values targets{ 1.0, 1.0, -1.0, -1.0 };

    // L = sum_i max(0, 1 - y_i z_i) / 4. The last coordinate is exactly
    // on the kink; this contract chooses its deterministic subgradient as 0.
    require_near(
        objective.sample_loss(logits, targets),
        2.25 / 4.0,
        1e-12,
        "Hinge loss must average max(0, 1 - y * z) across output coordinates"
    );
    require_values_near(
        objective.sample_loss_gradient(logits, targets),
        Values{ -0.25, 0.0, 0.25, 0.0 },
        1e-12,
        "Hinge loss must use -y / output_width inside the margin and zero otherwise"
    );

    require_invalid_argument(
        [&] { static_cast<void>(objective.sample_loss({ 0.0 }, { 0.5 })); },
        "Hinge loss must reject targets outside {-1,+1}"
    );
    require_invalid_argument(
        [&] { static_cast<void>(objective.sample_loss({}, {})); },
        "Hinge loss must reject empty inputs"
    );

    MLP network = make_zero_network({ 1, 1 });
    network.layer_activations = { Activations::Linear };
    const Dataset batch{ { { 1.0 }, { 1.0 } } };
    NetworkTangent tangent = make_zero_gradients_like(network);
    tangent.layers[0].weights[0] = 1.0;
    const ObjectiveConfig configuration = make_objective_config(objective);
    require_invalid_argument(
        [&] {
            static_cast<void>(objective_hessian_vector_product(
                network, batch, tangent, configuration
            ));
        },
        "Complete-objective HVP must reject non-smooth hinge loss"
    );
}

void test_binary_objectives_lower_training_loss()
{
    const Dataset dataset{
        { { -1.0 }, { -1.0 } },
        { { 1.0 }, { 1.0 } }
    };

    SGDOptions sgd_options;
    sgd_options.learning_rate_schedule = [](std::size_t) { return 0.1; };
    TrainingConfig training;
    training.max_iterations = 20;
    training.max_epochs.reset();

    for (const ObjectiveFunctions& objective : {
            make_exponential_objective(), make_hinge_objective() }) {
        MLP network = make_zero_network({ 1, 1 });
        network.layer_activations = { Activations::Linear };
        const ObjectiveConfig configuration = make_objective_config(objective);
        const double initial_loss = objective_loss(network, dataset, configuration);

        const TrainingReport report = train(
            network, dataset, make_sgd(sgd_options), training, configuration
        );

        require(report.steps == 20,
            "Binary-objective training must run the configured deterministic steps");
        require(report.final_loss < initial_loss,
            "A linear separable dataset must lower binary-objective training loss");
    }
}

} // namespace

int main()
{
    try {
        test_exponential_objective_contract();
        test_hinge_objective_contract();
        test_binary_objectives_lower_training_loss();
        std::cout << "[PASS] exponential and hinge objective contracts\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << '\n';
        return 1;
    }
}
