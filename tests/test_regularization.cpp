#include "mlp.hpp"
#include "objective_functions.hpp"
#include "optimizers.hpp"
#include "train.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

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

MLP make_regularization_network()
{
    MLP network = make_zero_network({ 3, 1 });
    network.layer_activations = { Activations::Linear };
    network.layers[0].weights = { -2.0, 0.0, 3.0 };
    network.layers[0].biases = { 4.0 };
    return network;
}

void test_l2_contract()
{
    const MLP network = make_regularization_network();
    const RegularizationTerm l2 = make_l2_regularization(0.25, false);
    const ObjectiveConfig objective = make_objective_config(
        make_mean_squared_error_objective(),
        l2
    );

    require(!l2.proximal, "L2 must be a gradient-based regularizer");
    require(l2.smooth, "L2 must be marked smooth");
    require(!l2.includes_biases, "L2 must retain the bias policy");
    require_near(l2.value(network), 13.0, 1e-12,
        "L2 value must be the unscaled sum of squared weights");
    require_near(regularization_loss(network, objective), 3.25, 1e-12,
        "L2 coefficient must be applied exactly once to the loss");

    NetworkGradients gradients = make_zero_gradients_like(network);
    add_regularization_gradients(network, objective, gradients);
    require_near(gradients.layers[0].weights[0], -1.0, 1e-12,
        "L2 weight gradient must be coefficient * 2 * weight");
    require_near(gradients.layers[0].weights[1], 0.0, 1e-12,
        "L2 gradient at a zero weight must be zero");
    require_near(gradients.layers[0].weights[2], 1.5, 1e-12,
        "L2 weight gradient must preserve sign");
    require_near(gradients.layers[0].biases[0], 0.0, 1e-12,
        "L2 must leave biases untouched by default");

    const RegularizationTerm l2_with_biases = make_l2_regularization(0.25, true);
    const ObjectiveConfig bias_objective = make_objective_config(
        make_mean_squared_error_objective(),
        l2_with_biases
    );
    require_near(l2_with_biases.value(network), 29.0, 1e-12,
        "L2 value must include squared biases when requested");

    gradients = make_zero_gradients_like(network);
    add_regularization_gradients(network, bias_objective, gradients);
    require_near(gradients.layers[0].biases[0], 2.0, 1e-12,
        "L2 bias gradient must be coefficient * 2 * bias");
}

void test_l1_subgradient_contract()
{
    const MLP network = make_regularization_network();
    const RegularizationTerm l1 = make_l1_regularization_subgradient(0.5, false);
    const ObjectiveConfig objective = make_objective_config(
        make_mean_squared_error_objective(),
        l1
    );

    require(!l1.proximal, "subgradient L1 must not be proximal");
    require(!l1.smooth, "L1 must be marked non-smooth");
    require_near(l1.value(network), 5.0, 1e-12,
        "L1 value must be the unscaled sum of absolute weights");
    require_near(regularization_loss(network, objective), 2.5, 1e-12,
        "L1 coefficient must be applied exactly once to the loss");

    NetworkGradients gradients = make_zero_gradients_like(network);
    add_regularization_gradients(network, objective, gradients);
    require_near(gradients.layers[0].weights[0], -0.5, 1e-12,
        "L1 must use -1 as the subgradient for negative weights");
    require_near(gradients.layers[0].weights[1], 0.0, 1e-12,
        "this library's chosen L1 subgradient at zero must be zero");
    require_near(gradients.layers[0].weights[2], 0.5, 1e-12,
        "L1 must use +1 as the subgradient for positive weights");
    require_near(gradients.layers[0].biases[0], 0.0, 1e-12,
        "L1 must leave biases untouched by default");
}

void test_smooth_l1_options_contract()
{
    const MLP network = make_regularization_network();

    L1RegularizationOptions epsilon_options;
    epsilon_options.method = L1Method::EpsilonSmooth;
    epsilon_options.epsilon = 2.0;
    const RegularizationTerm epsilon_l1 = make_l1_regularization(
        0.5,
        epsilon_options
    );
    const double expected_epsilon_value =
        std::sqrt(8.0) + std::sqrt(13.0) - 4.0;
    require(epsilon_l1.smooth, "epsilon-smoothed L1 must be marked smooth");
    require_near(epsilon_l1.value(network), expected_epsilon_value, 1e-12,
        "epsilon-smoothed L1 must use sqrt(theta^2 + epsilon^2) - epsilon");

    const ObjectiveConfig epsilon_objective = make_objective_config(
        make_mean_squared_error_objective(),
        epsilon_l1
    );
    NetworkGradients gradients = make_zero_gradients_like(network);
    add_regularization_gradients(network, epsilon_objective, gradients);
    require_near(gradients.layers[0].weights[0], -1.0 / std::sqrt(8.0), 1e-12,
        "epsilon-smoothed L1 gradient must be coefficient * theta / hypot(theta, epsilon)");
    require_near(gradients.layers[0].weights[1], 0.0, 1e-12,
        "epsilon-smoothed L1 gradient must be zero at zero");
    require_near(gradients.layers[0].weights[2], 1.5 / std::sqrt(13.0), 1e-12,
        "epsilon-smoothed L1 gradient must retain its smoothing parameter");

    L1RegularizationOptions log_cosh_options;
    log_cosh_options.method = L1Method::LogCoshSmooth;
    log_cosh_options.temperature = 2.0;
    const RegularizationTerm log_cosh_l1 = make_l1_regularization(
        0.5,
        log_cosh_options
    );
    const double expected_log_cosh_value =
        2.0 * std::log(std::cosh(1.0)) +
        2.0 * std::log(std::cosh(1.5));
    require(log_cosh_l1.smooth, "log-cosh L1 must be marked smooth");
    require_near(log_cosh_l1.value(network), expected_log_cosh_value, 1e-12,
        "log-cosh L1 must use temperature * log(cosh(theta / temperature))");

    gradients = make_zero_gradients_like(network);
    const ObjectiveConfig log_cosh_objective = make_objective_config(
        make_mean_squared_error_objective(),
        log_cosh_l1
    );
    add_regularization_gradients(network, log_cosh_objective, gradients);
    require_near(gradients.layers[0].weights[0], 0.5 * std::tanh(-1.0), 1e-12,
        "log-cosh L1 gradient must be coefficient * tanh(theta / temperature)");
    require_near(gradients.layers[0].weights[2], 0.5 * std::tanh(1.5), 1e-12,
        "log-cosh L1 gradient must retain its temperature");

    L1RegularizationOptions invalid_epsilon_options;
    invalid_epsilon_options.method = L1Method::EpsilonSmooth;
    invalid_epsilon_options.epsilon = 0.0;
    require_invalid_argument(
        [&invalid_epsilon_options] {
            static_cast<void>(make_l1_regularization(0.1, invalid_epsilon_options));
        },
        "epsilon-smoothed L1 must reject a non-positive epsilon"
    );
}

void test_elastic_net_contract()
{
    const MLP network = make_regularization_network();
    const RegularizationTerm elastic_net = make_elastic_net_regularization(
        0.5,
        0.25,
        false
    );
    const ObjectiveConfig objective = make_objective_config(
        make_mean_squared_error_objective(),
        elastic_net
    );

    // Elastic net owns both strengths, so its public aggregation coefficient
    // stays at one. This preserves the regularizer callback contract.
    require_near(elastic_net.coefficient, 1.0, 1e-12,
        "elastic net must not scale its two strengths a second time");
    require(!elastic_net.proximal, "elastic net must use the L1 subgradient path");
    require(!elastic_net.smooth, "elastic net is non-smooth because it contains L1");
    require_near(elastic_net.value(network), 5.75, 1e-12,
        "elastic-net value must be l1 * sum(abs(w)) + l2 * sum(w^2)");
    require_near(regularization_loss(network, objective), 5.75, 1e-12,
        "elastic-net strengths must be applied exactly once");

    NetworkGradients gradients = make_zero_gradients_like(network);
    add_regularization_gradients(network, objective, gradients);
    require_near(gradients.layers[0].weights[0], -1.5, 1e-12,
        "elastic-net gradient must be l1 * sign(w) + 2 * l2 * w");
    require_near(gradients.layers[0].weights[1], 0.0, 1e-12,
        "elastic-net must use zero for the L1 subgradient at zero");
    require_near(gradients.layers[0].weights[2], 2.0, 1e-12,
        "elastic-net gradient must combine its L1 and L2 parts");
    require_near(gradients.layers[0].biases[0], 0.0, 1e-12,
        "elastic net must leave biases untouched by default");

    const RegularizationTerm elastic_net_with_biases =
        make_elastic_net_regularization(0.5, 0.25, true);
    const ObjectiveConfig bias_objective = make_objective_config(
        make_mean_squared_error_objective(),
        elastic_net_with_biases
    );
    require_near(elastic_net_with_biases.value(network), 11.75, 1e-12,
        "elastic-net value must include biases when requested");

    gradients = make_zero_gradients_like(network);
    add_regularization_gradients(network, bias_objective, gradients);
    require_near(gradients.layers[0].biases[0], 2.5, 1e-12,
        "elastic-net bias gradient must combine L1 and L2 parts");

    require_invalid_argument(
        [] { static_cast<void>(make_elastic_net_regularization(-0.1, 0.25)); },
        "elastic net must reject a negative L1 coefficient"
    );
    require_invalid_argument(
        [] { static_cast<void>(make_elastic_net_regularization(0.1, -0.25)); },
        "elastic net must reject a negative L2 coefficient"
    );
}

void test_proximal_elastic_net_contract()
{
    ElasticNetOptions options;
    options.l1_coefficient = 0.5;
    options.l2_coefficient = 0.25;
    options.l1.method = L1Method::Proximal;

    MLP network = make_regularization_network();
    const RegularizationTerm elastic_net = make_elastic_net_regularization(options);
    const ObjectiveConfig objective = make_objective_config(
        make_mean_squared_error_objective(),
        elastic_net
    );

    require(elastic_net.proximal,
        "elastic net must enable a proximal update when proximal L1 is selected");
    require(static_cast<bool>(elastic_net.add_gradient),
        "proximal elastic net must retain its L2 gradient callback");
    require(!elastic_net.smooth,
        "proximal elastic net remains non-smooth because it contains L1");
    require_near(regularization_loss(network, objective), 5.75, 1e-12,
        "proximal elastic-net loss must include both L1 and L2 penalties");

    NetworkGradients gradients = make_zero_gradients_like(network);
    add_regularization_gradients(network, objective, gradients);
    require_near(gradients.layers[0].weights[0], -1.0, 1e-12,
        "proximal elastic net must add only its L2 gradient before thresholding");
    require_near(gradients.layers[0].weights[1], 0.0, 1e-12,
        "proximal elastic net L2 gradient must be zero at zero");
    require_near(gradients.layers[0].weights[2], 1.5, 1e-12,
        "proximal elastic net must retain its L2 gradient magnitude");

    apply_proximal_updates(network, objective, 0.2);
    require_near(network.layers[0].weights[0], -1.9, 1e-12,
        "proximal elastic net must soft-threshold negative weights");
    require_near(network.layers[0].weights[1], 0.0, 1e-12,
        "proximal elastic net must preserve an already sparse weight");
    require_near(network.layers[0].weights[2], 2.9, 1e-12,
        "proximal elastic net must soft-threshold positive weights");
    require_near(network.layers[0].biases[0], 4.0, 1e-12,
        "proximal elastic net must honor its bias policy");

    MLP trainer_network = make_zero_network({ 1, 1 });
    trainer_network.layer_activations = { Activations::Linear };
    trainer_network.layers[0].weights[0] = 0.15;
    trainer_network.layers[0].biases[0] = 0.8;
    const Dataset batch{ { { 0.0 }, { 0.0 } } };
    TrainingConfig training_config;
    training_config.max_iterations = 1;
    training_config.max_epochs.reset();
    SGDOptions sgd_options;
    sgd_options.learning_rate_schedule = [](std::size_t) { return 0.2; };

    const TrainingReport report = train(
        trainer_network,
        batch,
        make_sgd(sgd_options),
        training_config,
        objective
    );
    require(report.completed,
        "the trainer must support a regularizer with both gradient and proximal callbacks");
    require_near(trainer_network.layers[0].weights[0], 0.035, 1e-12,
        "the trainer must apply the L2 update before the L1 proximal threshold");
    require_near(trainer_network.layers[0].biases[0], 0.48, 1e-12,
        "the trainer must not threshold excluded biases");
}

void test_smooth_elastic_net_method_selection()
{
    const MLP network = make_regularization_network();

    ElasticNetOptions epsilon_options;
    epsilon_options.l1_coefficient = 0.5;
    epsilon_options.l2_coefficient = 0.25;
    epsilon_options.l1.method = L1Method::EpsilonSmooth;
    epsilon_options.l1.epsilon = 2.0;
    const RegularizationTerm epsilon_elastic_net =
        make_elastic_net_regularization(epsilon_options);
    const double expected_epsilon_value =
        0.5 * (std::sqrt(8.0) + std::sqrt(13.0) - 4.0) + 3.25;
    require(!epsilon_elastic_net.proximal,
        "smooth elastic net must use the ordinary gradient path");
    require(epsilon_elastic_net.smooth,
        "epsilon-smooth elastic net must be marked smooth");
    require_near(epsilon_elastic_net.value(network), expected_epsilon_value, 1e-12,
        "elastic net must use the selected epsilon-smoothed L1 value");

    ElasticNetOptions log_cosh_options;
    log_cosh_options.l1_coefficient = 0.5;
    log_cosh_options.l2_coefficient = 0.25;
    log_cosh_options.l1.method = L1Method::LogCoshSmooth;
    log_cosh_options.l1.temperature = 2.0;
    const RegularizationTerm log_cosh_elastic_net =
        make_elastic_net_regularization(log_cosh_options);
    const double expected_log_cosh_value =
        0.5 * (2.0 * std::log(std::cosh(1.0)) +
               2.0 * std::log(std::cosh(1.5))) + 3.25;
    require(log_cosh_elastic_net.smooth,
        "log-cosh elastic net must be marked smooth");
    require_near(log_cosh_elastic_net.value(network), expected_log_cosh_value, 1e-12,
        "elastic net must use the selected log-cosh L1 value");
}

} // namespace

int main()
{
    try {
        test_l2_contract();
        test_l1_subgradient_contract();
        test_smooth_l1_options_contract();
        test_elastic_net_contract();
        test_proximal_elastic_net_contract();
        test_smooth_elastic_net_method_selection();
        std::cout << "[PASS] configurable L1 and elastic-net contracts\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << '\n';
        return 1;
    }
}
