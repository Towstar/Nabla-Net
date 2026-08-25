#include "mlp.hpp"
#include "objective_functions.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

constexpr double finite_difference_step = 1e-5;

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
    const std::string& message
)
{
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void require_values_near(
    const Values& actual,
    const Values& expected,
    const double tolerance,
    const std::string& message
)
{
    require(actual.size() == expected.size(),
        (message + ": vector sizes must match").c_str());

    for (std::size_t index = 0; index < actual.size(); ++index) {
        require_near(
            actual[index],
            expected[index],
            tolerance,
            message + " at index " + std::to_string(index)
        );
    }
}

void require_gradients_near(
    const NetworkGradients& actual,
    const NetworkGradients& expected,
    const double tolerance,
    const std::string& message
)
{
    require(actual.layers.size() == expected.layers.size(),
        (message + ": layer counts must match").c_str());

    for (std::size_t layer_index = 0;
         layer_index < actual.layers.size();
         ++layer_index) {
        require_values_near(
            actual.layers[layer_index].weights,
            expected.layers[layer_index].weights,
            tolerance,
            message + " weights in layer " + std::to_string(layer_index)
        );
        require_values_near(
            actual.layers[layer_index].biases,
            expected.layers[layer_index].biases,
            tolerance,
            message + " biases in layer " + std::to_string(layer_index)
        );
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

Values central_difference_activation_forward(
    const ActivationFunction& activation,
    const Values& pre_activations,
    const Values& pre_activation_tangent
)
{
    Values plus(pre_activations);
    Values minus(pre_activations);
    for (std::size_t index = 0; index < pre_activations.size(); ++index) {
        plus[index] += finite_difference_step * pre_activation_tangent[index];
        minus[index] -= finite_difference_step * pre_activation_tangent[index];
    }

    const Values plus_result = activation.forward(plus);
    const Values minus_result = activation.forward(minus);
    Values result(pre_activations.size(), 0.0);
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] =
            (plus_result[index] - minus_result[index]) /
            (2.0 * finite_difference_step);
    }
    return result;
}

Values central_difference_activation_pullback(
    const ActivationFunction& activation,
    const Values& pre_activations,
    const Values& pre_activation_tangent,
    const Values& upstream_gradient,
    const Values& upstream_gradient_tangent
)
{
    Values plus_pre_activations(pre_activations);
    Values minus_pre_activations(pre_activations);
    Values plus_upstream(upstream_gradient);
    Values minus_upstream(upstream_gradient);

    for (std::size_t index = 0; index < pre_activations.size(); ++index) {
        plus_pre_activations[index] +=
            finite_difference_step * pre_activation_tangent[index];
        minus_pre_activations[index] -=
            finite_difference_step * pre_activation_tangent[index];
        plus_upstream[index] +=
            finite_difference_step * upstream_gradient_tangent[index];
        minus_upstream[index] -=
            finite_difference_step * upstream_gradient_tangent[index];
    }

    const Values plus_result =
        activation.backward(plus_pre_activations, plus_upstream);
    const Values minus_result =
        activation.backward(minus_pre_activations, minus_upstream);
    Values result(pre_activations.size(), 0.0);
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] =
            (plus_result[index] - minus_result[index]) /
            (2.0 * finite_difference_step);
    }
    return result;
}

MLP make_autodiff_network(const ActivationFunction& output_activation)
{
    MLP network = make_zero_network({ 2, 3, 2 });
    network.layer_activations = { Activations::GELU, output_activation };

    network.layers[0].weights = {
        0.23, -0.31,
        -0.17, 0.29,
        0.37, 0.11
    };
    network.layers[0].biases = { 0.07, -0.13, 0.19 };
    network.layers[1].weights = {
        0.41, -0.22, 0.16,
        -0.28, 0.35, 0.09
    };
    network.layers[1].biases = { -0.05, 0.12 };
    validate_network(network);
    return network;
}

NetworkTangent make_autodiff_tangent(const MLP& network)
{
    NetworkTangent tangent = make_zero_gradients_like(network);
    tangent.layers[0].weights = {
        0.11, -0.07,
        0.05, 0.13,
        -0.09, 0.04
    };
    tangent.layers[0].biases = { -0.08, 0.06, 0.02 };
    tangent.layers[1].weights = {
        -0.03, 0.12, -0.10,
        0.08, -0.04, 0.15
    };
    tangent.layers[1].biases = { 0.05, -0.11 };
    validate_gradients_like(network, tangent);
    return tangent;
}

MLP perturb_network(
    const MLP& network,
    const NetworkTangent& tangent,
    const double scale
)
{
    validate_gradients_like(network, tangent);
    MLP result = network;
    for (std::size_t layer_index = 0;
         layer_index < result.layers.size();
         ++layer_index) {
        for (std::size_t index = 0;
             index < result.layers[layer_index].weights.size();
             ++index) {
            result.layers[layer_index].weights[index] +=
                scale * tangent.layers[layer_index].weights[index];
        }
        for (std::size_t index = 0;
             index < result.layers[layer_index].biases.size();
             ++index) {
            result.layers[layer_index].biases[index] +=
                scale * tangent.layers[layer_index].biases[index];
        }
    }
    validate_network(result);
    return result;
}

NetworkGradients central_difference_sample_gradient(
    const MLP& network,
    const NetworkTangent& tangent,
    const Values& input,
    const Values& target,
    const ObjectiveFunctions& objective
)
{
    const MLP plus_network = perturb_network(
        network, tangent, finite_difference_step
    );
    const MLP minus_network = perturb_network(
        network, tangent, -finite_difference_step
    );
    const NetworkGradients plus = backward(
        plus_network,
        forward_pass(plus_network, input),
        target,
        objective
    );
    const NetworkGradients minus = backward(
        minus_network,
        forward_pass(minus_network, input),
        target,
        objective
    );

    NetworkGradients result = make_zero_gradients_like(network);
    for (std::size_t layer_index = 0;
         layer_index < result.layers.size();
         ++layer_index) {
        for (std::size_t index = 0;
             index < result.layers[layer_index].weights.size();
             ++index) {
            result.layers[layer_index].weights[index] =
                (plus.layers[layer_index].weights[index] -
                 minus.layers[layer_index].weights[index]) /
                (2.0 * finite_difference_step);
        }
        for (std::size_t index = 0;
             index < result.layers[layer_index].biases.size();
             ++index) {
            result.layers[layer_index].biases[index] =
                (plus.layers[layer_index].biases[index] -
                 minus.layers[layer_index].biases[index]) /
                (2.0 * finite_difference_step);
        }
    }
    return result;
}

NetworkGradients central_difference_objective_gradient(
    const MLP& network,
    const NetworkTangent& tangent,
    const Dataset& batch,
    const ObjectiveConfig& objective
)
{
    const MLP plus_network = perturb_network(
        network, tangent, finite_difference_step
    );
    const MLP minus_network = perturb_network(
        network, tangent, -finite_difference_step
    );
    const NetworkGradients plus =
        objective_gradients(plus_network, batch, objective);
    const NetworkGradients minus =
        objective_gradients(minus_network, batch, objective);

    NetworkGradients result = make_zero_gradients_like(network);
    for (std::size_t layer_index = 0;
         layer_index < result.layers.size();
         ++layer_index) {
        for (std::size_t index = 0;
             index < result.layers[layer_index].weights.size();
             ++index) {
            result.layers[layer_index].weights[index] =
                (plus.layers[layer_index].weights[index] -
                 minus.layers[layer_index].weights[index]) /
                (2.0 * finite_difference_step);
        }
        for (std::size_t index = 0;
             index < result.layers[layer_index].biases.size();
             ++index) {
            result.layers[layer_index].biases[index] =
                (plus.layers[layer_index].biases[index] -
                 minus.layers[layer_index].biases[index]) /
                (2.0 * finite_difference_step);
        }
    }
    return result;
}

void test_activation_jvps()
{
    const Values pre_activations{ -0.8, 0.0, 0.6 };
    const Values pre_activation_tangent{ 0.3, -0.4, 0.2 };
    const Values upstream_gradient{ -0.7, 0.5, 0.9 };
    const Values upstream_gradient_tangent{ 0.11, -0.25, 0.08 };

    require(
        static_cast<bool>(Activations::GELU.jvp),
        "GELU must supply an activation JVP callback"
    );
    require(
        static_cast<bool>(Activations::GELU.backward_jvp),
        "GELU must supply a differentiated-pullback callback"
    );

    require_values_near(
        Activations::GELU.jvp(pre_activations, pre_activation_tangent),
        central_difference_activation_forward(
            Activations::GELU, pre_activations, pre_activation_tangent
        ),
        2e-6,
        "GELU JVP must match a central difference of its forward operation"
    );
    require_values_near(
        Activations::GELU.backward_jvp(
            pre_activations,
            pre_activation_tangent,
            upstream_gradient,
            upstream_gradient_tangent
        ),
        central_difference_activation_pullback(
            Activations::GELU,
            pre_activations,
            pre_activation_tangent,
            upstream_gradient,
            upstream_gradient_tangent
        ),
        3e-6,
        "GELU differentiated pullback must match a central difference"
    );

    const ActivationFunction cubic = make_elementwise_activation(
        "cubic",
        [](const double z, ActivationParameter) { return z * z * z; },
        [](const double z, ActivationParameter) { return 3.0 * z * z; },
        [](const double z, ActivationParameter) { return 6.0 * z; }
    );
    require_values_near(
        cubic.jvp(pre_activations, pre_activation_tangent),
        central_difference_activation_forward(
            cubic, pre_activations, pre_activation_tangent
        ),
        1e-6,
        "make_elementwise_activation must build its JVP from the scalar derivative"
    );
    require_values_near(
        cubic.backward_jvp(
            pre_activations,
            pre_activation_tangent,
            upstream_gradient,
            upstream_gradient_tangent
        ),
        central_difference_activation_pullback(
            cubic,
            pre_activations,
            pre_activation_tangent,
            upstream_gradient,
            upstream_gradient_tangent
        ),
        1e-6,
        "make_elementwise_activation must build the differentiated pullback"
    );
}

void test_objective_loss_hvps()
{
    const Values logits{ 0.3, -0.7 };
    const Values target{ 0.8, 0.1 };
    const Values tangent{ 0.4, -0.6 };

    const ObjectiveFunctions bce = make_binary_cross_entropy_objective();
    require(static_cast<bool>(bce.sample_loss_hessian_vector_product),
        "BCE must supply a loss Hessian-vector-product callback");
    const Values bce_actual = bce.sample_loss_hessian_vector_product(
        logits, target, tangent
    );
    Values bce_expected(logits.size(), 0.0);
    for (std::size_t index = 0; index < logits.size(); ++index) {
        const double sigmoid = stable_sigmoid(logits[index]);
        bce_expected[index] =
            sigmoid * (1.0 - sigmoid) * tangent[index] /
            static_cast<double>(logits.size());
    }
    require_values_near(bce_actual, bce_expected, 1e-12,
        "BCE loss HVP must equal sigmoid'(z) times the logit tangent");

    const ObjectiveFunctions softmax = make_softmax_cross_entropy_objective();
    require(static_cast<bool>(softmax.sample_loss_hessian_vector_product),
        "softmax cross-entropy must supply a loss Hessian-vector-product callback");
    const Values softmax_actual = softmax.sample_loss_hessian_vector_product(
        logits, Values{ 1.0, 0.0 }, tangent
    );
    const Values probabilities = stable_softmax(logits);
    const double mean_tangent =
        probabilities[0] * tangent[0] + probabilities[1] * tangent[1];
    const Values softmax_expected{
        probabilities[0] * (tangent[0] - mean_tangent),
        probabilities[1] * (tangent[1] - mean_tangent)
    };
    require_values_near(softmax_actual, softmax_expected, 1e-12,
        "softmax cross-entropy loss HVP must use the softmax Jacobian");

    const ObjectiveFunctions mse = make_mean_squared_error_objective();
    require(static_cast<bool>(mse.sample_loss_hessian_vector_product),
        "activated-output MSE must supply a loss Hessian-vector-product callback");
    const Values mse_actual = mse.sample_loss_hessian_vector_product(
        logits, target, tangent
    );
    const Values mse_expected{ tangent[0], tangent[1] };
    require_values_near(mse_actual, mse_expected, 1e-12,
        "MSE loss HVP must equal 2/output_width times the output tangent");
}

void test_network_forward_jvp()
{
    const MLP network = make_autodiff_network(Activations::Linear);
    const Values input{ 0.35, -0.42 };
    const NetworkTangent parameter_tangent = make_autodiff_tangent(network);
    const ForwardCache primal_cache = forward_pass(network, input);
    const ForwardTangent tangent = forward_jvp(
        network, primal_cache, parameter_tangent
    );

    const MLP plus_network = perturb_network(
        network, parameter_tangent, finite_difference_step
    );
    const MLP minus_network = perturb_network(
        network, parameter_tangent, -finite_difference_step
    );
    const ForwardCache plus_cache = forward_pass(plus_network, input);
    const ForwardCache minus_cache = forward_pass(minus_network, input);

    require(tangent.activations.size() == primal_cache.activations.size(),
        "ForwardTangent activations must mirror ForwardCache");
    require(tangent.pre_activations.size() == primal_cache.pre_activations.size(),
        "ForwardTangent pre-activations must mirror ForwardCache");

    for (std::size_t layer_index = 0;
         layer_index < tangent.activations.size();
         ++layer_index) {
        Values numerical(tangent.activations[layer_index].size(), 0.0);
        for (std::size_t index = 0; index < numerical.size(); ++index) {
            numerical[index] =
                (plus_cache.activations[layer_index][index] -
                 minus_cache.activations[layer_index][index]) /
                (2.0 * finite_difference_step);
        }
        require_values_near(
            tangent.activations[layer_index],
            numerical,
            3e-6,
            "Forward JVP activation tangent must match a perturbed forward pass"
        );
    }

    for (std::size_t layer_index = 0;
         layer_index < tangent.pre_activations.size();
         ++layer_index) {
        Values numerical(tangent.pre_activations[layer_index].size(), 0.0);
        for (std::size_t index = 0; index < numerical.size(); ++index) {
            numerical[index] =
                (plus_cache.pre_activations[layer_index][index] -
                 minus_cache.pre_activations[layer_index][index]) /
                (2.0 * finite_difference_step);
        }
        require_values_near(
            tangent.pre_activations[layer_index],
            numerical,
            3e-6,
            "Forward JVP pre-activation tangent must match a perturbed forward pass"
        );
    }
}

void test_sample_loss_hvps()
{
    const Values input{ 0.35, -0.42 };

    const auto check_hvp = [&input](
        const MLP& network,
        const Values& target,
        const ObjectiveFunctions& objective,
        const char* name
    ) {
        const NetworkTangent tangent = make_autodiff_tangent(network);
        const ForwardCache cache = forward_pass(network, input);
        const NetworkGradients actual = loss_hessian_vector_product(
            network, cache, target, tangent, objective
        );
        const NetworkGradients numerical = central_difference_sample_gradient(
            network, tangent, input, target, objective
        );
        require_gradients_near(actual, numerical, 4e-5,
            std::string(name) + " HVP must match the central difference of backward");
    };

    check_hvp(
        make_autodiff_network(Activations::Linear),
        Values{ 0.2, 0.85 },
        make_binary_cross_entropy_objective(),
        "BCE-from-logits"
    );
    check_hvp(
        make_autodiff_network(Activations::Linear),
        Values{ 0.0, 1.0 },
        make_softmax_cross_entropy_objective(),
        "softmax cross-entropy"
    );
    check_hvp(
        make_autodiff_network(Activations::Sigmoid),
        Values{ 0.2, 0.85 },
        make_mean_squared_error_objective(),
        "activated-output MSE"
    );
}

void test_regularized_objective_hvp()
{
    const MLP network = make_autodiff_network(Activations::Sigmoid);
    const NetworkTangent tangent = make_autodiff_tangent(network);
    const Dataset batch{
        { { 0.35, -0.42 }, { 0.2, 0.85 } },
        { { -0.15, 0.27 }, { 0.75, 0.1 } }
    };
    const ObjectiveConfig l2_objective = make_objective_config(
        make_mean_squared_error_objective(),
        make_l2_regularization(0.25, true)
    );

    const NetworkGradients actual = objective_hessian_vector_product(
        network, batch, tangent, l2_objective
    );
    const NetworkGradients numerical = central_difference_objective_gradient(
        network, tangent, batch, l2_objective
    );
    require_gradients_near(actual, numerical, 5e-5,
        "L2 objective HVP must match the central difference of objective gradients");

    const RegularizationTerm l2 = make_l2_regularization(0.25, true);
    require(static_cast<bool>(l2.add_hessian_vector_product),
        "L2 must expose a regularization HVP callback");
    NetworkGradients l2_actual = make_zero_gradients_like(network);
    l2.add_hessian_vector_product(network, tangent, l2_actual);
    NetworkGradients l2_expected = make_zero_gradients_like(network);
    for (std::size_t layer_index = 0;
         layer_index < l2_expected.layers.size();
         ++layer_index) {
        for (std::size_t index = 0;
             index < l2_expected.layers[layer_index].weights.size();
             ++index) {
            l2_expected.layers[layer_index].weights[index] =
                2.0 * tangent.layers[layer_index].weights[index];
        }
        for (std::size_t index = 0;
             index < l2_expected.layers[layer_index].biases.size();
             ++index) {
            l2_expected.layers[layer_index].biases[index] =
                2.0 * tangent.layers[layer_index].biases[index];
        }
    }
    require_gradients_near(l2_actual, l2_expected, 1e-12,
        "L2's unscaled regularization HVP must equal 2 times the tangent");

    const ObjectiveConfig l1_subgradient_objective = make_objective_config(
        make_mean_squared_error_objective(),
        make_l1_regularization_subgradient(0.1)
    );
    require_invalid_argument(
        [&] {
            static_cast<void>(objective_hessian_vector_product(
                network, batch, tangent, l1_subgradient_objective
            ));
        },
        "objective HVPs must reject non-smooth subgradient L1"
    );

    L1RegularizationOptions proximal_options;
    proximal_options.method = L1Method::Proximal;
    const ObjectiveConfig l1_proximal_objective = make_objective_config(
        make_mean_squared_error_objective(),
        make_l1_regularization(0.1, proximal_options)
    );
    require_invalid_argument(
        [&] {
            static_cast<void>(objective_hessian_vector_product(
                network, batch, tangent, l1_proximal_objective
            ));
        },
        "objective HVPs must reject proximal L1"
    );
}

void test_l2_hvp_excludes_biases_by_default()
{
    const MLP network = make_autodiff_network(Activations::Sigmoid);
    const NetworkTangent tangent = make_autodiff_tangent(network);
    const RegularizationTerm l2 = make_l2_regularization(0.25, false);

    require(static_cast<bool>(l2.add_hessian_vector_product),
        "L2 must expose a regularization HVP callback");

    NetworkGradients actual = make_zero_gradients_like(network);
    l2.add_hessian_vector_product(network, tangent, actual);

    NetworkGradients expected = make_zero_gradients_like(network);
    for (std::size_t layer_index = 0;
         layer_index < expected.layers.size();
         ++layer_index) {
        for (std::size_t index = 0;
             index < expected.layers[layer_index].weights.size();
             ++index) {
            expected.layers[layer_index].weights[index] =
                2.0 * tangent.layers[layer_index].weights[index];
        }
        // expected bias entries deliberately remain zero.
    }

    require_gradients_near(actual, expected, 1e-12,
        "Default L2 HVP must leave bias entries unchanged");
}

void test_missing_jvp_callbacks_are_rejected()
{
    const ActivationFunction first_order_only{
        "first_order_only",
        [](const Values& values) { return values; },
        [](const Values&, const Values& upstream) { return upstream; }
    };
    MLP network = make_zero_network({ 1, 1 });
    network.layer_activations = { first_order_only };
    network.layers[0].weights[0] = 0.4;
    network.layers[0].biases[0] = -0.1;
    const Values input{ 0.2 };
    const Values target{ 0.7 };
    const ForwardCache cache = forward_pass(network, input);
    NetworkTangent one_layer_tangent = make_zero_gradients_like(network);
    one_layer_tangent.layers[0].weights[0] = 0.3;
    one_layer_tangent.layers[0].biases[0] = -0.2;

    require_invalid_argument(
        [&] {
            static_cast<void>(forward_jvp(network, cache, one_layer_tangent));
        },
        "forward_jvp must reject an activation without JVP callbacks"
    );
    require_invalid_argument(
        [&] {
            static_cast<void>(loss_hessian_vector_product(
                network,
                cache,
                target,
                one_layer_tangent,
                make_mean_squared_error_objective()
            ));
        },
        "loss_hessian_vector_product must reject an activation without JVP callbacks"
    );
}

} // namespace

int main()
{
    try {
        static_assert(std::is_same_v<NetworkTangent, NetworkGradients>);
        test_activation_jvps();
        test_objective_loss_hvps();
        test_network_forward_jvp();
        test_sample_loss_hvps();
        test_regularized_objective_hvp();
        test_l2_hvp_excludes_biases_by_default();
        test_missing_jvp_callbacks_are_rejected();
        std::cout << "[PASS] parameter-space JVP and HVP contracts\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << '\n';
        return 1;
    }
}
