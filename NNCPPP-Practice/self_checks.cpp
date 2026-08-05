#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "mlp.hpp"
#include "objective_functions.hpp"
#include "optimizers.hpp"
#include "wolfe_analysis.hpp"

namespace {
void require(const bool condition, const std::string_view message)
{
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void require_near(
    const double actual,
    const double expected,
    const double tolerance,
    const std::string_view message
)
{
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Function>
void require_throws(Function function, std::string_view message)
{
    try {
        function();
    }
    catch (const std::exception&) {
        return;
    }

    throw std::runtime_error(std::string(message));
}

class TestOptimizer final : public Optimizer
{
public:
    const char* name() const noexcept override
    {
        return "TestOptimizer";
    }

    void reset(const MLP&) override
    {
    }

    TrainingStepResult step(OptimizerContext&) override
    {
        return {};
    }
};

void require_invalid_wolfe_parameters(
    WolfeParameters parameters,
    const std::string_view message
)
{
    require(!wolfe_parameters_are_valid(parameters), message);
    require_throws(
        [&parameters] {
            validate_wolfe_parameters(parameters);
        },
        message
    );
}

bool same_parameters(const MLP& left, const MLP& right)
{
    if (left.layer_sizes != right.layer_sizes || left.layers.size() != right.layers.size()) {
        return false;
    }

    for (std::size_t k = 0; k < left.layers.size(); ++k) {
        const DenseLayer& left_layer = left.layers[k];
        const DenseLayer& right_layer = right.layers[k];

        if (left_layer.input_size != right_layer.input_size ||
            left_layer.output_size != right_layer.output_size ||
            left_layer.weights != right_layer.weights ||
            left_layer.biases != right_layer.biases) {
            return false;
        }
    }

    return true;
}

bool any_weight_differs(const MLP& left, const MLP& right)
{
    require(left.layers.size() == right.layers.size(),
        "networks must have the same layer count before comparing weights");

    for (std::size_t k = 0; k < left.layers.size(); ++k) {
        const Values& left_weights = left.layers[k].weights;
        const Values& right_weights = right.layers[k].weights;

        require(left_weights.size() == right_weights.size(),
            "networks must have the same weight count before comparing weights");

        for (std::size_t i = 0; i < left_weights.size(); ++i) {
            if (left_weights[i] != right_weights[i]) {
                return true;
            }
        }
    }

    return false;
}

bool all_parameters_finite(const MLP& network)
{
    for (const DenseLayer& layer : network.layers) {
        for (const double weight : layer.weights) {
            if (!std::isfinite(weight)) {
                return false;
            }
        }

        for (const double bias : layer.biases) {
            if (!std::isfinite(bias)) {
                return false;
            }
        }
    }

    return true;
}

bool all_biases_zero(const MLP& network)
{
    for (const DenseLayer& layer : network.layers) {
        for (const double bias : layer.biases) {
            if (bias != 0.0) {
                return false;
            }
        }
    }

    return true;
}

bool any_weight_nonzero(const MLP& network)
{
    for (const DenseLayer& layer : network.layers) {
        for (const double weight : layer.weights) {
            if (weight != 0.0) {
                return true;
            }
        }
    }

    return false;
}

bool all_cache_values_finite(const ForwardCache& cache)
{
    for (const Values& activation : cache.activations) {
        for (const double value : activation) {
            if (!std::isfinite(value)) {
                return false;
            }
        }
    }

    for (const Values& pre_activation : cache.pre_activations) {
        for (const double value : pre_activation) {
            if (!std::isfinite(value)) {
                return false;
            }
        }
    }

    return true;
}

bool all_gradient_values_finite(const NetworkGradients& gradients)
{
    for (const LayerGradients& layer_gradients : gradients.layers) {
        for (const double gradient : layer_gradients.weights) {
            if (!std::isfinite(gradient)) {
                return false;
            }
        }

        for (const double gradient : layer_gradients.biases) {
            if (!std::isfinite(gradient)) {
                return false;
            }
        }
    }

    return true;
}

void require_gradient_layout(
    const MLP& network,
    const NetworkGradients& gradients,
    const std::string_view message
)
{
    require(gradients.layers.size() == network.layers.size(), message);

    for (std::size_t layer_index = 0;
         layer_index < network.layers.size();
         ++layer_index) {
        require(
            gradients.layers[layer_index].weights.size() ==
                network.layers[layer_index].weights.size(),
            message
        );
        require(
            gradients.layers[layer_index].biases.size() ==
                network.layers[layer_index].biases.size(),
            message
        );
    }
}

void add_gradients_in_place(
    NetworkGradients& accumulated,
    const NetworkGradients& addend
)
{
    require(accumulated.layers.size() == addend.layers.size(),
        "gradient layer counts should match while accumulating");

    for (std::size_t layer_index = 0;
         layer_index < accumulated.layers.size();
         ++layer_index) {
        require(
            accumulated.layers[layer_index].weights.size() ==
                addend.layers[layer_index].weights.size(),
            "gradient weight counts should match while accumulating"
        );
        require(
            accumulated.layers[layer_index].biases.size() ==
                addend.layers[layer_index].biases.size(),
            "gradient bias counts should match while accumulating"
        );

        for (std::size_t parameter_index = 0;
             parameter_index < accumulated.layers[layer_index].weights.size();
             ++parameter_index) {
            accumulated.layers[layer_index].weights[parameter_index] +=
                addend.layers[layer_index].weights[parameter_index];
        }

        for (std::size_t parameter_index = 0;
             parameter_index < accumulated.layers[layer_index].biases.size();
             ++parameter_index) {
            accumulated.layers[layer_index].biases[parameter_index] +=
                addend.layers[layer_index].biases[parameter_index];
        }
    }
}

double check_gradient_l2_norm(const NetworkGradients& gradients)
{
    double sum_of_squares = 0.0;

    for (const LayerGradients& layer_gradients : gradients.layers) {
        for (const double gradient : layer_gradients.weights) {
            sum_of_squares += gradient * gradient;
        }

        for (const double gradient : layer_gradients.biases) {
            sum_of_squares += gradient * gradient;
        }
    }

    return std::sqrt(sum_of_squares);
}

double check_maximum_absolute_gradient(const NetworkGradients& gradients)
{
    double maximum = 0.0;

    for (const LayerGradients& layer_gradients : gradients.layers) {
        for (const double gradient : layer_gradients.weights) {
            maximum = std::max(maximum, std::abs(gradient));
        }

        for (const double gradient : layer_gradients.biases) {
            maximum = std::max(maximum, std::abs(gradient));
        }
    }

    return maximum;
}

Dataset make_xor_dataset()
{
    return Dataset{
        { { 0.0, 0.0 }, { 0.0 } },
        { { 0.0, 1.0 }, { 1.0 } },
        { { 1.0, 0.0 }, { 1.0 } },
        { { 1.0, 1.0 }, { 0.0 } }
    };
}

void run_architecture_checks()
{
    validate_network_architecture({ 2, 1 });
    validate_network_architecture({ 2, 4, 1 });
    validate_network_architecture({ 2, 3, 3, 1 });

    require_throws([] { validate_network_architecture({}); },
        "empty network architecture should be rejected");
    require_throws([] { validate_network_architecture({ 2 }); },
        "single-entry network architecture should be rejected");
    require_throws([] { validate_network_architecture({ 0, 4, 1 }); },
        "zero input width should be rejected");
    require_throws([] { validate_network_architecture({ 2, 0, 1 }); },
        "zero hidden width should be rejected");
    require_throws([] { validate_network_architecture({ 2, 4, 0 }); },
        "zero output width should be rejected");

    std::cout << "[PASS] network architecture validation\n";
}

void run_parameter_count_checks()
{
    require(parameter_count(make_zero_network({ 2, 4, 1 })) == 17,
        "parameter count for {2,4,1} should be 17");
    std::cout << "[PASS] parameter count {2,4,1}: 17\n";

    require(parameter_count(make_zero_network({ 2, 3, 3, 1 })) == 25,
        "parameter count for {2,3,3,1} should be 25");
    std::cout << "[PASS] parameter count {2,3,3,1}: 25\n";

    require(parameter_count(make_zero_network({ 2, 1 })) == 3,
        "parameter count for {2,1} should be 3");
    std::cout << "[PASS] parameter count {2,1}: 3\n";
}

void run_initialization_checks()
{
    const MLP first = make_mlp({ 2, 4, 1 }, 42);
    const MLP same_seed = make_mlp({ 2, 4, 1 }, 42);
    const MLP different_seed = make_mlp({ 2, 4, 1 }, 43);
    const MLP deeper = make_mlp({ 2, 3, 3, 1 }, 42);

    require(same_parameters(first, same_seed),
        "same network architecture and same seed should produce identical parameters");
    require(any_weight_differs(first, different_seed),
        "same network architecture and different seed should change at least one weight");
    require(all_parameters_finite(first), "initialized parameters should be finite");
    require(all_biases_zero(first), "initialized biases should be zero");
    require(any_weight_nonzero(first), "at least one initialized weight should be nonzero");
    require(parameter_count(first) == 17, "initialized {2,4,1} should have 17 parameters");
    require(parameter_count(deeper) == 25, "initialized {2,3,3,1} should have 25 parameters");

    std::cout << "[PASS] deterministic Xavier initialization\n";
}

void run_stable_sigmoid_checks()
{
    const double at_zero = stable_sigmoid(0.0);
    const double large_positive = stable_sigmoid(1000.0);
    const double large_negative = stable_sigmoid(-1000.0);

    require_near(at_zero, 0.5, 1e-12, "stable_sigmoid(0) should be 0.5");
    require(std::isfinite(large_positive),
        "stable_sigmoid(1000) should be finite");
    require_near(large_positive, 1.0, 1e-12,
        "stable_sigmoid(1000) should be near 1");
    require(std::isfinite(large_negative),
        "stable_sigmoid(-1000) should be finite");
    require_near(large_negative, 0.0, 1e-12,
        "stable_sigmoid(-1000) should be near 0");

    require(at_zero >= 0.0 && at_zero <= 1.0,
        "stable_sigmoid(0) should be in [0,1]");
    require(large_positive >= 0.0 && large_positive <= 1.0,
        "stable_sigmoid(1000) should be in [0,1]");
    require(large_negative >= 0.0 && large_negative <= 1.0,
        "stable_sigmoid(-1000) should be in [0,1]");

    std::cout << "[PASS] stable sigmoid\n";
}

void require_value_sizes(
    const std::vector<Values>& values,
    const std::vector<std::size_t>& expected_sizes,
    const std::string_view message
)
{
    require(values.size() == expected_sizes.size(), message);

    for (std::size_t i = 0; i < expected_sizes.size(); ++i) {
        require(values[i].size() == expected_sizes[i], message);
    }
}

void run_forward_cache_shape_checks()
{
    {
        const MLP network = make_zero_network({ 2, 4, 1 });
        const ForwardCache cache = forward_pass(network, { 0.0, 0.0 });

        require_value_sizes(cache.activations, { 2, 4, 1 },
            "{2,4,1} activation cache sizes should be 2, 4, 1");
        require_value_sizes(cache.pre_activations, { 4, 1 },
            "{2,4,1} pre-activation cache sizes should be 4, 1");
    }

    {
        const MLP network = make_zero_network({ 2, 3, 3, 1 });
        const ForwardCache cache = forward_pass(network, { 0.0, 0.0 });

        require_value_sizes(cache.activations, { 2, 3, 3, 1 },
            "{2,3,3,1} activation cache sizes should be 2, 3, 3, 1");
        require_value_sizes(cache.pre_activations, { 3, 3, 1 },
            "{2,3,3,1} pre-activation cache sizes should be 3, 3, 1");
    }

    {
        const MLP network = make_zero_network({ 2, 1 });
        const ForwardCache cache = forward_pass(network, { 0.0, 0.0 });

        require_value_sizes(cache.activations, { 2, 1 },
            "{2,1} activation cache sizes should be 2, 1");
        require_value_sizes(cache.pre_activations, { 1 },
            "{2,1} pre-activation cache sizes should be 1");
    }

    require_throws([] {
        const MLP network = make_zero_network({ 2, 4, 1 });
        static_cast<void>(forward_pass(network, { 0.0 }));
    }, "forward_pass should reject wrong input size");

    std::cout << "[PASS] forward cache shapes\n";
}

void run_forward_value_checks()
{
    {
        const MLP network = make_zero_network({ 2, 4, 1 });
        const ForwardCache cache = forward_pass(network, { 0.0, 0.0 });

        require(all_cache_values_finite(cache),
            "zero-network forward cache values should be finite");
        require_near(cache.activations.back()[0], 0.5, 1e-12,
            "zero-network output activation should be 0.5");
    }

    {
        MLP network = make_zero_network({ 2, 2, 1 });
        network.layers[0].weight(0, 0) = 1.0;
        network.layers[0].weight(0, 1) = -1.0;
        network.layers[0].biases[0] = 0.25;
        network.layers[0].weight(1, 0) = -0.5;
        network.layers[0].weight(1, 1) = 0.5;
        network.layers[0].biases[1] = -0.25;
        network.layers[1].weight(0, 0) = 2.0;
        network.layers[1].weight(0, 1) = -1.0;
        network.layers[1].biases[0] = 0.1;

        const ForwardCache cache = forward_pass(network, { 0.5, -0.25 });
        const double expected_hidden_z0 = 1.0;
        const double expected_hidden_z1 = -0.625;
        const double expected_hidden_a0 = std::tanh(expected_hidden_z0);
        const double expected_hidden_a1 = std::tanh(expected_hidden_z1);
        const double expected_output_z =
            0.1 + 2.0 * expected_hidden_a0 - expected_hidden_a1;

        require_near(cache.pre_activations[0][0], expected_hidden_z0, 1e-12,
            "first hidden pre-activation should match manual affine calculation");
        require_near(cache.pre_activations[0][1], expected_hidden_z1, 1e-12,
            "second hidden pre-activation should match manual affine calculation");
        require_near(cache.activations[0][0], 0.5, 1e-12,
            "input activation should preserve the first input coordinate");
        require_near(cache.activations[1][0], expected_hidden_a0, 1e-12,
            "hidden activation should use tanh");
        require_near(cache.activations[1][1], expected_hidden_a1, 1e-12,
            "hidden activation should use tanh");
        require_near(cache.pre_activations[1][0], expected_output_z, 1e-12,
            "output pre-activation should use final hidden activation");
        require_near(cache.activations[2][0], stable_sigmoid(expected_output_z), 1e-12,
            "output activation should use stable sigmoid");
        require(all_cache_values_finite(cache), "forward cache values should be finite");
    }

    {
        const MLP network = make_zero_network({ 2, 1 });
        const ForwardCache cache = forward_pass(network, { 1.0, 1.0 });

        require_near(cache.activations.back()[0], 0.5, 1e-12,
            "no-hidden zero network output should be 0.5");
        require(all_cache_values_finite(cache),
            "no-hidden forward cache values should be finite");
    }

    require_throws([] {
        const MLP network = make_zero_network({ 2, 1 });
        static_cast<void>(forward_pass(network, { std::nan(""), 0.0 }));
    }, "forward_pass should reject non-finite input values");

    std::cout << "[PASS] forward values\n";
}

void run_binary_cross_entropy_scalar_checks()
{
    const double log_two = std::log(2.0);

    require_near(binary_cross_entropy_from_logit(0.0, 0.0), log_two, 1e-12,
        "BCE(0,0) should be log(2)");
    require_near(binary_cross_entropy_from_logit(0.0, 1.0), log_two, 1e-12,
        "BCE(0,1) should be log(2)");

    const double large_positive_correct =
        binary_cross_entropy_from_logit(1000.0, 1.0);
    const double large_positive_wrong =
        binary_cross_entropy_from_logit(1000.0, 0.0);
    const double large_negative_correct =
        binary_cross_entropy_from_logit(-1000.0, 0.0);
    const double large_negative_wrong =
        binary_cross_entropy_from_logit(-1000.0, 1.0);

    require(std::isfinite(large_positive_correct),
        "BCE(1000,1) should be finite");
    require_near(large_positive_correct, 0.0, 1e-12,
        "BCE(1000,1) should be near 0");

    require(std::isfinite(large_positive_wrong),
        "BCE(1000,0) should be finite");
    require_near(large_positive_wrong, 1000.0, 1e-12,
        "BCE(1000,0) should be near 1000");

    require(std::isfinite(large_negative_correct),
        "BCE(-1000,0) should be finite");
    require_near(large_negative_correct, 0.0, 1e-12,
        "BCE(-1000,0) should be near 0");

    require(std::isfinite(large_negative_wrong),
        "BCE(-1000,1) should be finite");
    require_near(large_negative_wrong, 1000.0, 1e-12,
        "BCE(-1000,1) should be near 1000");

    require_throws([] {
        static_cast<void>(binary_cross_entropy_from_logit(0.0, -0.1));
    }, "BCE should reject target values below 0");
    require_throws([] {
        static_cast<void>(binary_cross_entropy_from_logit(0.0, 1.1));
    }, "BCE should reject target values above 1");
    require_throws([] {
        static_cast<void>(binary_cross_entropy_from_logit(
            std::numeric_limits<double>::infinity(),
            1.0
        ));
    }, "BCE should reject non-finite logits");
    require_throws([] {
        static_cast<void>(binary_cross_entropy_from_logit(
            0.0,
            std::numeric_limits<double>::quiet_NaN()
        ));
    }, "BCE should reject non-finite targets");

    std::cout << "[PASS] stable BCE scalar\n";
}

void run_sample_cost_checks()
{
    {
        const MLP network = make_zero_network({ 2, 1 });
        const ForwardCache cache = forward_pass(network, { 0.25, -0.5 });

        require_near(sample_cost(network, cache, { 0.0 }), std::log(2.0), 1e-12,
            "single-output sample cost should equal its BCE cost");
    }

    {
        MLP network = make_zero_network({ 1, 2 });
        network.layers[0].biases[0] = 2.0;
        network.layers[0].biases[1] = -1.0;
        const ForwardCache cache = forward_pass(network, { 0.0 });

        const double expected_cost = 0.5 * (
            binary_cross_entropy_from_logit(2.0, 1.0) +
            binary_cross_entropy_from_logit(-1.0, 0.0)
        );

        require_near(sample_cost(network, cache, { 1.0, 0.0 }), expected_cost, 1e-12,
            "multi-output sample cost should average BCE across output units");
    }

    require_throws([] {
        const MLP network = make_zero_network({ 2, 2 });
        const ForwardCache cache = forward_pass(network, { 0.0, 0.0 });
        static_cast<void>(sample_cost(network, cache, { 1.0 }));
    }, "sample_cost should reject target sizes that do not match the output width");

    std::cout << "[PASS] sample cost\n";
}

void run_batch_loss_checks()
{
    const MLP zero_network = make_zero_network({ 2, 1 });
    const Dataset xor_batch{
        { { 0.0, 0.0 }, { 0.0 } },
        { { 0.0, 1.0 }, { 1.0 } },
        { { 1.0, 0.0 }, { 1.0 } },
        { { 1.0, 1.0 }, { 0.0 } }
    };

    require_near(batch_loss(zero_network, xor_batch), std::log(2.0), 1e-12,
        "zero-network XOR batch loss should be log(2)");

    {
        MLP network = make_zero_network({ 1, 1 });
        network.layers[0].weight(0, 0) = 1.0;
        const Dataset batch{
            { { 0.0 }, { 0.0 } },
            { { 2.0 }, { 1.0 } }
        };
        const double expected_loss = 0.5 * (
            binary_cross_entropy_from_logit(0.0, 0.0) +
            binary_cross_entropy_from_logit(2.0, 1.0)
        );

        require_near(batch_loss(network, batch), expected_loss, 1e-12,
            "batch loss should average the sample costs");
    }

    require_throws([&zero_network] {
        static_cast<void>(batch_loss(zero_network, {}));
    }, "batch_loss should reject an empty batch");

    require_throws([&zero_network] {
        static_cast<void>(batch_loss(zero_network, { { { 0.0 }, { 0.0 } } }));
    }, "batch_loss should reject input sizes that do not match the network");

    require_throws([&zero_network] {
        static_cast<void>(batch_loss(zero_network, { { { 0.0, 0.0 }, { 0.0, 1.0 } } }));
    }, "batch_loss should reject target sizes that do not match the network");

    require_throws([&zero_network] {
        static_cast<void>(batch_loss(zero_network, {
            { { std::numeric_limits<double>::infinity(), 0.0 }, { 0.0 } }
        }));
    }, "batch_loss should reject non-finite inputs");

    require_throws([&zero_network] {
        static_cast<void>(batch_loss(zero_network, {
            { { 0.0, 0.0 }, { std::numeric_limits<double>::quiet_NaN() } }
        }));
    }, "batch_loss should reject non-finite targets");

    std::cout << "[PASS] batch loss\n";
}

void run_objective_checks()
{
    const MLP network = make_zero_network({ 1, 2 });
    MLP configured_network = network;
    configured_network.layers[0].biases = { 2.0, -1.0 };
    const ForwardCache cache = forward_pass(configured_network, { 0.0 });

    const ObjectiveFunctions bce = make_binary_cross_entropy_objective();
    validate_objective_functions(bce);
    std::cout << "[PASS] objective callback presence\n";

    const Values bce_gradient = bce.sample_loss_gradient(
        { 0.0, 2.0 },
        { 0.0, 1.0 }
    );
    require(bce_gradient.size() == 2,
        "BCE objective gradient should match the output width");
    require_near(
        bce_gradient[0],
        (stable_sigmoid(0.0) - 0.0) / 2.0,
        1e-12,
        "BCE objective gradient should use sigmoid(logit) minus target"
    );
    require_near(
        bce_gradient[1],
        (stable_sigmoid(2.0) - 1.0) / 2.0,
        1e-12,
        "BCE objective gradient should preserve output averaging"
    );
    require_throws([&bce] {
        static_cast<void>(bce.sample_loss_gradient({ 0.0 }, { 0.0, 1.0 }));
    }, "BCE objective gradient should reject mismatched vector sizes");
    require_throws([&bce] {
        static_cast<void>(bce.sample_loss_gradient(
            { 0.0 },
            { std::numeric_limits<double>::quiet_NaN() }
        ));
    }, "BCE objective gradient should reject non-finite targets");
    std::cout << "[PASS] BCE objective gradient callback\n";

    require_near(
        sample_cost(configured_network, cache, { 1.0, 0.0 }, bce),
        0.5 * (
            binary_cross_entropy_from_logit(2.0, 1.0) +
            binary_cross_entropy_from_logit(-1.0, 0.0)
        ),
        1e-12,
        "default BCE objective should match the existing sample cost"
    );

    bool saw_complete_logits = false;
    bool saw_complete_target = false;
    std::size_t custom_call_count = 0;

    ObjectiveFunctions custom;
    custom.sample_loss =
        [&saw_complete_logits, &saw_complete_target, &custom_call_count](
            const Values& logits,
            const Values& target
        ) {
            ++custom_call_count;
            saw_complete_logits = logits == Values{ 2.0, -1.0 };
            saw_complete_target = target == Values{ 1.0, 0.0 };
            return 4.0;
        };
    custom.sample_loss_gradient = [](const Values& logits, const Values&) {
        return Values(logits.size(), 0.0);
    };

    validate_objective_functions(custom);
    require_near(
        sample_cost(configured_network, cache, { 1.0, 0.0 }, custom),
        4.0,
        1e-12,
        "custom objective should return its sample loss value"
    );
    require(saw_complete_logits,
        "custom objective should receive every output logit");
    require(saw_complete_target,
        "custom objective should receive every target value");

    const Dataset two_sample_batch{
        { { 0.0 }, { 1.0, 0.0 } },
        { { 1.0 }, { 0.0, 1.0 } }
    };
    custom.sample_loss = [&custom_call_count](const Values&, const Values&) {
        ++custom_call_count;
        return custom_call_count == 2 ? 6.0 : 2.0;
    };
    custom_call_count = 0;

    require_near(
        batch_loss(configured_network, two_sample_batch, custom),
        4.0,
        1e-12,
        "objective-aware batch loss should average sample losses"
    );
    require(custom_call_count == 2,
        "objective-aware batch loss should evaluate every sample once");

    ObjectiveFunctions overflowing_batch = custom;
    overflowing_batch.sample_loss = [](const Values&, const Values&) {
        return std::numeric_limits<double>::max();
    };
    require_throws([&configured_network, &two_sample_batch, &overflowing_batch] {
        static_cast<void>(batch_loss(
            configured_network,
            two_sample_batch,
            overflowing_batch
        ));
    }, "objective-aware batch loss should reject non-finite accumulation");

    ObjectiveFunctions missing_loss = custom;
    missing_loss.sample_loss = {};
    require_throws([&missing_loss] {
        validate_objective_functions(missing_loss);
    }, "objective validation should reject a missing sample-loss callback");

    ObjectiveFunctions missing_gradient = custom;
    missing_gradient.sample_loss_gradient = {};
    require_throws([&missing_gradient] {
        validate_objective_functions(missing_gradient);
    }, "objective validation should reject a missing derivative callback");

    ObjectiveFunctions nonfinite_loss = custom;
    nonfinite_loss.sample_loss = [](const Values&, const Values&) {
        return std::numeric_limits<double>::quiet_NaN();
    };
    require_throws([&configured_network, &cache, &nonfinite_loss] {
        static_cast<void>(sample_cost(
            configured_network,
            cache,
            { 1.0, 0.0 },
            nonfinite_loss
        ));
    }, "objective-aware sample cost should reject non-finite loss results");

    std::cout << "[PASS] objective callback contract\n";
}

void run_mse_objective_checks()
{
    const ObjectiveFunctions mse = make_mean_squared_error_objective();
    validate_objective_functions(mse);

    // Convention under test:
    //
    //     loss = (1 / output_width) * sum((logit - target)^2)
    //     d_loss/d_logit = 2 * (logit - target) / output_width
    //
    // The callbacks receive output logits. MSE must not apply sigmoid here.
    const Values logits{ 1.0, -1.0 };
    const Values targets{ 0.5, 0.0 };

    require_near(
        mse.sample_loss(logits, targets),
        0.625,
        1e-12,
        "MSE objective should compute the documented mean-squared loss"
    );

    const Values gradient = mse.sample_loss_gradient(logits, targets);
    require(gradient.size() == logits.size(),
        "MSE objective gradient should match the output width");
    require_near(
        gradient[0],
        0.5,
        1e-12,
        "MSE objective gradient should use the first residual and width scaling"
    );
    require_near(
        gradient[1],
        -1.0,
        1e-12,
        "MSE objective gradient should preserve the second residual sign"
    );

    require_near(
        mse.sample_loss({ 2.0 }, { 2.0 }),
        0.0,
        1e-12,
        "MSE objective should be zero when logits equal targets"
    );

    const Values zero_gradient = mse.sample_loss_gradient({ 2.0 }, { 2.0 });
    require(zero_gradient.size() == 1,
        "MSE one-output gradient should contain one value");
    require_near(
        zero_gradient[0],
        0.0,
        1e-12,
        "MSE objective gradient should be zero when logits equal targets"
    );

    require_throws([] {
        const ObjectiveFunctions objective = make_mean_squared_error_objective();
        static_cast<void>(objective.sample_loss({}, {}));
    }, "MSE loss should reject empty vectors");

    require_throws([] {
        const ObjectiveFunctions objective = make_mean_squared_error_objective();
        static_cast<void>(objective.sample_loss_gradient({}, {}));
    }, "MSE gradient should reject empty vectors");

    require_throws([] {
        const ObjectiveFunctions objective = make_mean_squared_error_objective();
        static_cast<void>(objective.sample_loss({ 1.0 }, { 1.0, 2.0 }));
    }, "MSE loss should reject mismatched vector sizes");

    require_throws([] {
        const ObjectiveFunctions objective = make_mean_squared_error_objective();
        static_cast<void>(objective.sample_loss_gradient(
            { 1.0 },
            { 1.0, 2.0 }
        ));
    }, "MSE gradient should reject mismatched vector sizes");

    require_throws([] {
        const ObjectiveFunctions objective = make_mean_squared_error_objective();
        static_cast<void>(objective.sample_loss(
            { std::numeric_limits<double>::quiet_NaN() },
            { 0.0 }
        ));
    }, "MSE loss should reject NaN logits");

    require_throws([] {
        const ObjectiveFunctions objective = make_mean_squared_error_objective();
        static_cast<void>(objective.sample_loss_gradient(
            { std::numeric_limits<double>::infinity() },
            { 0.0 }
        ));
    }, "MSE gradient should reject infinite logits");

    require_throws([] {
        const ObjectiveFunctions objective = make_mean_squared_error_objective();
        static_cast<void>(objective.sample_loss(
            { 0.0 },
            { std::numeric_limits<double>::quiet_NaN() }
        ));
    }, "MSE loss should reject NaN targets");

    require_throws([] {
        const ObjectiveFunctions objective = make_mean_squared_error_objective();
        static_cast<void>(objective.sample_loss_gradient(
            { 0.0 },
            { -std::numeric_limits<double>::infinity() }
        ));
    }, "MSE gradient should reject infinite targets");

    MLP configured_network = make_zero_network({ 1, 2 });
    configured_network.layers[0].biases = { 1.0, -1.0 };
    const ForwardCache cache = forward_pass(configured_network, { 0.0 });

    require_near(
        sample_cost(configured_network, cache, targets, mse),
        0.625,
        1e-12,
        "MSE sample cost should evaluate the cache's output logits"
    );

    const Dataset batch{
        { { 0.0 }, { 0.5, 0.0 } },
        { { 0.0 }, { 1.0, -1.0 } }
    };
    require_near(
        batch_loss(configured_network, batch, mse),
        0.3125,
        1e-12,
        "MSE batch loss should average sample losses"
    );

    const NetworkGradients backward_gradients = backward(
        configured_network,
        cache,
        targets,
        mse
    );
    require_near(
        backward_gradients.layers[0].biases[0],
        0.5,
        1e-12,
        "MSE backward should use the first output-logit derivative"
    );
    require_near(
        backward_gradients.layers[0].biases[1],
        -1.0,
        1e-12,
        "MSE backward should use the second output-logit derivative"
    );

    std::cout << "[PASS] MSE objective value, gradient, validation, and integration\n";
}

void run_sgd_optimizer_checks()
{
    MLP network = make_zero_network({ 1, 1 });
    network.layers[0].weights = { 0.5 };
    network.layers[0].biases = { 0.25 };

    const Dataset batch{
        { { 1.0 }, { 0.0 } }
    };
    const ObjectiveConfig objective = make_objective_config(
        make_mean_squared_error_objective()
    );

    NetworkGradients gradient = make_zero_gradients_like(network);
    gradient.layers[0].weights[0] = 0.2;
    gradient.layers[0].biases[0] = 0.6;

    std::vector<std::size_t> observed_schedule_steps;
    SGDOptions options;
    options.learning_rate_schedule =
        [&observed_schedule_steps](const std::size_t update_index) {
            observed_schedule_steps.push_back(update_index);
            return update_index == 0 ? 0.5 : 0.25;
        };

    const OptimizerSpec configured_sgd = make_sgd(options);
    validate_optimizer_spec(configured_sgd);

    std::unique_ptr<Optimizer> optimizer = configured_sgd.make();
    require(optimizer != nullptr,
        "configured SGD factory should create an optimizer");
    require(std::string(optimizer->name()) == "SGD",
        "configured SGD should report its name");

    optimizer->reset(network);

    const double first_loss = objective_loss(
        network,
        batch,
        objective
    );
    OptimizerContext first_context{
        network,
        batch,
        objective,
        first_loss,
        gradient
    };

    const TrainingStepResult first_result =
        optimizer->step(first_context);

    require(first_result.updated,
        "SGD should report a successful update");
    require_near(first_result.previous_loss, first_loss, 1e-12,
        "SGD should preserve the supplied previous loss");
    require_near(first_result.gradient_norm, std::sqrt(0.4), 1e-12,
        "SGD should report the current gradient norm");
    require_near(network.layers[0].weights[0], 0.4, 1e-12,
        "SGD should update weights using the scheduled learning rate");
    require_near(network.layers[0].biases[0], -0.05, 1e-12,
        "SGD should update biases using the scheduled learning rate");
    require_near(first_result.new_loss, 0.1225, 1e-12,
        "SGD should report the post-update objective loss");

    const double second_loss = objective_loss(
        network,
        batch,
        objective
    );
    OptimizerContext second_context{
        network,
        batch,
        objective,
        second_loss,
        gradient
    };
    static_cast<void>(optimizer->step(second_context));

    require_near(network.layers[0].weights[0], 0.35, 1e-12,
        "SGD should use the schedule's second update rate");
    require_near(network.layers[0].biases[0], -0.2, 1e-12,
        "SGD should schedule bias updates consistently");

    optimizer->reset(network);
    const double reset_loss = objective_loss(network, batch, objective);
    OptimizerContext reset_context{
        network,
        batch,
        objective,
        reset_loss,
        gradient
    };
    static_cast<void>(optimizer->step(reset_context));

    require(observed_schedule_steps.size() == 3,
        "SGD should evaluate the schedule once per update");
    require(observed_schedule_steps[0] == 0 &&
            observed_schedule_steps[1] == 1 &&
            observed_schedule_steps[2] == 0,
        "SGD reset should restart the schedule at update zero");

    SGDOptions empty_schedule;
    empty_schedule.learning_rate_schedule = {};
    require_throws([&empty_schedule] {
        static_cast<void>(make_sgd(empty_schedule));
    }, "SGD should reject an empty learning-rate schedule");

    SGDOptions invalid_schedule;
    invalid_schedule.learning_rate_schedule = [](std::size_t) {
        return std::numeric_limits<double>::quiet_NaN();
    };
    const OptimizerSpec invalid_sgd = make_sgd(invalid_schedule);
    std::unique_ptr<Optimizer> invalid_optimizer = invalid_sgd.make();
    invalid_optimizer->reset(network);

    const MLP before_invalid_step = network;
    const double invalid_step_loss = objective_loss(
        network,
        batch,
        objective
    );
    OptimizerContext invalid_context{
        network,
        batch,
        objective,
        invalid_step_loss,
        gradient
    };

    require_throws([&invalid_optimizer, &invalid_context] {
        static_cast<void>(invalid_optimizer->step(invalid_context));
    }, "SGD should reject a non-finite scheduled learning rate");
    require(same_parameters(network, before_invalid_step),
        "invalid SGD schedules should not mutate the network");

    std::cout << "[PASS] configurable SGD and learning-rate schedules\n";
}

void run_regularization_coefficient_checks()
{
    const MLP network = make_zero_network({ 1, 1 });

    RegularizationTerm custom;
    custom.name = "coefficient contract";
    custom.value = [](const MLP&) {
        return 2.0;
    };
    custom.add_gradient = [](const MLP& network, NetworkGradients& gradients) {
        for (std::size_t layer_index = 0;
             layer_index < network.layers.size();
             ++layer_index) {
            for (double& value : gradients.layers[layer_index].weights) {
                value += 3.0;
            }
            for (double& value : gradients.layers[layer_index].biases) {
                value += 3.0;
            }
        }
    };
    custom.smooth = true;
    custom.coefficient = 0.5;

    const ObjectiveConfig custom_config = make_objective_config(
        make_binary_cross_entropy_objective(),
        { custom }
    );

    require_near(
        regularization_loss(network, custom_config),
        1.0,
        1e-12,
        "regularization loss should apply the coefficient exactly once"
    );

    NetworkGradients combined = make_zero_gradients_like(network);
    combined.layers[0].weights[0] = 1.0;
    combined.layers[0].biases[0] = 1.0;

    add_regularization_gradients(network, custom_config, combined);

    require_near(
        combined.layers[0].weights[0],
        2.5,
        1e-12,
        "regularization gradient should preserve data gradients and scale only the regularizer"
    );
    require_near(
        combined.layers[0].biases[0],
        2.5,
        1e-12,
        "regularization bias gradient should preserve data gradients and scale only the regularizer"
    );

    MLP regularized_network = make_zero_network({ 2, 1 });
    regularized_network.layers[0].weights = { 2.0, -3.0 };
    regularized_network.layers[0].biases = { 4.0 };

    const RegularizationTerm l2 = make_l2_regularization(0.1);
    require(l2.smooth,
        "L2 regularization should be marked smooth");
    require(!l2.includes_biases,
        "L2 regularization should exclude biases by default");
    require_near(
        l2.value(regularized_network),
        13.0,
        1e-12,
        "L2 value callback should return the unscaled sum of squares"
    );

    NetworkGradients l2_unscaled =
        make_zero_gradients_like(regularized_network);
    l2.add_gradient(regularized_network, l2_unscaled);
    require_near(
        l2_unscaled.layers[0].weights[0],
        4.0,
        1e-12,
        "L2 callback should return the unscaled derivative for the first weight"
    );
    require_near(
        l2_unscaled.layers[0].weights[1],
        -6.0,
        1e-12,
        "L2 callback should return the unscaled derivative for the second weight"
    );
    require_near(
        l2_unscaled.layers[0].biases[0],
        0.0,
        1e-12,
        "L2 callback should exclude biases by default"
    );

    const ObjectiveConfig l2_config = make_objective_config(
        make_binary_cross_entropy_objective(),
        { l2 }
    );
    require_near(
        regularization_loss(regularized_network, l2_config),
        1.3,
        1e-12,
        "L2 objective loss should apply the coefficient once"
    );

    const RegularizationTerm l1 = make_l1_regularization(
        0.1,
        false,
        L1Method::Subgradient
    );
    require(!l1.smooth,
        "L1 subgradient regularization should be marked nonsmooth");
    require(!l1.includes_biases,
        "L1 regularization should exclude biases by default");
    require_near(
        l1.value(regularized_network),
        5.0,
        1e-12,
        "L1 value callback should return the unscaled sum of absolute values"
    );

    NetworkGradients l1_unscaled =
        make_zero_gradients_like(regularized_network);
    l1.add_gradient(regularized_network, l1_unscaled);
    require_near(
        l1_unscaled.layers[0].weights[0],
        1.0,
        1e-12,
        "L1 callback should return the positive sign subgradient"
    );
    require_near(
        l1_unscaled.layers[0].weights[1],
        -1.0,
        1e-12,
        "L1 callback should return the negative sign subgradient"
    );
    require_near(
        l1_unscaled.layers[0].biases[0],
        0.0,
        1e-12,
        "L1 callback should exclude biases by default"
    );

    const RegularizationTerm l1_with_biases = make_l1_regularization(
        0.1,
        true,
        L1Method::Subgradient
    );
    require(l1_with_biases.includes_biases,
        "L1 regularization should preserve the bias-inclusion setting");
    require_near(
        l1_with_biases.value(regularized_network),
        9.0,
        1e-12,
        "L1 bias-inclusive value should include the bias magnitude"
    );

    require_throws([] {
        static_cast<void>(make_l2_regularization(-0.1));
    }, "L2 regularization should reject negative coefficients");
    require_throws([] {
        static_cast<void>(make_l1_regularization(
            std::numeric_limits<double>::quiet_NaN()
        ));
    }, "L1 regularization should reject NaN coefficients");

    std::cout << "[PASS] centralized regularization coefficients and L1/L2 contracts\n";
}

void require_zero_gradient_layout(
    const MLP& network,
    const NetworkGradients& gradients
)
{
    require(gradients.layers.size() == network.layers.size(),
        "zero gradients should have one layer per network layer");

    for (std::size_t layer_index = 0; layer_index < network.layers.size(); ++layer_index) {
        const DenseLayer& layer = network.layers[layer_index];
        const LayerGradients& layer_gradients = gradients.layers[layer_index];

        require(layer_gradients.weights.size() == layer.weights.size(),
            "zero gradient weight shape should match its layer");
        require(layer_gradients.biases.size() == layer.biases.size(),
            "zero gradient bias shape should match its layer");

        for (const double gradient : layer_gradients.weights) {
            require(gradient == 0.0, "zero gradient weights should start at zero");
        }

        for (const double gradient : layer_gradients.biases) {
            require(gradient == 0.0, "zero gradient biases should start at zero");
        }
    }
}

void run_zero_gradient_checks()
{
    {
        const MLP network = make_mlp({ 2, 4, 1 }, 42);
        const MLP original = network;
        const NetworkGradients gradients = make_zero_gradients_like(network);

        require_zero_gradient_layout(network, gradients);
        require(same_parameters(network, original),
            "make_zero_gradients_like should not change network parameters");
    }

    {
        const MLP network = make_mlp({ 2, 3, 3, 1 }, 42);
        const NetworkGradients gradients = make_zero_gradients_like(network);

        require_zero_gradient_layout(network, gradients);
    }

    {
        const MLP network = make_mlp({ 2, 1 }, 42);
        const NetworkGradients gradients = make_zero_gradients_like(network);

        require_zero_gradient_layout(network, gradients);
    }

    require_throws([] {
        MLP invalid_network = make_zero_network({ 2, 1 });
        invalid_network.layers[0].weights.pop_back();
        static_cast<void>(make_zero_gradients_like(invalid_network));
    }, "make_zero_gradients_like should reject invalid networks");

    std::cout << "[PASS] zero gradient layout\n";
}

void run_backward_checks()
{
    {
        MLP network = make_zero_network({ 2, 2 });
        network.layers[0].weight(0, 0) = 0.5;
        network.layers[0].weight(0, 1) = -0.25;
        network.layers[0].biases[0] = 0.1;
        network.layers[0].weight(1, 0) = -0.7;
        network.layers[0].weight(1, 1) = 0.3;
        network.layers[0].biases[1] = -0.2;

        const Values input{ 0.4, -0.6 };
        const Values target{ 1.0, 0.0 };
        const ForwardCache cache = forward_pass(network, input);
        const NetworkGradients gradients = backward(network, cache, target);

        const double delta_0 = (cache.activations[1][0] - target[0]) / 2.0;
        const double delta_1 = (cache.activations[1][1] - target[1]) / 2.0;

        require_near(gradients.layers[0].biases[0], delta_0, 1e-12,
            "output bias gradient should equal the first output delta");
        require_near(gradients.layers[0].biases[1], delta_1, 1e-12,
            "output bias gradient should equal the second output delta");
        require_near(gradients.layers[0].weights[0], delta_0 * input[0], 1e-12,
            "first output weight gradient should be delta times first input");
        require_near(gradients.layers[0].weights[1], delta_0 * input[1], 1e-12,
            "second first-row weight gradient should be delta times second input");
        require_near(gradients.layers[0].weights[2], delta_1 * input[0], 1e-12,
            "third output weight gradient should be delta times first input");
        require_near(gradients.layers[0].weights[3], delta_1 * input[1], 1e-12,
            "fourth output weight gradient should be delta times second input");
    }

    {
        MLP network = make_zero_network({ 1, 1, 1 });
        network.layers[0].weight(0, 0) = 0.4;
        network.layers[0].biases[0] = -0.1;
        network.layers[1].weight(0, 0) = 0.7;
        network.layers[1].biases[0] = -0.2;

        const Values input{ 0.5 };
        const Values target{ 1.0 };
        const ForwardCache cache = forward_pass(network, input);
        const NetworkGradients gradients = backward(network, cache, target);

        const double output_delta = cache.activations[2][0] - target[0];
        const double hidden_activation = cache.activations[1][0];
        const double hidden_delta = network.layers[1].weight(0, 0) * output_delta *
            (1.0 - hidden_activation * hidden_activation);

        require_near(gradients.layers[1].biases[0], output_delta, 1e-12,
            "final-layer bias gradient should equal the output delta");
        require_near(gradients.layers[1].weights[0], output_delta * hidden_activation, 1e-12,
            "final-layer weight gradient should be output delta times hidden activation");
        require_near(gradients.layers[0].biases[0], hidden_delta, 1e-12,
            "hidden bias gradient should equal the hidden delta");
        require_near(gradients.layers[0].weights[0], hidden_delta * input[0], 1e-12,
            "hidden weight gradient should be hidden delta times input");
    }

    require_throws([] {
        const MLP network = make_zero_network({ 2, 1 });
        const ForwardCache cache = forward_pass(network, { 0.0, 0.0 });
        static_cast<void>(backward(network, cache, { 0.0, 1.0 }));
    }, "backward should reject target sizes that do not match the output width");

    require_throws([] {
        const MLP network = make_zero_network({ 2, 1 });
        ForwardCache cache = forward_pass(network, { 0.0, 0.0 });
        cache.activations.pop_back();
        static_cast<void>(backward(network, cache, { 0.0 }));
    }, "backward should reject malformed forward caches");

    std::cout << "[PASS] analytic backward gradients\n";
}

void require_gradients_near(
    const MLP& network,
    const NetworkGradients& actual,
    const NetworkGradients& expected,
    const double tolerance,
    const std::string_view message
)
{
    require_gradient_layout(network, actual, message);
    require_gradient_layout(network, expected, message);

    for (std::size_t layer_index = 0;
         layer_index < network.layers.size();
         ++layer_index) {
        for (std::size_t parameter_index = 0;
             parameter_index < actual.layers[layer_index].weights.size();
             ++parameter_index) {
            require_near(
                actual.layers[layer_index].weights[parameter_index],
                expected.layers[layer_index].weights[parameter_index],
                tolerance,
                message
            );
        }

        for (std::size_t parameter_index = 0;
             parameter_index < actual.layers[layer_index].biases.size();
             ++parameter_index) {
            require_near(
                actual.layers[layer_index].biases[parameter_index],
                expected.layers[layer_index].biases[parameter_index],
                tolerance,
                message
            );
        }
    }
}

void run_objective_backward_checks()
{
    const ObjectiveFunctions bce = make_binary_cross_entropy_objective();

    {
        const MLP network = make_zero_network({ 1, 2 });
        const Values input{ 2.0 };
        const Values target{ 1.0, 0.0 };
        const ForwardCache cache = forward_pass(network, input);

        std::size_t callback_count = 0;
        ObjectiveFunctions custom = bce;
        custom.sample_loss_gradient = [&callback_count](
            const Values& logits,
            const Values& target_values
        ) {
            ++callback_count;
            require(logits.size() == 2,
                "objective backward should pass the complete output logits");
            require(target_values.size() == 2,
                "objective backward should pass the complete target");
            return Values{ 0.25, -0.5 };
        };

        const NetworkGradients gradients =
            backward(network, cache, target, custom);

        require(callback_count == 1,
            "objective gradient callback should be called exactly once");
        require_near(gradients.layers[0].biases[0], 0.25, 1e-12,
            "objective output bias gradient should use callback component zero");
        require_near(gradients.layers[0].biases[1], -0.5, 1e-12,
            "objective output bias gradient should use callback component one");
        require_near(gradients.layers[0].weights[0], 0.5, 1e-12,
            "objective weight gradient should equal delta times input");
        require_near(gradients.layers[0].weights[1], -1.0, 1e-12,
            "objective second weight gradient should equal delta times input");
    }

    {
        const MLP network = make_zero_network({ 1, 2 });
        const ForwardCache cache = forward_pass(network, { 0.0 });
        ObjectiveFunctions wrong_size = bce;
        wrong_size.sample_loss_gradient = [](const Values&, const Values&) {
            return Values{ 0.25, -0.5, 1.0 };
        };

        require_throws([&network, &cache, &wrong_size] {
            static_cast<void>(backward(
                network,
                cache,
                { 1.0, 0.0 },
                wrong_size
            ));
        }, "objective backward should reject a derivative with the wrong size");
    }

    {
        const MLP network = make_zero_network({ 1, 1 });
        const ForwardCache cache = forward_pass(network, { 0.0 });
        ObjectiveFunctions nonfinite = bce;
        nonfinite.sample_loss_gradient = [](const Values&, const Values&) {
            return Values{ std::numeric_limits<double>::quiet_NaN() };
        };

        require_throws([&network, &cache, &nonfinite] {
            static_cast<void>(backward(network, cache, { 1.0 }, nonfinite));
        }, "objective backward should reject non-finite derivatives");
    }

    {
        const MLP network = make_zero_network({ 1, 1 });
        const ForwardCache cache = forward_pass(network, { 0.0 });
        ObjectiveFunctions missing_gradient = bce;
        missing_gradient.sample_loss_gradient = {};

        require_throws([&network, &cache, &missing_gradient] {
            static_cast<void>(backward(
                network,
                cache,
                { 1.0 },
                missing_gradient
            ));
        }, "objective backward should reject a missing derivative callback");
    }

    {
        MLP network = make_zero_network({ 1, 1, 1 });
        network.layers[0].weight(0, 0) = 0.4;
        network.layers[0].biases[0] = -0.1;
        network.layers[1].weight(0, 0) = 0.7;
        network.layers[1].biases[0] = -0.2;

        const ForwardCache cache = forward_pass(network, { 0.5 });
        const Values target{ 1.0 };
        const NetworkGradients legacy = backward(network, cache, target);
        const NetworkGradients objective_aware =
            backward(network, cache, target, bce);

        require_gradients_near(
            network,
            objective_aware,
            legacy,
            1e-12,
            "BCE objective-aware backward should match legacy backward"
        );
    }

    std::cout << "[PASS] objective-aware backward validation and output gradients\n";
}

void run_objective_batch_gradient_checks()
{
    const ObjectiveFunctions bce = make_binary_cross_entropy_objective();
    const Dataset batch{
        { { 0.0, 0.0 }, { 0.0, 1.0 } },
        { { 0.0, 1.0 }, { 1.0, 0.0 } },
        { { 1.0, 0.0 }, { 1.0, 1.0 } }
    };

    {
        const MLP network = make_mlp({ 2, 2, 2 }, 42);
        const MLP original = network;
        const NetworkGradients actual = batch_gradients(network, batch, bce);
        NetworkGradients expected = make_zero_gradients_like(network);

        for (const Sample& sample : batch) {
            const ForwardCache cache = forward_pass(network, sample.input);
            const NetworkGradients sample_gradients =
                backward(network, cache, sample.target, bce);
            add_gradients_in_place(expected, sample_gradients);
        }

        for (LayerGradients& layer_gradients : expected.layers) {
            for (double& gradient : layer_gradients.weights) {
                gradient /= static_cast<double>(batch.size());
            }
            for (double& gradient : layer_gradients.biases) {
                gradient /= static_cast<double>(batch.size());
            }
        }

        require_gradients_near(
            network,
            actual,
            expected,
            1e-12,
            "objective-aware batch gradients should average objective-aware sample gradients"
        );
        require(all_gradient_values_finite(actual),
            "objective-aware batch gradients should be finite");
        require(same_parameters(network, original),
            "objective-aware batch gradients should not modify network parameters");
    }

    {
        const MLP network = make_zero_network({ 1, 2 });
        const Dataset two_sample_batch{
            { { 2.0 }, { 1.0, 0.0 } },
            { { 4.0 }, { 0.0, 1.0 } }
        };
        std::size_t callback_count = 0;
        ObjectiveFunctions custom = bce;
        custom.sample_loss_gradient = [&callback_count](
            const Values& logits,
            const Values& target
        ) {
            ++callback_count;
            require(logits.size() == 2 && target.size() == 2,
                "objective-aware batch gradients should pass complete sample vectors");
            return Values{ 0.25, -0.5 };
        };

        const NetworkGradients gradients =
            batch_gradients(network, two_sample_batch, custom);

        require(callback_count == two_sample_batch.size(),
            "objective-aware batch gradients should evaluate one derivative per sample");
        require_near(gradients.layers[0].biases[0], 0.25, 1e-12,
            "objective-aware batch bias gradient should be averaged across samples");
        require_near(gradients.layers[0].biases[1], -0.5, 1e-12,
            "objective-aware second batch bias gradient should be averaged across samples");
        require_near(gradients.layers[0].weights[0], 0.75, 1e-12,
            "objective-aware batch weight gradient should average sample inputs");
        require_near(gradients.layers[0].weights[1], -1.5, 1e-12,
            "objective-aware second batch weight gradient should average sample inputs");
    }

    {
        const MLP network = make_zero_network({ 1, 1 });
        require_throws([&network, &bce] {
            static_cast<void>(batch_gradients(network, {}, bce));
        }, "objective-aware batch gradients should reject an empty batch");

        ObjectiveFunctions missing_gradient = bce;
        missing_gradient.sample_loss_gradient = {};
        require_throws([&network, &missing_gradient] {
            static_cast<void>(batch_gradients(
                network,
                { { { 0.0 }, { 1.0 } } },
                missing_gradient
            ));
        }, "objective-aware batch gradients should reject missing callbacks");
    }

    std::cout << "[PASS] objective-aware batch gradients\n";
}

void run_xor_training_checks()
{
    const ObjectiveFunctions objective = make_binary_cross_entropy_objective();
    const Dataset xor_batch = make_xor_dataset();
    MLP network = make_mlp({ 2, 4, 1 }, 42);

    const double initial_loss = batch_loss(network, xor_batch, objective);
    constexpr double learning_rate = 1.0;
    constexpr std::size_t maximum_iterations = 20000;

    for (std::size_t iteration = 0;
         iteration < maximum_iterations;
         ++iteration) {
        const NetworkGradients gradients =
            batch_gradients(network, xor_batch, objective);
        apply_gradient(network, gradients, learning_rate);
    }

    const double final_loss = batch_loss(network, xor_batch, objective);

    require(final_loss < initial_loss,
        "fixed full-batch XOR training should reduce the objective");
    require(final_loss < 0.05,
        "fixed full-batch XOR training should reach a low final loss");

    for (const Sample& sample : xor_batch) {
        const ForwardCache cache = forward_pass(network, sample.input);
        const double prediction = cache.activations.back().front();
        const bool predicted_label = prediction >= 0.5;
        const bool expected_label = sample.target.front() >= 0.5;

        require(predicted_label == expected_label,
            "fixed full-batch XOR training should classify every sample");
    }

    std::cout << "[PASS] deterministic fixed full-batch XOR training: "
              << initial_loss << " -> " << final_loss << '\n';
}

void run_batch_gradient_checks()
{
    const Dataset xor_batch = make_xor_dataset();

    {
        const MLP network = make_mlp({ 2, 4, 1 }, 42);
        const MLP original = network;
        const NetworkGradients actual = batch_gradients(network, xor_batch);
        NetworkGradients expected = make_zero_gradients_like(network);

        for (const Sample& sample : xor_batch) {
            const ForwardCache cache = forward_pass(network, sample.input);
            const NetworkGradients sample_gradients =
                backward(network, cache, sample.target);
            add_gradients_in_place(expected, sample_gradients);
        }

        for (LayerGradients& layer_gradients : expected.layers) {
            for (double& gradient : layer_gradients.weights) {
                gradient /= static_cast<double>(xor_batch.size());
            }

            for (double& gradient : layer_gradients.biases) {
                gradient /= static_cast<double>(xor_batch.size());
            }
        }

        require_gradients_near(
            network,
            actual,
            expected,
            1e-12,
            "batch gradients should equal the average of per-sample gradients"
        );
        require(all_gradient_values_finite(actual),
            "batch gradients should be finite");
        const double actual_l2_norm = gradient_l2_norm(actual);
        require_near(
            actual_l2_norm,
            check_gradient_l2_norm(actual),
            1e-12,
            "public gradient L2 norm should match the independent calculation"
        );
        require(actual_l2_norm >= 0.0 &&
                    std::isfinite(actual_l2_norm),
            "batch gradient L2 norm should be finite and nonnegative");
        require(check_maximum_absolute_gradient(actual) >= 0.0 &&
                    std::isfinite(check_maximum_absolute_gradient(actual)),
            "maximum absolute gradient should be finite and nonnegative");
        require(same_parameters(network, original),
            "batch_gradients should not modify network parameters");
    }

    require_throws([] {
        NetworkGradients gradients;
        gradients.layers.push_back({});
        gradients.layers[0].weights.push_back(
            std::numeric_limits<double>::infinity()
        );
        static_cast<void>(gradient_l2_norm(gradients));
    }, "gradient_l2_norm should reject non-finite gradients");

    {
        const MLP network = make_mlp({ 2, 4, 1 }, 42);
        Dataset reversed_batch = xor_batch;
        std::reverse(reversed_batch.begin(), reversed_batch.end());

        const NetworkGradients forward_order =
            batch_gradients(network, xor_batch);
        const NetworkGradients reverse_order =
            batch_gradients(network, reversed_batch);

        require_gradients_near(
            network,
            forward_order,
            reverse_order,
            1e-12,
            "reordering a batch should not change its average gradient"
        );
    }

    {
        const MLP network = make_mlp({ 2, 3, 3, 1 }, 42);
        const NetworkGradients gradients = batch_gradients(network, xor_batch);
        require_gradient_layout(network, gradients,
            "deeper batch gradients should match the network layout");
        require(all_gradient_values_finite(gradients),
            "deeper batch gradients should be finite");
    }

    {
        const MLP network = make_mlp({ 2, 1 }, 42);
        const NetworkGradients gradients = batch_gradients(network, xor_batch);
        require_gradient_layout(network, gradients,
            "no-hidden batch gradients should match the network layout");
        require(all_gradient_values_finite(gradients),
            "no-hidden batch gradients should be finite");
    }

    const MLP network = make_mlp({ 2, 4, 1 }, 42);
    require_throws([&network] {
        static_cast<void>(batch_gradients(network, {}));
    }, "batch_gradients should reject an empty batch");
    require_throws([&network] {
        static_cast<void>(batch_gradients(
            network,
            { { { 0.0 }, { 0.0 } } }
        ));
    }, "batch_gradients should reject an input-size mismatch");
    require_throws([&network] {
        static_cast<void>(batch_gradients(
            network,
            { { { 0.0, 0.0 }, { 0.0, 1.0 } } }
        ));
    }, "batch_gradients should reject a target-size mismatch");

    std::cout << "[PASS] full-batch gradients\n";
}

struct GradientCheckSummary {
    std::size_t checked_parameter_count{};
    double maximum_absolute_error{};
    double maximum_relative_error{};
    std::string worst_parameter;
};

double finite_difference_for_parameter(
    MLP& network,
    const Dataset& batch,
    double& parameter,
    const double epsilon
)
{
    const double original = parameter;

    try {
        parameter = original + epsilon;
        const double plus_loss = batch_loss(network, batch);

        parameter = original - epsilon;
        const double minus_loss = batch_loss(network, batch);

        parameter = original;
        return (plus_loss - minus_loss) / (2.0 * epsilon);
    }
    catch (...) {
        parameter = original;
        throw;
    }
}

void record_gradient_check_entry(
    const std::string& descriptor,
    const double analytic,
    const double numeric,
    GradientCheckSummary& summary
)
{
    require(std::isfinite(analytic),
        "analytic gradient-check values should be finite");
    require(std::isfinite(numeric),
        "numeric gradient-check values should be finite");

    const double absolute_error = std::abs(analytic - numeric);
    const double denominator = std::max(
        1.0,
        std::max(std::abs(analytic), std::abs(numeric))
    );
    const double relative_error = absolute_error / denominator;

    ++summary.checked_parameter_count;
    if (absolute_error > summary.maximum_absolute_error) {
        summary.maximum_absolute_error = absolute_error;
    }
    if (relative_error > summary.maximum_relative_error ||
        summary.checked_parameter_count == 1) {
        summary.maximum_relative_error = relative_error;
        summary.worst_parameter = descriptor;
    }

    std::cout << "[GRADIENT] " << descriptor
              << " analytic=" << analytic
              << " numeric=" << numeric
              << " abs_error=" << absolute_error
              << " rel_error=" << relative_error << '\n';
}

void run_single_gradient_check(
    const std::vector<std::size_t>& architecture,
    const std::size_t expected_parameter_count,
    const std::string_view architecture_name
)
{
    constexpr double epsilon = 1e-5;
    constexpr double tolerance = 1e-5;

    MLP network = make_mlp(architecture, 42);
    const MLP original = network;
    const Dataset xor_batch = make_xor_dataset();
    const NetworkGradients analytic = batch_gradients(network, xor_batch);
    GradientCheckSummary summary;

    for (std::size_t layer_index = 0;
         layer_index < network.layers.size();
         ++layer_index) {
        DenseLayer& layer = network.layers[layer_index];

        for (std::size_t parameter_index = 0;
             parameter_index < layer.weights.size();
             ++parameter_index) {
            const double numeric = finite_difference_for_parameter(
                network,
                xor_batch,
                layer.weights[parameter_index],
                epsilon
            );
            const std::string descriptor =
                "layer " + std::to_string(layer_index) +
                " weight[" + std::to_string(parameter_index) + "]";

            record_gradient_check_entry(
                descriptor,
                analytic.layers[layer_index].weights[parameter_index],
                numeric,
                summary
            );
        }

        for (std::size_t parameter_index = 0;
             parameter_index < layer.biases.size();
             ++parameter_index) {
            const double numeric = finite_difference_for_parameter(
                network,
                xor_batch,
                layer.biases[parameter_index],
                epsilon
            );
            const std::string descriptor =
                "layer " + std::to_string(layer_index) +
                " bias[" + std::to_string(parameter_index) + "]";

            record_gradient_check_entry(
                descriptor,
                analytic.layers[layer_index].biases[parameter_index],
                numeric,
                summary
            );
        }
    }

    require(summary.checked_parameter_count == expected_parameter_count,
        "gradient checker should visit every parameter exactly once");
    require(summary.checked_parameter_count == parameter_count(network),
        "gradient checker count should match parameter_count");
    require(summary.maximum_relative_error < tolerance,
        "analytic and numeric gradients should agree within tolerance");
    require(same_parameters(network, original),
        "gradient checker should restore every network parameter");

    std::cout << "[PASS] gradient check " << architecture_name
              << " parameters=" << summary.checked_parameter_count
              << " max_abs_error=" << summary.maximum_absolute_error
              << " max_rel_error=" << summary.maximum_relative_error
              << " worst=" << summary.worst_parameter << '\n';
}

void run_gradient_check_checks()
{
    run_single_gradient_check({ 2, 4, 1 }, 17, "{2,4,1}");
    run_single_gradient_check({ 2, 3, 3, 1 }, 25, "{2,3,3,1}");
    run_single_gradient_check({ 2, 1 }, 3, "{2,1}");
}

void run_vector_algebra_checks()
{
    const MLP network = make_zero_network({ 2, 2, 1 });
    NetworkGradients left = make_zero_gradients_like(network);
    NetworkGradients right = make_zero_gradients_like(network);

    left.layers[0].weights = { 1.0, 2.0, 3.0, 4.0 };
    left.layers[0].biases = { 5.0, 6.0 };
    left.layers[1].weights = { 7.0, 8.0 };
    left.layers[1].biases = { 9.0 };

    right.layers[0].weights = { 2.0, 3.0, 4.0, 5.0 };
    right.layers[0].biases = { 6.0, 7.0 };
    right.layers[1].weights = { 8.0, 9.0 };
    right.layers[1].biases = { 10.0 };

    require_near(network_vector_dot(left, right), 330.0, 1e-12,
        "structured dot product should sum matching weights and biases once");

    require_throws([&left] {
        NetworkGradients wrong_shape = left;
        wrong_shape.layers[0].weights.pop_back();
        static_cast<void>(network_vector_dot(left, wrong_shape));
    }, "structured dot product should reject weight shape mismatches");

    require_throws([&left] {
        NetworkGradients nonfinite = left;
        nonfinite.layers[1].biases[0] = std::numeric_limits<double>::quiet_NaN();
        static_cast<void>(network_vector_dot(left, nonfinite));
    }, "structured dot product should reject non-finite values");

    const NetworkGradients negative = make_negative_gradient_direction(left);
    require(left.layers[0].weights[0] == 1.0 &&
                left.layers[0].biases[0] == 5.0,
        "negative-gradient construction should not modify its input");
    require(negative.layers[0].weights[0] == -1.0 &&
                negative.layers[0].biases[0] == -5.0 &&
                negative.layers[1].weights[1] == -8.0,
        "negative-gradient construction should negate every component");

    const double gradient_norm = gradient_l2_norm(left);
    require_near(
        network_vector_dot(left, negative),
        -(gradient_norm * gradient_norm),
        1e-12,
        "gradient and negative-gradient dot product should equal negative norm squared"
    );
    require_near(downhill_cosine(left, negative), 1.0, 1e-12,
        "negative-gradient direction should have downhill cosine one");
    require(is_downhill_direction(left, negative, 0.0),
        "negative-gradient direction should be downhill");
    require(is_downhill_direction(left, negative, 0.99),
        "negative-gradient direction should exceed a lower cosine threshold");
    require(!is_downhill_direction(left, negative, 1.0),
        "a cosine equal to the threshold should not exceed it");

    require(!is_downhill_direction(left, left, 0.0),
        "the positive-gradient direction should be rejected as uphill");

    NetworkGradients nonfinite_bias = left;
    nonfinite_bias.layers[0].biases[0] = std::numeric_limits<double>::infinity();
    require_throws([&nonfinite_bias] {
        static_cast<void>(make_negative_gradient_direction(nonfinite_bias));
    }, "negative-gradient construction should reject non-finite biases");

    const NetworkGradients zero = make_zero_gradients_like(network);
    require_throws([&zero, &left] {
        static_cast<void>(downhill_cosine(zero, left));
    }, "downhill cosine should reject a zero gradient norm");
    require_throws([&zero, &left] {
        static_cast<void>(downhill_cosine(left, zero));
    }, "downhill cosine should reject a zero direction norm");
    require(!is_downhill_direction(zero, left, 0.0),
        "zero gradient should not be considered downhill");
    require(!is_downhill_direction(left, zero, 0.0),
        "zero direction should not be considered downhill");

    std::cout << "[PASS] structured vector algebra\n";
}

void run_gradient_application_checks()
{
    require(learning_rate_is_valid(0.1),
        "positive finite learning rate should be valid");
    require(!learning_rate_is_valid(0.0),
        "zero learning rate should be invalid for a descent step");
    require(!learning_rate_is_valid(-0.1),
        "negative learning rate should be invalid");
    require(!learning_rate_is_valid(std::numeric_limits<double>::quiet_NaN()),
        "NaN learning rate should be invalid");
    require(!learning_rate_is_valid(std::numeric_limits<double>::infinity()),
        "infinite learning rate should be invalid");

    {
        MLP network = make_zero_network({ 2, 1 });
        network.layers[0].weight(0, 0) = 1.5;
        network.layers[0].weight(0, 1) = -2.0;
        network.layers[0].biases[0] = 0.25;

        NetworkGradients gradients = make_zero_gradients_like(network);
        gradients.layers[0].weights[0] = 0.4;
        gradients.layers[0].weights[1] = -0.6;
        gradients.layers[0].biases[0] = -0.5;

        apply_gradient(network, gradients, 0.25);

        require_near(network.layers[0].weight(0, 0), 1.4, 1e-12,
            "manual weight update should use parameter minus learning rate times gradient");
        require_near(network.layers[0].weight(0, 1), -1.85, 1e-12,
            "row-major second weight update should use the matching gradient");
        require_near(network.layers[0].biases[0], 0.375, 1e-12,
            "manual bias update should use parameter minus learning rate times gradient");
    }

    {
        MLP network = make_zero_network({ 2, 2, 1 });
        NetworkGradients gradients = make_zero_gradients_like(network);

        gradients.layers[0].weights = { 0.1, 0.2, 0.3, 0.4 };
        gradients.layers[0].biases = { 0.5, 0.6 };
        gradients.layers[1].weights = { 0.7, 0.8 };
        gradients.layers[1].biases = { 0.9 };

        apply_gradient(network, gradients, 0.1);

        require_near(network.layers[0].weight(1, 1), -0.04, 1e-12,
            "gradient application should update the second hidden row correctly");
        require_near(network.layers[0].biases[1], -0.06, 1e-12,
            "gradient application should update the second hidden bias correctly");
        require_near(network.layers[1].weight(0, 1), -0.08, 1e-12,
            "gradient application should update the output row correctly");
        require_near(network.layers[1].biases[0], -0.09, 1e-12,
            "gradient application should update the output bias correctly");
    }

    {
        MLP network = make_zero_network({ 2, 1 });
        network.layers[0].weight(0, 0) = 1.0;
        network.layers[0].weight(0, 1) = -1.0;
        network.layers[0].biases[0] = 0.5;

        const MLP original = network;
        NetworkGradients gradients = make_zero_gradients_like(network);
        gradients.layers[0].weights[0] = 0.25;
        gradients.layers[0].weights[1] = -0.25;
        gradients.layers[0].biases[0] = -0.5;

        apply_gradient(network, gradients, 0.2);

        require(network.layers[0].weight(0, 0) < original.layers[0].weight(0, 0),
            "positive weight gradient should decrease its parameter");
        require(network.layers[0].weight(0, 1) > original.layers[0].weight(0, 1),
            "negative weight gradient should increase its parameter");
        require(network.layers[0].biases[0] > original.layers[0].biases[0],
            "negative bias gradient should increase its parameter");
        require(!same_parameters(network, original),
            "a nonzero gradient should change network parameters");
        require(all_parameters_finite(network),
            "ordinary gradient updates should leave parameters finite");
    }

    {
        MLP network = make_zero_network({ 2, 1 });
        const NetworkGradients gradients = make_zero_gradients_like(network);

        require_throws([&network, &gradients] {
            apply_gradient(network, gradients, 0.0);
        }, "zero learning rate should be rejected");
        require_throws([&network, &gradients] {
            apply_gradient(network, gradients, -0.1);
        }, "negative learning rate should be rejected");
    }

    {
        MLP network = make_zero_network({ 2, 1 });
        const NetworkGradients gradients = make_zero_gradients_like(network);

        require_throws([&network, &gradients] {
            apply_gradient(network, gradients, std::numeric_limits<double>::quiet_NaN());
        }, "NaN learning rate should be rejected");
        require_throws([&network, &gradients] {
            apply_gradient(network, gradients, std::numeric_limits<double>::infinity());
        }, "infinite learning rate should be rejected");
    }

    {
        MLP network = make_zero_network({ 2, 1 });

        NetworkGradients missing_layer = make_zero_gradients_like(network);
        missing_layer.layers.pop_back();
        require_throws([&network, &missing_layer] {
            apply_gradient(network, missing_layer, 0.1);
        }, "gradient layer-count mismatch should be rejected");

        NetworkGradients wrong_weight_shape = make_zero_gradients_like(network);
        wrong_weight_shape.layers[0].weights.pop_back();
        require_throws([&network, &wrong_weight_shape] {
            apply_gradient(network, wrong_weight_shape, 0.1);
        }, "gradient weight shape mismatch should be rejected");

        NetworkGradients wrong_bias_shape = make_zero_gradients_like(network);
        wrong_bias_shape.layers[0].biases.push_back(0.0);
        require_throws([&network, &wrong_bias_shape] {
            apply_gradient(network, wrong_bias_shape, 0.1);
        }, "gradient bias shape mismatch should be rejected");
    }

    {
        MLP network = make_zero_network({ 2, 1 });
        NetworkGradients nan_gradient = make_zero_gradients_like(network);
        nan_gradient.layers[0].weights[0] = std::numeric_limits<double>::quiet_NaN();
        require_throws([&network, &nan_gradient] {
            apply_gradient(network, nan_gradient, 0.1);
        }, "NaN gradient should be rejected");

        NetworkGradients infinite_gradient = make_zero_gradients_like(network);
        infinite_gradient.layers[0].biases[0] = std::numeric_limits<double>::infinity();
        require_throws([&network, &infinite_gradient] {
            apply_gradient(network, infinite_gradient, 0.1);
        }, "infinite gradient should be rejected");
    }

    {
        MLP network = make_zero_network({ 1, 1 });
        network.layers[0].weights[0] = std::numeric_limits<double>::max();
        NetworkGradients gradients = make_zero_gradients_like(network);
        gradients.layers[0].weights[0] = -std::numeric_limits<double>::max();
        const MLP original = network;

        require_throws([&network, &gradients] {
            apply_gradient(network, gradients, 1.0);
        }, "an update that produces a non-finite parameter should be rejected");
        require(same_parameters(network, original),
            "a rejected update should not partially modify the network");
    }

    {
        MLP network = make_zero_network({ 1, 1, 1 });
        network.layers[1].weights[0] = std::numeric_limits<double>::max();
        NetworkGradients gradients = make_zero_gradients_like(network);
        gradients.layers[0].weights[0] = 1.0;
        gradients.layers[1].weights[0] = -std::numeric_limits<double>::max();
        const MLP original = network;

        require_throws([&network, &gradients] {
            apply_gradient(network, gradients, 1.0);
        }, "a later overflow should reject the whole update");
        require(same_parameters(network, original),
            "a later overflow should not leave earlier layers updated");
    }

    std::cout << "[PASS] gradient application checks\n";
}

void run_direction_application_checks()
{
    MLP network = make_zero_network({ 2, 1 });
    network.layers[0].weights = { 1.0, -2.0 };
    network.layers[0].biases = { 0.5 };

    NetworkDirection direction = make_zero_gradients_like(network);
    direction.layers[0].weights = { 0.4, -0.6 };
    direction.layers[0].biases = { -0.5 };

    const MLP original = network;
    apply_direction(network, direction, 0.25);

    require_near(network.layers[0].weights[0], 1.1, 1e-12,
        "direction application should add scale times the first weight direction");
    require_near(network.layers[0].weights[1], -2.15, 1e-12,
        "direction application should add scale times the second weight direction");
    require_near(network.layers[0].biases[0], 0.375, 1e-12,
        "direction application should add scale times the bias direction");

    const MLP candidate = make_candidate_network(original, direction, 0.25);
    require_near(candidate.layers[0].weights[0], 1.1, 1e-12,
        "candidate construction should apply the direction to a copy");
    require_near(candidate.layers[0].weights[1], -2.15, 1e-12,
        "candidate construction should preserve row-major direction indexing");
    require_near(candidate.layers[0].biases[0], 0.375, 1e-12,
        "candidate construction should update biases");
    require(original.layers[0].weights[0] == 1.0 &&
                original.layers[0].weights[1] == -2.0 &&
                original.layers[0].biases[0] == 0.5,
        "candidate construction should not modify the original network");

    {
        NetworkDirection wrong_shape = direction;
        wrong_shape.layers.pop_back();
        require_throws([&original, &wrong_shape] {
            MLP copy = original;
            apply_direction(copy, wrong_shape, 0.25);
        }, "direction application should reject a layer-count mismatch");
    }

    {
        MLP overflow_network = make_zero_network({ 1, 1 });
        overflow_network.layers[0].weights[0] = std::numeric_limits<double>::max();
        NetworkDirection overflow_direction = make_zero_gradients_like(overflow_network);
        overflow_direction.layers[0].weights[0] = 1.0;
        const MLP overflow_original = overflow_network;

        require_throws([&overflow_network, &overflow_direction] {
            apply_direction(
                overflow_network,
                overflow_direction,
                std::numeric_limits<double>::max()
            );
        }, "direction application should reject non-finite updated parameters");
        require(same_parameters(overflow_network, overflow_original),
            "rejected direction application should be transactional");
    }

    std::cout << "[PASS] structured direction application and candidates\n";
}

void run_wolfe_parameter_checks()
{
    const WolfeParameters defaults{};

    require(wolfe_parameters_are_valid(defaults),
        "default Wolfe parameters should be valid");
    validate_wolfe_parameters(defaults);

    {
        WolfeParameters invalid = defaults;
        invalid.sufficient_decrease = 0.0;
        require_invalid_wolfe_parameters(
            invalid,
            "zero sufficient-decrease constant should be rejected"
        );
    }

    {
        WolfeParameters invalid = defaults;
        invalid.sufficient_decrease = 0.95;
        invalid.curvature = 0.9;
        require_invalid_wolfe_parameters(
            invalid,
            "sufficient-decrease constant must be below curvature constant"
        );
    }

    {
        WolfeParameters invalid = defaults;
        invalid.curvature = 1.0;
        require_invalid_wolfe_parameters(
            invalid,
            "curvature constant equal to one should be rejected"
        );
    }

    {
        WolfeParameters invalid = defaults;
        invalid.shrink_factor = 0.0;
        require_invalid_wolfe_parameters(
            invalid,
            "zero shrink factor should be rejected"
        );
    }

    {
        WolfeParameters invalid = defaults;
        invalid.shrink_factor = 1.0;
        require_invalid_wolfe_parameters(
            invalid,
            "unit shrink factor should be rejected"
        );
    }

    {
        WolfeParameters invalid = defaults;
        invalid.maximum_step = 0.0;
        require_invalid_wolfe_parameters(
            invalid,
            "zero maximum step should be rejected"
        );
    }

    {
        WolfeParameters invalid = defaults;
        invalid.maximum_trials = 0;
        require_invalid_wolfe_parameters(
            invalid,
            "zero maximum trials should be rejected"
        );
    }

    {
        WolfeParameters invalid = defaults;
        invalid.minimum_downhill_cosine = 1.1;
        require_invalid_wolfe_parameters(
            invalid,
            "downhill-cosine threshold above one should be rejected"
        );
    }

    std::cout << "[PASS] Wolfe parameter validation\n";
}

void run_wolfe_scalar_checks()
{
    {
        const bool accepted = satisfies_sufficient_decrease(
            0.8,
            1.0,
            0.5,
            -2.0,
            0.1
        );
        require(accepted,
            "Armijo should accept a candidate satisfying sufficient decrease");
    }

    {
        const bool accepted = satisfies_sufficient_decrease(
            0.95,
            1.0,
            0.5,
            -2.0,
            0.1
        );
        require(!accepted,
            "Armijo should reject a candidate failing sufficient decrease");
    }

    {
        require_throws([] {
            static_cast<void>(satisfies_sufficient_decrease(
                0.8,
                1.0,
                0.5,
                0.0,
                0.1
            ));
        }, "Armijo should reject a non-downhill initial derivative");

        require_throws([] {
            static_cast<void>(satisfies_sufficient_decrease(
                0.8,
                1.0,
                0.0,
                -2.0,
                0.1
            ));
        }, "Armijo should reject a non-positive step size");

        require_throws([] {
            static_cast<void>(satisfies_sufficient_decrease(
                std::numeric_limits<double>::quiet_NaN(),
                1.0,
                0.5,
                -2.0,
                0.1
            ));
        }, "Armijo should reject a non-finite candidate loss");
    }

    {
        const bool accepted = satisfies_strong_curvature(
            1.0,
            -2.0,
            0.9
        );
        require(accepted,
            "strong curvature should accept a candidate within the curvature bound");
    }

    {
        const bool accepted = satisfies_strong_curvature(
            1.9,
            -2.0,
            0.9
        );
        require(!accepted,
            "strong curvature should reject a candidate outside the curvature bound");
    }

    {
        require_throws([] {
            static_cast<void>(satisfies_strong_curvature(
                1.0,
                0.0,
                0.9
            ));
        }, "strong curvature should reject a non-downhill initial derivative");

        require_throws([] {
            static_cast<void>(satisfies_strong_curvature(
                1.0,
                -2.0,
                1.0
            ));
        }, "strong curvature should reject a curvature constant equal to one");

        require_throws([] {
            static_cast<void>(satisfies_strong_curvature(
                std::numeric_limits<double>::infinity(),
                -2.0,
                0.9
            ));
        }, "strong curvature should reject a non-finite candidate derivative");
    }

    {
        WolfeEvaluation evaluation;
        evaluation.sufficient_decrease = true;
        evaluation.strong_curvature = true;
        require(satisfies_strong_wolfe(evaluation),
            "strong Wolfe should require both scalar conditions");

        evaluation.strong_curvature = false;
        require(!satisfies_strong_wolfe(evaluation),
            "strong Wolfe should reject a failed curvature condition");
    }

    std::cout << "[PASS] Armijo and Strong-Wolfe scalar checks\n";
}

void run_wolfe_candidate_checks()
{
    const MLP network = make_mlp({ 2, 4, 1 }, 42);
    const Dataset batch = make_xor_dataset();
    const WolfeParameters parameters{};
    const ObjectiveFunctions objective = make_binary_cross_entropy_objective();
    const double current_loss = batch_loss(network, batch, objective);
    const NetworkGradients current_gradient =
        batch_gradients(network, batch, objective);
    const NetworkDirection direction =
        make_negative_gradient_direction(current_gradient);
    constexpr double step_size = 0.001;
    const MLP original = network;

    const WolfeEvaluation evaluation = evaluate_wolfe_candidate(
        network,
        batch,
        current_loss,
        current_gradient,
        direction,
        step_size,
        parameters,
        objective
    );

    const MLP expected_candidate =
        make_candidate_network(network, direction, step_size);
    const double expected_loss =
        batch_loss(expected_candidate, batch, objective);
    const NetworkGradients expected_gradient =
        batch_gradients(expected_candidate, batch, objective);
    const double expected_initial_slope =
        network_vector_dot(current_gradient, direction);
    const double expected_candidate_slope =
        network_vector_dot(expected_gradient, direction);

    require(same_parameters(evaluation.candidate_network, expected_candidate),
        "Wolfe candidate evaluation should construct the requested candidate");
    require_near(evaluation.candidate_loss, expected_loss, 1e-12,
        "Wolfe candidate evaluation should use the candidate loss");
    require_gradients_near(
        expected_candidate,
        evaluation.candidate_gradient,
        expected_gradient,
        1e-12,
        "Wolfe candidate evaluation should use the candidate gradient"
    );
    require_near(evaluation.initial_directional_derivative,
        expected_initial_slope, 1e-12,
        "Wolfe candidate evaluation should report the initial slope");
    require_near(evaluation.candidate_directional_derivative,
        expected_candidate_slope, 1e-12,
        "Wolfe candidate evaluation should report the candidate slope");
    require(
        evaluation.sufficient_decrease == satisfies_sufficient_decrease(
            expected_loss,
            current_loss,
            step_size,
            expected_initial_slope,
            parameters.sufficient_decrease
        ),
        "Wolfe candidate evaluation should report the Armijo result"
    );
    require(
        evaluation.strong_curvature == satisfies_strong_curvature(
            expected_candidate_slope,
            expected_initial_slope,
            parameters.curvature
        ),
        "Wolfe candidate evaluation should report the curvature result"
    );
    require(same_parameters(network, original),
        "Wolfe candidate evaluation should not mutate the current network");

    bool found_curvature_satisfied_trial = false;
    for (const double trial_step : Values{ 0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0 }) {
        const WolfeEvaluation trial = evaluate_wolfe_candidate(
            network,
            batch,
            current_loss,
            current_gradient,
            direction,
            trial_step,
            parameters,
            objective
        );
        const bool expected_curvature = satisfies_strong_curvature(
            trial.candidate_directional_derivative,
            trial.initial_directional_derivative,
            parameters.curvature
        );

        if (expected_curvature) {
            found_curvature_satisfied_trial = true;
            require(trial.strong_curvature,
                "Wolfe candidate evaluation should use the candidate slope for curvature");
            break;
        }
    }

    require(found_curvature_satisfied_trial,
        "candidate-evaluation test data should include a curvature-satisfying trial");

    {
        const MLP custom_network = make_zero_network({ 1, 1 });
        const Dataset custom_batch{
            { { 1.0 }, { 1.0 } }
        };
        ObjectiveFunctions custom = make_binary_cross_entropy_objective();
        custom.sample_loss = [](const Values&, const Values&) {
            return 0.5;
        };
        custom.sample_loss_gradient = [](
            const Values& logits,
            const Values& target
        ) {
            static_cast<void>(target);
            return Values{ logits.front() < -0.1 ? 0.0 : 0.25 };
        };

        const double custom_loss =
            batch_loss(custom_network, custom_batch, custom);
        const NetworkGradients custom_gradient =
            batch_gradients(custom_network, custom_batch, custom);
        const NetworkDirection custom_direction =
            make_negative_gradient_direction(custom_gradient);
        const WolfeEvaluation custom_evaluation =
            evaluate_wolfe_candidate(
                custom_network,
                custom_batch,
                custom_loss,
                custom_gradient,
                custom_direction,
                1.0,
                parameters,
                custom
            );

        require(
            custom_evaluation.strong_curvature ==
                satisfies_strong_curvature(
                    custom_evaluation.candidate_directional_derivative,
                    custom_evaluation.initial_directional_derivative,
                    parameters.curvature
                ),
            "Wolfe curvature should use the supplied custom objective gradient"
        );
    }

    {
        WolfeParameters invalid = parameters;
        invalid.shrink_factor = 1.0;
        require_throws([&network, &batch, &current_loss, &current_gradient,
                        &direction, &objective, &invalid] {
            static_cast<void>(evaluate_wolfe_candidate(
                network,
                batch,
                current_loss,
                current_gradient,
                direction,
                step_size,
                invalid,
                objective
            ));
        }, "Wolfe candidate evaluation should validate Wolfe parameters");
    }

    require_throws([&network, &batch, &current_loss, &current_gradient, &parameters, &objective, step_size] {
        const NetworkDirection uphill = current_gradient;
        static_cast<void>(evaluate_wolfe_candidate(
            network,
            batch,
            current_loss,
            current_gradient,
            uphill,
            step_size,
            parameters,
            objective
        ));
    }, "Wolfe candidate evaluation should reject an uphill direction");

    std::cout << "[PASS] Wolfe candidate evaluation\n";
}

void run_backtracking_wolfe_checks()
{
    const MLP network = make_mlp({ 2, 4, 1 }, 42);
    const MLP original = network;
    const Dataset batch = make_xor_dataset();
    const WolfeParameters parameters{};
    const ObjectiveFunctions objective = make_binary_cross_entropy_objective();
    const double current_loss = batch_loss(network, batch, objective);
    const NetworkGradients current_gradient =
        batch_gradients(network, batch, objective);
    const NetworkDirection direction =
        make_negative_gradient_direction(current_gradient);

    const LineSearchResult result = backtracking_wolfe_stepsize(
        network,
        batch,
        current_loss,
        current_gradient,
        direction,
        parameters,
        objective
    );

    require(!result.history.empty(),
        "Backtracking Wolfe should record every trial");
    require(result.history.size() <= parameters.maximum_trials,
        "Backtracking Wolfe should respect the maximum trial count");
    require_near(result.history.front().step_size, parameters.maximum_step,
        1e-12, "Backtracking Wolfe should start at the maximum step");

    for (std::size_t i = 1; i < result.history.size(); ++i) {
        require_near(
            result.history[i].step_size,
            result.history[i - 1].step_size * parameters.shrink_factor,
            1e-12,
            "Backtracking Wolfe should shrink the step by the configured factor"
        );
    }

    if (result.status == LineSearchStatus::StrongWolfeSatisfied ||
        result.status == LineSearchStatus::SufficientDecreaseFallback) {
        require(result.step_size > 0.0,
            "An accepted Wolfe result should have a positive step size");
        require(result.selected_loss <= current_loss,
            "An accepted Wolfe result should not increase the loss");
        validate_gradients_like(network, result.selected_gradient);
        require(!same_parameters(result.selected_network, original),
            "An accepted Wolfe result should select an updated network");
    }
    else {
        require(result.step_size == 0.0,
            "A failed Wolfe search should not report an accepted step");
        require(same_parameters(result.selected_network, original),
            "A failed Wolfe search should preserve the current network");
    }

    require(same_parameters(network, original),
        "Backtracking Wolfe should not mutate the current network");

    require_throws([&network, &batch, &current_loss, &current_gradient,
                    &parameters, &objective] {
        static_cast<void>(backtracking_wolfe_stepsize(
            network,
            batch,
            current_loss,
            current_gradient,
            current_gradient,
            parameters,
            objective
        ));
    }, "Backtracking Wolfe should reject an uphill direction");

    {
        WolfeParameters invalid = parameters;
        invalid.maximum_trials = 0;
        require_throws([&network, &batch, &current_loss, &current_gradient,
                        &direction, &invalid, &objective] {
            static_cast<void>(backtracking_wolfe_stepsize(
                network,
                batch,
                current_loss,
                current_gradient,
                direction,
                invalid,
                objective
            ));
        }, "Backtracking Wolfe should validate the trial limit");
    }

    std::cout << "[PASS] backtracking Wolfe step search\n";
}
}

// -----------------------------------------------------------------------------
// Phase 1 exercise skeletons
// -----------------------------------------------------------------------------
// These are intentionally documentation-only scaffolds. They identify the
// checks to implement while the corresponding declarations in mlp.hpp remain
// unimplemented. Do not add them to run_all_self_checks until their behavior
// has been implemented in mlp.cpp.

void phase_one_objective_config_skeleton()
{
    // TODO: verify that make_objective_config stores the data objective and
    // regularization terms, and that validate_objective_config rejects empty
    // callbacks, invalid coefficients, and malformed regularizers.
}

void phase_one_regularization_skeleton()
{
    // The include_biases argument in each regularization factory controls the
    // parameter set being regularized:
    //
    //   false (the default): regularize weights only;
    //   true:                regularize weights and biases.
    //
    // The RegularizationTerm::includes_biases metadata should mirror that
    // choice. The const on the current bool value parameter only prevents
    // reassigning the local copy inside the factory; it does not change the
    // caller's value or make the setting immutable after construction.

    // Suggested shared fixture:
    //   MLP network = make_zero_network({ 2, 2, 1 });
    //   assign distinct finite nonzero weights and biases to every layer;
    //   NetworkGradients gradients = make_zero_gradients_like(network);
    //
    // L2 skeleton:
    //   - make_l2_regularization(coefficient, false) has the expected name,
    //     smooth=true, includes_biases=false, and coefficient metadata.
    //   - Its value callback includes weights but excludes biases.
    //   - Its gradient callback adds the unscaled weight derivative and leaves
    //     bias gradients unchanged; aggregation applies the coefficient.
    //   - Repeat with include_biases=true and verify bias loss/gradients.
    //   - Reject negative, NaN, and infinite coefficients.

    // L1 skeleton:
    //   - smooth=false and the include_biases metadata is preserved.
    //   - Verify sign-subgradient behavior for negative, positive, and zero
    //     parameters; the derivative at exactly zero is defined as zero.
    //   - Verify weights-only and weights-plus-biases variants.
    //   - Reject negative, NaN, and infinite coefficients.

    // Elastic Net skeleton:
    //   - Store and validate both nonnegative coefficients.
    //   - Verify value equals the L1 contribution plus the L2 contribution.
    //   - Verify gradient equals the corresponding sum of subgradients.
    //   - Mark the combined term nonsmooth because it contains L1.
    //   - Verify include_biases behavior for both components.

    // Composition/contract skeleton:
    //   - make_objective_config stores the data objective and regularizers.
    //   - validate_objective_config rejects missing callbacks, missing
    //     regularization callbacks, empty names, invalid coefficients, and
    //     malformed callback results.
    //   - regularization_loss is added once after batch data loss averaging,
    //     not once per sample and not divided by batch size.
    //   - add_regularization_gradients adds each regularizer once while
    //     preserving the network-gradient shape and finite-value contract.
    //   - objective_loss and objective_gradients agree with those components.
}

void phase_one_optimizer_spec_skeleton()
{
    bool factory_called = false;

    OptimizerFactory factory = [&factory_called] {
        factory_called = true;
        return std::make_unique<TestOptimizer>();
    };

    const OptimizerSpec custom = make_custom_optimizer(
        "TestOptimizer",
        OptimizerRequirement::MiniBatchCompatible,
        factory
    );

    require(custom.name == "TestOptimizer",
        "custom optimizer should preserve its name");
    require(
        custom.requirement == OptimizerRequirement::MiniBatchCompatible,
        "custom optimizer should preserve its requirement"
    );
    require(static_cast<bool>(custom.make),
        "custom optimizer should preserve its factory");
    require(!factory_called,
        "validating an optimizer spec should not eagerly invoke its factory");

    validate_optimizer_spec(custom);

    std::unique_ptr<Optimizer> instance = custom.make();

    require(factory_called,
        "custom optimizer factory should be invoked when requested");
    require(instance != nullptr,
        "custom optimizer factory should produce an optimizer instance");
    require(std::string(instance->name()) == "TestOptimizer",
        "custom optimizer factory should produce the expected optimizer");

    require_throws([] {
        static_cast<void>(make_custom_optimizer(
            "",
            OptimizerRequirement::MiniBatchCompatible,
            [] {
                return std::make_unique<TestOptimizer>();
            }
        ));
    }, "empty custom optimizer names should be rejected");

    require_throws([] {
        static_cast<void>(make_custom_optimizer(
            "MissingFactory",
            OptimizerRequirement::MiniBatchCompatible,
            {}
        ));
    }, "empty custom optimizer factories should be rejected");

    require_throws([] {
        static_cast<void>(make_custom_optimizer(
            "InvalidRequirement",
            static_cast<OptimizerRequirement>(-1),
            [] {
                return std::make_unique<TestOptimizer>();
            }
        ));
    }, "invalid optimizer requirements should be rejected");
}

void phase_one_training_config_skeleton()
{
    TrainingConfig full_batch{};
    validate_training_config(full_batch);

    require(full_batch.epochs == 1,
        "default training configuration should use one epoch");
    require(full_batch.batch_size == 0,
        "zero batch size should represent full-dataset batches");
    require(full_batch.gradient_tolerance == 0.0,
        "zero gradient tolerance should disable early stopping");

    TrainingConfig mini_batch{};
    mini_batch.epochs = 5;
    mini_batch.batch_size = 4;
    mini_batch.shuffle = true;
    mini_batch.shuffle_seed = 1234;
    mini_batch.gradient_tolerance = 1e-6;

    validate_training_config(mini_batch);

    TrainingConfig zero_epochs{};
    zero_epochs.epochs = 0;

    require_throws([&zero_epochs] {
        validate_training_config(zero_epochs);
    }, "zero training epochs should be rejected");

    TrainingConfig negative_tolerance{};
    negative_tolerance.gradient_tolerance = -1.0;

    require_throws([&negative_tolerance] {
        validate_training_config(negative_tolerance);
    }, "negative gradient tolerance should be rejected");

    TrainingConfig nan_tolerance{};
    nan_tolerance.gradient_tolerance =
        std::numeric_limits<double>::quiet_NaN();

    require_throws([&nan_tolerance] {
        validate_training_config(nan_tolerance);
    }, "NaN gradient tolerance should be rejected");

    TrainingConfig infinite_tolerance{};
    infinite_tolerance.gradient_tolerance =
        std::numeric_limits<double>::infinity();

    require_throws([&infinite_tolerance] {
        validate_training_config(infinite_tolerance);
    }, "infinite gradient tolerance should be rejected");
}

void run_phase_one_stub_checks()
{
    const ObjectiveConfig default_objective = make_objective_config();
    validate_objective_config(default_objective);

    require_throws([] {
        static_cast<void>(make_elastic_net_regularization(0.1, 0.1));
    }, "Elastic Net should remain an explicit exercise stub");

    require_throws([] {
        validate_objective_config({});
    }, "objective-config validation should remain an explicit exercise stub");

    require_throws([] {
        static_cast<void>(regularization_loss(
            make_zero_network({ 1, 1 }),
            ObjectiveConfig{}
        ));
    }, "regularization loss should remain an explicit exercise stub");

    require_throws([] {
        NetworkGradients gradients;
        add_regularization_gradients(
            make_zero_network({ 1, 1 }),
            ObjectiveConfig{},
            gradients
        );
    }, "regularization gradients should remain an explicit exercise stub");

    require_throws([] {
        static_cast<void>(objective_loss(
            make_zero_network({ 1, 1 }),
            Dataset{ { { 0.0 }, { 0.0 } } },
            ObjectiveConfig{}
        ));
    }, "objective loss should remain an explicit exercise stub");

    require_throws([] {
        static_cast<void>(objective_gradients(
            make_zero_network({ 1, 1 }),
            Dataset{ { { 0.0 }, { 0.0 } } },
            ObjectiveConfig{}
        ));
    }, "objective gradients should remain an explicit exercise stub");

    require_throws([] {
        static_cast<void>(make_custom_optimizer(
            "exercise",
            OptimizerRequirement::MiniBatchCompatible,
            {}
        ));
    }, "custom optimizer construction should remain an explicit exercise stub");

    phase_one_optimizer_spec_skeleton();
    phase_one_training_config_skeleton();

    validate_optimizer_spec(Optimizers::SGD);
    require(static_cast<bool>(Optimizers::SGD.make),
        "implemented SGD should expose a factory");

    std::unique_ptr<Optimizer> sgd = Optimizers::SGD.make();
    require(sgd != nullptr,
        "built-in SGD factory should create an optimizer");
    require(std::string(sgd->name()) == "SGD",
        "built-in SGD should report its name");

    require(!Optimizers::AdaGrad.make,
        "AdaGrad exercise spec should expose an empty factory until implemented");
    require(!Optimizers::RMSProp.make,
        "RMSProp exercise spec should expose an empty factory until implemented");
    require(!Optimizers::Adam.make,
        "Adam exercise spec should expose an empty factory until implemented");
    require(!Optimizers::AdamW.make,
        "AdamW exercise spec should expose an empty factory until implemented");
    require(!Optimizers::LBFGS.make,
        "LBFGS exercise spec should expose an empty factory until implemented");

    std::cout << "[SCAFFOLD] Remaining Phase 1 API stubs are linkable and intentionally unimplemented\n";
}

void run_all_self_checks()
{
    run_architecture_checks();
    run_parameter_count_checks();
    run_initialization_checks();
    run_stable_sigmoid_checks();
    run_forward_cache_shape_checks();
    run_forward_value_checks();
    run_binary_cross_entropy_scalar_checks();
    run_sample_cost_checks();
    run_batch_loss_checks();
    run_objective_checks();
    run_mse_objective_checks();
    run_sgd_optimizer_checks();
    run_regularization_coefficient_checks();
    run_zero_gradient_checks();
    run_backward_checks();
    run_objective_backward_checks();
    run_objective_batch_gradient_checks();
    run_xor_training_checks();
    run_batch_gradient_checks();
    run_gradient_check_checks();
    run_vector_algebra_checks();
    run_gradient_application_checks();
    run_direction_application_checks();
    run_wolfe_parameter_checks();
    run_wolfe_scalar_checks();
    run_wolfe_candidate_checks();
    run_backtracking_wolfe_checks();
    run_phase_one_stub_checks();
}
