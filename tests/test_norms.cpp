#include <nablanet/nablanet.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace nablanet;

namespace {
void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, const char* message)
{
    require(std::isfinite(actual), message);
    if (expected == 0.0) {
        require(actual == 0.0, message);
    } else {
        require(std::abs(actual / expected - 1.0) <= 1e-12, message);
    }
}

template<class Exception, class Action>
void throws(Action action, const char* message)
{
    try {
        action();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_vectors()
{
    const double infinity = std::numeric_limits<double>::infinity();
    const auto l2 = make_lp_norm(2.0);
    near(vector_norm({ -3.0, 4.0 }, make_lp_norm(1.0)), 7.0, "L1");
    near(vector_norm({ -3.0, 4.0 }, l2), 5.0, "L2 is not squared L2");
    near(vector_norm({ -3.0, 4.0 }, make_lp_norm(infinity)), 4.0, "L infinity");
    near(vector_norm({ -8.0, 1.0 }, make_lp_norm(1.5)),
        std::pow(std::pow(8.0, 1.5) + 1.0, 2.0 / 3.0), "fractional p");

    for (double p : { 1.0, 1.5, 2.0, 3.0, 1000.0, infinity }) {
        const auto norm = make_lp_norm(p);
        near(vector_norm({}, norm), 0.0, "empty vector");
        near(norm.evaluate({}), 0.0, "empty direct callback");
        near(vector_norm({ 0.0, -0.0 }, norm), 0.0, "zero vector");
        near(vector_norm({ -3.0, 4.0 }, norm),
            vector_norm({ 3.0, -4.0 }, norm), "sign symmetry");
        near(vector_norm({ -6.0, 8.0 }, norm),
            2.0 * vector_norm({ -3.0, 4.0 }, norm), "homogeneity");
        const double xy = vector_norm({ 2.0, 6.0 }, norm);
        require(xy <= vector_norm({ 3.0, 4.0 }, norm) +
            vector_norm({ -1.0, 2.0 }, norm) + 1e-12, "triangle inequality");
    }

    near(vector_norm({ 3e200, 4e200 }, l2), 5e200, "large representable norm");
    near(vector_norm({ 3e-200, 4e-200 }, l2), 5e-200, "small representable norm");
    near(vector_norm({ 1e300, 1e-300 }, l2), 1e300, "mixed magnitudes");
    const double tiny = std::numeric_limits<double>::denorm_min();
    require(vector_norm({ tiny }, l2) == tiny, "subnormal singleton");
    const double largest = std::numeric_limits<double>::max();
    require(vector_norm({ largest }, l2) == largest, "maximum singleton");
    near(vector_norm({ 1.0, 1.0 }, make_lp_norm(largest)), 1.0, "very large p");
    throws<std::overflow_error>([&] { vector_norm({ largest, largest }, l2); },
        "true overflow must be reported");
    throws<std::overflow_error>([&] {
        vector_norm({ largest, largest }, make_lp_norm(1.0));
    }, "L1 overflow must be reported");

    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (double p : { -infinity, -1.0, 0.0, 0.5, nan }) {
        throws<std::invalid_argument>([&] { make_lp_norm(p); }, "invalid p");
    }
    for (double value : { infinity, -infinity, nan }) {
        throws<std::invalid_argument>([&] { vector_norm({ value }, l2); },
            "nonfinite input");
        throws<std::invalid_argument>([&] { l2.evaluate({ value }); },
            "direct factory callback validates input");
    }
}

void test_contract_and_layout()
{
    const auto l1 = make_lp_norm(1.0);
    const auto l2 = make_lp_norm(2.0);
    throws<std::invalid_argument>([] { vector_norm({}, {}); }, "missing callback");
    throws<std::invalid_argument>([] {
        validate_norm_function({ "", [](const Values&) { return 0.0; } });
    }, "missing name");
    throws<std::invalid_argument>([] {
        validate_norm_function({ "missing", {} });
    }, "named missing callback");
    for (double result : { -1.0, -std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::quiet_NaN() }) {
        throws<std::invalid_argument>([&] {
            vector_norm({ 1.0 }, { "invalid", [result](const Values&) { return result; } });
        }, "invalid custom result");
    }
    throws<std::overflow_error>([] {
        vector_norm({ 1.0 }, { "overflow", [](const Values&) {
            return std::numeric_limits<double>::infinity();
        } });
    }, "custom overflow");

    Values observed;
    int calls = 0;
    const NormFunction custom{ "custom L1", [&](const Values& values) {
        ++calls;
        observed = values;
        double sum = 0.0;
        for (double value : values) { sum += std::abs(value); }
        return sum;
    } };
    near(vector_norm({}, custom), 0.0, "custom empty input");
    require(calls == 0, "empty input bypasses callback");
    throws<std::invalid_argument>([&] {
        vector_norm({ std::numeric_limits<double>::infinity() }, custom);
    }, "validate before custom callback");
    require(calls == 0, "invalid input must not reach callback");

    MLP network = make_zero_network({ 2, 1, 1 });
    network.layers[0].weights = { -1.0, 2.0 };
    network.layers[0].biases = { 3.0 };
    network.layers[1].weights = { 4.0 };
    network.layers[1].biases = { -5.0 };
    near(network_norm(network, custom), 15.0, "all parameters");
    require(observed == Values({ -1, 2, 3, 4, -5 }) && calls == 1,
        "one global vector, layer/weight/bias order");
    near(network_norm(network, custom, false), 7.0, "weight-only parameters");
    require(observed == Values({ -1, 2, 4 }), "bias exclusion order");
    near(network_norm(network, l2), std::sqrt(55.0), "global L2, not sum of layer norms");
    require(network.layers[0].weights == Values({ -1, 2 }) &&
        network.layers[1].biases == Values({ -5 }), "network stays unchanged");

    NetworkGradients gradients;
    gradients.layers = { { { -1, 2 }, { 3 } }, { { 4 }, { -5 } } };
    near(network_norm(gradients, custom), 15.0, "gradient components");
    require(observed == Values({ -1, 2, 3, 4, -5 }), "gradient flattening order");
    near(network_norm(gradients, custom, false), 7.0, "weight-only gradients");
    require(observed == Values({ -1, 2, 4 }), "gradient bias exclusion order");
    near(network_norm(gradients, l2), gradient_l2_norm(gradients), "legacy L2 agreement");
    near(network_norm(gradients, make_lp_norm(std::numeric_limits<double>::infinity())),
        maximum_absolute_gradient(gradients), "legacy infinity agreement");
    near(network_norm(NetworkGradients{}, l1), 0.0, "empty gradients");
    gradients.layers[0].biases[0] = std::numeric_limits<double>::infinity();
    throws<std::invalid_argument>([&] { network_norm(gradients, l1); }, "invalid gradient");
    near(network_norm(gradients, l1, false), 7.0, "excluded gradient bias not inspected");
    gradients.layers[0].weights[0] = std::numeric_limits<double>::quiet_NaN();
    throws<std::invalid_argument>([&] { network_norm(gradients, l1, false); }, "invalid weight gradient");

    network.layers[0].biases[0] = std::numeric_limits<double>::infinity();
    throws<std::invalid_argument>([&] { network_norm(network, l1, false); },
        "whole network is validated even if biases are excluded");
    network.layers[0].biases[0] = 3.0;
    network.layers[0].weights.pop_back();
    throws<std::invalid_argument>([&] { network_norm(network, l1); }, "malformed network");
    throws<std::invalid_argument>([&] { network_norm(MLP{}, l1); }, "empty network is invalid");
}
} // namespace

int main()
{
    try {
        test_vectors();
        test_contract_and_layout();
        std::cout << "[PASS] standalone norms and contract adapters\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
