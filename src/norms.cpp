#include <nablanet/norms.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace nablanet {
namespace {

void validate_values(const Values& values)
{
    for (const double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Norm inputs must be finite.");
        }
    }
}

double evaluate_lp(const Values& values, const double p)
{
    validate_values(values);
    double maximum = 0.0;
    for (const double value : values) {
        maximum = std::max(maximum, std::abs(value));
    }
    if (maximum == 0.0 || std::isinf(p)) {
        return maximum;
    }

    // Scale before exponentiation: neither |x|^p nor the sum of raw powers
    // needs to be representable. Wider accumulation helps where supported.
    long double sum = 0.0L;
    for (const double value : values) {
        const long double ratio =
            std::abs(static_cast<long double>(value)) / maximum;
        sum += std::pow(ratio, static_cast<long double>(p));
    }
    const long double result = static_cast<long double>(maximum) *
        std::pow(sum, 1.0L / static_cast<long double>(p));
    if (!std::isfinite(result) ||
        result > std::numeric_limits<double>::max()) {
        throw std::overflow_error("L-p norm is not representable as a double.");
    }
    return static_cast<double>(result);
}

} // namespace

NormFunction make_lp_norm(const double p)
{
    if (std::isnan(p) || p < 1.0) {
        throw std::invalid_argument("Norm exponent must be >= 1 or positive infinity.");
    }
    std::ostringstream name;
    name << "L";
    if (std::isinf(p)) {
        name << "inf";
    } else {
        name.precision(std::numeric_limits<double>::max_digits10);
        name << p;
    }
    return { name.str(), [p](const Values& values) {
        return evaluate_lp(values, p);
    } };
}

void validate_norm_function(const NormFunction& norm)
{
    if (norm.name.empty() || !norm.evaluate) {
        throw std::invalid_argument("A norm requires a name and evaluation callback.");
    }
}

double vector_norm(const Values& values, const NormFunction& norm)
{
    validate_norm_function(norm);
    validate_values(values);
    if (values.empty()) {
        return 0.0;
    }
    const double result = norm.evaluate(values);
    if (std::isnan(result) || result < 0.0) {
        throw std::invalid_argument("A norm result must be nonnegative and not NaN.");
    }
    if (!std::isfinite(result)) {
        throw std::overflow_error("Norm result is not representable as a double.");
    }
    return result;
}

double network_norm(
    const MLP& network, const NormFunction& norm, const bool include_biases)
{
    validate_norm_function(norm);
    validate_network(network);
    Values values;
    values.reserve(parameter_count(network));
    for (const DenseLayer& layer : network.layers) {
        validate_values(layer.weights);
        validate_values(layer.biases);
        values.insert(values.end(), layer.weights.begin(), layer.weights.end());
        if (include_biases) {
            values.insert(values.end(), layer.biases.begin(), layer.biases.end());
        }
    }
    return vector_norm(values, norm);
}

double network_norm(
    const NetworkGradients& gradients, const NormFunction& norm,
    const bool include_biases)
{
    validate_norm_function(norm);
    Values values;
    for (const LayerGradients& layer : gradients.layers) {
        values.insert(values.end(), layer.weights.begin(), layer.weights.end());
        if (include_biases) {
            values.insert(values.end(), layer.biases.begin(), layer.biases.end());
        }
    }
    return vector_norm(values, norm);
}

} // namespace nablanet
