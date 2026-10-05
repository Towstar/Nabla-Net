#pragma once

#include "mlp.hpp"

#include <functional>
#include <string>

namespace nablanet {

/// A named, read-only norm on a vector. Custom callbacks must satisfy the
/// mathematical norm axioms; validation can only check the callable and result.
/// Use vector_norm (or the network helpers) to validate custom evaluations.
struct NormFunction {
    std::string name;
    std::function<double(const Values&)> evaluate;
};

/// Creates the true L-p norm for finite p >= 1 or positive infinity.
/// This is not a regularizer: there is no coefficient, derivative, or update.
/// The factory callback itself also checks inputs and overflow.
NormFunction make_lp_norm(double p);

/// Requires a nonempty name and an evaluation callback.
void validate_norm_function(const NormFunction& norm);

/// Evaluates a norm on finite values. Empty vectors return zero without calling
/// the callback. Invalid inputs and negative/NaN results throw invalid_argument;
/// positive infinite results throw overflow_error. Callback exceptions propagate.
double vector_norm(const Values& values, const NormFunction& norm);

/// Validates the network, then evaluates one vector containing each layer's
/// weights followed by its biases (if selected), in stored order.
double network_norm(
    const MLP& network,
    const NormFunction& norm,
    bool include_biases = true
);

/// Evaluates one vector of gradient components in the same order as parameters.
/// Selected components must be finite; no reference network/shape is required.
/// Empty gradients return zero. Excluded biases are not inspected.
double network_norm(
    const NetworkGradients& gradients,
    const NormFunction& norm,
    bool include_biases = true
);

} // namespace nablanet
