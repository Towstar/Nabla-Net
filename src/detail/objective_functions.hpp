#pragma once

#include "../../include/nablanet/objective_functions.hpp"

namespace nablanet {

// Internal objective-composition helpers used by train() and tests. Public
// callers evaluate complete objectives through objective_loss(),
// objective_gradients(), and objective_hessian_vector_product().
void add_regularization_gradients(
    const MLP& network,
    const ObjectiveConfig& objective,
    NetworkGradients& gradients
);

void apply_proximal_updates(
    MLP& network,
    const ObjectiveConfig& objective,
    double effective_step_size
);

} // namespace nablanet
