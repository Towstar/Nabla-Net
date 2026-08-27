#pragma once

#include "optimizers.hpp"

namespace nablanet {

/// <summary>
/// Trains a network according to the optimizer and training configuration.
/// </summary>
TrainingReport train(
    MLP& network,
    const Dataset& dataset,
    const OptimizerSpec& optimizer,
    const TrainingConfig& config,
    const ObjectiveConfig& objective
);

} // namespace nablanet
