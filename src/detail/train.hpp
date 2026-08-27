#pragma once

#include "../../include/nablanet/train.hpp"

#include <cstddef>
#include <random>
#include <vector>

namespace nablanet {

// Internal batching and validation helpers used to implement train().
void validate_training_inputs(
    const MLP& network,
    const Dataset& dataset,
    const OptimizerSpec& optimizer,
    const TrainingConfig& config,
    const ObjectiveConfig& objective
);

std::size_t get_effective_batch_size(
    const TrainingConfig& config,
    std::size_t dataset_size
);

std::vector<std::size_t> make_epoch_order(
    std::size_t dataset_size,
    const TrainingConfig& config,
    std::mt19937& random_engine
);

Dataset make_batch(
    const Dataset& dataset,
    const std::vector<std::size_t>& order,
    std::size_t begin,
    std::size_t end
);

double parameter_change(const MLP& before, const MLP& after);

} // namespace nablanet
