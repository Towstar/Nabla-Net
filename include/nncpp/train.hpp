#pragma once

#include "optimizers.hpp"

#include <cstddef>
#include <random>
#include <vector>

void validate_training_inputs(const MLP& network, const Dataset& dataset, const OptimizerSpec& optimizer, const TrainingConfig& config, const ObjectiveConfig& objective);

std::size_t get_effective_batch_size(const TrainingConfig& config, const std::size_t dataset_size);

std::vector<std::size_t> make_epoch_order(const std::size_t dataset_size, const TrainingConfig& config, std::mt19937& random_engine);

Dataset make_batch(const Dataset& dataset, const std::vector<std::size_t>& order, const std::size_t begin, const std::size_t end);

double parameter_change(const MLP& before, const MLP& after);

TrainingReport train(
    MLP& network,
    const Dataset& dataset,
    const OptimizerSpec& optimizer,
    const TrainingConfig& config,
    const ObjectiveConfig& objective
);
