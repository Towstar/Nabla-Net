#pragma once

#include "optimizers.hpp"

#include <cstddef>
#include <random>
#include <vector>

/// <summary>
/// Validates the network, data, optimizer, and configuration for training.
/// </summary>
void validate_training_inputs(const MLP& network, const Dataset& dataset, const OptimizerSpec& optimizer, const TrainingConfig& config, const ObjectiveConfig& objective);

/// <summary>
/// Calculates the batch size that will be used for one update.
/// </summary>
std::size_t get_effective_batch_size(const TrainingConfig& config, const std::size_t dataset_size);

/// <summary>
/// Creates the sample ordering for an epoch, shuffling when configured.
/// </summary>
std::vector<std::size_t> make_epoch_order(const std::size_t dataset_size, const TrainingConfig& config, std::mt19937& random_engine);

/// <summary>
/// Copies the requested interval of an epoch ordering into a training batch.
/// </summary>
Dataset make_batch(const Dataset& dataset, const std::vector<std::size_t>& order, const std::size_t begin, const std::size_t end);

/// <summary>
/// Measures the largest absolute change between two network parameter sets.
/// </summary>
double parameter_change(const MLP& before, const MLP& after);

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
