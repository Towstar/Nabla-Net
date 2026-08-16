#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "objective_functions.hpp"

enum class OptimizerRequirement {
    MiniBatchCompatible,
    DeterministicFullBatch
};

struct OptimizerContext {
    MLP& network;
    const Dataset& batch;
    const ObjectiveConfig& objective;
    double current_loss{};
    NetworkGradients current_gradient;
};

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual const char* name() const noexcept = 0;
    virtual void reset(const MLP& network) = 0;
    virtual TrainingStepResult step(OptimizerContext& context) = 0;
};

// stores a function that creates a new optimizer instance
using OptimizerFactory = std::function<std::unique_ptr<Optimizer>()>;

struct OptimizerSpec {
    std::string name;
    OptimizerRequirement requirement{
        OptimizerRequirement::MiniBatchCompatible
    };
    OptimizerFactory make;
};

using LearningRateSchedule =
std::function<double(std::size_t update_index)>;

struct SGDOptions {
    LearningRateSchedule learning_rate_schedule{
        [](std::size_t) {
            return 0.01;
        }
    };
};

OptimizerSpec make_sgd(
    SGDOptions options = {}
);

namespace Optimizers {
    extern const OptimizerSpec SGD;
    extern const OptimizerSpec AdaGrad;
    extern const OptimizerSpec RMSProp;
    extern const OptimizerSpec Adam;
    extern const OptimizerSpec AdamW;
    extern const OptimizerSpec LBFGS;
}

enum class BatchMode {
    FullBatch,
    MiniBatch,
    Stochastic
};

enum class TrainingStopReason {
    NotStarted,
    MaxIterations,
    MaxEpochs,
    GradientTolerance,
    ParameterChangeTolerance,
    OptimizerStopped
};

struct TrainingConfig {
    BatchMode batch_mode{ BatchMode::FullBatch };
    std::size_t batch_size{};

    // At least one limit must be enabled. The default is one epoch, which
    // preserves the behavior of the original training scaffold.
    std::optional<std::size_t> max_iterations{};
    std::optional<std::size_t> max_epochs{ 1 };

    std::optional<double> gradient_tolerance{};
    std::optional<double> parameter_change_tolerance{};

    bool shuffle{ false };
    std::uint32_t shuffle_seed{ 0 };
    bool record_history{ true };
};

struct TrainingReport {
    bool completed{};
    TrainingStopReason stop_reason{ TrainingStopReason::NotStarted };
    std::string optimizer_name;
    std::size_t epochs_completed{};
    std::size_t steps{};

    double initial_loss{};
    double final_loss{};
    double final_gradient_norm{};
    double final_parameter_change{};

    std::vector<double> losses;
    std::vector<double> gradient_norms;
    std::vector<double> parameter_changes;
};

OptimizerSpec make_custom_optimizer(
    std::string name,
    OptimizerRequirement requirement,
    OptimizerFactory factory
);

void validate_optimizer_spec(
    const OptimizerSpec& optimizer
);

void validate_training_config(
    const TrainingConfig& config
);

TrainingReport train(
    MLP& network,
    const Dataset& dataset,
    const OptimizerSpec& optimizer,
    const TrainingConfig& config,
    const ObjectiveConfig& objective
);
