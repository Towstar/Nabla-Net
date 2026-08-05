#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
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

struct TrainingConfig {
    std::size_t epochs{ 1 };
    std::size_t batch_size{};
    bool shuffle{ false };
    std::uint32_t shuffle_seed{ 0 };
    double gradient_tolerance{};
};

struct TrainingReport {
    bool completed{};
    std::size_t epochs_completed{};
    std::size_t steps{};
    std::vector<double> losses;
    std::vector<double> gradient_norms;
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

void train(
    MLP& network,
    const Dataset& dataset,
    const OptimizerSpec& optimizer,
    const TrainingConfig& config,
    const ObjectiveConfig& objective
);
