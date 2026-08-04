#pragma once

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

using OptimizerFactory = std::function<std::unique_ptr<Optimizer>()>;

struct OptimizerSpec {
    std::string name;
    OptimizerRequirement requirement{
        OptimizerRequirement::MiniBatchCompatible
    };
    OptimizerFactory make;
};

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
    std::size_t batch_size{}; // zero means full-dataset batches
    bool shuffle{ false };
    std::uint32_t shuffle_seed{ 0 };
    double gradient_tolerance{}; // zero means disabled
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
