#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <iosfwd>
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
    bool supports_proximal{ false };
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

struct MomentumSGDOptions {
    LearningRateSchedule learning_rate_schedule{
        [](std::size_t) {
            return 0.01;
        }
    };

    double momentum{ 0.5 };
};

struct AdaGradOptions {
    LearningRateSchedule learning_rate_schedule{
        [](std::size_t) {
            return 0.01;
        }
    };

    double epsilon{ 1e-8 };
};

struct RMSPropOptions {
    LearningRateSchedule learning_rate_schedule{
        [](std::size_t) {return 0.01; }
    };
    double decay = 0.99;
    double epsilon = 1e-8;
};


struct AdamOptions {
    // Default options as suggested in the original paper.
    LearningRateSchedule learning_rate_schedule{
        [](std::size_t) {return 0.001; }
    };
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
};

struct AdamWOptions {
    LearningRateSchedule learning_rate_schedule{
        [](std::size_t) {
            return 0.001;
        }
    };

    double beta1{ 0.9 };
    double beta2{ 0.999 };
    double epsilon{ 1e-8 };

    double weight_decay{ 0.01 };
    bool decay_biases{ false };
};

struct CAdamWOptions {
    LearningRateSchedule learning_rate_schedule{
        [](std::size_t) {
            return 0.001;
        }
    };

    double beta1{ 0.9 };
    double beta2{ 0.999 };
    double epsilon{ 1e-8 };
    double weight_decay{ 0.01 };
    bool decay_biases{ false };
    double cautious_epsilon{ 1.0 };
};

struct LBFGSOptions {
    LearningRateSchedule learning_rate_schedule{
        [](std::size_t) {return 0.01; }
    };
    double decay = 0.99;
    double epsilon = 1e-8;
};

OptimizerSpec make_sgd(
    SGDOptions options = {}
);

OptimizerSpec make_momentum_sgd(MomentumSGDOptions options = {});

OptimizerSpec make_adagrad(AdaGradOptions = {});

OptimizerSpec make_rmsprop(RMSPropOptions options = {});

OptimizerSpec make_adam(AdamOptions options = {});

OptimizerSpec make_adamw(AdamWOptions options = {});

OptimizerSpec make_cadamw(CAdamWOptions options = {});

OptimizerSpec make_lbfgs(LBFGSOptions options = {});

namespace Optimizers {
    extern const OptimizerSpec SGD;
    extern const OptimizerSpec MomentumSGD;
    extern const OptimizerSpec AdaGrad;
    extern const OptimizerSpec RMSProp;
    extern const OptimizerSpec Adam;
    extern const OptimizerSpec AdamW;
    extern const OptimizerSpec CAdamW;
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

/// <summary>
/// <para>BatchMode : FullBatch, MiniBatch, or Stochastic</para>
/// <para>Batch Size : Size of Training Batch</para>
/// <para>Max Iterations (Optional) : Maximum Number of Training Iterations</para>
/// <para>Max Epochs (Optional, Default 1) : Maxmimum Number of Training Epochs (Full Passes Through Dataset)</para>
/// <para>Gradient Tolerance (Optional) : Epsilon for Infinity Norm of Gradient </para>
/// <para>Parameter Change Tolerance (Optional) : Epsilon for Infinity Norm of Step</para>
/// <para>Shuffle (Default False) : To Shuffle Dataset or Not</para>
/// <para>Shuffle Seed (Default 0) : Seed for Shuffling</para>
/// <para>Record History (Default True) : T/F Whether to record history or not</para>
/// </summary>
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

/// <summary>
/// <para>Completed: T/F did training complete?</para>
/// <para>Stop Reason: TrainingStopReason</para>
/// <para>Optimizer Name: String associated with Optimizer</para>
/// <para>Epochs Completed: Number of epochs completed</para>
/// <para>Steps: Number of steps taken</para>
/// <para>Initial Loss: The initial loss</para>
/// <para>Final Loss: The final loss after training</para>
/// <para>Final Gradient Norm: The infinity norm of the gradient vector</para>
/// <para>Final Parameter Change: The final change in parameter</para>
/// <para>Losses: Vector of doubles of actual individual losses, size of [ToDo]</para>
/// <para>Gradient Norms: Vector of the Gradient Norms</para>
/// <para>Parameter Changes: Vector of the Parameter Changes</para>
/// </summary>
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
    OptimizerFactory factory,
    bool supports_proximal = false
);

void validate_optimizer_spec(
    const OptimizerSpec& optimizer
);

void validate_training_config(
    const TrainingConfig& config
);

std::ostream& operator<<(std::ostream& output, const TrainingReport& report);

std::string trainingStopReasonToString(TrainingStopReason reason);
