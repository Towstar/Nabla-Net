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

/// <summary>
/// Direct data members of <c>OptimizerContext</c>:
/// <para><c>network</c> (<c>MLP</c>).</para>
/// <para><c>batch</c> (<c>const Dataset</c>).</para>
/// <para><c>objective</c> (<c>const ObjectiveConfig</c>).</para>
/// <para><c>current_loss</c> (<c>double</c>).</para>
/// <para><c>current_gradient</c> (<c>NetworkGradients</c>).</para>
/// </summary>
struct OptimizerContext {
    MLP& network;
    const Dataset& batch;
    const ObjectiveConfig& objective;
    double current_loss{};
    NetworkGradients current_gradient;
};

class Optimizer {
public:
    /// <summary>
    /// Destroys an optimizer through its polymorphic interface.
    /// </summary>
    virtual ~Optimizer() = default;

    /// <summary>
    /// Returns the optimizer's stable display name.
    /// </summary>
    virtual const char* name() const noexcept = 0;

    /// <summary>
    /// Resets state for a new network or training run.
    /// </summary>
    virtual void reset(const MLP& network) = 0;

    /// <summary>
    /// Performs one optimizer update using the supplied training context.
    /// </summary>
    virtual TrainingStepResult step(OptimizerContext& context) = 0;
};

// stores a function that creates a new optimizer instance
using OptimizerFactory = std::function<std::unique_ptr<Optimizer>()>;

/// <summary>
/// Direct data members of <c>OptimizerSpec</c>:
/// <para><c>name</c> (<c>std::string</c>).</para>
/// <para><c>requirement</c> (<c>OptimizerRequirement</c>).</para>
/// <para><c>make</c> (<c>OptimizerFactory</c>).</para>
/// <para><c>supports_proximal</c> (<c>bool</c>).</para>
/// </summary>
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

/// <summary>
/// Direct data members of <c>SGDOptions</c>:
/// <para><c>learning_rate_schedule</c> (<c>LearningRateSchedule</c>).</para>
/// </summary>
struct SGDOptions {
    LearningRateSchedule learning_rate_schedule{
        [](std::size_t) {
            return 0.01;
        }
    };
};

/// <summary>
/// Direct data members of <c>MomentumSGDOptions</c>:
/// <para><c>learning_rate_schedule</c> (<c>LearningRateSchedule</c>).</para>
/// <para><c>momentum</c> (<c>double</c>).</para>
/// </summary>
struct MomentumSGDOptions {
    LearningRateSchedule learning_rate_schedule{
        [](std::size_t) {
            return 0.01;
        }
    };

    double momentum{ 0.5 };
};

/// <summary>
/// Direct data members of <c>AdaGradOptions</c>:
/// <para><c>learning_rate_schedule</c> (<c>LearningRateSchedule</c>).</para>
/// <para><c>epsilon</c> (<c>double</c>).</para>
/// </summary>
struct AdaGradOptions {
    LearningRateSchedule learning_rate_schedule{
        [](std::size_t) {
            return 0.01;
        }
    };

    double epsilon{ 1e-8 };
};

/// <summary>
/// Direct data members of <c>RMSPropOptions</c>:
/// <para><c>learning_rate_schedule</c> (<c>LearningRateSchedule</c>).</para>
/// <para><c>decay</c> (<c>double</c>).</para>
/// <para><c>epsilon</c> (<c>double</c>).</para>
/// </summary>
struct RMSPropOptions {
    LearningRateSchedule learning_rate_schedule{
        [](std::size_t) {return 0.01; }
    };
    double decay = 0.99;
    double epsilon = 1e-8;
};


/// <summary>
/// Direct data members of <c>AdamOptions</c>:
/// <para><c>learning_rate_schedule</c> (<c>LearningRateSchedule</c>).</para>
/// <para><c>beta1</c> (<c>double</c>).</para>
/// <para><c>beta2</c> (<c>double</c>).</para>
/// <para><c>epsilon</c> (<c>double</c>).</para>
/// </summary>
struct AdamOptions {
    // Default options as suggested in the original paper.
    LearningRateSchedule learning_rate_schedule{
        [](std::size_t) {return 0.001; }
    };
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
};

/// <summary>
/// Direct data members of <c>AdamWOptions</c>:
/// <para><c>learning_rate_schedule</c> (<c>LearningRateSchedule</c>).</para>
/// <para><c>beta1</c> (<c>double</c>).</para>
/// <para><c>beta2</c> (<c>double</c>).</para>
/// <para><c>epsilon</c> (<c>double</c>).</para>
/// <para><c>weight_decay</c> (<c>double</c>).</para>
/// <para><c>decay_biases</c> (<c>bool</c>).</para>
/// </summary>
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

/// <summary>
/// Direct data members of <c>CAdamWOptions</c>:
/// <para><c>learning_rate_schedule</c> (<c>LearningRateSchedule</c>).</para>
/// <para><c>beta1</c> (<c>double</c>).</para>
/// <para><c>beta2</c> (<c>double</c>).</para>
/// <para><c>epsilon</c> (<c>double</c>).</para>
/// <para><c>weight_decay</c> (<c>double</c>).</para>
/// <para><c>decay_biases</c> (<c>bool</c>).</para>
/// <para><c>cautious_epsilon</c> (<c>double</c>).</para>
/// </summary>
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

/// Selects how L-BFGS handles an Armijo-only line-search result.
enum class LBFGSLineSearchPolicy {
    StrongWolfeRequired,
    ArmijoFallbackWithoutHistory
};

/// Options for the exact-curvature damped Newton--CG reference optimizer.
/// It solves (H + damping I)p = -gradient through Hessian-vector products.
struct NewtonCGOptions {
    double damping{ 1e-3 };
    std::size_t maximum_cg_iterations{ 50 };
    double absolute_residual_tolerance{ 1e-10 };
    double relative_residual_tolerance{ 1e-4 };
    double negative_curvature_tolerance{ 1e-12 };
    WolfeParameters line_search{};
};

/// Options for deterministic full-batch L-BFGS.
struct LBFGSOptions {
    std::size_t history_size{ 10 };
    double curvature_tolerance{ 1e-10 };
    WolfeParameters line_search{};
    LBFGSLineSearchPolicy line_search_policy{LBFGSLineSearchPolicy::StrongWolfeRequired};
};

/// <summary>
/// Creates a stochastic-gradient-descent optimizer specification.
/// </summary>
OptimizerSpec make_sgd(
    SGDOptions options = {}
);

/// <summary>
/// Creates a momentum stochastic-gradient-descent optimizer specification.
/// </summary>
OptimizerSpec make_momentum_sgd(MomentumSGDOptions options = {});

/// <summary>
/// Creates an AdaGrad optimizer specification.
/// </summary>
OptimizerSpec make_adagrad(AdaGradOptions = {});

/// <summary>
/// Creates an RMSProp optimizer specification.
/// </summary>
OptimizerSpec make_rmsprop(RMSPropOptions options = {});

/// <summary>
/// Creates an Adam optimizer specification.
/// </summary>
OptimizerSpec make_adam(AdamOptions options = {});

/// <summary>
/// Creates an AdamW optimizer specification.
/// </summary>
OptimizerSpec make_adamw(AdamWOptions options = {});

/// <summary>
/// Creates a cautious AdamW optimizer specification.
/// </summary>
OptimizerSpec make_cadamw(CAdamWOptions options = {});

/// <summary>
/// Creates a damped Newton--CG optimizer specification.
/// </summary>
OptimizerSpec make_newton_cg(NewtonCGOptions options = {});

/// <summary>
/// Creates an L-BFGS optimizer specification.
/// </summary>
OptimizerSpec make_lbfgs(LBFGSOptions options = {});

namespace Optimizers {
    extern const OptimizerSpec SGD;
    extern const OptimizerSpec MomentumSGD;
    extern const OptimizerSpec AdaGrad;
    extern const OptimizerSpec RMSProp;
    extern const OptimizerSpec Adam;
    extern const OptimizerSpec AdamW;
    extern const OptimizerSpec CAdamW;
    extern const OptimizerSpec NewtonCG;
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

/// <summary>
/// Creates a validated specification for a caller-provided optimizer factory.
/// </summary>
OptimizerSpec make_custom_optimizer(
    std::string name,
    OptimizerRequirement requirement,
    OptimizerFactory factory,
    bool supports_proximal = false
);

/// <summary>
/// Validates an optimizer specification and its factory callback.
/// </summary>
void validate_optimizer_spec(
    const OptimizerSpec& optimizer
);

/// <summary>
/// Validates the stopping, batching, and shuffle settings for training.
/// </summary>
void validate_training_config(
    const TrainingConfig& config
);

/// <summary>
/// Writes a human-readable training report to an output stream.
/// </summary>
std::ostream& operator<<(std::ostream& output, const TrainingReport& report);

/// <summary>
/// Converts a training stop reason to a display string.
/// </summary>
std::string trainingStopReasonToString(TrainingStopReason reason);
