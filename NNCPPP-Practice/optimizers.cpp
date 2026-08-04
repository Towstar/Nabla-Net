#include "optimizers.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace
{
    class SGDOptimizer final : public Optimizer
    {
    public:
        const char* name() const noexcept override;
        void reset(const MLP& network) override;
        TrainingStepResult step(OptimizerContext& context) override;
    };

    class AdaGradOptimizer final : public Optimizer
    {
    public:
        const char* name() const noexcept override;
        void reset(const MLP& network) override;
        TrainingStepResult step(OptimizerContext& context) override;
    };

    class RMSPropOptimizer final : public Optimizer
    {
    public:
        const char* name() const noexcept override;
        void reset(const MLP& network) override;
        TrainingStepResult step(OptimizerContext& context) override;
    };

    class AdamOptimizer final : public Optimizer
    {
    public:
        const char* name() const noexcept override;
        void reset(const MLP& network) override;
        TrainingStepResult step(OptimizerContext& context) override;
    };

    class AdamWOptimizer final : public Optimizer
    {
    public:
        const char* name() const noexcept override;
        void reset(const MLP& network) override;
        TrainingStepResult step(OptimizerContext& context) override;
    };

    class LBFGSOptimizer final : public Optimizer
    {
    public:
        const char* name() const noexcept override;
        void reset(const MLP& network) override;
        TrainingStepResult step(OptimizerContext& context) override;
    };

    std::unique_ptr<Optimizer> make_sgd_optimizer();
    std::unique_ptr<Optimizer> make_adagrad_optimizer();
    std::unique_ptr<Optimizer> make_rmsprop_optimizer();
    std::unique_ptr<Optimizer> make_adam_optimizer();
    std::unique_ptr<Optimizer> make_adamw_optimizer();
    std::unique_ptr<Optimizer> make_lbfgs_optimizer();
}

namespace Optimizers {
    const OptimizerSpec SGD{
        "SGD",
        OptimizerRequirement::MiniBatchCompatible,
        {}
    };

    const OptimizerSpec AdaGrad{
        "AdaGrad",
        OptimizerRequirement::MiniBatchCompatible,
        {}
    };

    const OptimizerSpec RMSProp{
        "RMSProp",
        OptimizerRequirement::MiniBatchCompatible,
        {}
    };

    const OptimizerSpec Adam{
        "Adam",
        OptimizerRequirement::MiniBatchCompatible,
        {}
    };

    const OptimizerSpec AdamW{
        "AdamW",
        OptimizerRequirement::MiniBatchCompatible,
        {}
    };

    const OptimizerSpec LBFGS{
        "LBFGS",
        OptimizerRequirement::DeterministicFullBatch,
        {}
    };
}

namespace
{
    bool is_valid_optimizer_requirement(
        const OptimizerRequirement requirement
    ) noexcept
    {
        switch (requirement)
        {
        case OptimizerRequirement::MiniBatchCompatible:
        case OptimizerRequirement::DeterministicFullBatch:
            return true;

        default:
            return false;
        }
    }
}

OptimizerSpec make_custom_optimizer(
    std::string name,
    const OptimizerRequirement requirement,
    OptimizerFactory factory
)
{
    OptimizerSpec spec{
        std::move(name),
        requirement,
        std::move(factory)
    };

    validate_optimizer_spec(spec);
    return spec;
}

void validate_optimizer_spec(const OptimizerSpec& optimizer)
{
    if (optimizer.name.empty())
    {
        throw std::invalid_argument(
            "Optimizer name cannot be empty."
        );
    }

    if (!is_valid_optimizer_requirement(optimizer.requirement))
    {
        throw std::invalid_argument(
            "Optimizer has an invalid requirement."
        );
    }

    if (!optimizer.make)
    {
        throw std::invalid_argument(
            "Optimizer factory cannot be empty."
        );
    }
}

void validate_training_config(const TrainingConfig& config)
{
    if (config.epochs == 0)
    {
        throw std::invalid_argument(
            "Training epochs must be positive."
        );
    }

    if (!std::isfinite(config.gradient_tolerance) ||
        config.gradient_tolerance < 0.0)
    {
        throw std::invalid_argument(
            "Gradient tolerance must be non-negative and finite."
        );
    }

    // batch_size == 0 intentionally means full-dataset batches.
}
