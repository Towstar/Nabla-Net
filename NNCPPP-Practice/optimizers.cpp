#include "optimizers.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

#pragma region Optimizer Implementations and Factories

namespace
{
    class SGDOptimizer final : public Optimizer
    {
    public:
        explicit SGDOptimizer(LearningRateSchedule schedule)
            : learning_rate_schedule_(std::move(schedule))
        {
            if (!learning_rate_schedule_) {
                throw std::invalid_argument(
                    "SGD learning-rate schedule cannot be empty."
                );
            }
        }

        const char* name() const noexcept override
        {
            return "SGD";
        }

        void reset(const MLP& network) override
        {
            validate_network(network);
            update_index_ = 0;
        }

        TrainingStepResult step(OptimizerContext& context) override
        {
            validate_network(context.network);
            validate_objective_config(context.objective);

            if (context.batch.empty()) {
                throw std::invalid_argument(
                    "SGD batch cannot be empty."
                );
            }

            validate_gradients_like(
                context.network,
                context.current_gradient
            );

            if (!std::isfinite(context.current_loss)) {
                throw std::invalid_argument(
                    "Current loss must be finite."
                );
            }

            return _step(context);
        }

    private:
        LearningRateSchedule learning_rate_schedule_;
        std::size_t update_index_{};

        TrainingStepResult _step(OptimizerContext& context)
        {
            const double learning_rate =
                learning_rate_schedule_(update_index_);

            if (!learning_rate_is_valid(learning_rate)) {
                throw std::invalid_argument(
                    "Learning-rate schedule produced an invalid rate."
                );
            }

            TrainingStepResult result{};
            result.previous_loss = context.current_loss;
            result.gradient_norm =
                gradient_l2_norm(context.current_gradient);

            MLP candidate = context.network;

            apply_gradient(
                candidate,
                context.current_gradient,
                learning_rate
            );

            result.new_loss = objective_loss(
                candidate,
                context.batch,
                context.objective
            );

            context.network = std::move(candidate);
            ++update_index_;

            result.updated = true;
            return result;
        }
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

    std::unique_ptr<Optimizer> make_sgd_instance(
        LearningRateSchedule schedule
    )
    {
        return std::make_unique<SGDOptimizer>(
            std::move(schedule)
        );
    }

    std::unique_ptr<Optimizer> make_adagrad_optimizer();
    std::unique_ptr<Optimizer> make_rmsprop_optimizer();
    std::unique_ptr<Optimizer> make_adam_optimizer();
    std::unique_ptr<Optimizer> make_adamw_optimizer();
    std::unique_ptr<Optimizer> make_lbfgs_optimizer();
}

OptimizerSpec make_sgd(SGDOptions options)
{
    if (!options.learning_rate_schedule) {
        throw std::invalid_argument(
            "SGD learning-rate schedule cannot be empty."
        );
    }

    const LearningRateSchedule schedule =
        options.learning_rate_schedule;

    return make_custom_optimizer(
        "SGD",
        OptimizerRequirement::MiniBatchCompatible,
        [schedule] {
            return make_sgd_instance(schedule);
        }
    );
}

#pragma endregion

#pragma region Built-in Optimizer Specifications

namespace Optimizers {
    const OptimizerSpec SGD = make_sgd();

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

#pragma endregion

#pragma region Optimizer Validation Helpers

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

#pragma endregion

#pragma region Custom Optimizer API

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

#pragma endregion

#pragma region Training Configuration

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

#pragma endregion
