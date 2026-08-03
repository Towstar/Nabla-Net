#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <functional>

using Values = std::vector<double>;

struct Sample {
    Values input;
    Values target;
};

using Dataset = std::vector<Sample>;

struct DenseLayer {
    std::size_t input_size{};
    std::size_t output_size{};
    Values weights;
    Values biases;

    double& weight(std::size_t output_index, std::size_t input_index);
    const double& weight(std::size_t output_index, std::size_t input_index) const;
};

struct MLP {
    std::vector<std::size_t> layer_sizes;
    std::vector<DenseLayer> layers;
};

struct LayerGradients {
    Values weights;
    Values biases;
};

struct NetworkGradients {
    std::vector<LayerGradients> layers;
};

using SampleLossFunction = std::function<double(
    const Values& output_logits,
    const Values& target
)>;

using SampleLossGradientFunction = std::function<Values(
    const Values& output_logits,
    const Values& target
)>;

struct ObjectiveFunctions {
    SampleLossFunction sample_loss;
    SampleLossGradientFunction sample_loss_gradient;
};

using NetworkDirection = NetworkGradients;

enum class LineSearchStatus {
    StrongWolfeSatisfied,
    SufficientDecreaseFallback,
    Failed
};

struct WolfeParameters {
    double sufficient_decrease{ 1e-4 };
    double curvature{ 0.9 };
    double shrink_factor{ 0.5 };
    double maximum_step{ 1.0 };
    std::size_t maximum_trials{ 30 };
    double minimum_downhill_cosine{ 0.0 };
};

struct LineSearchTrial {
    double step_size{};
    double loss{};
    double directional_derivative{};
    bool sufficient_decrease{};
    bool strong_curvature{};
};

struct WolfeEvaluation {
    double step_size{};
    double candidate_loss{};
    double initial_directional_derivative{};
    double candidate_directional_derivative{};
    bool sufficient_decrease{};
    bool strong_curvature{};
    MLP candidate_network;
    NetworkGradients candidate_gradient;
	ObjectiveFunctions objective;
};

struct LineSearchResult {
    LineSearchStatus status{ LineSearchStatus::Failed };
    double step_size{};
    double selected_loss{};
    double initial_directional_derivative{};
    double selected_directional_derivative{};
    MLP selected_network;
    NetworkGradients selected_gradient;
    std::vector<LineSearchTrial> history;
};

struct TrainingStepResult {
    bool updated{};
    double previous_loss{};
    double new_loss{};
    double gradient_norm{};
    LineSearchResult line_search;
};

// Phase 1 exercise contracts. These declarations describe the objective,
// regularization, and trainer boundaries; their behavior is intentionally
// left for the learner to implement in mlp.cpp.
struct RegularizationTerm {
    std::string name;
    std::function<double(const MLP&)> value;
    std::function<void(const MLP&, NetworkGradients&)> add_gradient;
    bool smooth{ true };
    bool includes_biases{ false };
	double coefficient{ 0.0 };
};

struct ObjectiveConfig {
    ObjectiveFunctions data_objective;
    std::vector<RegularizationTerm> regularizers;
};

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

/// <summary>
/// Stores intermediate values to compute gradients without recomputing.
/// </summary>
struct ForwardCache {
    std::vector<Values> activations;
    std::vector<Values> pre_activations;
};

void validate_network_architecture(const std::vector<std::size_t>& layer_sizes);
void validate_network(const MLP& network);
std::size_t parameter_count(const MLP& network);

MLP make_zero_network(std::vector<std::size_t> layer_sizes);
MLP make_mlp(const std::vector<std::size_t>& layer_sizes, std::uint32_t seed);
double stable_sigmoid(double x);
ForwardCache forward_pass(const MLP& network, const Values& input);
double binary_cross_entropy_from_logit(double logit, double target);

ObjectiveFunctions make_binary_cross_entropy_objective();

Values stable_softmax(const Values& logits);

double cross_entropy_from_logits(
    const Values& logits,
    const Values& targets
);

ObjectiveFunctions make_softmax_cross_entropy_objective();

ObjectiveConfig make_objective_config(
    const ObjectiveFunctions& data_objective =
        make_binary_cross_entropy_objective(),
    std::vector<RegularizationTerm> regularizers = {}
);

RegularizationTerm make_l2_regularization(
    double coefficient,
    bool include_biases = false
);

RegularizationTerm make_l1_regularization(
    double coefficient,
    bool include_biases = false
);

RegularizationTerm make_elastic_net_regularization(
    double l1_coefficient,
    double l2_coefficient,
    bool include_biases = false
);

void validate_objective_config(
    const ObjectiveConfig& objective
);

double regularization_loss(
    const MLP& network,
    const ObjectiveConfig& objective
);

void add_regularization_gradients(
    const MLP& network,
    const ObjectiveConfig& objective,
    NetworkGradients& gradients
);

double objective_loss(
    const MLP& network,
    const Dataset& batch,
    const ObjectiveConfig& objective
);

NetworkGradients objective_gradients(
    const MLP& network,
    const Dataset& batch,
    const ObjectiveConfig& objective
);

void validate_objective_functions(
    const ObjectiveFunctions& objective
);

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

double sample_cost(
    const MLP& network,
    const ForwardCache& cache,
    const Values& target
);

double sample_cost(
    const MLP& network,
    const ForwardCache& cache,
    const Values& target,
    const ObjectiveFunctions& objective
);

double batch_loss(const MLP& network, const Dataset& batch);

double batch_loss(
    const MLP& network,
    const Dataset& batch,
    const ObjectiveFunctions& objective
);
NetworkGradients make_zero_gradients_like(const MLP& network);

NetworkGradients backward(const MLP& network, const ForwardCache& cache, const Values& target);
NetworkGradients backward(const MLP& network, const ForwardCache& cache, const Values& target, const ObjectiveFunctions& objective);

NetworkGradients batch_gradients(
    const MLP& network,
    const Dataset& batch
);

NetworkGradients batch_gradients(
    const MLP& network,
    const Dataset& batch,
    const ObjectiveFunctions& objective
);

double gradient_l2_norm(
    const NetworkGradients& gradients
);

double maximum_absolute_gradient(
    const NetworkGradients& gradients
);

bool learning_rate_is_valid(double learning_rate) noexcept;

void validate_gradients_like(
    const MLP& network,
    const NetworkGradients& gradients
);

bool wolfe_parameters_are_valid(
    const WolfeParameters& parameters
) noexcept;

void validate_wolfe_parameters(
    const WolfeParameters& parameters
);

void apply_gradient(
    MLP& network,
    const NetworkGradients& gradients,
    double learning_rate
);

double network_vector_dot(
    const NetworkGradients& left,
    const NetworkGradients& right
);

NetworkDirection make_negative_gradient_direction(
    const NetworkGradients& gradient
);

double downhill_cosine(
    const NetworkGradients& gradient,
    const NetworkDirection& direction
);

bool is_downhill_direction(
    const NetworkGradients& gradient,
    const NetworkDirection& direction,
    double minimum_cosine = 0.0
);

void apply_direction(
    MLP& network,
    const NetworkDirection& direction,
    double scale
);

MLP make_candidate_network(
    const MLP& current_network,
    const NetworkDirection& direction,
    double step_size
);

bool satisfies_sufficient_decrease(
    double candidate_loss,
    double current_loss,
    double step_size,
    double initial_directional_derivative,
    double sufficient_decrease_constant
);

bool satisfies_strong_curvature(
    double candidate_directional_derivative,
    double initial_directional_derivative,
    double curvature_constant
);

bool satisfies_strong_wolfe(
    const WolfeEvaluation& evaluation
) noexcept;

WolfeEvaluation evaluate_wolfe_candidate(
    const MLP& current_network,
    const Dataset& batch,
    double current_loss,
	const NetworkGradients& current_gradient,
	const NetworkDirection& direction,
	double step_size,
	const WolfeParameters& parameters,
	const ObjectiveFunctions& objective = make_binary_cross_entropy_objective()
);

LineSearchResult backtracking_wolfe_stepsize(
    const MLP& current_network,
    const Dataset& batch,
    double current_loss,
    const NetworkGradients& current_gradient,
    const NetworkDirection& direction,
    const WolfeParameters& parameters,
    const ObjectiveFunctions& objective = make_binary_cross_entropy_objective()
);

TrainingStepResult take_wolfe_gradient_step(
    MLP& network,
    const Dataset& batch,
    const WolfeParameters& parameters,
    double gradient_tolerance,
	const ObjectiveFunctions& objective = make_binary_cross_entropy_objective()
);

const char* line_search_status_name(
    LineSearchStatus status
) noexcept;
