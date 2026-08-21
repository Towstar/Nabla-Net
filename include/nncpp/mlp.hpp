#pragma once

#include "common.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <functional>

using Values = std::vector<double>;

// Activation scaffolding. The scalar aliases are used by the helper for
// elementwise activations. The public activation callbacks operate on whole
// vectors so vector-valued activations such as softmax can be added later.
using ActivationParameter = std::optional<double>;

using ScalarActivationForward = std::function<double(
    double input,
    ActivationParameter parameter
)>;

using ScalarActivationDerivative = std::function<double(
    double input,
    ActivationParameter parameter
)>;

using ActivationForward = std::function<Values(
    const Values& pre_activations
)>;

// Computes J(z)^T * upstream_gradient.
using ActivationBackward = std::function<Values(
    const Values& pre_activations,
    const Values& upstream_gradient
)>;

struct ActivationFunction {
    std::string name;
    ActivationForward forward;
    ActivationBackward backward;
};

ActivationFunction make_elementwise_activation(
    std::string name,
    ScalarActivationForward forward,
    ScalarActivationDerivative derivative,
    ActivationParameter parameter = std::nullopt
);

namespace Activations {
    extern const ActivationFunction Tanh;
    extern const ActivationFunction Linear;
    extern const ActivationFunction ReLU;
    extern const ActivationFunction Sigmoid;
    extern const ActivationFunction LeakyReLU;
    extern const ActivationFunction ELU;
    extern const ActivationFunction GELU;
    extern const ActivationFunction SiLU;
    extern const ActivationFunction Mish;
}

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
    // One activation descriptor per dense layer.  An empty vector preserves
    // the compatibility defaults: tanh for hidden layers and sigmoid for the
    // output layer.
    std::vector<ActivationFunction> layer_activations;
};

enum class InitializationType {
    XavierGlorot,
    KaimingHe,
    LeCunn,
    LeCun = LeCunn,
    Zero
};

struct NetworkSpec {
    std::vector<std::size_t> layer_sizes;
    std::uint32_t seed{ 0 };
    std::vector<ActivationFunction> layer_activations;
    InitializationType initialization_type{
        InitializationType::XavierGlorot
    };
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

enum class ObjectiveInputDomain {
    PreActivation,
    ActivatedOutput
};

struct ObjectiveFunctions {
    SampleLossFunction sample_loss;
    SampleLossGradientFunction sample_loss_gradient;
    ObjectiveInputDomain input_domain{
        ObjectiveInputDomain::PreActivation
    };
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
    std::optional<double> effective_step_size{};
    LineSearchResult line_search;
};

struct RegularizationTerm {
    std::string name;
    // Callbacks return the unscaled penalty and unscaled gradient. The
    // objective aggregation layer applies coefficient exactly once.
    std::function<double(const MLP&)> value;
    std::function<void(const MLP&, NetworkGradients&)> add_gradient;
    // Proximal terms contribute their value to the objective but do not add a
    // subgradient. The trainer invokes this callback after a base optimizer
    // step, passing the effective step size.
    bool proximal{ false };
    std::function<void(MLP&, double effective_step_size)> proximal_update;
    bool smooth{ true };
    bool includes_biases{ false };
	double coefficient{ 0.0 };
};

/// <summary>
/// <para>ObjectiveFunction data_objective</para>
/// <para>regularizers (vector of regularization terms)</para>
/// </summary>
struct ObjectiveConfig {
    ObjectiveFunctions data_objective;
    std::vector<RegularizationTerm> regularizers;
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
void initialize_network(
    MLP& network,
    InitializationType initialization_type,
    const std::vector<std::size_t>& layer_sizes,
    std::uint32_t seed
);

MLP make_mlp(
    const std::vector<std::size_t>& layer_sizes,
    std::uint32_t seed,
    InitializationType initialization_type =
        InitializationType::XavierGlorot
);

MLP make_mlp(const NetworkSpec& specification);

double stable_sigmoid(double x);
ForwardCache forward_pass(const MLP& network, const Values& input);
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


void apply_gradient(
    MLP& network,
    const NetworkGradients& gradients,
    double learning_rate
);
