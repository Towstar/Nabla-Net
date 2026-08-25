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

using ScalarActivationSecondDerivative = std::function<double(
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

// Computes J(z) * pre_activation_tangent.
using ActivationJvp = std::function<Values(
    const Values& pre_activations,
    const Values& pre_activation_tangent
)>;

// Computes the directional derivative of J(z)^T * upstream_gradient.
using ActivationBackwardJvp = std::function<Values(
    const Values& pre_activations,
    const Values& pre_activation_tangent,
    const Values& upstream_gradient,
    const Values& upstream_gradient_tangent
)>;

/// <summary>
/// Direct data members of <c>ActivationFunction</c>:
/// <para><c>name</c> (<c>std::string</c>).</para>
/// <para><c>forward</c> (<c>ActivationForward</c>).</para>
/// <para><c>backward</c> (<c>ActivationBackward</c>).</para>
/// <para><c>jvp</c> (<c>ActivationJvp</c>, optional).</para>
/// <para><c>backward_jvp</c> (<c>ActivationBackwardJvp</c>, optional).</para>
/// </summary>
struct ActivationFunction {
    std::string name;
    ActivationForward forward;
    ActivationBackward backward;
    // First-order custom activations may leave these empty. JVP- and
    // HVP-based APIs reject such an activation with std::invalid_argument.
    ActivationJvp jvp;
    ActivationBackwardJvp backward_jvp;
};

/// <summary>
/// Creates a vector activation from scalar forward and derivative callbacks.
/// </summary>
ActivationFunction make_elementwise_activation(
    std::string name,
    ScalarActivationForward forward,
    ScalarActivationDerivative derivative,
    ActivationParameter parameter = std::nullopt
);

/// <summary>
/// Creates a twice-differentiable vector activation from scalar callbacks.
/// </summary>
ActivationFunction make_elementwise_activation(
    std::string name,
    ScalarActivationForward forward,
    ScalarActivationDerivative derivative,
    ScalarActivationSecondDerivative second_derivative,
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

/// <summary>
/// Direct data members of <c>Sample</c>:
/// <para><c>input</c> (<c>Values</c>).</para>
/// <para><c>target</c> (<c>Values</c>).</para>
/// </summary>
struct Sample {
    Values input;
    Values target;
};

using Dataset = std::vector<Sample>;

/// <summary>
/// Direct data members of <c>DenseLayer</c>:
/// <para><c>input_size</c> (<c>std::size_t</c>).</para>
/// <para><c>output_size</c> (<c>std::size_t</c>).</para>
/// <para><c>weights</c> (<c>Values</c>).</para>
/// <para><c>biases</c> (<c>Values</c>).</para>
/// </summary>
struct DenseLayer {
    std::size_t input_size{};
    std::size_t output_size{};
    Values weights;
    Values biases;

    /// <summary>
    /// Returns mutable access to a weight by output and input index.
    /// </summary>
    double& weight(std::size_t output_index, std::size_t input_index);

    /// <summary>
    /// Returns read-only access to a weight by output and input index.
    /// </summary>
    const double& weight(std::size_t output_index, std::size_t input_index) const;
};

/// <summary>
/// Direct data members of <c>MLP</c>:
/// <para><c>layer_sizes</c> (<c>std::vector&lt;std::size_t&gt;</c>).</para>
/// <para><c>layers</c> (<c>std::vector&lt;DenseLayer&gt;</c>).</para>
/// <para><c>layer_activations</c> (<c>std::vector&lt;ActivationFunction&gt;</c>).</para>
/// </summary>
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

/// <summary>
/// Direct data members of <c>NetworkSpec</c>:
/// <para><c>layer_sizes</c> (<c>std::vector&lt;std::size_t&gt;</c>).</para>
/// <para><c>seed</c> (<c>std::uint32_t</c>).</para>
/// <para><c>layer_activations</c> (<c>std::vector&lt;ActivationFunction&gt;</c>).</para>
/// <para><c>initialization_type</c> (<c>InitializationType</c>).</para>
/// </summary>
struct NetworkSpec {
    std::vector<std::size_t> layer_sizes;
    std::uint32_t seed{ 0 };
    std::vector<ActivationFunction> layer_activations;
    InitializationType initialization_type{
        InitializationType::XavierGlorot
    };
};

/// <summary>
/// Direct data members of <c>LayerGradients</c>:
/// <para><c>weights</c> (<c>Values</c>).</para>
/// <para><c>biases</c> (<c>Values</c>).</para>
/// </summary>
struct LayerGradients {
    Values weights;
    Values biases;
};

/// <summary>
/// Direct data members of <c>NetworkGradients</c>:
/// <para><c>layers</c> (<c>std::vector&lt;LayerGradients&gt;</c>).</para>
/// </summary>
struct NetworkGradients {
    std::vector<LayerGradients> layers;
};

// A parameter-space tangent shares the weight and bias layout of a gradient.
// This is intentionally distinct from NetworkDirection, which remains the
// public alias used by line-search code.
using NetworkTangent = NetworkGradients;

using SampleLossFunction = std::function<double(
    const Values& output_logits,
    const Values& target
)>;

using SampleLossGradientFunction = std::function<Values(
    const Values& output_logits,
    const Values& target
)>;

// Computes the Hessian-vector product of a scalar sample loss with respect
// to its immediate objective input. The target is held constant.
using SampleLossHessianVectorProduct = std::function<Values(
    const Values& objective_input,
    const Values& target,
    const Values& objective_input_tangent
)>;

enum class ObjectiveInputDomain {
    PreActivation,
    ActivatedOutput
};

/// <summary>
/// Direct data members of <c>ObjectiveFunctions</c>:
/// <para><c>sample_loss</c> (<c>SampleLossFunction</c>).</para>
/// <para><c>sample_loss_gradient</c> (<c>SampleLossGradientFunction</c>).</para>
/// <para><c>sample_loss_hessian_vector_product</c> (<c>SampleLossHessianVectorProduct</c>, optional).</para>
/// <para><c>input_domain</c> (<c>ObjectiveInputDomain</c>).</para>
/// </summary>
struct ObjectiveFunctions {
    SampleLossFunction sample_loss;
    SampleLossGradientFunction sample_loss_gradient;
    // First-order-only objectives remain valid for forward/backward APIs.
    // Parameter-HVP APIs require this loss-input HVP callback and reject
    // objectives that omit it.
    SampleLossHessianVectorProduct sample_loss_hessian_vector_product;
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

/// <summary>
/// Direct data members of <c>WolfeParameters</c>:
/// <para><c>sufficient_decrease</c> (<c>double</c>).</para>
/// <para><c>curvature</c> (<c>double</c>).</para>
/// <para><c>shrink_factor</c> (<c>double</c>).</para>
/// <para><c>maximum_step</c> (<c>double</c>).</para>
/// <para><c>maximum_trials</c> (<c>std::size_t</c>).</para>
/// <para><c>minimum_downhill_cosine</c> (<c>double</c>).</para>
/// </summary>
struct WolfeParameters {
    double sufficient_decrease{ 1e-4 };
    double curvature{ 0.9 };
    double shrink_factor{ 0.5 };
    double maximum_step{ 1.0 };
    std::size_t maximum_trials{ 30 };
    double minimum_downhill_cosine{ 0.0 };
};

/// <summary>
/// Direct data members of <c>LineSearchTrial</c>:
/// <para><c>step_size</c> (<c>double</c>).</para>
/// <para><c>loss</c> (<c>double</c>).</para>
/// <para><c>directional_derivative</c> (<c>double</c>).</para>
/// <para><c>sufficient_decrease</c> (<c>bool</c>).</para>
/// <para><c>strong_curvature</c> (<c>bool</c>).</para>
/// </summary>
struct LineSearchTrial {
    double step_size{};
    double loss{};
    double directional_derivative{};
    bool sufficient_decrease{};
    bool strong_curvature{};
};

/// <summary>
/// Direct data members of <c>WolfeEvaluation</c>:
/// <para><c>step_size</c> (<c>double</c>).</para>
/// <para><c>candidate_loss</c> (<c>double</c>).</para>
/// <para><c>initial_directional_derivative</c> (<c>double</c>).</para>
/// <para><c>candidate_directional_derivative</c> (<c>double</c>).</para>
/// <para><c>sufficient_decrease</c> (<c>bool</c>).</para>
/// <para><c>strong_curvature</c> (<c>bool</c>).</para>
/// <para><c>candidate_network</c> (<c>MLP</c>).</para>
/// <para><c>candidate_gradient</c> (<c>NetworkGradients</c>).</para>
/// <para><c>objective</c> (<c>ObjectiveFunctions</c>).</para>
/// </summary>
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

/// <summary>
/// Direct data members of <c>LineSearchResult</c>:
/// <para><c>status</c> (<c>LineSearchStatus</c>).</para>
/// <para><c>step_size</c> (<c>double</c>).</para>
/// <para><c>selected_loss</c> (<c>double</c>).</para>
/// <para><c>initial_directional_derivative</c> (<c>double</c>).</para>
/// <para><c>selected_directional_derivative</c> (<c>double</c>).</para>
/// <para><c>selected_network</c> (<c>MLP</c>).</para>
/// <para><c>selected_gradient</c> (<c>NetworkGradients</c>).</para>
/// <para><c>history</c> (<c>std::vector&lt;LineSearchTrial&gt;</c>).</para>
/// </summary>
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

/// <summary>
/// Direct data members of <c>TrainingStepResult</c>:
/// <para><c>updated</c> (<c>bool</c>).</para>
/// <para><c>previous_loss</c> (<c>double</c>).</para>
/// <para><c>new_loss</c> (<c>double</c>).</para>
/// <para><c>gradient_norm</c> (<c>double</c>).</para>
/// <para><c>effective_step_size</c> (<c>std::optional&lt;double&gt;</c>).</para>
/// <para><c>line_search</c> (<c>LineSearchResult</c>).</para>
/// </summary>
struct TrainingStepResult {
    bool updated{};
    double previous_loss{};
    double new_loss{};
    double gradient_norm{};
    std::optional<double> effective_step_size{};
    LineSearchResult line_search;
};

/// <summary>
/// Direct data members of <c>RegularizationTerm</c>:
/// <para><c>name</c> (<c>std::string</c>).</para>
/// <para><c>value</c> (<c>std::function&lt;double(const MLP&amp;)&gt;</c>).</para>
/// <para><c>add_gradient</c> (<c>std::function&lt;void(const MLP&amp;, NetworkGradients&amp;)&gt;</c>).</para>
/// <para><c>add_hessian_vector_product</c> (<c>std::function&lt;void(const MLP&amp;, const NetworkTangent&amp;, NetworkGradients&amp;)&gt;</c>, optional).</para>
/// <para><c>proximal</c> (<c>bool</c>).</para>
/// <para><c>proximal_update</c> (<c>std::function&lt;void(MLP&amp;, double effective_step_size)&gt;</c>).</para>
/// <para><c>smooth</c> (<c>bool</c>).</para>
/// <para><c>includes_biases</c> (<c>bool</c>).</para>
/// <para><c>coefficient</c> (<c>double</c>).</para>
/// </summary>
struct RegularizationTerm {
    std::string name;
    // Callbacks return the unscaled penalty and unscaled gradient. The
    // objective aggregation layer applies coefficient exactly once.
    std::function<double(const MLP&)> value;
    // Optional for a purely proximal term. A composite term, such as
    // proximal elastic net, may use this callback for its smooth component
    // and proximal_update for its non-smooth component.
    std::function<void(const MLP&, NetworkGradients&)> add_gradient;
    // Adds the unscaled regularization Hessian-vector product. It is used
    // only by complete-objective HVP APIs; the aggregation layer applies
    // coefficient exactly once, just as it does for add_gradient.
    std::function<void(const MLP&, const NetworkTangent&, NetworkGradients&)> add_hessian_vector_product;
    // The trainer invokes this callback after a base optimizer step, passing
    // the effective step size. It may coexist with add_gradient when the term
    // combines a smooth and a non-smooth penalty.
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

/// <summary>
/// Stores parameter-space tangents corresponding to a ForwardCache.
/// activations[0] is the all-zero input tangent because this API currently
/// differentiates only with respect to network parameters.
/// </summary>
struct ForwardTangent {
    std::vector<Values> activations;
    std::vector<Values> pre_activations;
};

/// <summary>
/// Validates a sequence of layer widths for a feed-forward network.
/// </summary>
void validate_network_architecture(const std::vector<std::size_t>& layer_sizes);

/// <summary>
/// Validates a network's layer shapes, parameters, and activation descriptors.
/// </summary>
void validate_network(const MLP& network);

/// <summary>
/// Counts the trainable weights and biases in a network.
/// </summary>
std::size_t parameter_count(const MLP& network);

/// <summary>
/// Creates a network with the requested architecture and zero-valued parameters.
/// </summary>
MLP make_zero_network(std::vector<std::size_t> layer_sizes);

/// <summary>
/// Initializes an existing network according to the selected strategy and seed.
/// </summary>
void initialize_network(
    MLP& network,
    InitializationType initialization_type,
    const std::vector<std::size_t>& layer_sizes,
    std::uint32_t seed
);

/// <summary>
/// Creates and initializes a multilayer perceptron from architecture settings.
/// </summary>
MLP make_mlp(
    const std::vector<std::size_t>& layer_sizes,
    std::uint32_t seed,
    InitializationType initialization_type =
        InitializationType::XavierGlorot
);

/// <summary>
/// Creates and initializes a multilayer perceptron from a network specification.
/// </summary>
MLP make_mlp(const NetworkSpec& specification);

/// <summary>
/// Evaluates the logistic sigmoid while avoiding exponential overflow.
/// </summary>
double stable_sigmoid(double x);

/// <summary>
/// Runs a network forward and stores intermediate activation values.
/// </summary>
ForwardCache forward_pass(const MLP& network, const Values& input);

/// <summary>
/// Propagates a parameter-space tangent through a previously computed forward pass.
/// </summary>
ForwardTangent forward_jvp(
    const MLP& network,
    const ForwardCache& primal_cache,
    const NetworkTangent& parameter_tangent
);

/// <summary>
/// Creates zero-valued gradients with the same layout as a network.
/// </summary>
NetworkGradients make_zero_gradients_like(const MLP& network);

/// <summary>
/// Backpropagates the default binary-cross-entropy objective for one sample.
/// </summary>
NetworkGradients backward(const MLP& network, const ForwardCache& cache, const Values& target);

/// <summary>
/// Backpropagates a specified objective function for one sample.
/// </summary>
NetworkGradients backward(const MLP& network, const ForwardCache& cache, const Values& target, const ObjectiveFunctions& objective);

/// <summary>
/// Computes the default BCE-from-logits sample-loss Hessian-vector product.
/// </summary>
NetworkGradients loss_hessian_vector_product(
    const MLP& network,
    const ForwardCache& primal_cache,
    const Values& target,
    const NetworkTangent& parameter_tangent
);

/// <summary>
/// Computes one sample loss Hessian-vector product for a parameter tangent.
/// </summary>
NetworkGradients loss_hessian_vector_product(
    const MLP& network,
    const ForwardCache& primal_cache,
    const Values& target,
    const NetworkTangent& parameter_tangent,
    const ObjectiveFunctions& objective
);

/// <summary>
/// Averages default-objective gradients across a non-empty batch.
/// </summary>
NetworkGradients batch_gradients(
    const MLP& network,
    const Dataset& batch
);

/// <summary>
/// Averages gradients for a specified objective across a non-empty batch.
/// </summary>
NetworkGradients batch_gradients(
    const MLP& network,
    const Dataset& batch,
    const ObjectiveFunctions& objective
);

/// <summary>
/// Computes the Euclidean norm of all gradient components.
/// </summary>
double gradient_l2_norm(
    const NetworkGradients& gradients
);

/// <summary>
/// Returns the largest absolute gradient component.
/// </summary>
double maximum_absolute_gradient(
    const NetworkGradients& gradients
);

/// <summary>
/// Determines whether a learning rate is finite and strictly positive.
/// </summary>
bool learning_rate_is_valid(double learning_rate) noexcept;

/// <summary>
/// Validates that gradients match a network's parameter layout.
/// </summary>
void validate_gradients_like(
    const MLP& network,
    const NetworkGradients& gradients
);

/// <summary>
/// Applies a scaled gradient update to a network in place.
/// </summary>
void apply_gradient(
    MLP& network,
    const NetworkGradients& gradients,
    double learning_rate
);
