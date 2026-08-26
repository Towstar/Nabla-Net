#include <nncpp/mlp.hpp>
#include <nncpp/objective_functions.hpp>

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr double finite_difference_step = 1e-6;

void require(const bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(
    const double actual,
    const double expected,
    const double tolerance,
    const std::string& message
)
{
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void require_values_near(
    const Values& actual,
    const Values& expected,
    const double tolerance,
    const char* message
)
{
    require(actual.size() == expected.size(),
        "HVP and finite-difference vectors must have matching sizes.");

    for (std::size_t index = 0; index < actual.size(); ++index) {
        require_near(
            actual[index], expected[index], tolerance,
            std::string(message) + " at index " + std::to_string(index)
        );
    }
}

template <typename Action>
void require_invalid_argument(Action&& action, const char* message)
{
    try {
        action();
    } catch (const std::invalid_argument&) {
        return;
    }

    throw std::runtime_error(message);
}

Values central_difference_loss_gradient(
    const ObjectiveFunctions& objective,
    const Values& objective_input,
    const Values& target,
    const Values& objective_input_tangent
)
{
    Values plus(objective_input);
    Values minus(objective_input);
    for (std::size_t index = 0; index < objective_input.size(); ++index) {
        plus[index] += finite_difference_step * objective_input_tangent[index];
        minus[index] -= finite_difference_step * objective_input_tangent[index];
    }

    const Values plus_gradient = objective.sample_loss_gradient(plus, target);
    const Values minus_gradient = objective.sample_loss_gradient(minus, target);
    Values result(objective_input.size(), 0.0);
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] =
            (plus_gradient[index] - minus_gradient[index]) /
            (2.0 * finite_difference_step);
    }

    return result;
}

void require_hvp_matches_loss_gradient_difference(
    const ObjectiveFunctions& objective,
    const Values& objective_input,
    const Values& target,
    const Values& objective_input_tangent,
    const char* objective_name
)
{
    require(static_cast<bool>(objective.sample_loss_hessian_vector_product),
        "Objective must provide a loss Hessian-vector-product callback.");

    const Values actual = objective.sample_loss_hessian_vector_product(
        objective_input, target, objective_input_tangent
    );
    const Values numerical = central_difference_loss_gradient(
        objective, objective_input, target, objective_input_tangent
    );

    require_values_near(actual, numerical, 2e-7, objective_name);
}

void test_bce_loss_hvp()
{
    const ObjectiveFunctions objective = make_binary_cross_entropy_objective();
    const Values logits{ 0.3, -0.7 };
    const Values target{ 0.8, 0.1 };
    const Values logits_tangent{ 0.4, -0.6 };

    require_hvp_matches_loss_gradient_difference(
        objective, logits, target, logits_tangent,
        "BCE loss HVP must match a central difference of its loss gradient"
    );

    require_invalid_argument(
        [&] {
            static_cast<void>(objective.sample_loss_hessian_vector_product(
                logits, target, Values{ 0.1 }
            ));
        },
        "BCE loss HVP must reject a mismatched tangent"
    );
}

void test_softmax_cross_entropy_loss_hvp()
{
    const ObjectiveFunctions objective = make_softmax_cross_entropy_objective();
    const Values logits{ 0.4, -0.7, 1.1 };
    const Values target{ 0.0, 1.0, 0.0 };
    const Values logits_tangent{ 0.3, -0.5, 0.2 };

    require_hvp_matches_loss_gradient_difference(
        objective, logits, target, logits_tangent,
        "Softmax cross-entropy loss HVP must match a central difference of its loss gradient"
    );

    require_invalid_argument(
        [&] {
            static_cast<void>(objective.sample_loss_hessian_vector_product(
                logits,
                Values{ 0.2, 0.2, 0.2 },
                logits_tangent
            ));
        },
        "Softmax cross-entropy loss HVP must reject targets that do not sum to one"
    );
}

void test_activated_output_mse_loss_hvp()
{
    const ObjectiveFunctions objective = make_mean_squared_error_objective();
    const Values outputs{ 0.31, 0.74 };
    const Values target{ 0.2, 0.85 };
    const Values output_tangent{ -0.4, 0.6 };

    require_hvp_matches_loss_gradient_difference(
        objective, outputs, target, output_tangent,
        "Activated-output MSE loss HVP must match a central difference of its loss gradient"
    );

    require_invalid_argument(
        [&] {
            static_cast<void>(objective.sample_loss_hessian_vector_product(
                outputs,
                target,
                Values{
                    std::numeric_limits<double>::quiet_NaN(),
                    0.1
                }
            ));
        },
        "MSE loss HVP must reject a non-finite output tangent"
    );
}

} // namespace

int main()
{
    try {
        test_bce_loss_hvp();
        test_softmax_cross_entropy_loss_hvp();
        test_activated_output_mse_loss_hvp();
        std::cout << "[PASS] objective loss Hessian-vector products\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << '\n';
        return 1;
    }
}
