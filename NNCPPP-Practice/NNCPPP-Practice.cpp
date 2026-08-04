#include <exception>
#include <iostream>

#include "mlp.hpp"
#include "objective_functions.hpp"

void run_all_self_checks();

int main()
{
    try {
        std::cout << "C++ Configurable MLP Backpropagation Lab\n";
        run_all_self_checks();
        const Dataset training_data{
            { { -1.0, -1.0 }, { 1.0, 0.0, 0.0 } },
            { { -0.8, -1.2 }, { 1.0, 0.0, 0.0 } },
            { { -1.2, -0.8 }, { 1.0, 0.0, 0.0 } },
            { { -0.9, -1.1 }, { 1.0, 0.0, 0.0 } },

            { { 1.0, -1.0 }, { 0.0, 1.0, 0.0 } },
            { { 0.8, -1.2 }, { 0.0, 1.0, 0.0 } },
            { { 1.2, -0.8 }, { 0.0, 1.0, 0.0 } },
            { { 0.9, -1.1 }, { 0.0, 1.0, 0.0 } },

            { { 0.0, 1.0 }, { 0.0, 0.0, 1.0 } },
            { { -0.2, 0.8 }, { 0.0, 0.0, 1.0 } },
            { { 0.2, 1.2 }, { 0.0, 0.0, 1.0 } },
            { { 0.0, 0.8 }, { 0.0, 0.0, 1.0 } }
        };

        std::cout << "Loaded " << training_data.size()
                  << " three-class training samples.\n";

        MLP network = make_mlp({ 2, 6, 3 }, 727);
        auto grads = batch_gradients(network, training_data, make_softmax_cross_entropy_objective());
		std::cout << "Batch gradients computed successfully.\n";
        apply_gradient(network, grads, 0.01);

    }
    catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << '\n';
        return 1;
    }


    return 0;
}
