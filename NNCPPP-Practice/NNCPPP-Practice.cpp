#include <exception>
#include <iostream>

#include "mlp.hpp"
#include "objective_functions.hpp"
#include "optimizers.hpp"
#include "self_checks.hpp"
#include "train.hpp"

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

        TrainingConfig config;
        config.batch_mode = BatchMode::FullBatch;
        config.batch_size = 0;
        config.max_iterations = 100;
        config.max_epochs = 3;
        config.gradient_tolerance = 0.001;
        config.parameter_change_tolerance = 0.0001;
        config.shuffle = true;
        config.shuffle_seed = 41;
        config.record_history = true;

        const ObjectiveFunctions data_objective =
            make_softmax_cross_entropy_objective();
        const ObjectiveConfig objective = make_objective_config(
            data_objective,
            make_l2_regularization(0.0001)
        );

        std::cout << train(
            network,
            training_data,
            make_sgd(),
            config,
            objective
        );

    }
    catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << '\n';
        return 1;
    }
    return 0;
}
