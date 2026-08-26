#include <iostream>

#include <nncpp/nncpp.hpp>

int main()
{
    const MLP network = make_mlp({ 2, 3, 1 }, 7);

    // Choose an L1 strategy independently of the elastic-net strengths.
    // Proximal L1 combines a normal L2 gradient with post-step L1
    // soft-thresholding in train().
    ElasticNetOptions regularization;
    regularization.l1_coefficient = 1e-4;
    regularization.l2_coefficient = 5e-4;
    regularization.l1.method = L1Method::Proximal;
    regularization.l1.include_biases = false;
    const ObjectiveConfig objective = make_objective_config(
        make_binary_cross_entropy_objective(),
        make_elastic_net_regularization(regularization)
    );

    std::cout << "NablaNet library example: "
              << parameter_count(network) << " parameters; "
              << objective.regularizers.front().name << " configured\n";
    return 0;
}
