#include <iostream>

#include <nncpp/nncpp.hpp>

int main()
{
    const MLP network = make_mlp({ 2, 3, 1 }, 7);
    std::cout << "NeuralNetworkCPP library example: "
              << parameter_count(network) << " parameters\n";
    return 0;
}
