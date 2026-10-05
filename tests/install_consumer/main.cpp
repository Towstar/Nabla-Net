#include <nablanet/nablanet.hpp>

int main()
{
    nablanet::MLP network = nablanet::make_zero_network({ 1, 1 });
    network.layers[0].weights[0] = 3.0;
    network.layers[0].biases[0] = 4.0;
    const auto l2 = nablanet::make_lp_norm(2.0);
    return nablanet::parameter_count(network) == 2 &&
        nablanet::network_norm(network, l2) == 5.0 &&
        nablanet::vector_norm({ 3.0, 4.0 }, l2) == 5.0 &&
        nablanet::network_norm(nablanet::make_zero_gradients_like(network), l2) == 0.0
        ? 0 : 1;
}
