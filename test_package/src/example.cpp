#include <nablanet/nablanet.hpp>

int main()
{
    const nablanet::MLP network = nablanet::make_zero_network({ 1, 1 });
    const nablanet::ObjectiveFunctions objective =
        nablanet::make_exponential_objective();

    return nablanet::parameter_count(network) == 2 && objective.sample_loss ? 0 : 1;
}
