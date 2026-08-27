#include <nablanet/nablanet.hpp>

int main()
{
    const nablanet::MLP network = nablanet::make_zero_network({ 1, 1 });
    return nablanet::parameter_count(network) == 2 ? 0 : 1;
}
