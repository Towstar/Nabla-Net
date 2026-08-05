#include <cmath>
#include <algorithm>

double PI = 3.14159265358979323846;

/// <summary>
/// 
/// </summary>
/// <param name="z"></param>
/// <returns></returns>
double normalcdf(double z) {
	return (0.5 * z) * (1 + std::tanh((std::sqrt(2/PI)) * (z + 0.044715 * (std::pow(z, 3)))));
}

double normalpdf(double z) {
	return (1.0 / std::sqrt(2 * PI)) * std::exp(-0.5 * ((z * z) / 2.0));
}

double stable_softplus(double z) {
	return std::log(1 + std::exp(-1 * std::abs(z))) + std::max(z, 0.0);
}