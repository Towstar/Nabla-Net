#include "detail/common.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

/// <summary>
///
/// </summary>
/// <param name="z"></param>
/// <returns></returns>
double normalcdf(double z) {
    if (!std::isfinite(z)) {
        throw std::invalid_argument("normalcdf input must be finite.");
    }
    return 0.5 * std::erfc(-z / std::sqrt(2.0));
}

double normalpdf(double z) {
    if (!std::isfinite(z)) {
        throw std::invalid_argument("normalpdf input must be finite.");
    }
    constexpr double pi = 3.14159265358979323846;
    return std::exp(-0.5 * z * z) / std::sqrt(2.0 * pi);
}

double stable_softplus(double z) {
	return std::log(1 + std::exp(-1 * std::abs(z))) + std::max(z, 0.0);
}

// SVD (Jacobi & BDC)

// FFT (for spectral layers & image processing)
