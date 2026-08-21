#include "common.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require_near(double actual, double expected, double tolerance, const char* message)
{
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}
}

int main()
{
    try {
        const double pi = 3.14159265358979323846;
        require_near(normalcdf(0.0), 0.5, 1e-15, "normalcdf(0) must equal 0.5");
        require_near(
            normalpdf(0.0),
            1.0 / std::sqrt(2.0 * pi),
            1e-15,
            "normalpdf(0) must equal the standard-normal density"
        );
        require_near(normalcdf(1.25), 1.0 - normalcdf(-1.25), 1e-15,
            "normalcdf must be symmetric");
        require_near(normalpdf(1.25), normalpdf(-1.25), 1e-15,
            "normalpdf must be even");

        bool rejected_nonfinite = false;
        try {
            static_cast<void>(normalcdf(std::numeric_limits<double>::quiet_NaN()));
        } catch (const std::invalid_argument&) {
            rejected_nonfinite = true;
        }
        if (!rejected_nonfinite) {
            throw std::runtime_error("normalcdf must reject non-finite input");
        }

        std::cout << "[PASS] normal distribution helpers\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << '\n';
        return 1;
    }
}
