#include "self_checks.hpp"

#include <exception>
#include <iostream>

int main()
{
    try {
        run_all_self_checks();
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << '\n';
        return 1;
    }
}
