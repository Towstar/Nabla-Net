#include <exception>
#include <iostream>

void run_all_self_checks();

int main()
{
    try {
        std::cout << "C++ Configurable MLP Backpropagation Lab\n";
        run_all_self_checks();
    }
    catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << '\n';
        return 1;
    }

    return 0;
}
