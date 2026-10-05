#include <nablanet/nablanet.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>

namespace {

constexpr std::size_t input_size = 16;
constexpr std::size_t sample_count = 8192;
constexpr std::size_t warmup_iterations = 3;
constexpr std::size_t measured_iterations = 20;
constexpr std::uint32_t network_seed = 727;
constexpr std::uint32_t dataset_seed = 2026;

nablanet::Dataset make_benchmark_dataset()
{
    std::mt19937 random_engine(dataset_seed);
    std::uniform_real_distribution<double> input_distribution(-1.0, 1.0);

    nablanet::Dataset batch;
    batch.reserve(sample_count);

    for (std::size_t sample_index = 0; sample_index < sample_count; ++sample_index) {
        nablanet::Sample sample;
        sample.input.resize(input_size);

        // A simple binary classification rule gives each input one 0/1 target.
        double score = 0.0;
        for (std::size_t feature_index = 0; feature_index < input_size; ++feature_index) {
            const double value = input_distribution(random_engine);
            sample.input[feature_index] = value;
            score += (feature_index % 2 == 0) ? value : -value;
        }
        sample.target = {score > 0.0 ? 1.0 : 0.0};
        batch.push_back(std::move(sample));
    }

    return batch;
}

double loss_sum_range(
    const nablanet::MLP& network,
    const nablanet::Dataset& batch,
    const nablanet::ObjectiveFunctions& objective,
    std::size_t begin,
    std::size_t end
) {
    if (begin > end || end > batch.size()) {
        throw std::invalid_argument("Invalid range for loss_sum_range.");
    }
    double sum = 0.0;
    for (std::size_t index = begin; index < end; ++index) {
        const nablanet::Sample& sample = batch[index];
        const nablanet::ForwardCache cache = nablanet::forward_pass(network, sample.input);
        sum += nablanet::sample_cost(network, cache, sample.target, objective);
    }
    return sum;
}

} // namespace

int main()
{
#ifndef NDEBUG
    std::cerr << "Warning: assertions are enabled; use a Release build for timing.\n";
#endif

    // Setup and warmup are outside the measured region.
    const nablanet::MLP network = nablanet::make_mlp(
        {input_size, 32, 1}, network_seed, nablanet::InitializationType::KaimingHe
    );
    const nablanet::Dataset batch = make_benchmark_dataset();
    const nablanet::ObjectiveFunctions objective = nablanet::make_binary_cross_entropy_objective();
    const double reference_loss = nablanet::batch_loss(network, batch, objective);
    constexpr std::size_t range_counts[] = {4, 3};

    // Check both equal and unequal ranges before warmup and timing.
    for (const std::size_t range_count : range_counts) {
        double partitioned_sum = 0.0;
        for (std::size_t range = 0; range < range_count; ++range) {
            const std::size_t begin = batch.size() * range / range_count;
            const std::size_t end = batch.size() * (range + 1) / range_count;

            partitioned_sum += loss_sum_range(
                network, batch, objective, begin, end
            );
        }

        // Divide once so unequal ranges get the correct weight.
        const double partitioned_loss = partitioned_sum / static_cast<double>(batch.size());
        const double error = std::abs(partitioned_loss - reference_loss);
        std::cout << "Partition error (" << range_count << " ranges): " << error << '\n';
        if (!std::isfinite(error) || error > 1e-12) {
            std::cerr << "Partitioned loss does not match the serial reference.\n";
            return 1;
        }
    }

    double warmup_loss = 0.0;
    for (std::size_t iteration = 0; iteration < warmup_iterations; ++iteration) {
        warmup_loss = nablanet::batch_loss(network, batch, objective);
    }
    if (!std::isfinite(warmup_loss)) {
        std::cerr << "Warmup produced a non-finite loss.\n";
        return 1;
    }

    double checksum = 0.0;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < measured_iterations; ++iteration) {
        checksum += nablanet::batch_loss(network, batch, objective);
    }
    const auto end = std::chrono::steady_clock::now();

    if (!std::isfinite(checksum)) {
        std::cerr << "Timed calls produced a non-finite checksum.\n";
        return 1;
    }

    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double average_ms = elapsed_ms / static_cast<double>(measured_iterations);

    // Printing the accumulated losses keeps the timed results observable.
    std::cout << "NablaNet serial batch_loss baseline\n"
              << "Network: 16 -> 32 -> 1 (" << nablanet::parameter_count(network) << " parameters)\n"
              << "Network seed: " << network_seed << "; dataset seed: " << dataset_seed << '\n'
              << "Samples: " << batch.size() << "; inputs per sample: " << input_size << '\n'
              << "Warmup calls: " << warmup_iterations << "; timed calls: " << measured_iterations << '\n'
              << std::fixed << std::setprecision(3)
              << "Average ms per batch_loss: " << average_ms << '\n'
              << std::setprecision(17)
              << "Loss per batch: " << warmup_loss << '\n'
              << "Checksum (sum of timed losses): " << checksum << '\n';
    return 0;
}
