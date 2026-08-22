#include "examples.hpp"

#include "common.hpp"
#include "mlp.hpp"
#include "objective_functions.hpp"
#include "optimizers.hpp"
#include "train.hpp"
#include "wolfe_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

/// <summary>
/// Direct data members of <c>NumericRows</c>:
/// <para><c>features</c> (<c>std::vector&lt;Values&gt;</c>).</para>
/// <para><c>targets</c> (<c>std::vector&lt;double&gt;</c>).</para>
/// </summary>
struct NumericRows {
    std::vector<Values> features;
    std::vector<double> targets;
};

/// <summary>
/// Direct data members of <c>DatasetSplit</c>:
/// <para><c>training</c> (<c>Dataset</c>).</para>
/// <para><c>validation</c> (<c>Dataset</c>).</para>
/// </summary>
struct DatasetSplit {
    Dataset training;
    Dataset validation;
};

/// <summary>
/// Direct data members of <c>MinMaxScaler</c>:
/// <para><c>minimum</c> (<c>Values</c>).</para>
/// <para><c>range</c> (<c>Values</c>).</para>
/// </summary>
struct MinMaxScaler {
    Values minimum;
    Values range;

    Values transform(const Values& row) const
    {
        if (row.size() != minimum.size()) {
            throw std::runtime_error(
                "Feature row width does not match the fitted scaler."
            );
        }

        Values transformed;
        transformed.reserve(row.size());
        for (std::size_t index = 0; index < row.size(); ++index) {
            transformed.push_back(
                range[index] <= 1e-12
                    ? 0.0
                    : (row[index] - minimum[index]) / range[index]
            );
        }
        return transformed;
    }
};

MinMaxScaler fit_min_max_scaler(const std::vector<Values>& feature_rows)
{
    if (feature_rows.empty() || feature_rows.front().empty()) {
        throw std::runtime_error(
            "Cannot fit a feature scaler to an empty feature set."
        );
    }

    const std::size_t width = feature_rows.front().size();
    MinMaxScaler scaler;
    scaler.minimum.assign(width, std::numeric_limits<double>::max());
    scaler.range.assign(width, 0.0);
    Values maximum(width, std::numeric_limits<double>::lowest());

    for (const Values& row : feature_rows) {
        if (row.size() != width) {
            throw std::runtime_error(
                "Feature rows do not have a consistent width."
            );
        }
        for (std::size_t column = 0; column < width; ++column) {
            scaler.minimum[column] =
                std::min(scaler.minimum[column], row[column]);
            maximum[column] = std::max(maximum[column], row[column]);
        }
    }

    for (std::size_t column = 0; column < width; ++column) {
        scaler.range[column] = maximum[column] - scaler.minimum[column];
    }

    return scaler;
}

/// <summary>
/// Direct data members of <c>TargetScaler</c>:
/// <para><c>minimum</c> (<c>double</c>).</para>
/// <para><c>range</c> (<c>double</c>).</para>
/// </summary>
struct TargetScaler {
    double minimum{};
    double range{};

    double transform(const double target) const
    {
        return range <= 1e-12
            ? 0.0
            : (target - minimum) / range;
    }
};

TargetScaler fit_target_scaler(const std::vector<double>& targets)
{
    if (targets.empty()) {
        throw std::runtime_error("Cannot fit a target scaler to no targets.");
    }

    const auto [minimum_it, maximum_it] =
        std::minmax_element(targets.begin(), targets.end());
    return TargetScaler{
        *minimum_it,
        *maximum_it - *minimum_it
    };
}

Dataset make_dataset_from_rows(
    const NumericRows& rows,
    const MinMaxScaler* feature_scaler,
    const TargetScaler* target_scaler
)
{
    if (rows.features.empty() || rows.features.size() != rows.targets.size()) {
        throw std::runtime_error(
            "Dataset rows must be non-empty and have matching sizes."
        );
    }

    Dataset dataset;
    dataset.reserve(rows.features.size());
    for (std::size_t index = 0; index < rows.features.size(); ++index) {
        Values features = feature_scaler == nullptr
            ? rows.features[index]
            : feature_scaler->transform(rows.features[index]);
        const double target = target_scaler == nullptr
            ? rows.targets[index]
            : target_scaler->transform(rows.targets[index]);
        dataset.push_back({ std::move(features), Values{ target } });
    }
    return dataset;
}

DatasetSplit make_deterministic_split(
    NumericRows rows,
    const bool scale_features,
    const bool scale_targets,
    const std::uint32_t seed
)
{
    if (rows.features.size() < 2 ||
        rows.features.size() != rows.targets.size()) {
        throw std::runtime_error(
            "A dataset split requires at least two matching rows."
        );
    }

    std::vector<std::size_t> order(rows.features.size());
    for (std::size_t index = 0; index < order.size(); ++index) {
        order[index] = index;
    }
    std::mt19937 random_engine(seed);
    std::shuffle(order.begin(), order.end(), random_engine);

    const std::size_t validation_count =
        std::max<std::size_t>(1, rows.features.size() / 5);
    const std::size_t training_count =
        rows.features.size() - validation_count;

    NumericRows training_rows;
    NumericRows validation_rows;
    training_rows.features.reserve(training_count);
    training_rows.targets.reserve(training_count);
    validation_rows.features.reserve(validation_count);
    validation_rows.targets.reserve(validation_count);

    for (std::size_t position = 0; position < order.size(); ++position) {
        const std::size_t source_index = order[position];
        NumericRows& destination = position < training_count
            ? training_rows
            : validation_rows;
        destination.features.push_back(
            std::move(rows.features[source_index])
        );
        destination.targets.push_back(rows.targets[source_index]);
    }

    std::optional<MinMaxScaler> feature_scaler;
    if (scale_features) {
        feature_scaler = fit_min_max_scaler(training_rows.features);
    }

    std::optional<TargetScaler> target_scaler;
    if (scale_targets) {
        target_scaler = fit_target_scaler(training_rows.targets);
    }

    return DatasetSplit{
        make_dataset_from_rows(
            training_rows,
            feature_scaler.has_value() ? &*feature_scaler : nullptr,
            target_scaler.has_value() ? &*target_scaler : nullptr
        ),
        make_dataset_from_rows(
            validation_rows,
            feature_scaler.has_value() ? &*feature_scaler : nullptr,
            target_scaler.has_value() ? &*target_scaler : nullptr
        )
    };
}

TrainingConfig make_mini_batch_config(
    const std::size_t batch_size,
    const std::size_t epochs,
    const std::uint32_t shuffle_seed
)
{
    TrainingConfig config;
    config.batch_mode = BatchMode::MiniBatch;
    config.batch_size = batch_size;
    config.max_iterations.reset();
    config.max_epochs = epochs;
    config.shuffle = true;
    config.shuffle_seed = shuffle_seed;
    config.record_history = true;
    return config;
}

TrainingConfig make_full_batch_config(const std::size_t epochs)
{
    TrainingConfig config;
    config.batch_mode = BatchMode::FullBatch;
    config.batch_size = 0;
    config.max_iterations.reset();
    config.max_epochs = epochs;
    config.shuffle = false;
    config.record_history = true;
    return config;
}

class ExampleAdamOptimizer final : public Optimizer
{
public:
    const char* name() const noexcept override
    {
        return "ExampleAdam";
    }

    void reset(const MLP& network) override
    {
        validate_network(network);
        first_moment_ = make_zero_gradients_like(network);
        second_moment_ = make_zero_gradients_like(network);
        update_index_ = 0;
    }

    TrainingStepResult step(OptimizerContext& context) override
    {
        validate_network(context.network);
        validate_objective_config(context.objective);
        validate_gradients_like(context.network, context.current_gradient);

        if (context.batch.empty()) {
            throw std::invalid_argument(
                "Example Adam cannot train on an empty batch."
            );
        }
        if (!std::isfinite(context.current_loss)) {
            throw std::invalid_argument(
                "Example Adam requires a finite current loss."
            );
        }

        ++update_index_;
        const double first_correction =
            1.0 - std::pow(beta1_, static_cast<double>(update_index_));
        const double second_correction =
            1.0 - std::pow(beta2_, static_cast<double>(update_index_));

        if (!std::isfinite(first_correction) ||
            !std::isfinite(second_correction) ||
            first_correction <= 0.0 ||
            second_correction <= 0.0) {
            throw std::runtime_error(
                "Example Adam bias correction is not finite."
            );
        }

        NetworkGradients next_first = first_moment_;
        NetworkGradients next_second = second_moment_;
        MLP candidate = context.network;

        for (std::size_t layer_index = 0;
             layer_index < candidate.layers.size();
             ++layer_index) {
            const LayerGradients& gradient =
                context.current_gradient.layers[layer_index];
            LayerGradients& first = next_first.layers[layer_index];
            LayerGradients& second = next_second.layers[layer_index];
            DenseLayer& layer = candidate.layers[layer_index];

            for (std::size_t index = 0;
                 index < layer.weights.size();
                 ++index) {
                const double value = gradient.weights[index];
                first.weights[index] =
                    beta1_ * first.weights[index] + (1.0 - beta1_) * value;
                second.weights[index] =
                    beta2_ * second.weights[index] +
                    (1.0 - beta2_) * value * value;

                const double first_hat =
                    first.weights[index] / first_correction;
                const double second_hat =
                    second.weights[index] / second_correction;
                const double update =
                    learning_rate_ * first_hat /
                    (std::sqrt(second_hat) + epsilon_);
                layer.weights[index] -= update;

                if (!std::isfinite(layer.weights[index])) {
                    throw std::overflow_error(
                        "Example Adam produced a non-finite weight."
                    );
                }
            }

            for (std::size_t index = 0;
                 index < layer.biases.size();
                 ++index) {
                const double value = gradient.biases[index];
                first.biases[index] =
                    beta1_ * first.biases[index] + (1.0 - beta1_) * value;
                second.biases[index] =
                    beta2_ * second.biases[index] +
                    (1.0 - beta2_) * value * value;

                const double first_hat =
                    first.biases[index] / first_correction;
                const double second_hat =
                    second.biases[index] / second_correction;
                const double update =
                    learning_rate_ * first_hat /
                    (std::sqrt(second_hat) + epsilon_);
                layer.biases[index] -= update;

                if (!std::isfinite(layer.biases[index])) {
                    throw std::overflow_error(
                        "Example Adam produced a non-finite bias."
                    );
                }
            }
        }

        validate_network(candidate);
        TrainingStepResult result{};
        result.updated = true;
        result.previous_loss = context.current_loss;
        result.gradient_norm = gradient_l2_norm(context.current_gradient);
        result.new_loss = objective_loss(
            candidate,
            context.batch,
            context.objective
        );

        context.network = std::move(candidate);
        first_moment_ = std::move(next_first);
        second_moment_ = std::move(next_second);
        return result;
    }

private:
    NetworkGradients first_moment_;
    NetworkGradients second_moment_;
    std::size_t update_index_{};
    double learning_rate_{ 0.01 };
    double beta1_{ 0.9 };
    double beta2_{ 0.999 };
    double epsilon_{ 1e-8 };
};

OptimizerSpec make_example_adam()
{
    return make_custom_optimizer(
        "ExampleAdam",
        OptimizerRequirement::MiniBatchCompatible,
        [] {
            return std::make_unique<ExampleAdamOptimizer>();
        }
    );
}

void print_training_report(
    const std::string& name,
    const DatasetSplit& split,
    const MLP& network,
    const TrainingReport& report,
    const ObjectiveConfig& objective
)
{
    std::cout << "\n" << name << "\n";
    std::cout << "------------------------------\n";
    std::cout << "Training samples: " << split.training.size() << "\n";
    std::cout << "Validation samples: " << split.validation.size() << "\n";
    std::cout << "Input width: "
              << split.training.front().input.size() << "\n";
    std::cout << report;
    std::cout << "VALIDATION LOSS: "
              << objective_loss(network, split.validation, objective) << "\n";
}

NumericRows load_college_majors_rows(const std::size_t desired_rows)
{
    const auto path = locate_data_file(
        "local-notes/CollegeMajorsDataset/college_majors_2026.csv"
    );
    CsvReader reader(path);
    const auto columns = make_column_index(reader.headers());

    const std::vector<std::string> feature_names{
        "institution_admission_rate",
        "institution_avg_sat",
        "institution_undergrad_enrollment",
        "institution_tuition_in_state_usd",
        "institution_tuition_out_state_usd",
        "median_debt_usd",
        "occupation_employment_2024_thousands",
        "occupation_growth_pct_2024_34"
    };
    const std::size_t target_column =
        require_column(columns, "earnings_vs_national_pct");

    std::vector<std::size_t> feature_columns;
    for (const std::string& name : feature_names) {
        feature_columns.push_back(require_column(columns, name));
    }

    std::vector<Values> features;
    std::vector<double> targets;
    CsvRow row;

    while (features.size() < desired_rows && reader.next(row)) {
        Values feature_row;
        feature_row.reserve(feature_columns.size());
        bool valid = true;

        for (const std::size_t column : feature_columns) {
            const auto value = parse_optional_number(row[column]);
            if (!value.has_value()) {
                valid = false;
                break;
            }
            feature_row.push_back(*value);
        }

        const auto target = parse_optional_number(row[target_column]);
        if (!target.has_value()) {
            valid = false;
        }

        if (valid) {
            features.push_back(std::move(feature_row));
            targets.push_back(*target);
        }
    }

    if (desired_rows != std::numeric_limits<std::size_t>::max() &&
        features.size() < desired_rows) {
        throw std::runtime_error(
            "Could not collect enough complete college-major rows."
        );
    }

    return NumericRows{ std::move(features), std::move(targets) };
}

NumericRows load_esports_rows(const std::size_t desired_rows)
{
    const auto path = locate_data_file(
        "local-notes/EsportsGamingBiometricDataset/"
        "esports_gaming_biometrics_250k.csv"
    );
    CsvReader reader(path);
    const auto columns = make_column_index(reader.headers());

    const std::vector<std::string> feature_names{
        "Age",
        "PlayTime_Hours",
        "Sleep_Hours",
        "Energy_Drinks",
        "Screen_Brightness",
        "APM",
        "Heart_Rate_BPM",
        "HRV_Score",
        "Reaction_Time_ms"
    };
    const std::size_t target_column =
        require_column(columns, "Match_Outcome");

    std::vector<std::size_t> feature_columns;
    for (const std::string& name : feature_names) {
        feature_columns.push_back(require_column(columns, name));
    }

    std::vector<Values> features;
    std::vector<double> targets;
    CsvRow row;

    while (features.size() < desired_rows && reader.next(row)) {
        Values feature_row;
        feature_row.reserve(feature_columns.size());
        bool valid = true;

        for (const std::size_t column : feature_columns) {
            const auto value = parse_optional_number(row[column]);
            if (!value.has_value()) {
                valid = false;
                break;
            }
            feature_row.push_back(*value);
        }

        double target = 0.0;
        if (row[target_column] == "Win") {
            target = 1.0;
        } else if (row[target_column] != "Loss") {
            valid = false;
        }

        if (valid) {
            features.push_back(std::move(feature_row));
            targets.push_back(target);
        }
    }

    if (desired_rows != std::numeric_limits<std::size_t>::max() &&
        features.size() < desired_rows) {
        throw std::runtime_error(
            "Could not collect enough complete esports rows."
        );
    }

    return NumericRows{ std::move(features), std::move(targets) };
}

NumericRows load_mushroom_rows(const std::size_t desired_rows)
{
    const auto path = locate_data_file(
        "local-notes/MushroomDataset/mushrooms.csv"
    );
    CsvReader reader(path);
    const auto columns = make_column_index(reader.headers());
    const std::size_t target_column = require_column(columns, "class");

    std::vector<CsvRow> raw_rows;
    CsvRow row;
    while (raw_rows.size() < desired_rows && reader.next(row)) {
        raw_rows.push_back(std::move(row));
    }

    if (raw_rows.size() < desired_rows) {
        throw std::runtime_error(
            "Could not collect enough mushroom rows."
        );
    }

    std::vector<std::size_t> feature_columns;
    for (std::size_t column = 0; column < reader.headers().size(); ++column) {
        if (column != target_column) {
            feature_columns.push_back(column);
        }
    }

    std::vector<std::map<std::string, std::size_t>> categories(
        feature_columns.size()
    );
    for (const CsvRow& raw_row : raw_rows) {
        for (std::size_t feature_index = 0;
             feature_index < feature_columns.size();
             ++feature_index) {
            categories[feature_index].emplace(
                raw_row[feature_columns[feature_index]],
                categories[feature_index].size()
            );
        }
    }

    std::vector<std::size_t> offsets(categories.size(), 0);
    std::size_t feature_width = 0;
    for (std::size_t index = 0; index < categories.size(); ++index) {
        offsets[index] = feature_width;
        feature_width += categories[index].size();
    }

    std::vector<Values> features;
    std::vector<double> targets;
    features.reserve(raw_rows.size());
    targets.reserve(raw_rows.size());

    for (const CsvRow& raw_row : raw_rows) {
        Values feature_row(feature_width, 0.0);
        for (std::size_t feature_index = 0;
             feature_index < feature_columns.size();
             ++feature_index) {
            const auto& category_map = categories[feature_index];
            const auto found = category_map.find(
                raw_row[feature_columns[feature_index]]
            );
            if (found == category_map.end()) {
                throw std::runtime_error(
                    "Mushroom category was not found during encoding."
                );
            }
            feature_row[offsets[feature_index] + found->second] = 1.0;
        }

        if (raw_row[target_column] == "p") {
            targets.push_back(1.0);
        } else if (raw_row[target_column] == "e") {
            targets.push_back(0.0);
        } else {
            throw std::runtime_error(
                "Unexpected mushroom class: " + raw_row[target_column]
            );
        }
        features.push_back(std::move(feature_row));
    }

    return NumericRows{ std::move(features), std::move(targets) };
}

void print_wolfe_result(
    const std::size_t iteration,
    const TrainingStepResult& result
)
{
    std::cout << "Wolfe step " << iteration
              << ": updated=" << (result.updated ? "yes" : "no")
              << ", previous_loss=" << result.previous_loss
              << ", new_loss=" << result.new_loss;

    if (result.line_search.status != LineSearchStatus::Failed) {
        std::cout << ", status="
                  << line_search_status_name(result.line_search.status)
                  << ", selected_step="
                  << result.line_search.step_size;
    }
    std::cout << '\n';

    for (std::size_t trial_index = 0;
         trial_index < result.line_search.history.size();
         ++trial_index) {
        const LineSearchTrial& trial =
            result.line_search.history[trial_index];
        std::cout << "  trial " << trial_index
                  << ": step=" << trial.step_size
                  << ", loss=" << trial.loss
                  << ", sufficient_decrease="
                  << (trial.sufficient_decrease ? "PASS" : "FAIL")
                  << ", strong_curvature="
                  << (trial.strong_curvature ? "PASS" : "FAIL")
                  << '\n';
    }
}

} // namespace

void run_college_majors_example()
{
    // Keep the console example quick while retaining a deterministic sample
    // from the real dataset. Increase this cap when extending the example.
    constexpr std::size_t example_rows = 8192;
    std::cout << "\nCollege majors example: loading data...\n"
              << std::flush;
    const DatasetSplit split = make_deterministic_split(
        load_college_majors_rows(example_rows),
        true,
        true,
        2026
    );
    std::cout << "Prepared " << split.training.size()
              << " training rows and " << split.validation.size()
              << " validation rows.\n" << std::flush;
    MLP network = make_mlp({ 8, 12, 1 }, 2026);
    const ObjectiveConfig objective = make_objective_config(
        make_mean_squared_error_objective()
    );

    const TrainingReport report = train(
        network,
        split.training,
        make_example_adam(),
        make_full_batch_config(6),
        objective
    );

    print_training_report(
        "College majors example: full-batch adaptive regression",
        split,
        network,
        report,
        objective
    );
}

void run_esports_biometrics_example()
{
    // Keep the console example quick while retaining a deterministic sample
    // from the real dataset. Increase this cap when extending the example.
    constexpr std::size_t example_rows = 8192;
    const DatasetSplit split = make_deterministic_split(
        load_esports_rows(example_rows),
        true,
        false,
        2027
    );
    MLP network = make_mlp({ 9, 12, 1 }, 2027);
    const ObjectiveConfig objective = make_objective_config(
        make_binary_cross_entropy_objective()
    );

    const TrainingConfig config = make_mini_batch_config(64, 5, 17);
    const SGDOptions options{
        [](const std::size_t) {
            return 0.25;
        }
    };

    const TrainingReport report = train(
        network,
        split.training,
        make_sgd(options),
        config,
        objective
    );

    print_training_report(
        "Esports biometrics example: mini-batch classification",
        split,
        network,
        report,
        objective
    );
}

void run_mushroom_example()
{
    const DatasetSplit split = make_deterministic_split(
        load_mushroom_rows(8124),
        false,
        false,
        2028
    );
    NetworkSpec specification;
    specification.layer_sizes = {
        split.training.front().input.size(),
        24,
        1
    };
    specification.seed = 2028;
    specification.layer_activations = {
        Activations::Tanh,
        Activations::Sigmoid
    };

    MLP network = make_mlp(specification);
    const ObjectiveFunctions objective =
        make_binary_cross_entropy_objective();

    WolfeParameters parameters;
    parameters.sufficient_decrease = 1e-4;
    parameters.curvature = 0.9;
    parameters.shrink_factor = 0.5;
    parameters.maximum_step = 1.0;
    parameters.maximum_trials = 20;

    std::cout << "\nMushroom example: deterministic full-batch Wolfe descent\n";
    std::cout << "-----------------------------------------------\n";
    std::cout << "Training samples: " << split.training.size() << '\n';
    std::cout << "Validation samples: " << split.validation.size() << '\n';
    std::cout << "Input width: "
              << split.training.front().input.size() << '\n';

    constexpr std::size_t wolfe_steps = 8;
    for (std::size_t iteration = 1;
         iteration <= wolfe_steps;
         ++iteration) {
        const TrainingStepResult result = take_wolfe_gradient_step(
            network,
            split.training,
            parameters,
            1e-8,
            objective
        );
        print_wolfe_result(iteration, result);
        if (!result.updated) {
            break;
        }
    }

    std::cout << "Mushroom training loss: "
              << batch_loss(network, split.training, objective) << '\n';
    std::cout << "Mushroom validation loss: "
              << batch_loss(network, split.validation, objective) << '\n';
}
