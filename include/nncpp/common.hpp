#pragma once

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

using CsvRow = std::vector<std::string>;

// Small, dependency-free CSV utilities shared by executable examples and
// future data-ingestion demonstrations.  These utilities validate field
// counts and support quoted fields with escaped quotes.
class CsvReader {
public:
    /// <summary>
    /// Opens a CSV file and reads its header record.
    /// </summary>
    explicit CsvReader(const std::filesystem::path& path);

    /// <summary>
    /// Returns the header fields read from the CSV file.
    /// </summary>
    const CsvRow& headers() const noexcept;

    /// <summary>
    /// Reads the next CSV record into the supplied row.
    /// </summary>
    bool next(CsvRow& row);

private:
    /// <summary>
    /// Reads one logical CSV record, including embedded newlines in quoted fields.
    /// </summary>
    bool read_record(std::string& record);

    /// <summary>
    /// Parses a CSV record into unescaped field values.
    /// </summary>
    static CsvRow parse_record(const std::string& record);

    std::filesystem::path path_;
    std::ifstream input_;
    CsvRow headers_;
};

/// <summary>
/// Resolves a data file by searching from the current directory upward.
/// </summary>
std::filesystem::path locate_data_file(
    const std::filesystem::path& relative_path
);

/// <summary>
/// Builds a lookup from CSV header names to column indices.
/// </summary>
std::unordered_map<std::string, std::size_t> make_column_index(
    const CsvRow& headers
);

/// <summary>
/// Returns a required column index or throws when the column is absent.
/// </summary>
std::size_t require_column(
    const std::unordered_map<std::string, std::size_t>& columns,
    const std::string& name
);

/// <summary>
/// Determines whether a CSV field represents a missing value.
/// </summary>
bool is_missing_csv_value(const std::string& value);

/// <summary>
/// Parses a finite numeric CSV field, or returns no value for a missing field.
/// </summary>
std::optional<double> parse_optional_number(
    const std::string& value
);

/// <summary>
/// Evaluates the standard normal cumulative distribution function.
/// </summary>
double normalcdf(double z);

/// <summary>
/// Evaluates the standard normal probability density function.
/// </summary>
double normalpdf(double z);

/// <summary>
/// Evaluates softplus while avoiding exponential overflow.
/// </summary>
double stable_softplus(double z);
