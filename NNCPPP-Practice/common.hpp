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
    explicit CsvReader(const std::filesystem::path& path);

    const CsvRow& headers() const noexcept;
    bool next(CsvRow& row);

private:
    bool read_record(std::string& record);
    static CsvRow parse_record(const std::string& record);

    std::filesystem::path path_;
    std::ifstream input_;
    CsvRow headers_;
};

std::filesystem::path locate_data_file(
    const std::filesystem::path& relative_path
);

std::unordered_map<std::string, std::size_t> make_column_index(
    const CsvRow& headers
);

std::size_t require_column(
    const std::unordered_map<std::string, std::size_t>& columns,
    const std::string& name
);

bool is_missing_csv_value(const std::string& value);

std::optional<double> parse_optional_number(
    const std::string& value
);

double normalcdf(double z);
double normalpdf(double z);
double stable_softplus(double z);
