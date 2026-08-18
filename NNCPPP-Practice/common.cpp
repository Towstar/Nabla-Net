#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

double PI = 3.14159265358979323846;

CsvReader::CsvReader(const std::filesystem::path& path)
    : path_(path), input_(path)
{
    if (!input_) {
        throw std::runtime_error(
            "Could not open CSV file: " + path.string()
        );
    }

    std::string header_record;
    if (!read_record(header_record)) {
        throw std::runtime_error(
            "CSV file is empty: " + path.string()
        );
    }
    headers_ = parse_record(header_record);
}

const CsvRow& CsvReader::headers() const noexcept
{
    return headers_;
}

bool CsvReader::next(CsvRow& row)
{
    std::string record;
    if (!read_record(record)) {
        return false;
    }

    row = parse_record(record);
    if (row.size() != headers_.size()) {
        throw std::runtime_error(
            "CSV row has " + std::to_string(row.size()) +
            " fields but the header has " +
            std::to_string(headers_.size()) + " fields in " +
            path_.string()
        );
    }
    return true;
}

bool CsvReader::read_record(std::string& record)
{
    // The local-notes files are line-oriented CSV files.  Read complete lines
    // here, then let parse_record handle quoted commas and escaped quotes.
    return static_cast<bool>(std::getline(input_, record));
}

CsvRow CsvReader::parse_record(const std::string& record)
{
    CsvRow fields;
    std::string field;
    bool in_quotes = false;

    for (std::size_t index = 0; index < record.size(); ++index) {
        const char character = record[index];

        if (in_quotes) {
            if (character == '"') {
                if (index + 1 < record.size() &&
                    record[index + 1] == '"') {
                    field.push_back('"');
                    ++index;
                } else {
                    in_quotes = false;
                }
            } else {
                field.push_back(character);
            }
            continue;
        }

        if (character == '"') {
            if (!field.empty()) {
                throw std::runtime_error(
                    "Unexpected quote in unquoted CSV field."
                );
            }
            in_quotes = true;
        } else if (character == ',') {
            fields.push_back(std::move(field));
            field.clear();
        } else if (character != '\r') {
            field.push_back(character);
        }
    }

    if (in_quotes) {
        throw std::runtime_error(
            "Unterminated quoted field in CSV record."
        );
    }

    fields.push_back(std::move(field));
    return fields;
}

std::filesystem::path locate_data_file(
    const std::filesystem::path& relative_path
)
{
    std::filesystem::path current =
        std::filesystem::absolute(std::filesystem::current_path());

    for (std::size_t depth = 0; depth < 10; ++depth) {
        const std::vector<std::filesystem::path> candidates{
            current / relative_path,
            current / "NNCPPP-Practice" / relative_path,
            current / "NeuralNetworkCPP" / "NNCPPP-Practice" /
                relative_path
        };

        for (const auto& candidate : candidates) {
            if (std::filesystem::is_regular_file(candidate)) {
                return candidate;
            }
        }

        const std::filesystem::path parent = current.parent_path();
        if (parent == current) {
            break;
        }
        current = parent;
    }

    throw std::runtime_error(
        "Could not locate example data file '" +
        relative_path.string() +
        "'. Run the program from the project tree containing local-notes."
    );
}

std::unordered_map<std::string, std::size_t> make_column_index(
    const CsvRow& headers
)
{
    std::unordered_map<std::string, std::size_t> result;
    for (std::size_t index = 0; index < headers.size(); ++index) {
        if (!result.emplace(headers[index], index).second) {
            throw std::runtime_error(
                "CSV header contains a duplicate column: " + headers[index]
            );
        }
    }
    return result;
}

std::size_t require_column(
    const std::unordered_map<std::string, std::size_t>& columns,
    const std::string& name
)
{
    const auto found = columns.find(name);
    if (found == columns.end()) {
        throw std::runtime_error("Required CSV column is missing: " + name);
    }
    return found->second;
}

bool is_missing_csv_value(const std::string& value)
{
    return value.empty() ||
        value == "NA" ||
        value == "N/A" ||
        value == "null" ||
        value == "not_available" ||
        value == "privacy_suppressed" ||
        value == "Not reported to IPEDS";
}

std::optional<double> parse_optional_number(const std::string& value)
{
    if (is_missing_csv_value(value)) {
        return std::nullopt;
    }

    try {
        std::size_t consumed = 0;
        const double parsed = std::stod(value, &consumed);
        if (consumed != value.size() || !std::isfinite(parsed)) {
            return std::nullopt;
        }
        return parsed;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

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
