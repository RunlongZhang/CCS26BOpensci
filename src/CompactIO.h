#ifndef COMPACTIO_H
#define COMPACTIO_H

#include <vector>
#include <string>
#include <fstream>
#include <cstdint>    // For int8_t, uint64_t
#include <stdexcept>
#include <print>      // C++23/26
#include <format>     // C++23/26

// The container for our data in RAM
struct CompactDataset {
    uint64_t count;   // Number of hyperplanes
    uint64_t dim;     // Dimension
    std::vector<int8_t> data; // Flattened array [a1, a2, ..., ad, a1, a2, ...]

    // Helper to get coefficient at hyperplane (row) and dimension (col)
    int8_t at(size_t row, size_t col) const {
        return data[row * dim + col];
    }
};

class CompactIO {
public:
    // Dumps the data from RAM to a binary file
    static void save(const std::string& filename, const CompactDataset& ds) {
        // std::ios::binary is CRITICAL. It stops OS from corrupting data.
        std::ofstream out(filename, std::ios::binary);
        if (!out) throw std::runtime_error("Could not open file for writing");

        // 1. Write Header (Metadata)
        out.write(reinterpret_cast<const char*>(&ds.count), sizeof(ds.count));
        out.write(reinterpret_cast<const char*>(&ds.dim), sizeof(ds.dim));

        // 2. Write Data (Raw Bytes)
        // We dump the entire vector's internal memory in one command.
        out.write(reinterpret_cast<const char*>(ds.data.data()), ds.data.size() * sizeof(int8_t));
    }

    // Loads the data from a binary file into RAM
    static CompactDataset load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) throw std::runtime_error(std::format("Could not open file '{}'", filename));

        CompactDataset ds;

        // 1. Read Header
        in.read(reinterpret_cast<char*>(&ds.count), sizeof(ds.count));
        in.read(reinterpret_cast<char*>(&ds.dim), sizeof(ds.dim));

        // 2. Allocate Memory
        size_t total_size = ds.count * ds.dim;
        ds.data.resize(total_size);

        // 3. Read Data
        // We read the entire file's data block directly into the vector's memory.
        in.read(reinterpret_cast<char*>(ds.data.data()), total_size * sizeof(int8_t));

        if (!in) {
            std::println("Warning: File read ended early. File might be truncated.");
        }

        return ds;
    }
};

#endif // COMPACTIO_H