#include "FunctionPairGenerator.h"

#include "CompactIO.h"    // for CompactDataset, CompactIO::save
#include <random>
#include <string>
#include <format>
#include <print>
#include <cstdint>
#include <vector>

namespace Generator {

    void generate_functions_and_pairs(std::size_t n, std::size_t dim) {
        std::println(">>> Running Function + Pairwise Generator");
        std::println("Target Count: {} functions", n);
        std::println("Target Dim:   {}D", dim);

        // RNG: A in [0,100]
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, 100);

        // ---------------- 1. Generate base functions Fi ----------------
        CompactDataset funcs;
        funcs.count = n;
        funcs.dim   = dim;
        funcs.data.reserve(n * dim);

        // Function id = i+1 (1-based) by position in funcs.data
        for (std::size_t i = 0; i < n; ++i) {
            bool all_zero = true;

            for (std::size_t d = 0; d < dim; ++d) {
                int val = distrib(gen);       // [0,100]
                if (val != 0) all_zero = false;
                funcs.data.push_back(static_cast<int8_t>(val));
            }

            // Avoid degenerate all-zero function
            if (all_zero) {
                // remove last dim entries and re-generate this function
                funcs.data.resize(funcs.data.size() - dim);
                --i;
                continue;
            }
        }

        // --------------- 2. Generate pairwise Fi = Fj ------------------
        const std::size_t n_pairs = n * (n - 1) / 2;

        CompactDataset pairs;
        pairs.count = n_pairs;
        pairs.dim   = dim;
        pairs.data.reserve(n_pairs * dim);

        // Accessor: Ai,k (i is 1-based function id, k is 0..dim-1)
        auto get_A = [&](std::size_t i_id, std::size_t k) -> int8_t {
            // i_id: 1-based -> index = i_id - 1
            return funcs.data[(i_id - 1) * dim + k];
        };

        // Store in lexicographic order (1,2), (1,3), ..., (1,n), (2,3), ...
        for (std::size_t i_id = 1; i_id <= n; ++i_id) {
            for (std::size_t j_id = i_id + 1; j_id <= n; ++j_id) {
                for (std::size_t k = 0; k < dim; ++k) {
                    int ai = static_cast<int>(get_A(i_id, k));
                    int aj = static_cast<int>(get_A(j_id, k));
                    int diff = ai - aj;  // in [-100,100], safe in int8_t
                    pairs.data.push_back(static_cast<int8_t>(diff));
                }
            }
        }

        // -------------------- 3. Save to disk --------------------------
        try {
            std::string func_filename = std::format("{}_functions_{}d.bin", n, dim);
            std::string pair_filename = std::format("{}_pairwise_{}d.bin", n, dim);

            CompactIO::save(func_filename, funcs);
            CompactIO::save(pair_filename, pairs);

            std::println("SUCCESS: Saved functions to   '{}'", func_filename);
            std::println("SUCCESS: Saved pairwise to    '{}'", pair_filename);
            std::println("Total pairwise hyperplanes:    {}", n_pairs);
        } catch (const std::exception& e) {
            std::println("Error saving file(s): {}", e.what());
        }
    }

    // New wrapper entry point
    void run(std::size_t n_functions, std::size_t dim) {
        generate_functions_and_pairs(n_functions, dim);
    }

} // namespace Generator