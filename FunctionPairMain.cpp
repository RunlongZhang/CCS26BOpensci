 #include "FunctionPairGenerator.h"
#include "CompactIO.h"

#include <string>
#include <format>
#include <print>
#include <cstdint>
#include <cstdlib>
#include <algorithm>

int main(int argc, char** argv) {
    // Default test parameters
    std::size_t n   = 5;  // number of functions
    std::size_t dim = 3;  // dimension

    if (argc >= 3) {
        n   = static_cast<std::size_t>(std::stoul(argv[1]));
        dim = static_cast<std::size_t>(std::stoul(argv[2]));
    }

    // std::println("=== Function + Pairwise Generator Test ===");
    // std::println("Requested: {} functions in {}D", n, dim);

    // 1) Run the generator (creates the two .bin files)
    Generator::run(n, dim);

    // 2) Load the generated datasets back
    const std::string func_file = std::format("{}_functions_{}d.bin", n, dim);
    const std::string pair_file = std::format("{}_pairwise_{}d.bin", n, dim);

    CompactDataset funcs;
    CompactDataset pairs;

    try {
        funcs = CompactIO::load(func_file);
        pairs = CompactIO::load(pair_file);
    } catch (const std::exception& e) {
        // std::println("ERROR: Failed to load generated files: {}", e.what());
        return 1;
    }

    // std::println("\n--- Loaded Datasets ---");
    // std::println("Functions file:    '{}'", func_file);
    // std::println("  count = {}, dim = {}", funcs.count, funcs.dim);
    // std::println("Pairwise file:     '{}'", pair_file);
    // std::println("  count = {}, dim = {}", pairs.count, pairs.dim);

    const std::size_t expected_pairs = n * (n - 1) / 2;
    if (funcs.count != n || funcs.dim != dim) {
        // std::println("WARNING: Functions dataset header mismatch!");
    }
    if (pairs.count != expected_pairs || pairs.dim != dim) {
        // std::println("WARNING: Pairwise dataset header mismatch!");
    }

    // Helper to access Ai,k (1-based i, 0-based k)
    auto get_A = [&](std::size_t i_id, std::size_t k) -> int {
        return static_cast<int>(funcs.data[(i_id - 1) * dim + k]);
    };
    // Helper to access H_ij,k (1-based i,j, 0-based k)
    auto get_H = [&](std::size_t i_id, std::size_t j_id, std::size_t k) -> int {
        std::size_t idx = Generator::pair_index(i_id, j_id, n);
        return static_cast<int>(pairs.data[idx * dim + k]);
    };

    // 3) Verify that each pairwise hyperplane equals Ai - Aj
    bool all_ok = true;
    for (std::size_t i_id = 1; i_id <= n; ++i_id) {
        for (std::size_t j_id = i_id + 1; j_id <= n; ++j_id) {
            for (std::size_t k = 0; k < dim; ++k) {
                int expected = get_A(i_id, k) - get_A(j_id, k);
                int actual   = get_H(i_id, j_id, k);
                if (expected != actual) {
                    all_ok = false;
                    // std::println(
                    //     "MISMATCH for pair (F{}, F{}) at dim {}: expected {}, got {}",
                    //     i_id, j_id, k, expected, actual
                    // );
                }
            }
        }
    }

    if (all_ok) {
        // std::println("\n✅ All pairwise hyperplanes H_ij match Ai - Aj exactly.");
    } else {
        // std::println("\n❌ Some pairwise hyperplanes did not match Ai - Aj.");
    }

    // 4) Print some sample coefficients for visual inspection
    // std::println("\n--- Sample Functions (Ai) ---");
    std::size_t max_show_funcs = std::min<std::size_t>(n, 3);
    for (std::size_t i_id = 1; i_id <= max_show_funcs; ++i_id) {
        // std::print("F{}: [", i_id);
        for (std::size_t k = 0; k < dim; ++k) {
            // std::print("{}", get_A(i_id, k));
            if (k + 1 < dim) {
                // std::print(", ");
            }
        }
        // std::println("]");
    }

    // std::println("\n--- Sample Pairwise Hyperplanes (Ai - Aj) ---");
    std::size_t max_i = std::min<std::size_t>(n, 3);
    for (std::size_t i_id = 1; i_id <= max_i; ++i_id) {
        for (std::size_t j_id = i_id + 1; j_id <= std::min<std::size_t>(n, i_id + 3); ++j_id) {
            // std::print("H_{}{} (F{} = F{}): [", i_id, j_id, i_id, j_id);
            for (std::size_t k = 0; k < dim; ++k) {
                // std::print("{}", get_H(i_id, j_id, k));
                if (k + 1 < dim) {
                    // std::print(", ");
                }
            }
            // std::println("]");
        }
    }

    // 5) Show how to list all hyperplanes involving a specific Fi
    if (n >= 3) {
        std::size_t target_i = std::min<std::size_t>(3, n);
        // std::println("\n--- All hyperplanes involving F{} ---", target_i);

        for (std::size_t j_id = target_i + 1; j_id <= n; ++j_id) {
            std::size_t idx = Generator::pair_index(target_i, j_id, n);
            // std::println("Pair (F{}, F{}) -> pair_index = {}", target_i, j_id, idx);
        }
        for (std::size_t k_id = 1; k_id < target_i; ++k_id) {
            std::size_t idx = Generator::pair_index(k_id, target_i, n);
            // std::println("Pair (F{}, F{}) -> pair_index = {}", k_id, target_i, idx);
        }
    }

    // std::println("\n=== Test Completed ===");
    return all_ok ? 0 : 1;
}