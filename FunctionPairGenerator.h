#pragma once

#include <cstddef> // for std::size_t

namespace Generator {

    /**
     * Entry point for this generator (similar to other Generator::run functions).
     * n_functions: number of base functions Fi to generate
     * dim:         dimension of the space
     */
    void run(std::size_t n_functions, std::size_t dim);

    /**
     * Generate:
     *  1) n base functions Fi(x) = Ai x + 0, with Ai ∈ [0,100]^dim (integers)
     *     - stored in "<n>_functions_<dim>d.bin"
     *     - function ids are 1..n by position
     *
     *  2) All pairwise equality hyperplanes Fi = Fj  (i < j):
     *       (Ai - Aj) x = 0
     *     - stored in "<n>_pairwise_<dim>d.bin"
     *     - in lexicographic order of pairs: (1,2), (1,3), ..., (1,n), (2,3), ..., (n-1,n)
     */
    void generate_functions_and_pairs(std::size_t n, std::size_t dim);

    /**
     * Compute the 0-based index of the pair (i, j) in the pairwise array,
     * assuming:
     *   - 1 <= i < j <= n
     *   - pairs are stored in lexicographic order:
     *       (1,2), (1,3), ..., (1,n),
     *       (2,3), (2,4), ..., (2,n),
     *       ...
     *       (n-1,n)
     *
     * This is useful when you want to retrieve "Fi-related" hyperplanes from
     * the flat CompactDataset for the pairwise file.
     */
    inline std::size_t pair_index(std::size_t i, std::size_t j, std::size_t n) {
        // number of pairs in rows 1..(i-1), where row k has (n - k) pairs
        std::size_t before = (i - 1) * (2 * n - i) / 2;
        return before + (j - i - 1);
    }

} // namespace Generator