#pragma once
#define NOMINMAX
#include <windows.h>
#include <print>
#include <vector>
#include <string>
#include <format>
#include <stdexcept>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <Eigen/Dense>
#include <psapi.h>
#include <cstdlib>

// --- Project Headers ---
#include "CompactIO.h"
#include "PolytopeStructs.h"
#include "PolytopeOps.h"
#include "MFsTreeFull.h"
#include "FunctionPairGenerator.h"

int main(int argc, char* argv[]) {
    // ---------------------------------------------------------
    // 0. Argument Parsing
    // ---------------------------------------------------------
    if (argc != 4) {
        std::println("Usage: {} <count> <dim> <ADS>", argv[0]);
        return 1;
    }

    int n_functions = std::stoi(argv[1]);
    int dim = std::stoi(argv[2]);
    int ads = std::stoi(argv[3]);
    std::string filename = std::format("{}_pairwise_{}d.bin", n_functions, dim);

    namespace fs = std::filesystem;

    auto ensure_logs_dir = []() -> fs::path {
        fs::path logs_dir = "logs_FsTree_Storage";
        std::error_code ec;
        fs::path res_dir = "results";
        fs::create_directories(logs_dir, ec);
        fs::create_directories(res_dir, ec);
        return logs_dir;
        };

    // NEW: log file result_{dim}
    fs::path perf_dir = ensure_logs_dir();
    std::string log_filename = std::format("results/mfstree_result_{}_{}_{}", n_functions, dim, ads);
    std::ofstream log(log_filename, std::ios::trunc);
    if (!log) {
        throw std::runtime_error(std::format("Failed to open log file: {}", log_filename));
    }

    // NEW: per-fi FC + relevant-leaf log
    const std::string perf_base = std::format("FsTreePerf_FC_{}_{}", dim, n_functions);
    fs::path perf_path = perf_dir / (perf_base + ".txt");
    std::ofstream perf_log(perf_path, std::ios::trunc);
    if (!perf_log) {
        throw std::runtime_error(std::format("Failed to open perf log file: {}", perf_path.string()));
    }
    perf_log << "fi\tfc_sec\trelevant_leaves\n";
    perf_log.flush();

    std::println("==========================================");
    std::println("FS-Tree Solver (Iterative Build)        ");
    std::println("Target File: {}", filename);
    std::println("==========================================");

    try {
        // ---------------------------------------------------------
        // 1. Load Data
        // ---------------------------------------------------------
        auto t0 = std::chrono::high_resolution_clock::now();
        CompactDataset ds = CompactIO::load(filename);
        {
            const std::size_t expected_pairs = static_cast<std::size_t>(n_functions) * (static_cast<std::size_t>(n_functions) - 1) / 2;
            if (ds.count != expected_pairs) {
                throw std::runtime_error(std::format(
                    "Pairwise file count mismatch: expected {} (n*(n-1)/2 for n={}), got {}",
                    expected_pairs, n_functions, ds.count));
            }
            if (static_cast<int>(ds.dim) != dim) {
                throw std::runtime_error(std::format(
                    "Pairwise file dim mismatch: expected {}, got {}",
                    dim, ds.dim));
            }
        }

        // Convert to Eigen (Batch)
        std::vector<Eigen::VectorXd> planes;
        planes.reserve(ds.count);
        for (size_t i = 0; i < ds.count; ++i) {
            Eigen::VectorXd h(ds.dim);
            for (size_t j = 0; j < ds.dim; ++j) {
                h[j] = ds.at(i, j);
            }
            planes.push_back(h);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        std::println("[1] Loaded {} planes ({:.4f}s)", ds.count, std::chrono::duration<double>(t1 - t0).count());

        // ---------------------------------------------------------
        // 2. Initialize Root Domain
        // ---------------------------------------------------------
        // Create Standard Cube [0, 1]^d
        Polytope root_poly = create_hypercube(dim);

        std::println("[2] Initialized Root Domain ([0,1]^{})", dim);

        // ---------------------------------------------------------
        // 3. Build Tree (Grouped by Function)
        // ---------------------------------------------------------
        std::println("[3] Building Tree (grouped insertion)...");
        auto t2 = std::chrono::high_resolution_clock::now();

        ITreeBuilder builder(root_poly);
        builder.global_planes = &planes;

        const int bar_width = 50;
        const std::size_t total_pairs = static_cast<std::size_t>(n_functions) * (static_cast<std::size_t>(n_functions) - 1) / 2;
        std::size_t planes_inserted = 0;

        // Insert in groups: {F1-related}, {F2-related}, ..., {Fn-related}
        for (int fi = 1; fi <= n_functions; ++fi) {
            builder.current_function_id = fi; // Fi is 1-based

            // Fi-related hyperplanes are all pairwise planes (min(fi,fj), max(fi,fj)) for fj != fi

            for (int fj = fi + 1; fj <= n_functions; ++fj) {
                int a = fi;
                int b = fj;

                std::size_t plane_index = Generator::pair_index(a, b, n_functions);
                // std::print("cur_index: {}\n", plane_index);

                int unique_h_id = (int)plane_index + 1;

                // Progress bar (over total pairs processed across all groups)
                float progress = static_cast<float>(planes_inserted + 1) /
                    static_cast<float>(total_pairs);
                int pos = static_cast<int>(bar_width * progress);

                std::print("\r    Progress: [");
                for (int j = 0; j < bar_width; ++j) {
                    if (j < pos) std::print("=");
                    else if (j == pos) std::print(">");
                    else std::print(" ");
                }
                std::print("] {:.1f}% ({}/{})", progress * 100.0, planes_inserted + 1,
                    total_pairs);
                fflush(stdout);

                // Skip if this plane does NOT partition the root polytope
                int cls = classify_polytope_against_plane(root_poly, planes[plane_index]);
                if (cls != 2) {
                    planes_inserted++;
                    continue;
                }

                // Group-aware insertion
                builder.insert_dfs_non_recursive(root_poly, planes[plane_index], unique_h_id, fi, n_functions);
                planes_inserted++;
            }

            // After finishing the Fi-group, mark all current leaf nodes as Fi-relevant if they are not relevant yet.
            builder.mark_all_leaves_relevant(fi);
            // const int relevant_leaves = builder.count_relevant_leaves();
            // std::println("[Fi={}] relevant_leaves = {}", fi, relevant_leaves);


            // Per-fi performance snapshot: cumulative FC time + current relevant leaves
            // {
            //     const double fc_sec = builder.get_fc_time_sec();
            //     const int relevant_leaves = builder.count_relevant_leaves();
            //     perf_log << std::format("{}\t{:.6f}\t{}\n", fi, fc_sec, relevant_leaves);
            //     perf_log.flush();
            // }
        }

        builder.populateLeaf(n_functions, dim, true, ds);
        //recursively builds authentication structure
        if (ads == 1)
        {
            auto flat_view = Eigen::Map<const Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(ds.data.data(), n_functions, dim);
            Eigen::MatrixXd d_matrix = flat_view.cast<double>();
            builder.create_ADS_recursive(n_functions, dim, true, ds, d_matrix);
        }
        //signature goes here
        //skipped for performance evaluation


        double fc_sec = builder.get_fc_time_sec();
        int relevant_leaves = builder.count_relevant_leaves();
        perf_log << std::format("{}\t{:.6f}\t{}\n", n_functions, fc_sec, relevant_leaves);
        perf_log.flush();

        std::println("\r    Comparisons {}/{}... Done.", planes_inserted,
            total_pairs);

        auto t3 = std::chrono::high_resolution_clock::now();

        // ---------------------------------------------------------
        // 4. Statistics & Output (final snapshot)
        // ---------------------------------------------------------
        double total_time = std::chrono::duration<double>(t3 - t2).count();
        auto total_nodes_final = builder.count_nodes();
        auto leaf_cells_final = builder.count_leaves();
        auto depth_final = builder.compute_depth();
        bool validADS = builder.checkHashConsRec();
        size_t store = builder.estimateStorage(total_nodes_final, leaf_cells_final, relevant_leaves, dim, n_functions, ads);
        //Eigen::VectorXd sam = builder.get_sample();

        std::println("\n=== Results ===");
        std::println("Time:         {:.4f} s", total_time);
        std::println("Total Nodes:  {}", total_nodes_final);
        std::println("Leaf Cells:   {}", leaf_cells_final);
        std::println("Tree Depth:   {}", depth_final);
        std::println("Relevant Leaves: {}", relevant_leaves);
        std::println("Merkle-Tree Validity: {}", validADS);
        std::println("Storage: {} bytes", store);
        //std::println("Sample Point: {}", sam);

        // Also write final result to log
        log << "=== Final Results ===\n";
        log << std::format("Time:         {:.4f} s\n", total_time);
        log << std::format("Total Nodes:  {}\n", total_nodes_final);
        log << std::format("Leaf Cells:   {}\n", leaf_cells_final);
        log << std::format("Tree Depth:   {}\n", depth_final);
        log << std::format("Relevant Leaves: {}\n", relevant_leaves);
        log << std::format("Merkle-Tree Validity: {}\n", validADS);
        log << std::format("Storage: {} bytes", store);
        //log << std::format("Sample Point: {}", sam);
        log.flush();

        std::println("[Log] PERF: {}", perf_path.string());

    }
    catch (const std::exception& e) {
        std::println("Error: {}", e.what());
        return 1;
    }

    return 0;
}