#pragma once
#define NOMINMAX
#include <windows.h>
#include <print>
#include <vector>
#include <string>
#include <format>
#include <stdexcept>
#include <chrono>
#include <Eigen/Dense>
#include <fstream>
#include <filesystem>
#include <psapi.h>
#include <cstdlib>

#include "CompactIO.h"
#include "ITreeSimplex.h"

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;

    auto ensure_logs_dir = []() -> fs::path {
        fs::path logs_dir = "logs";
        fs::path res_dir = "results";
        if (!fs::exists(logs_dir)) {
            fs::create_directory(logs_dir);
            fs::create_directory(res_dir);
        }
        return logs_dir;
    };

    if (argc != 4) {
        std::println("Usage: {} <count> <dim> <ADS>", argv[0]);
        return 1;
    }

    int n_planes = std::stoi(argv[1]);
    int dim = std::stoi(argv[2]);
    int ads = std::stoi(argv[3]);
    std::string filename = std::format("{}_pairwise_{}d.bin", n_planes, dim);

    // NEW: log file result_{dim}
    fs::path log_dir = ensure_logs_dir();
    std::string log_filename = std::format("results/itree_result_{}_{}_{}", n_planes, dim, ads);
    std::ofstream log(log_filename, std::ios::trunc);
    if (!log) {
        throw std::runtime_error(std::format("Failed to open log file: {}", log_filename));
    }

    std::println("==========================================");
    std::println("   I-Tree Solver (HiGHS LP Engine)        ");
    std::println("==========================================");

    try {
        // 1. Load Data
        auto t0 = std::chrono::high_resolution_clock::now();
        CompactDataset ds = CompactIO::load(filename);

        std::vector<Eigen::VectorXd> planes;
        planes.reserve(ds.count);
        for (size_t i = 0; i < ds.count; ++i) {
            Eigen::VectorXd h(ds.dim);
            for (size_t j = 0; j < ds.dim; ++j) h[j] = ds.at(i, j);

            // std::print("Plane {:4d}: [", (int)i + 1);
            // for (int k = 0; k < h.size(); ++k) {
            //     std::print("{}{}", h[k], (k + 1 < h.size() ? ", " : ""));
            // }
            // std::println("]");

            planes.push_back(h);
        }
        std::println("[1] Loaded {} planes", ds.count);

        // 2. Build Tree
        std::println("[2] Building Tree...");
        auto t2 = std::chrono::high_resolution_clock::now();

        LPTreeBuilder builder(dim);

        const int bar_width = 50;

        const std::string base = std::format("ITreeSimplexPerf_FC_{}_{}", dim, n_planes);
        fs::path txt_path = log_dir / (base + ".txt");

        std::ofstream fc_log(txt_path);

        // Scale log: how many intersections processed by time
        const std::string scale_base = std::format("ITreeSimplexPerf_Scale_{}_{}", dim, n_planes);
        fs::path scale_path = log_dir / (scale_base + ".txt");
        std::ofstream scale_log(scale_path);

        // Log checkpoints at 10..50 minutes (step 5), and final at 55 minutes
        auto start_time = t2;
        std::chrono::minutes next_checkpoint(5);
        const std::chrono::minutes checkpoint_step(5);
        const std::chrono::minutes last_checkpoint(50);
        const std::chrono::minutes hard_stop(360);

        for (size_t i = 0; i < planes.size(); ++i) {
            int unique_h_id = (int)i + 1;

            if (i % 10 == 0 || i == planes.size() - 1) {
                float progress = (float)(i + 1) / planes.size();
                int pos = (int)(bar_width * progress);
                std::print("\r    Progress: [");
                for (int j = 0; j < bar_width; ++j) {
                    if (j < pos) std::print("=");
                    else if (j == pos) std::print(">");
                    else std::print(" ");
                }
                std::print("] {:.1f}% ({}/{})", progress * 100.0, i + 1, planes.size());
                fflush(stdout);
            }

            builder.insert(planes[i]);

            // FC log every 50 inserts
            if ((i + 1) % 1 == 0) {
                fc_log << std::format("{}\t{:.6f}\n", i + 1, builder.get_lp_cut_time_sec());
                fc_log.flush();
            }

            // Scale checkpoints based on elapsed time since start_time (t2)
            auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(
                std::chrono::high_resolution_clock::now() - start_time);

            // Emit all checkpoints we have passed (10..50 step 5)
            while (elapsed >= next_checkpoint && next_checkpoint <= last_checkpoint) {
                scale_log << std::format("{}\t{}\n", next_checkpoint.count(), i + 1);
                scale_log.flush();
                next_checkpoint += checkpoint_step;
            }

            // Hard stop at 55 minutes: log and exit
            if (elapsed >= hard_stop) {
                scale_log << std::format("{}\t{}\n", hard_stop.count(), i + 1);
                scale_log.flush();

                std::println("\n[Scale] Reached {} minutes. Processed {} intersections. Exiting.", hard_stop.count(), (int)i + 1);
                std::println("[Log] SCALE: {}", scale_path.string());

                // Ensure FC final log line also exists
                if (((i + 1) % 50) != 0) {
                    fc_log << std::format("{}\t{:.6f}\n", i + 1, builder.get_lp_cut_time_sec());
                }

                return 0;
            }
        }

        builder.populateLeaf(n_planes, dim, true, ds);

        if (ads == 1)
        {
            builder.createADSRecursive(n_planes, dim, true, ds);
        }

        if (planes.size() % 50 != 0) {
            fc_log << std::format("{}\t{:.6f}\n", planes.size(), builder.get_lp_cut_time_sec());
        }
        // If we finished early (before 55 minutes), write the last reached checkpoint info if any
        scale_log.flush();
        scale_log.close();

        std::println("");

        auto t3 = std::chrono::high_resolution_clock::now();
        auto tnodes = builder.count_nodes();
        auto tleaf = builder.count_leaves();
        auto depth = builder.compute_depth();
        size_t store = builder.estimateStorage(tnodes, tleaf, dim, n_planes, ads);

        // 3. Results
        std::println("\n=== Results ===");
        std::println("Time:         {:.4f} s", std::chrono::duration<double>(t3-t2).count());
        std::println("Total Nodes:  {}", tnodes);
        std::println("Leaf Cells:   {}", tleaf);
        std::println("Tree Depth:   {}", depth);
        std::println("Merkle Valid: {}", builder.checkHashConsRec());
        std::println("Storage: {} bytes", store);


        std::println("Feasibility Chekcing Time:     {:.4f} s",
                     builder.get_lp_cut_time_sec());

        // Also write final result to log
        log << "=== Final Results ===\n";
        log << std::format("Time:         {:.4f} s\n", std::chrono::duration<double>(t3 - t2).count());
        log << std::format("Total Nodes:  {}\n", tnodes);
        log << std::format("Leaf Cells:   {}\n", tleaf);
        log << std::format("Tree Depth:   {}\n", depth);
        log << std::format("Merkle-Tree Validity: {}\n", builder.checkHashConsRec());
        log << std::format("Storage: {} bytes\n", store);
        log << std::format("Feasibility Chekcing Time:     {:.4f} s", builder.get_lp_cut_time_sec());
        //log << std::format("Sample Point: {}", sam);
        log.flush();

        std::println("[Log] TXT: {}", txt_path.string());
        std::println("[Log] SCALE: {}", scale_path.string());

    } catch (const std::exception& e) {
        std::println("Error: {}", e.what());
        return 1;
    }

    return 0;
}