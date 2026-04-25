// Format.cpp
// Reads result files from /results, extracts statistics, and writes a combined summary.
//
// Usage:
//   Format <itree:0/1> <mfstree:0/1> <vamfstree:0/1> <d> <m:0/1>
//
// Parameters:
//   itree/mfstree/vamfstree : 1 = process that tree type, 0 = skip
//   d                       : dimension filter (matches filename component _d_)
//   m                       : Merkle filter (0 = non-Merkle, 1 = Merkle)
//
// Input files  : results/{name}_result_{n}_{d}_{m}
// Output file  : results/full_{d}_{m}

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Per-file statistics
// ---------------------------------------------------------------------------
struct FileStats {
    int       n              = 0;
    double    time           = 0.0;
    long long storage        = 0;
    long long mstorage       = 0;   // only populated when m == 0
    std::optional<double>    verify_time;
    std::optional<long long> verify_storage;
};

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

// Returns the first double found after the colon on a line such as
//   "Time:            1.2345 s"
double parse_double_after_colon(const std::string& line) {
    auto pos = line.find(':');
    if (pos == std::string::npos) return 0.0;
    std::istringstream iss(line.substr(pos + 1));
    double val = 0.0;
    iss >> val;
    return val;
}

// Returns the first long long found after the colon on a line such as
//   "Storage:         67890 bytes (0.01 MB)"
long long parse_longlong_after_colon(const std::string& line) {
    auto pos = line.find(':');
    if (pos == std::string::npos) return 0LL;
    std::istringstream iss(line.substr(pos + 1));
    long long val = 0LL;
    iss >> val;
    return val;
}

// ---------------------------------------------------------------------------
// File parser
// ---------------------------------------------------------------------------
FileStats parse_file(const fs::path& path,
                     int  n,
                     int  d,
                     bool compute_mstorage,
                     bool is_itree)
{
    std::ifstream f(path);
    FileStats s;
    s.n = n;

    double    verify_time_raw    = -1.0;
    long long verify_storage_raw = -1LL;
    double    tree_depth         =  0.0;
    long long rleafs             =  0LL;

    std::string line;
    while (std::getline(f, line)) {
        // "Verify Time:" and "Verify Storage:" must be checked before
        // "Time:" and "Storage:" so the shorter prefixes don't match first.
        if (line.starts_with("Verify Time:")) {
            verify_time_raw = parse_double_after_colon(line);
        } else if (line.starts_with("Verify Storage:")) {
            verify_storage_raw = parse_longlong_after_colon(line);
        } else if (line.starts_with("Time:")) {
            s.time = parse_double_after_colon(line);
        } else if (line.starts_with("Storage:")) {
            s.storage = parse_longlong_after_colon(line);
        } else if (line.starts_with("Tree Depth:")) {
            tree_depth = parse_double_after_colon(line);
        } else if (line.starts_with("Leaf Cells:")) {
            rleafs = parse_longlong_after_colon(line);
        }
    }

    // Verify time=
    if (verify_time_raw >= 0.0)
        s.verify_time = verify_time_raw;

    if (verify_storage_raw >= 0LL)
        s.verify_storage = verify_storage_raw;

    // mstorage only for non-Merkle runs (m == 0)
    //   itree    : rleafs * ((2 * n) - 1) * 32
    //   mfstree / vamfstree : rleafs * 5 * 32
    if (compute_mstorage) {
        if (is_itree)
            s.mstorage = rleafs * (2LL * n - 1) * 32LL;
        else
            s.mstorage = rleafs * 5LL * 32LL;
    }

    return s;
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

void write_main_group(std::ofstream& out,
                      const std::string& label,
                      const std::vector<FileStats>& stats,
                      bool compute_mstorage)
{
    out << "=== " << label << " ===\n";
    for (const auto& s : stats) {
        if (compute_mstorage)
            out << std::format("{}: {:.6f}, {}, {}\n",
                               s.n, s.time, s.storage, s.mstorage);
        else
            out << std::format("{}: {:.6f}, {}\n",
                               s.n, s.time, s.storage);
    }
}

void write_verify_group(std::ofstream& out,
                        const std::string& label,
                        const std::vector<FileStats>& stats)
{
    out << "=== " << label << " (verify) ===\n";
    for (const auto& s : stats) {
        double    vt = s.verify_time.value_or(0.0);
        long long vs = s.verify_storage.value_or(0LL);
        out << std::format("{}: {:.6f}, {}\n", s.n, vt, vs);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <itree:0/1> <mfstree:0/1> <vamfstree:0/1> <d> <m:0/1>\n";
        return 1;
    }

    bool use_itree     = std::string(argv[1]) == "1";
    bool use_mfstree   = std::string(argv[2]) == "1";
    bool use_vamfstree = std::string(argv[3]) == "1";
    int  d             = std::stoi(argv[4]);
    int  m             = std::stoi(argv[5]);
    bool compute_mstorage = (m == 0);

    struct TreeType {
        std::string name;
        bool        enabled;
        bool        is_itree;
    };

    const std::vector<TreeType> tree_types = {
        {"itree",     use_itree,     true},
        {"mfstree",   use_mfstree,   false},
        {"vamfstree", use_vamfstree, false},
    };

    // Open output file
    std::string out_path = std::format("results/full_{}_{}", d, m);
    std::ofstream out(out_path);
    if (!out) {
        std::cerr << "Could not open output file: " << out_path << "\n";
        return 1;
    }

    bool first_block = true;

    for (const auto& tt : tree_types) {
        if (!tt.enabled) continue;

        // ------------------------------------------------------------------
        // Collect all matching result files for this tree type
        // ------------------------------------------------------------------
        std::vector<std::pair<int, fs::path>> matched;  // (n, path)

        for (const auto& entry : fs::directory_iterator("results")) {
            if (!entry.is_regular_file()) continue;

            std::string fn     = entry.path().filename().string();
            std::string prefix = tt.name + "_result_";
            if (!fn.starts_with(prefix)) continue;

            // remainder must be exactly "{n}_{d}_{m}"
            std::string rest = fn.substr(prefix.size());
            std::vector<std::string> parts;
            {
                std::istringstream iss(rest);
                std::string tok;
                while (std::getline(iss, tok, '_'))
                    parts.push_back(tok);
            }
            if (parts.size() != 3) continue;

            try {
                int fn_n = std::stoi(parts[0]);
                int fn_d = std::stoi(parts[1]);
                int fn_m = std::stoi(parts[2]);
                if (fn_d == d && fn_m == m)
                    matched.emplace_back(fn_n, entry.path());
            } catch (...) {
                continue;
            }
        }

        if (matched.empty()) continue;

        // Sort ascending by n
        std::sort(matched.begin(), matched.end());

        // ------------------------------------------------------------------
        // Parse each file
        // ------------------------------------------------------------------
        std::vector<FileStats> all_stats;
        for (auto& [n, path] : matched)
            all_stats.push_back(parse_file(path, n, d, compute_mstorage, tt.is_itree));

        // ------------------------------------------------------------------
        // Write main group
        // ------------------------------------------------------------------
        if (!first_block) out << "\n---\n\n";
        first_block = false;

        write_main_group(out, tt.name, all_stats, compute_mstorage);

        // ------------------------------------------------------------------
        // Write verify group as a separate block (only if any file has data)
        // ------------------------------------------------------------------
        bool has_verify = std::any_of(all_stats.begin(), all_stats.end(),
            [](const FileStats& s) {
                return s.verify_time.has_value() || s.verify_storage.has_value();
            });

        if (has_verify) {
            // Filter to entries that actually have verify data
            std::vector<FileStats> verify_stats;
            for (const auto& s : all_stats)
                if (s.verify_time.has_value() || s.verify_storage.has_value())
                    verify_stats.push_back(s);

            out << "\n---\n\n";
            write_verify_group(out, tt.name, verify_stats);
        }
    }

    out.close();
    std::cout << "Written to " << out_path << "\n";
    return 0;
}
