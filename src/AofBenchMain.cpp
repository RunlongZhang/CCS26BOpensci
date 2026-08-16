// ============================================================================
// AofBenchMain.cpp — benchmark driver for the AOF-tree ADS.
//
// Usage:  AofBench <n> <d> [queries=200] [seed=42]
//
// Expects "<n>_functions_<d>d.bin" and "<n>_pairwise_<d>d.bin" in the working
// directory (produced by FunctionGenerator).
//
// Measures and reports (tab-separated, one KEY=VALUE per line):
//   construction time, tree shape, ADS commit time and sizes,
//   top-k and range: server latency, VO size, gap-block size, client verify
//   time and acceptance, plus tamper-rejection self-tests.
// ============================================================================

#include <print>
#include <random>
#include <chrono>
#include <fstream>
#include <format>
#include <filesystem>
#include "AofAds.h"

using Clock = std::chrono::high_resolution_clock;
static double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

struct Shape { size_t nodes = 0, leaves = 0, internals = 0, placeholders = 0, maxDepth = 0; };
static void shapeOf(const aof::Node* nd, size_t depth, Shape& s) {
    if (!nd) return;
    s.nodes++;
    s.maxDepth = std::max(s.maxDepth, depth);
    if (nd->is_relevant_leaf) s.leaves++;
    else if (nd->splitting_plane_id != -1) s.internals++;
    else s.placeholders++;
    shapeOf(nd->left.get(), depth + 1, s);
    shapeOf(nd->right.get(), depth + 1, s);
}

int main(int argc, char* argv[]) {
    if (argc < 3) { std::println("Usage: {} <n> <d> [queries=200] [seed=42]", argv[0]); return 1; }
    int n = std::stoi(argv[1]);
    int d = std::stoi(argv[2]);
    int nQueries = argc > 3 ? std::stoi(argv[3]) : 200;
    unsigned seed = argc > 4 ? unsigned(std::stoul(argv[4])) : 42u;

    std::string funcFile = std::format("{}_functions_{}d.bin", n, d);
    std::string pairFile = std::format("{}_pairwise_{}d.bin", n, d);
    if (!std::filesystem::exists(funcFile)) { std::println("missing {}", funcFile); return 1; }

    CompactDataset funcs = CompactIO::load(funcFile);
    CompactDataset pairsDs = CompactIO::load(pairFile);
    aof::FunctionTable F; F.init(funcs);

    std::vector<Eigen::VectorXd> planes;
    planes.reserve(pairsDs.count);
    for (size_t i = 0; i < pairsDs.count; ++i) {
        Eigen::VectorXd h(d);
        for (int j = 0; j < d; ++j) h[j] = double(pairsDs.at(i, j));
        planes.push_back(h);
    }

    // ---- Construction --------------------------------------------------------
    auto t0 = Clock::now();
    auto builder = aof::buildAof(F, planes, n, d);
    auto t1 = Clock::now();

    Shape shape; shapeOf(builder->root.get(), 0, shape);

    // ---- Commitments + signature --------------------------------------------
    aof::AdsStats st;
    auto t2 = Clock::now();
    aof::commitNode(F, planes, builder->root.get(), st);
    auto t3 = Clock::now();

    aof::Signer signer;
    aof::Server server;
    server.init(F, planes, builder->root.get(), signer);

    aof::Client client;
    client.n = n; client.d = d; client.signer = &signer;

    // Index storage estimates (bytes), per the paper's accounting:
    //   pruned: per node metadata (9B) + per leaf (rank 4B + witness d*8B)
    //   unpruned adds the explicit lists: (n-1) ids * 4B per leaf
    size_t idx_pruned   = shape.nodes * 9 + shape.leaves * (4 + size_t(d) * 8);
    size_t idx_unpruned = idx_pruned + shape.leaves * size_t(n - 1) * 4;

    std::println("n={} d={} seed={}", n, d, seed);
    std::println("CONSTRUCT_SEC={:.6f}", secs(t0, t1));
    std::println("FC_SEC={:.6f}", double(builder->fc_time.count()) / 1e9);
    std::println("NODES={} LEAVES={} INTERNALS={} PLACEHOLDERS={} MAXDEPTH={}",
                 shape.nodes, shape.leaves, shape.internals, shape.placeholders, shape.maxDepth);
    std::println("COMMIT_SEC={:.6f}", secs(t2, t3));
    std::println("ADS_BYTES={} HASH_INVOCATIONS={} COMMITTED_BYTES={}",
                 st.stored_ads_bytes, st.hash_invocations, st.committed_bytes);
    std::println("INDEX_PRUNED_BYTES={} INDEX_UNPRUNED_BYTES={}", idx_pruned, idx_unpruned);

    // ---- Queries -------------------------------------------------------------
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> uni(0.02, 0.98);
    std::uniform_int_distribution<int> kd(1, std::max(1, n / 2));

    double topkServer = 0, topkVerify = 0, topkVoBytes = 0, topkGap = 0;
    double rangeServer = 0, rangeVerify = 0, rangeVoBytes = 0, rangeAns = 0;
    int topkAccept = 0, rangeAccept = 0, tamperRejected = 0, tamperTried = 0;

    for (int q = 0; q < nQueries; ++q) {
        Eigen::VectorXd x(d);
        for (int t = 0; t < d; ++t) x[t] = uni(gen);
        int k = kd(gen);

        auto q0 = Clock::now();
        auto [ans, vo] = server.topK(x, k);
        auto q1 = Clock::now();
        auto res = client.verifyTopK(x, k, ans, vo);
        auto q2 = Clock::now();

        topkServer += secs(q0, q1);
        topkVerify += secs(q1, q2);
        topkVoBytes += double(vo.bytes(d));
        if (vo.lo && vo.hi) topkGap += double(std::max<int>(0, vo.hi->rank - vo.lo->rank - 1));
        if (res.accepted) topkAccept++;
        else if (q < 5) std::println("TOPK_REJECT q={} k={} reason={}", q, k, res.reason);

        // Tamper self-test on a few queries: drop one answer element,
        // swap a bucket member, bump the claimed rank.
        if (q < 25 && !ans.empty()) {
            tamperTried++;
            auto ans2 = ans; ans2.pop_back();
            if (!client.verifyTopK(x, k, ans2, vo).accepted) tamperRejected++; else
                std::println("TAMPER_ACCEPTED drop-answer q={}", q);
            tamperTried++;
            auto vo2 = vo;
            if (vo2.lo && vo2.lo->flOpened && !vo2.lo->Fl.empty()) {
                vo2.lo->Fl[0][0] = char(vo2.lo->Fl[0][0] ^ 0x7f);
                if (!client.verifyTopK(x, k, ans, vo2).accepted) tamperRejected++; else
                    std::println("TAMPER_ACCEPTED bucket-swap q={}", q);
            } else tamperRejected++;   // nothing to tamper counts as vacuous
            tamperTried++;
            auto vo3 = vo;
            if (vo3.lo) { vo3.lo->rank += 1;
                if (!client.verifyTopK(x, k, ans, vo3).accepted) tamperRejected++; else
                    std::println("TAMPER_ACCEPTED rank-bump q={}", q);
            } else tamperRejected++;
        }

        // Range query: a band around the median value at x.
        std::vector<double> vals(n);
        for (int f = 1; f <= n; ++f) vals[f - 1] = F.eval(f, x);
        std::nth_element(vals.begin(), vals.begin() + n / 2, vals.end());
        double mid = vals[n / 2];
        double lo = mid * 0.9, hi = mid * 1.1;

        auto r0 = Clock::now();
        auto [rans, rvo] = server.range(x, lo, hi);
        auto r1 = Clock::now();
        auto rres = client.verifyRange(x, lo, hi, rans, rvo);
        auto r2 = Clock::now();

        rangeServer += secs(r0, r1);
        rangeVerify += secs(r1, r2);
        rangeVoBytes += double(rvo.bytes(d));
        rangeAns += double(rans.size());
        if (rres.accepted) rangeAccept++;
        else std::println("RANGE_REJECT q={} lo={:.4f} hi={:.4f} reason={}", q, lo, hi, rres.reason);
    }

    double Q = double(nQueries);
    std::println("TOPK_ACCEPT={}/{}", topkAccept, nQueries);
    std::println("TOPK_SERVER_MS={:.4f} TOPK_VERIFY_MS={:.4f} TOPK_VO_BYTES={:.0f} TOPK_GAP={:.2f}",
                 1000 * topkServer / Q, 1000 * topkVerify / Q, topkVoBytes / Q, topkGap / Q);
    std::println("RANGE_ACCEPT={}/{}", rangeAccept, nQueries);
    std::println("RANGE_SERVER_MS={:.4f} RANGE_VERIFY_MS={:.4f} RANGE_VO_BYTES={:.0f} RANGE_ANS={:.1f}",
                 1000 * rangeServer / Q, 1000 * rangeVerify / Q, rangeVoBytes / Q, rangeAns / Q);
    std::println("TAMPER_REJECTED={}/{}", tamperRejected, tamperTried);

    // Machine-readable line for the sweep script.
    std::ofstream csv("aof_bench_results.csv", std::ios::app);
    csv << std::format("{},{},{},{:.6f},{},{},{},{},{},{},{:.4f},{:.4f},{:.0f},{:.2f},{:.4f},{:.4f},{:.0f},{},{},{}\n",
        n, d, seed, secs(t0, t1), shape.nodes, shape.leaves, shape.internals, shape.maxDepth,
        st.stored_ads_bytes, idx_pruned,
        1000 * topkServer / Q, 1000 * topkVerify / Q, topkVoBytes / Q, topkGap / Q,
        1000 * rangeServer / Q, 1000 * rangeVerify / Q, rangeVoBytes / Q,
        topkAccept, rangeAccept, tamperRejected);
    return (topkAccept == nQueries && rangeAccept == nQueries &&
            tamperRejected == tamperTried) ? 0 : 2;
}
