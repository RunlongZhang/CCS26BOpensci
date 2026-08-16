#pragma once
// ============================================================================
// AofAds.h — AOF-tree ADS.
//
// Implements, on top of the polytope machinery (PolytopeOps.h):
//   1. The routing-based AOF-tree builder with the paper's witness handling:
//      - facet-centroid witnesses (not first-vertex) at split children,
//      - a derived strictly-interior witness per node (nudged off the creating
//        facet by half the minimum ancestor-constraint slack),
//      - robust sign evaluation for routing (fallback blending when a pair
//        hyperplane passes exactly through the witness).
//   2. The commitment scheme:
//      - type-tagged SHA-256 hashing (leaf / internal / component / absent),
//      - leaf commitment over six components:
//          H(enc(f_p)) || H(enc(r_p)) || H(enc(F_l)) || H(enc(F_r)) || h(vl) || h(vr)
//      - internal-node commitment binding both coefficient vectors:
//          H(TAG_INT || enc(f_i) || enc(f_j) || h(vl) || h(vr))
//      - absent child slots: h(bot) = H(TAG_ABSENT)  (a hash of a known input,
//        NOT a fixed constant).
//   3. A signed root:  sigma = Sign_RSA( TAG_ROOT || enc(D) || n || d || nu || h_root ).
//   4. Query processing (top-k and range) with rank/value bracketing, the
//      gap block B, VO assembly, and a fully independent client-side verifier
//      per Algorithm VerifyTopK of the paper.
// ============================================================================

#include <memory>
#include <vector>
#include <string>
#include <array>
#include <optional>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <chrono>
#include <Eigen/Dense>
#include <openssl/evp.h>
#include <openssl/rsa.h>
#include "PolytopeStructs.h"
#include "PolytopeOps.h"
#include "FunctionPairGenerator.h"
#include "CompactIO.h"

namespace aof {

using Hash = std::array<uint8_t, 32>;
constexpr double kEps = 1e-9;

// ---------------------------------------------------------------------------
// Domain-separation tags (one byte each, per the paper's commitment spec).
// ---------------------------------------------------------------------------
enum Tag : uint8_t {
    TAG_COMPONENT = 0x01,
    TAG_LEAF      = 0x02,
    TAG_INTERNAL  = 0x03,
    TAG_ABSENT    = 0x04,
    TAG_ROOT      = 0x05,
};

// ---------------------------------------------------------------------------
// SHA-256 helpers.
// ---------------------------------------------------------------------------
class Sha256 {
    EVP_MD_CTX* ctx_ = nullptr;
    const EVP_MD* md_ = nullptr;
public:
    Sha256() { ctx_ = EVP_MD_CTX_new(); md_ = EVP_get_digestbyname("sha256"); reset(); }
    ~Sha256() { EVP_MD_CTX_free(ctx_); }
    void reset() { EVP_DigestInit_ex(ctx_, md_, nullptr); }
    void update(const void* p, size_t len) { EVP_DigestUpdate(ctx_, p, len); }
    void update(uint8_t b) { EVP_DigestUpdate(ctx_, &b, 1); }
    void update(const Hash& h) { EVP_DigestUpdate(ctx_, h.data(), h.size()); }
    Hash final() { Hash out; unsigned int len = 0; EVP_DigestFinal_ex(ctx_, out.data(), &len); reset(); return out; }
    static Hash absent() {
        static Hash h = [] { Sha256 s; s.update(uint8_t(TAG_ABSENT)); return s.final(); }();
        return h;
    }
};

// ---------------------------------------------------------------------------
// Function table: n functions, d int8 coefficients each (homogeneous A·x).
// enc(f) = the d coefficient bytes (fixed-width, injective for distinct fns).
// Sets are encoded as concatenated coefficient blocks in increasing index
// order, prefixed by the 4-byte member count.
// ---------------------------------------------------------------------------
struct FunctionTable {
    int n = 0, d = 0;
    const CompactDataset* ds = nullptr;             // function coefficients
    Eigen::MatrixXd M;                              // n x d, double copy for evaluation

    void init(const CompactDataset& funcs) {
        ds = &funcs; n = int(funcs.count); d = int(funcs.dim);
        M.resize(n, d);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < d; ++j)
                M(i, j) = double(funcs.at(i, j));
    }
    // 1-based function id -> coefficient byte block.
    const int8_t* coeffs(int fid) const { return ds->data.data() + size_t(fid - 1) * d; }
    double eval(int fid, const Eigen::VectorXd& x) const { return M.row(fid - 1).dot(x); }
};

inline void encAppendU32(std::string& out, uint32_t v) {
    out.push_back(char(v & 0xff)); out.push_back(char((v >> 8) & 0xff));
    out.push_back(char((v >> 16) & 0xff)); out.push_back(char((v >> 24) & 0xff));
}
inline std::string encFunction(const FunctionTable& F, int fid) {
    return std::string(reinterpret_cast<const char*>(F.coeffs(fid)), F.d);
}
inline std::string encSet(const FunctionTable& F, const std::vector<int>& members /*sorted ids*/) {
    std::string out; encAppendU32(out, uint32_t(members.size()));
    for (int fid : members) out.append(encFunction(F, fid));
    return out;
}
inline Hash hashComponent(const std::string& enc) {
    Sha256 s; s.update(uint8_t(TAG_COMPONENT)); s.update(enc.data(), enc.size()); return s.final();
}

// ---------------------------------------------------------------------------
// Pair-plane helpers.  Planes are stored for pairs (i, j), i < j, as A_i - A_j.
// ---------------------------------------------------------------------------
struct PairInfo { int i, j; };                       // i < j, 1-based
inline PairInfo unrankPair(size_t plane_index, size_t n) {
    // Inverse of Generator::pair_index (lexicographic order).
    size_t i = 1;
    while (Generator::pair_index(i + 1, i + 2, n) <= plane_index && i + 1 < n) ++i;
    // linear fallback below is exact and O(n); fine for benchmark scales
    for (size_t a = 1; a < n; ++a) {
        size_t lo = Generator::pair_index(a, a + 1, n);
        size_t hi = lo + (n - a) - 1;
        if (plane_index >= lo && plane_index <= hi)
            return { int(a), int(a + 1 + (plane_index - lo)) };
    }
    throw std::runtime_error("unrankPair: index out of range");
}

// ---------------------------------------------------------------------------
// AOF-tree node (interleaved single-tree representation).
//   internal node:  splitting_plane_id > 0, not relevant leaf
//   PO-leaf:        is_relevant_leaf, target_function_id = pivot
//   placeholder:    neither (absent child slot; commits to h(bot))
// ---------------------------------------------------------------------------
struct Node {
    int  splitting_plane_id = -1;   // 1-based plane id (pair_index + 1)
    int  target_function_id = -1;   // pivot id for PO-leaves
    bool is_relevant_leaf   = false;

    Eigen::VectorXd facet_centroid;   // centroid of the creating fragment
    Eigen::VectorXd interior_witness; // strictly interior point of the region

    int   creating_plane_id = -1;   // plane whose cut created this node's region
    int   side = 0;                 // -1: left (f_i < f_j), +1: right, 0: root
    Node* parent = nullptr;

    std::unique_ptr<Node> left, right;
    Hash h{};                       // commitment (filled by commit pass)
    bool is_leaf() const { return !left && !right; }
    bool is_placeholder() const { return !is_relevant_leaf && splitting_plane_id == -1; }
};

// ---------------------------------------------------------------------------
// Builder: routing-based construction (grouped insertion) with the paper's
// witness handling.  Semantically equivalent to Algorithms 1-2 of the paper
// under index order; see Section 3's implementation remark.
// ---------------------------------------------------------------------------
class Builder {
public:
    std::unique_ptr<Node> root;
    const std::vector<Eigen::VectorXd>* planes = nullptr;  // pair planes (A_i - A_j)
    const FunctionTable* F = nullptr;
    int dim = 0;
    std::chrono::nanoseconds fc_time{0};

    Builder(const FunctionTable& table, const std::vector<Eigen::VectorXd>& pairPlanes, int d) {
        F = &table; planes = &pairPlanes; dim = d;
        root = std::make_unique<Node>();
        root->interior_witness = Eigen::VectorXd::Constant(d, 0.5);  // box center
        root->facet_centroid = root->interior_witness;
    }

    // Signed distance-style slack of point x against the node's region
    // constraints (ancestor cuts + unit box), and the minimal slack.
    double minSlackAt(const Node* node, const Eigen::VectorXd& x) const {
        double slack = 1e300;
        for (int t = 0; t < dim; ++t) {
            slack = std::min(slack, x[t]);          // x_t >= 0
            slack = std::min(slack, 1.0 - x[t]);    // x_t <= 1
        }
        for (const Node* a = node; a && a->parent; a = a->parent) {
            const Eigen::VectorXd& hpl = (*planes)[a->creating_plane_id - 1];
            double v = hpl.dot(x);
            // left child region: hpl(x) <= 0  -> slack = -v ; right: v
            slack = std::min(slack, a->side < 0 ? -v : v);
        }
        return slack;
    }

    // Is x strictly inside the region of `node` (box + ancestor cuts), and,
    // if requireSide, strictly on node's own side of its creating plane?
    bool strictlyInterior(const Node* node, const Eigen::VectorXd& x, bool requireSide) const {
        for (int t = 0; t < dim; ++t)
            if (x[t] <= 0.0 || x[t] >= 1.0) return false;
        const Node* start = requireSide ? node : node->parent;
        for (const Node* a = start; a && a->parent; a = a->parent) {
            if (a->creating_plane_id <= 0) continue;
            double v = (*planes)[a->creating_plane_id - 1].dot(x);
            if ((a->side < 0 ? -v : v) <= 0.0) return false;
        }
        return true;
    }

    // Derive a strictly interior witness for a freshly created child whose
    // region hangs off `creating` plane on side `side`, given a point c in
    // the relative interior of the created facet.
    Eigen::VectorXd interiorWitness(Node* child, const Eigen::VectorXd& c) const {
        const Eigen::VectorXd& hpl = (*planes)[child->creating_plane_id - 1];
        Eigen::VectorXd nrm = hpl.normalized();
        Eigen::VectorXd dir = (child->side < 0) ? Eigen::VectorXd(-nrm) : nrm;
        // Initial step: half the minimum slack of c against the other constraints.
        double slack = 1e300;
        for (int t = 0; t < dim; ++t) {
            slack = std::min(slack, std::max(c[t], 0.0));
            slack = std::min(slack, std::max(1.0 - c[t], 0.0));
        }
        for (const Node* a = child->parent; a && a->parent; a = a->parent) {
            if (a->creating_plane_id <= 0) continue;
            const Eigen::VectorXd& ah = (*planes)[a->creating_plane_id - 1];
            double v = ah.dot(c);
            slack = std::min(slack, a->side < 0 ? -v : v);
        }
        double delta = std::max(slack * 0.5, 1e-7);
        // Shrink until strictly valid (bounded retries; then blend toward the
        // box center as a last resort).
        for (int it = 0; it < 40; ++it) {
            Eigen::VectorXd cand = c + delta * dir;
            if (strictlyInterior(child, cand, true)) return cand;
            delta *= 0.5;
        }
        Eigen::VectorXd center = Eigen::VectorXd::Constant(dim, 0.5);
        Eigen::VectorXd cand = c + 1e-7 * dir;
        for (int it = 0; it < 40; ++it) {
            cand = 0.5 * (cand + center);
            Eigen::VectorXd probe = cand + 1e-9 * dir;
            if (strictlyInterior(child, probe, true)) return probe;
        }
        return c + 1e-7 * dir;   // best effort; flagged downstream by checks
    }

    // Robust sign of pair plane `hpl` over the region of PO-leaf `leaf`,
    // per the paper's implementation remark: evaluate at the facet-relint
    // witness (a valid closure point of the region by construction); pairs
    // whose hyperplane contains the facet are classified structurally; the
    // rare near-zero non-parallel case falls back to the derived interior
    // point.
    double signOver(const Node* leaf, const Eigen::VectorXd& hpl) const {
        return signOverStatic(*planes, leaf, hpl);
    }
    static double signOverStatic(const std::vector<Eigen::VectorXd>& planes,
                                 const Node* leaf, const Eigen::VectorXd& hpl) {
        double v = hpl.dot(leaf->facet_centroid);
        if (std::abs(v) > kEps) return v;
        if (leaf->creating_plane_id > 0) {
            const Eigen::VectorXd& cp = planes[leaf->creating_plane_id - 1];
            double scale = hpl.dot(cp) / cp.squaredNorm();
            Eigen::VectorXd resid = hpl - scale * cp;
            if (resid.norm() < 1e-9 * (hpl.norm() + 1.0)) {
                // hpl == scale * cp identically: the region's side of cp
                // determines the sign of hpl over the whole region.
                return (leaf->side < 0 ? -scale : scale);
            }
        }
        return hpl.dot(leaf->interior_witness);
    }

    struct Job { Node* node; Polytope frag; };

    void insertPlane(const Polytope& rootDomain, const Eigen::VectorXd& h_vec,
                     int h_id, int fi, size_t n) {
        Polytope initial = slice_polytope(rootDomain, h_vec, h_id);
        if (initial.vertices.empty()) return;

        std::vector<Job> stack;
        stack.push_back({ root.get(), std::move(initial) });

        while (!stack.empty()) {
            Job job = std::move(stack.back()); stack.pop_back();
            Node* cur = job.node;

            if (cur->is_relevant_leaf) {
                // Route to the side of this pivot that fi belongs to.
                int t = cur->target_function_id;               // t < fi always
                size_t pi = Generator::pair_index(size_t(t), size_t(fi), n);
                double val = signOver(cur, (*planes)[pi]);      // sign of f_t - f_fi
                Node* next = (val < 0.0) ? cur->left.get() : cur->right.get();
                if (next) stack.push_back({ next, std::move(job.frag) });
                continue;
            }

            if (cur->is_leaf()) {
                // Split here.
                cur->splitting_plane_id = h_id;
                cur->target_function_id = fi;

                cur->left  = std::make_unique<Node>();
                cur->right = std::make_unique<Node>();
                cur->left->parent = cur;  cur->right->parent = cur;
                cur->left->creating_plane_id = h_id;  cur->right->creating_plane_id = h_id;
                cur->left->side = -1;                 cur->right->side = +1;
                cur->left->target_function_id = fi;   cur->right->target_function_id = fi;

                // Facet centroid witness (relative interior of the new facet).
                // The origin lies on every homogeneous pair plane and in every
                // region's closure, but slice_polytope excludes it; include it
                // implicitly so the centroid is pulled off the box boundary.
                Eigen::VectorXd c = Eigen::VectorXd::Zero(dim);
                for (const auto& v : job.frag.vertices) {
                    for (int t = 0; t < dim; ++t) c[t] += v.position[t];
                }
                c /= double(job.frag.vertices.size() + 1);
                cur->left->facet_centroid  = c;  cur->right->facet_centroid = c;
                cur->left->interior_witness  = interiorWitness(cur->left.get(), c);
                cur->right->interior_witness = interiorWitness(cur->right.get(), c);
                continue;
            }

            // Internal node: classify fragment against the existing plane.
            const Eigen::VectorXd& hex = (*planes)[cur->splitting_plane_id - 1];
            auto t0 = std::chrono::high_resolution_clock::now();
            int cls = classify_polytope_against_plane(job.frag, hex);
            auto t1 = std::chrono::high_resolution_clock::now();
            fc_time += t1 - t0;

            if (cls == -1) { if (cur->left)  stack.push_back({ cur->left.get(),  std::move(job.frag) }); continue; }
            if (cls ==  1) { if (cur->right) stack.push_back({ cur->right.get(), std::move(job.frag) }); continue; }

            auto s0 = std::chrono::high_resolution_clock::now();
            auto parts = split_polytope(job.frag, hex, cur->splitting_plane_id);
            auto s1 = std::chrono::high_resolution_clock::now();
            fc_time += s1 - s0;
            if (cur->right) stack.push_back({ cur->right.get(), std::move(parts.first) });
            if (cur->left)  stack.push_back({ cur->left.get(),  std::move(parts.second) });
        }
    }

    void markRelevant(int fi) {
        std::vector<Node*> st{ root.get() };
        while (!st.empty()) {
            Node* nd = st.back(); st.pop_back();
            if (!nd->is_leaf()) {
                if (nd->is_relevant_leaf) { st.push_back(nd->left.get()); st.push_back(nd->right.get()); }
                else { if (nd->left) st.push_back(nd->left.get()); if (nd->right) st.push_back(nd->right.get()); }
                continue;
            }
            if (!nd->is_relevant_leaf && nd->target_function_id == fi) {
                nd->is_relevant_leaf = true;
                nd->left = std::make_unique<Node>();  nd->left->parent = nd;
                nd->right = std::make_unique<Node>(); nd->right->parent = nd;
                // Placeholders inherit region witnesses for later routing decisions.
                nd->left->creating_plane_id = nd->creating_plane_id;
                nd->right->creating_plane_id = nd->creating_plane_id;
                nd->left->side = nd->side; nd->right->side = nd->side;
                nd->left->facet_centroid = nd->facet_centroid;
                nd->right->facet_centroid = nd->facet_centroid;
                nd->left->interior_witness = nd->interior_witness;
                nd->right->interior_witness = nd->interior_witness;
            }
        }
    }

    // Degenerate base case: no plane ever crossed -> root is a bare leaf.
    void finalizeDegenerate() {
        if (root->is_leaf() && !root->is_relevant_leaf) {
            root->is_relevant_leaf = true;
            root->target_function_id = 1;
            root->left = std::make_unique<Node>();  root->left->parent = root.get();
            root->right = std::make_unique<Node>(); root->right->parent = root.get();
        }
    }
};

inline std::unique_ptr<Builder> buildAof(const FunctionTable& F,
                                         const std::vector<Eigen::VectorXd>& planes,
                                         int n, int d) {
    Polytope box = create_hypercube(d);
    auto b = std::make_unique<Builder>(F, planes, d);
    for (int fi = 1; fi <= n; ++fi) {
        for (int fj = fi + 1; fj <= n; ++fj) {
            size_t pi = Generator::pair_index(size_t(fi), size_t(fj), size_t(n));
            if (classify_polytope_against_plane(box, planes[pi]) != 2) continue;
            b->insertPlane(box, planes[pi], int(pi) + 1, fi, size_t(n));
        }
        b->markRelevant(fi);
    }
    b->finalizeDegenerate();
    return b;
}

// ---------------------------------------------------------------------------
// Tuple computation at a PO-leaf, per the paper's implementation remark:
// each pair (f, pivot) is classified by the sign of its difference over the
// leaf's region — facet-centroid evaluation with structural handling of
// facet-containing hyperplanes (Builder::signOverStatic).
// ---------------------------------------------------------------------------
struct Tuple { std::vector<int> Fl, Fr; int rank = 0; };

inline Tuple computeTuple(const FunctionTable& F,
                          const std::vector<Eigen::VectorXd>& planes,
                          const Node* leaf) {
    Tuple t;
    int p = leaf->target_function_id;
    for (int f = 1; f <= F.n; ++f) {
        if (f == p) continue;
        int i = std::min(f, p), j = std::max(f, p);
        const Eigen::VectorXd& w0 = planes[Generator::pair_index(size_t(i), size_t(j), size_t(F.n))];
        double v = Builder::signOverStatic(planes, leaf, w0);   // sign of A_i - A_j
        double vf = (f == i) ? v : -v;                          // sign of A_f - A_p
        if (vf > 0.0) t.Fl.push_back(f); else t.Fr.push_back(f);
    }
    t.rank = int(t.Fl.size()) + 1;
    return t;
}

// ---------------------------------------------------------------------------
// Commitment pass (bottom-up) + ADS statistics.
// ---------------------------------------------------------------------------
struct AdsStats {
    size_t leaves = 0, internals = 0, placeholders = 0;
    size_t hash_invocations = 0;
    size_t committed_bytes = 0;      // total bytes fed to H across the ADS
    size_t stored_ads_bytes = 0;     // per-node stored hash + metadata (pruned index excluded)
};

inline Hash commitNode(const FunctionTable& F, const std::vector<Eigen::VectorXd>& planes,
                       Node* nd, AdsStats& st) {
    if (!nd || nd->is_placeholder()) {
        if (nd) { nd->h = Sha256::absent(); st.placeholders++; }
        return Sha256::absent();
    }
    Hash hl = commitNode(F, planes, nd->left.get(), st);
    Hash hr = commitNode(F, planes, nd->right.get(), st);

    Sha256 s;
    if (nd->is_relevant_leaf) {
        Tuple t = computeTuple(F, planes, nd);
        std::string encP = encFunction(F, nd->target_function_id);
        std::string encR; encAppendU32(encR, uint32_t(t.rank));
        std::string encL = encSet(F, t.Fl);
        std::string encRr = encSet(F, t.Fr);
        Hash c1 = hashComponent(encP), c2 = hashComponent(encR),
             c3 = hashComponent(encL), c4 = hashComponent(encRr);
        s.update(uint8_t(TAG_LEAF));
        s.update(c1); s.update(c2); s.update(c3); s.update(c4);
        s.update(hl); s.update(hr);
        st.leaves++; st.hash_invocations += 5;
        st.committed_bytes += encP.size() + encR.size() + encL.size() + encRr.size() + 1 + 6 * 32;
        st.stored_ads_bytes += 6 * 32;   // stored: own hash is derivable; store component hashes
    } else {
        PairInfo pr = unrankPair(size_t(nd->splitting_plane_id - 1), size_t(F.n));
        s.update(uint8_t(TAG_INTERNAL));
        std::string ei = encFunction(F, pr.i), ej = encFunction(F, pr.j);
        s.update(ei.data(), ei.size()); s.update(ej.data(), ej.size());
        s.update(hl); s.update(hr);
        st.internals++; st.hash_invocations += 1;
        st.committed_bytes += 1 + ei.size() + ej.size() + 2 * 32;
        st.stored_ads_bytes += 32;
    }
    nd->h = s.final();
    return nd->h;
}

// ---------------------------------------------------------------------------
// Signed root (RSA-2048 via OpenSSL EVP).
// ---------------------------------------------------------------------------
struct Signer {
    EVP_PKEY* key = nullptr;
    Signer() {
        key = EVP_PKEY_Q_keygen(nullptr, nullptr, "RSA", size_t(2048));
        if (!key) throw std::runtime_error("RSA keygen failed");
    }
    ~Signer() { EVP_PKEY_free(key); }

    static std::string rootMessage(int n, int d, uint32_t version, const Hash& root) {
        std::string m; m.push_back(char(TAG_ROOT));
        // enc(D): unit box, canonical: d, then lo/hi per axis as doubles.
        encAppendU32(m, uint32_t(d));
        for (int t = 0; t < d; ++t) { double lo = 0.0, hi = 1.0;
            m.append(reinterpret_cast<const char*>(&lo), 8);
            m.append(reinterpret_cast<const char*>(&hi), 8); }
        encAppendU32(m, uint32_t(n)); encAppendU32(m, uint32_t(d)); encAppendU32(m, version);
        m.append(reinterpret_cast<const char*>(root.data()), root.size());
        return m;
    }
    std::vector<uint8_t> sign(const std::string& msg) const {
        EVP_MD_CTX* c = EVP_MD_CTX_new();
        EVP_DigestSignInit(c, nullptr, EVP_sha256(), nullptr, key);
        size_t len = 0;
        EVP_DigestSign(c, nullptr, &len, reinterpret_cast<const uint8_t*>(msg.data()), msg.size());
        std::vector<uint8_t> sig(len);
        EVP_DigestSign(c, sig.data(), &len, reinterpret_cast<const uint8_t*>(msg.data()), msg.size());
        sig.resize(len); EVP_MD_CTX_free(c);
        return sig;
    }
    bool verify(const std::string& msg, const std::vector<uint8_t>& sig) const {
        EVP_MD_CTX* c = EVP_MD_CTX_new();
        EVP_DigestVerifyInit(c, nullptr, EVP_sha256(), nullptr, key);
        int ok = EVP_DigestVerify(c, sig.data(), sig.size(),
                                  reinterpret_cast<const uint8_t*>(msg.data()), msg.size());
        EVP_MD_CTX_free(c);
        return ok == 1;
    }
};

// ---------------------------------------------------------------------------
// Verification objects.  The client holds no function table: every function
// on the wire is its coefficient block (= enc(f)), and identity IS the block.
// ---------------------------------------------------------------------------
using Block = std::string;                          // enc(f): d coefficient bytes

struct PathInternal {              // internal comparison node on the root path
    Block ci, cj;                  // committed coefficient vectors of the pair
    int dir = 0;                   // -1 descended left (f_i < f_j), +1 right
    Hash sibling{};                // hash of the non-descended child
};
struct PathAncestorLeaf {          // PO-leaf traversed on the way down
    Hash c1{}, c2{}, c3{}, c4{};   // the four component hashes
    int dir = 0;                   // which child PO-tree the path descends into
    Hash sibling{};                // hash of the non-descended child
};
struct PathEntry {
    bool isLeafEntry = false;
    PathInternal in{};
    PathAncestorLeaf al{};
};
struct OpenedLeaf {
    Block pivot;                                    // pivot coefficient block
    int rank = 0;
    bool flOpened = false, frOpened = false;
    std::vector<Block> Fl, Fr;                      // opened buckets, committed order
    Hash hFl{}, hFr{};                              // component hashes when unopened
    Hash childL{}, childR{};                        // child subtree hashes
    std::vector<PathEntry> path;                    // root -> this leaf
};
struct VO {
    std::optional<OpenedLeaf> lo, hi;               // rank brackets (top-k) / value brackets (range)
    std::vector<uint8_t> signature;
    uint32_t version = 1;
    size_t bytes(int d) const {
        size_t b = signature.size() + 4;
        auto leafBytes = [&](const OpenedLeaf& L) {
            size_t s = d + 4;                                       // pivot coeffs + rank
            if (L.flOpened) s += 4 + L.Fl.size() * d; else s += 32;
            if (L.frOpened) s += 4 + L.Fr.size() * d; else s += 32;
            s += 64;                                                // child hashes
            for (const auto& e : L.path) s += e.isLeafEntry ? (4 * 32 + 1 + 32) : (2 * size_t(d) + 1 + 32);
            return s;
        };
        if (lo) b += leafBytes(*lo);
        if (hi) b += leafBytes(*hi);
        return b;
    }
};

inline double evalBlock(const Block& blk, const Eigen::VectorXd& x) {
    double v = 0.0;
    for (int t = 0; t < int(blk.size()); ++t) v += double(int8_t(blk[t])) * x[t];
    return v;
}
inline std::string encBucket(const std::vector<Block>& members) {
    std::string out; encAppendU32(out, uint32_t(members.size()));
    for (const auto& m : members) out.append(m);
    return out;
}

// ---------------------------------------------------------------------------
// Server-side query processing.
// ---------------------------------------------------------------------------
class Server {
public:
    const FunctionTable* F = nullptr;
    const std::vector<Eigen::VectorXd>* planes = nullptr;
    Node* root = nullptr;
    std::vector<uint8_t> rootSig;
    uint32_t version = 1;

    void init(const FunctionTable& table, const std::vector<Eigen::VectorXd>& p,
              Node* r, const Signer& s) {
        F = &table; planes = &p; root = r;
        rootSig = s.sign(Signer::rootMessage(F->n, F->d, version, r->h));
    }

    Block blockOf(int fid) const { return encFunction(*F, fid); }
    std::vector<Block> blocksOf(const std::vector<int>& ids) const {
        std::vector<Block> out; out.reserve(ids.size());
        for (int f : ids) out.push_back(blockOf(f));
        return out;
    }

    PathEntry internalEntry(const Node* nd, const Eigen::VectorXd& x, double& val) const {
        PairInfo pr = unrankPair(size_t(nd->splitting_plane_id - 1), size_t(F->n));
        val = (*planes)[nd->splitting_plane_id - 1].dot(x);
        PathEntry e; e.isLeafEntry = false;
        e.in.ci = blockOf(pr.i); e.in.cj = blockOf(pr.j);
        e.in.dir = (val < 0.0) ? -1 : +1;
        e.in.sibling = (val < 0.0) ? nd->right->h : nd->left->h;
        return e;
    }
    PathEntry leafEntry(const Node* nd, const Tuple& t, int dir) const {
        PathEntry e; e.isLeafEntry = true;
        e.al.c1 = hashComponent(blockOf(nd->target_function_id));
        { std::string er; encAppendU32(er, uint32_t(t.rank)); e.al.c2 = hashComponent(er); }
        e.al.c3 = hashComponent(encBucket(blocksOf(t.Fl)));
        e.al.c4 = hashComponent(encBucket(blocksOf(t.Fr)));
        e.al.dir = dir;
        e.al.sibling = (dir < 0) ? nd->right->h : nd->left->h;
        return e;
    }
    OpenedLeaf openLeaf(const Node* leaf, const std::vector<PathEntry>& path, const Tuple& t) const {
        OpenedLeaf L;
        L.pivot = blockOf(leaf->target_function_id);
        L.rank = t.rank;
        L.hFl = hashComponent(encBucket(blocksOf(t.Fl)));
        L.hFr = hashComponent(encBucket(blocksOf(t.Fr)));
        L.childL = leaf->left ? leaf->left->h : Sha256::absent();
        L.childR = leaf->right ? leaf->right->h : Sha256::absent();
        L.path = path;
        return L;
    }

    // Top-k query: returns (answer blocks, VO).
    std::pair<std::vector<Block>, VO> topK(const Eigen::VectorXd& x, int k) const {
        VO vo; vo.signature = rootSig; vo.version = version;
        std::vector<PathEntry> path;
        Node* nd = root;
        const Node* loN = nullptr; const Node* hiN = nullptr;
        Tuple loT, hiT;
        std::vector<PathEntry> loPath, hiPath;

        while (nd && !nd->is_placeholder()) {
            if (!nd->is_relevant_leaf) {
                double v; PathEntry e = internalEntry(nd, x, v);
                path.push_back(e);
                nd = (v < 0.0) ? nd->left.get() : nd->right.get();
                continue;
            }
            Tuple t = computeTuple(*F, *planes, nd);
            if (t.rank <= k) {
                if (!loN || loT.rank < t.rank) { loN = nd; loT = t; loPath = path; }
                path.push_back(leafEntry(nd, t, +1));
                nd = nd->right.get();
            } else {
                if (!hiN || hiT.rank > t.rank) { hiN = nd; hiT = t; hiPath = path; }
                path.push_back(leafEntry(nd, t, -1));
                nd = nd->left.get();
            }
        }

        if (loN) {
            OpenedLeaf L = openLeaf(loN, loPath, loT);
            L.flOpened = true; L.Fl = blocksOf(loT.Fl);
            if (!hiN) { L.frOpened = true; L.Fr = blocksOf(loT.Fr); }
            vo.lo = std::move(L);
        }
        if (hiN) {
            OpenedLeaf H = openLeaf(hiN, hiPath, hiT);
            H.flOpened = true; H.Fl = blocksOf(hiT.Fl);
            vo.hi = std::move(H);
        }

        // Answer assembly (client recomputes this independently; use the SAME
        // block-evaluation arithmetic as the client to avoid summation-order
        // rounding differences at boundaries).
        std::vector<Block> base, gap;
        if (loN) { base = blocksOf(loT.Fl); base.push_back(blockOf(loN->target_function_id)); }
        if (hiN) {
            std::vector<Block> excl = base; std::sort(excl.begin(), excl.end());
            for (const Block& blk : blocksOf(hiT.Fl))
                if (!std::binary_search(excl.begin(), excl.end(), blk)) gap.push_back(blk);
        } else if (loN) gap = blocksOf(loT.Fr);
        std::sort(gap.begin(), gap.end(), [&](const Block& a, const Block& b) {
            double va = evalBlock(a, x), vb = evalBlock(b, x);
            return va != vb ? va > vb : a < b;
        });
        std::vector<Block> answer = base;
        for (size_t i = 0; i < gap.size() && int(answer.size()) < k; ++i) answer.push_back(gap[i]);
        return { answer, vo };
    }

    // Range query q = (x, a, b): explore the slice, keep tightest value brackets.
    std::pair<std::vector<Block>, VO> range(const Eigen::VectorXd& x, double a, double b) const {
        VO vo; vo.signature = rootSig; vo.version = version;
        const Node* plusN = nullptr; const Node* minusN = nullptr;
        const Node* anyN = nullptr;                          // fallback slice leaf
        Tuple plusT, minusT, anyT;
        std::vector<PathEntry> plusPath, minusPath, anyPath;

        std::vector<std::pair<Node*, std::vector<PathEntry>>> st;
        st.push_back({ root, {} });
        while (!st.empty()) {
            auto [nd, path] = std::move(st.back()); st.pop_back();
            while (nd && !nd->is_placeholder()) {
                if (!nd->is_relevant_leaf) {
                    double v; PathEntry e = internalEntry(nd, x, v);
                    path.push_back(e);
                    nd = (v < 0.0) ? nd->left.get() : nd->right.get();
                    continue;
                }
                Tuple t = computeTuple(*F, *planes, nd);
                double pv = evalBlock(blockOf(nd->target_function_id), x);  // client-identical
                if (!anyN) { anyN = nd; anyT = t; anyPath = path; }
                if (pv > b && (!plusN  || evalBlock(blockOf(plusN->target_function_id), x)  > pv)) { plusN  = nd; plusT  = t; plusPath  = path; }
                if (pv < a && (!minusN || evalBlock(blockOf(minusN->target_function_id), x) < pv)) { minusN = nd; minusT = t; minusPath = path; }
                if (nd->right && !nd->right->is_placeholder()) {
                    auto pathR = path; pathR.push_back(leafEntry(nd, t, +1));
                    st.push_back({ nd->right.get(), std::move(pathR) });
                }
                path.push_back(leafEntry(nd, t, -1));
                nd = nd->left.get();
            }
        }

        if (minusN) { OpenedLeaf L = openLeaf(minusN, minusPath, minusT); L.flOpened = true; L.Fl = blocksOf(minusT.Fl); vo.lo = std::move(L); }
        if (plusN)  {
            OpenedLeaf H = openLeaf(plusN, plusPath, plusT);
            H.flOpened = true; H.Fl = blocksOf(plusT.Fl);
            if (!minusN) { H.frOpened = true; H.Fr = blocksOf(plusT.Fr); }   // C = F_r(p_+)
            vo.hi = std::move(H);
        }
        if (!plusN && !minusN && anyN) {
            // Both brackets absent: every slice pivot's value lies inside
            // [a, b].  Open both buckets of one slice leaf, giving C = F.
            OpenedLeaf L = openLeaf(anyN, anyPath, anyT);
            L.flOpened = true; L.Fl = blocksOf(anyT.Fl);
            L.frOpened = true; L.Fr = blocksOf(anyT.Fr);
            vo.lo = std::move(L);
        }

        std::vector<Block> answer;
        for (int f = 1; f <= F->n; ++f) {
            Block blk = blockOf(f);
            double v = evalBlock(blk, x);      // client-identical arithmetic
            if (v >= a && v <= b) answer.push_back(std::move(blk));
        }
        return { answer, vo };
    }
};

// ---------------------------------------------------------------------------
// Client-side verification: a fully independent implementation of the
// paper's VerifyTopK / VerifyRange.  The client knows only (n, d, D = unit
// box, pk); everything else comes from the query, the answer, and the VO.
// ---------------------------------------------------------------------------
class Client {
public:
    int n = 0, d = 0;
    const Signer* signer = nullptr;    // stands in for the public key
    uint32_t highWater = 0;

    struct Result { bool accepted = false; std::string reason; };

    static Hash leafHashFrom(const OpenedLeaf& L, const Hash& c1, const Hash& c2,
                             const Hash& c3, const Hash& c4) {
        Sha256 s; s.update(uint8_t(TAG_LEAF));
        s.update(c1); s.update(c2); s.update(c3); s.update(c4);
        s.update(L.childL); s.update(L.childR);
        return s.final();
    }
    static Hash foldPath(const OpenedLeaf& L, Hash h) {
        for (auto it = L.path.rbegin(); it != L.path.rend(); ++it) {
            Sha256 s;
            if (it->isLeafEntry) {
                s.update(uint8_t(TAG_LEAF));
                s.update(it->al.c1); s.update(it->al.c2); s.update(it->al.c3); s.update(it->al.c4);
                if (it->al.dir < 0) { s.update(h); s.update(it->al.sibling); }
                else                { s.update(it->al.sibling); s.update(h); }
            } else {
                s.update(uint8_t(TAG_INTERNAL));
                s.update(it->in.ci.data(), it->in.ci.size());
                s.update(it->in.cj.data(), it->in.cj.size());
                if (it->in.dir < 0) { s.update(h); s.update(it->in.sibling); }
                else                { s.update(it->in.sibling); s.update(h); }
            }
            h = s.final();
        }
        return h;
    }

    // Path-predicate re-evaluation at the client's own x; rejects on tie.
    bool checkPath(const OpenedLeaf& L, const Eigen::VectorXd& x, std::string& why) const {
        for (const auto& e : L.path) {
            if (e.isLeafEntry) continue;                     // no predicate at ancestor leaves
            double v = evalBlock(e.in.ci, x) - evalBlock(e.in.cj, x);
            if (std::abs(v) <= kEps) { why = "boundary query (tie on path predicate)"; return false; }
            if ((v < 0.0 && e.in.dir != -1) || (v > 0.0 && e.in.dir != +1)) {
                why = "path predicate inconsistent with x"; return false;
            }
        }
        return true;
    }

    // Verify one opened leaf: recompute component hashes for opened buckets,
    // check rank consistency, fold to the root hash.
    bool checkLeaf(const OpenedLeaf& L, Hash& rootOut, std::string& why) const {
        Hash c1 = hashComponent(L.pivot);
        std::string er; encAppendU32(er, uint32_t(L.rank));
        Hash c2 = hashComponent(er);
        Hash c3 = L.flOpened ? hashComponent(encBucket(L.Fl)) : L.hFl;
        Hash c4 = L.frOpened ? hashComponent(encBucket(L.Fr)) : L.hFr;
        if (L.flOpened && L.rank != int(L.Fl.size()) + 1) { why = "rank != |Fl|+1"; return false; }
        if (L.frOpened && L.rank != n - int(L.Fr.size()))  { why = "rank != n-|Fr|"; return false; }
        rootOut = foldPath(L, leafHashFrom(L, c1, c2, c3, c4));
        return true;
    }

    Result verifyTopK(const Eigen::VectorXd& x, int k,
                      const std::vector<Block>& answer, const VO& vo) const {
        Result res;
        for (int t = 0; t < d; ++t)
            if (x[t] < 0.0 || x[t] > 1.0) { res.reason = "x outside D"; return res; }
        if (!vo.lo && !vo.hi) { res.reason = "no opened leaf"; return res; }

        std::optional<Hash> rootHash;
        for (const OpenedLeaf* Lp : { vo.lo ? &*vo.lo : nullptr, vo.hi ? &*vo.hi : nullptr }) {
            if (!Lp) continue;
            std::string why;
            if (!checkPath(*Lp, x, why)) { res.reason = why; return res; }
            Hash r;
            if (!checkLeaf(*Lp, r, why)) { res.reason = why; return res; }
            if (rootHash && std::memcmp(rootHash->data(), r.data(), 32) != 0) {
                res.reason = "opened leaves fold to different roots"; return res;
            }
            rootHash = r;
        }
        std::string msg = Signer::rootMessage(n, d, vo.version, *rootHash);
        if (!signer->verify(msg, vo.signature)) { res.reason = "signature invalid"; return res; }
        if (vo.version < highWater) { res.reason = "stale version"; return res; }

        int rlo = vo.lo ? vo.lo->rank : 0;
        int rhi = vo.hi ? vo.hi->rank : n + 1;
        if (!(rlo <= k && k < rhi)) { res.reason = "bracket condition fails"; return res; }
        if (vo.lo && !vo.lo->flOpened) { res.reason = "Fl(lo) not opened"; return res; }
        if (vo.hi && !vo.hi->flOpened) { res.reason = "Fl(hi) not opened"; return res; }
        if (vo.lo && !vo.hi && !vo.lo->frOpened) { res.reason = "Fr(lo) not opened (no upper bracket)"; return res; }

        // Gap block.
        std::vector<Block> B;
        std::vector<Block> base;
        if (vo.lo) { base = vo.lo->Fl; base.push_back(vo.lo->pivot); }
        if (vo.hi) {
            std::vector<Block> excl = base; std::sort(excl.begin(), excl.end());
            for (const Block& blk : vo.hi->Fl)
                if (!std::binary_search(excl.begin(), excl.end(), blk)) B.push_back(blk);
        } else {
            B = vo.lo->Fr;
        }
        for (const Block& blk : B)
            for (const Block& other : B)
                if (&blk != &other && std::abs(evalBlock(blk, x) - evalBlock(other, x)) <= kEps) {
                    res.reason = "tie inside gap block"; return res;
                }
        std::sort(B.begin(), B.end(), [&](const Block& a2, const Block& b2) {
            double va = evalBlock(a2, x), vb = evalBlock(b2, x);
            return va != vb ? va > vb : a2 < b2;
        });
        std::vector<Block> expect = base;
        for (size_t i = 0; i < B.size() && int(expect.size()) < k; ++i) expect.push_back(B[i]);

        std::vector<Block> got = answer, want = expect;
        std::sort(got.begin(), got.end()); std::sort(want.begin(), want.end());
        if (got != want) { res.reason = "answer mismatch"; return res; }
        res.accepted = true; res.reason = "ok";
        return res;
    }

    Result verifyRange(const Eigen::VectorXd& x, double a, double b,
                       const std::vector<Block>& answer, const VO& vo) const {
        Result res;
        for (int t = 0; t < d; ++t)
            if (x[t] < 0.0 || x[t] > 1.0) { res.reason = "x outside D"; return res; }
        if (!vo.lo && !vo.hi) { res.reason = "no opened leaf"; return res; }

        std::optional<Hash> rootHash;
        for (const OpenedLeaf* Lp : { vo.lo ? &*vo.lo : nullptr, vo.hi ? &*vo.hi : nullptr }) {
            if (!Lp) continue;
            std::string why;
            if (!checkPath(*Lp, x, why)) { res.reason = why; return res; }
            Hash r;
            if (!checkLeaf(*Lp, r, why)) { res.reason = why; return res; }
            if (rootHash && std::memcmp(rootHash->data(), r.data(), 32) != 0) {
                res.reason = "opened leaves fold to different roots"; return res;
            }
            rootHash = r;
        }
        std::string msg = Signer::rootMessage(n, d, vo.version, *rootHash);
        if (!signer->verify(msg, vo.signature)) { res.reason = "signature invalid"; return res; }

        // Bracket validity by value at the client's own x.
        if (vo.hi && evalBlock(vo.hi->pivot, x) <= b) { res.reason = "upper bracket not above b"; return res; }
        if (vo.lo && !vo.lo->frOpened && evalBlock(vo.lo->pivot, x) >= a) { res.reason = "lower bracket not below a"; return res; }

        // Candidate band C.
        std::vector<Block> C;
        if (vo.lo && vo.lo->frOpened && vo.lo->flOpened) {
            // degenerate full opening: C = Fl ∪ {p} ∪ Fr
            C = vo.lo->Fl; C.push_back(vo.lo->pivot);
            C.insert(C.end(), vo.lo->Fr.begin(), vo.lo->Fr.end());
        } else if (vo.lo && vo.hi) {
            std::vector<Block> excl = vo.hi->Fl; excl.push_back(vo.hi->pivot);
            std::sort(excl.begin(), excl.end());
            for (const Block& blk : vo.lo->Fl)
                if (!std::binary_search(excl.begin(), excl.end(), blk)) C.push_back(blk);
        } else if (vo.lo) {
            C = vo.lo->Fl;                       // no pivot above b exists
        } else if (vo.hi && vo.hi->frOpened) {
            C = vo.hi->Fr;                       // no pivot below a exists
        } else {
            res.reason = "missing bracket opening"; return res;
        }

        std::vector<Block> expect;
        for (const Block& blk : C) {
            double v = evalBlock(blk, x);
            if (v >= a && v <= b) expect.push_back(blk);
        }
        std::vector<Block> got = answer, want = expect;
        std::sort(got.begin(), got.end()); std::sort(want.begin(), want.end());
        if (got != want) { res.reason = "answer mismatch"; return res; }
        res.accepted = true; res.reason = "ok";
        return res;
    }
};

} // namespace aof
