#pragma once
#include <memory>
#include <vector>
#include <print>
#include <queue>
#include <chrono>
#include <string>
#include <format>
#include <filesystem>
#include <Eigen/Dense>
#include "PolytopeStructs.h"
#include "PolytopeOps.h"
#include "FunctionPairGenerator.h"
#include "CompactIO.h"
#include "MerkleTree.h"

struct LinearConstraint {
    Eigen::VectorXd normal;
    double rhs;
};

struct ITreeNode {
    // The hyperplane that splits this node's domain.
    // -1 means "no splitting plane yet" (true leaf).
    int splitting_plane_id = -1;

    // Function this leaf is currently "relevant" for (Fi).
    // -1 means "no target function associated".
    int target_function_id = -1;

    // True iff this node is currently a relevant leaf for some Fi
    // AND should hold a sample point.
    bool is_relevant_leaf = false;

    // A sample point inside this node's subdomain.
    // Only meaningful when is_relevant_leaf == true.
    Eigen::VectorXd sample_point;

    std::unique_ptr<ITreeNode> left;
    std::unique_ptr<ITreeNode> right;

    std::unique_ptr<MerkleNode> hroot;

    // Leaf = has no splitting plane yet.
    // bool is_leaf() const { return splitting_plane_id == -1; }
    bool is_leaf() const { return left == nullptr && right == nullptr; }
};

inline void clear_relevance(ITreeNode& node) {
    node.is_relevant_leaf = false;
    node.target_function_id = -1;
    node.sample_point.resize(0); // empty vector
}

struct InsertionJob {
    ITreeNode* node;
    Polytope fragment;
};

// Rough memory usage estimate for a Polytope in bytes.
inline std::size_t estimate_polytope_memory(const Polytope& P) {
    std::size_t bytes = 0;

    // Vertices (shallow estimate: Vertex object only; Eigen storage is dynamic and not fully counted here)
    bytes += P.vertices.size() * sizeof(Vertex);

    // Edges
    bytes += P.edges.size() * sizeof(Edge);

    // Constraint map (key + vector header)
    bytes += P.constraints.size() * (sizeof(int) + sizeof(std::vector<int>));
    for (const auto& kv : P.constraints) {
        bytes += kv.second.size() * sizeof(int);
    }

    return bytes;
}

class ITreeBuilder {
public:
    // FC time for FsTree: classify_polytope_against_plane(...) + split_polytope(...)
    // Accumulated across the whole run (all inserted planes).
    std::chrono::nanoseconds fc_time_ns{ 0 };

    double get_fc_time_sec() const {
        return static_cast<double>(fc_time_ns.count()) / 1e9;
    }

    std::unique_ptr<ITreeNode> root;
    const std::vector<Eigen::VectorXd>* global_planes = nullptr;

    // NEW: current function whose group we are inserting.
    int current_function_id = -1;

    ITreeBuilder(const Polytope& root_domain) {
        root = std::make_unique<ITreeNode>();
    }

    // DFS (non-recursive) insertion with group-aware semantics.
    void insert_dfs_non_recursive(const Polytope& root_domain,
        const Eigen::VectorXd& h_vec,
        int h_id,
        int fi,
        std::size_t n_functions,
        std::size_t& bytes,
        std::chrono::nanoseconds& vtime,
        int dim) {

        if (!global_planes) {
            throw std::runtime_error("ITreeBuilder::insert_dfs_non_recursive: global_planes is nullptr");
        }

        // 1. Initial slice
        Polytope initial_fragment = slice_polytope(root_domain, h_vec, h_id);
        if (initial_fragment.vertices.empty()) {
            // New plane does not intersect domain
            return;
        }

        // Explicit stack for DFS
        std::vector<InsertionJob> stack;
        stack.push_back({ root.get(), std::move(initial_fragment) });

        while (!stack.empty()) {
            InsertionJob job = std::move(stack.back());
            stack.pop_back();

            ITreeNode* current_node = job.node;

            // === CASE 1: Node is a relevant leaf for some Fj ===
            if (current_node->is_relevant_leaf) {
                int tree_target_function = current_node->target_function_id;
                std::size_t plane_index = Generator::pair_index(tree_target_function, fi, n_functions);

                const Eigen::VectorXd& h_pair = (*global_planes)[plane_index];

                // Use the current hyperplane h_vec and the sample point P
                const Eigen::VectorXd& P = current_node->sample_point;
                double val = h_pair.dot(P); // h(P)

                // This node is becoming internal
                // current_node->splitting_plane_id = h_id;
                // clear_relevance(*current_node); // remove sample + target function from internal node

                // Decide which child receives this fragment (Fi vs Fj relation;
                // using sign of h(P) as a proxy for f_i - f_j > 0 vs < 0).
                if (val < 0.0) {
                    // Insert into LEFT subtree
                    stack.push_back({ current_node->left.get(), std::move(job.fragment) });
                }
                else if (val > 0.0) {
                    // Insert into RIGHT subtree
                    stack.push_back({ current_node->right.get(), std::move(job.fragment) });
                }
                else {
                    continue;
                }
                continue;
            }

            // === CASE 2: Pure leaf with no relevance yet ===
            if (current_node->is_leaf()) {
                // Attach the new splitting plane here
                current_node->splitting_plane_id = h_id;
                current_node->sample_point.resize(0);

                current_node->target_function_id = fi;

                current_node->left = std::make_unique<ITreeNode>();
                current_node->right = std::make_unique<ITreeNode>();

                // Pick a sample point from the current fragment for this group
                // and assign it to both children as "relevant leaves"
                if (!job.fragment.vertices.empty()) {
                    const Vertex& v0 = job.fragment.vertices.front();
                    Eigen::VectorXd P(v0.position.size());
                    for (int d = 0; d < P.size(); ++d) {
                        P[d] = v0.position[d];
                    }

                    current_node->left->sample_point = P;
                    current_node->left->target_function_id = fi;

                    current_node->right->sample_point = P;
                    current_node->right->target_function_id = fi;
                }

                // No need to push children onto the stack for this plane:
                // we've fully propagated this hyperplane at this node.
                continue;
            }

            // === CASE 3: Internal node -> classify fragment using existing plane ===
            int existing_id = current_node->splitting_plane_id;
            const Eigen::VectorXd& h_existing = (*global_planes)[existing_id - 1];

            using FCClock = std::chrono::high_resolution_clock;

            FarkasCertificate cert;

            auto fc0 = FCClock::now();
            int cls = classify_polytope_against_plane_v3(job.fragment, h_existing, &cert);
            auto fc1 = FCClock::now();
            fc_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(fc1 - fc0);

            {
                auto v0 = std::chrono::high_resolution_clock::now();
                [[maybe_unused]] bool cert_valid = verify_farkas_certificate(
                    job.fragment, h_existing, cert, GEOM_EPS, /*full_check=*/true);
                auto v1 = std::chrono::high_resolution_clock::now();
                vtime += std::chrono::duration_cast<std::chrono::nanoseconds>(v1 - v0);

                // Storage estimation: P vertices + H + cert
                std::size_t p_bytes;
                std::size_t cert_bytes;
                if (cert.type == FarkasCertificate::Type::Partition)
                {
                    //already accounted for by certificate
                    p_bytes = 0;
                    cert_bytes = sizeof(FarkasCertificate) / 3 * 2;
                }
                else
                {
                    p_bytes = job.fragment.vertices.size() * static_cast<std::size_t>(dim) * sizeof(double);
                    cert_bytes = sizeof(FarkasCertificate) / 3;
                }
                //std::size_t h_bytes = static_cast<std::size_t>(dim) * sizeof(double);
                bytes += p_bytes + cert_bytes;
            }

            if (cls == -1) {
                // Entire fragment on <= side
                if (current_node->left) {
                    stack.push_back({ current_node->left.get(), std::move(job.fragment) });
                }
                continue;
            }

            if (cls == 1) {
                // Entire fragment on >= side
                if (current_node->right) {
                    stack.push_back({ current_node->right.get(), std::move(job.fragment) });
                }
                continue;
            }

            // CASE 3C: True split
            auto sp0 = FCClock::now();
            auto split_result = split_polytope(job.fragment, h_existing, existing_id);
            auto sp1 = FCClock::now();
            fc_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(sp1 - sp0);
            Polytope p_pos = std::move(split_result.first);   // H >= 0
            Polytope p_neg = std::move(split_result.second);  // H <= 0

            if (current_node->right) {
                stack.push_back({ current_node->right.get(), std::move(p_pos) });
            }
            if (current_node->left) {
                stack.push_back({ current_node->left.get(), std::move(p_neg) });
            }
        }
    }

    // Check whether a point P satisfies the given inequality set in standard form
    // a^T x <= b for each LinearConstraint.
    bool point_satisfies_constraints(const Eigen::VectorXd& P,
        const std::vector<LinearConstraint>& constraints,
        double eps = 1e-9) const {
        for (const auto& lc : constraints) {
            double val = lc.normal.dot(P) - lc.rhs; // a^T P - b
            if (val > eps) return false;
        }
        return true;
    }

    int count_nodes(const ITreeNode* node = nullptr) {
        if (!node) node = root.get();
        int c = 1;
        if (node->left) c += count_nodes(node->left.get());
        if (node->right) c += count_nodes(node->right.get());
        return c;
    }

    int count_leaves(const ITreeNode* node = nullptr) {
        if (!node) node = root.get();
        if (node->is_leaf()) return 1;
        return count_leaves(node->left.get()) + count_leaves(node->right.get());
    }

    // Count leaf nodes whose is_relevant_leaf == true (ITERATIVE to avoid stack overflow)
    int count_relevant_leaves() const {
        if (!root) return 0;

        int cnt = 0;
        std::vector<const ITreeNode*> st;
        st.push_back(root.get());

        while (!st.empty()) {
            const ITreeNode* node = st.back();
            st.pop_back();
            if (!node) continue;
            if (node->is_relevant_leaf) { cnt += 1; }
            if (node->is_leaf()) {
                // if (node->is_relevant_leaf) cnt += 1;
                continue;
            }

            if (node->left)  st.push_back(node->left.get());
            if (node->right) st.push_back(node->right.get());
        }
        return cnt;
    }

    // Compute maximum depth of the I-Tree
    int compute_depth(const ITreeNode* node = nullptr) const {
        if (!node) node = root.get();
        if (!node) return 0;
        if (!node->left && !node->right) return 1;
        int left_depth = node->left ? compute_depth(node->left.get()) : 0;
        int right_depth = node->right ? compute_depth(node->right.get()) : 0;
        return 1 + std::max(left_depth, right_depth);
    }

    //temp helper
    Eigen::VectorXd get_sample(const ITreeNode* node = nullptr) const
    {
        if (!node) node = root.get();
        Eigen::VectorXd v(2);
        while (node->left)
        {
            if (node->is_relevant_leaf)
            {
                v = node->sample_point;
            }
            node = node->left.get();
        }

        return v;
    }

    void mark_all_leaves_relevant(int fi) {
        if (!root) return;
        std::vector<ITreeNode*> st;
        st.push_back(root.get());

        while (!st.empty()) {
            ITreeNode* node = st.back();
            st.pop_back();
            if (!node) continue;

            // Internal node: keep traversing
            if (!node->is_leaf()) {
                if (node->right) st.push_back(node->right.get());
                if (node->left)  st.push_back(node->left.get());
                continue;
            }

            // Leaf node: set relevance if not already relevant
            // Only mark leaves that already belong to Fi.
            if (!node->is_relevant_leaf && node->target_function_id == fi) {
                node->is_relevant_leaf = true;

                //TODO
                //This should be fixed at some point
                //Making empty nodes at the bottom of the tree to avoid deadlock at 67% progress
                if (!node->left)  node->left = std::make_unique<ITreeNode>();
                if (!node->right) node->right = std::make_unique<ITreeNode>();
            }
        }
    }

    void create_ADS_recursive(int n, int dim, bool firstIt, const CompactDataset& ds, Eigen::MatrixXd& d_matrix, ITreeNode* node = nullptr)
    {
        //get the root if this is first iteration, else return
        if (!node && !firstIt)
        {
            return;
        }
        else if (!node && firstIt)
        {
            node = root.get();
        }

        if (node->left != nullptr) create_ADS_recursive(n, dim, false, ds, d_matrix, node->left.get());
        if (node->right != nullptr) create_ADS_recursive(n, dim, false, ds, d_matrix, node->right.get());
        node->hroot = std::make_unique<MerkleNode>();
        if (node->target_function_id == -1) return;

        //leaf node case
        if (node->is_relevant_leaf)
        {
            int fi = node->target_function_id;

            //calculated values of all functions relative to sample point
            Eigen::VectorXd fVals = d_matrix * node->sample_point;

            //fi value
            int fiVal = fVals(fi - 1);

            //stores coefficients of fi to string for hashing
            int offset = (fi - 1) * dim;
            std::string fiA;
            fiA.reserve(dim);
            fiA.append(reinterpret_cast<const char*>(&ds.data[offset]), dim);

            //stores coefficients of functions belonging to Fl and Fr to string for hashing
            std::string Fl;
            std::string Fr;
            Fl.reserve(n * dim);
            Fr.reserve(n * dim);
            for (int i = 0; i < n; i++)
            {
                //skip adding function to string if current function is fi
                if (i == fi - 1)
                {
                    continue;
                }

                const char* row_ptr = reinterpret_cast<const char*>(&ds.data[i * dim]);

                //add to Fl
                if (fVals(i) > fiVal)
                {
                    Fl.append(row_ptr, dim);
                }

                //add to Fr
                else if (fVals(i) <= fiVal)
                {
                    Fr.append(row_ptr, dim);
                }
            }

            //checks to see if this relevant leaf is a bottom layer leaf or an internal leaf
            std::vector<std::string> q;
            size_t dataIdx = 0;
            if (node->left->hroot->hash.empty() && node->right->hroot->hash.empty())
            {
                q.reserve(3);
                q.push_back(std::move(Fl));
                q.push_back(std::move(Fr));
                q.push_back(std::move(fiA));
                makeEmptyTree(2, 0, node->hroot.get(), q, dataIdx);
            }
            else //case internal leaf
            {
                q.reserve(5);
                q.push_back(std::string(reinterpret_cast<char*>(node->left->hroot->hash.data()), 32));
                q.push_back(std::string(reinterpret_cast<char*>(node->right->hroot->hash.data()), 32));
                q.push_back(std::move(Fl));
                q.push_back(std::move(Fr));
                q.push_back(std::move(fiA));
                makeEmptyTree(3, 0, node->hroot.get(), q, dataIdx);
            }
            hashData(node->hroot.get());
        }
        //internal node case
        else
        {
            computeInternalHash(node->hroot.get(), node->left->hroot.get(), node->right->hroot.get());
        }
    }

    //check to ensure that the ADS has been constructed
    //Only checks the root of the Merkle-tree of each node
    //If Merkle-root contains an empty hash, then all leaf nodes of Merkle-tree are also empty
    bool checkHashConsRec(ITreeNode* node = nullptr)
    {
        if (!node)
        {
            node = root.get();
        }

        bool l = false, r = false;

        if (node->left != nullptr)
        {
            l = checkHashConsRec(node->left.get());
        }
        if (node->right != nullptr)
        {
            r = checkHashConsRec(node->right.get());
        }

        //if current node is a filler node at below the leaf nodes, then return true (skip)
        //if hash is empty always return false
        //if hash is not empty and child nodes are also not empty then return true
        if (node->hroot == nullptr) return false;
        if (node->left == nullptr && node->right == nullptr) return true;
        else if (!node->hroot->is_set) return false;
        else return (l && r);
    }

    size_t estimateStorage(int tnodes, int tleaf, int dim, int n, int ads)
    {
        size_t metad = (size_t)tnodes * 9;
        size_t pstore = (size_t)tleaf * dim * sizeof(double);
        size_t mnodes = 5;
        size_t thashes = (size_t)tleaf * mnodes;
        size_t mstore = thashes * 32;

        if (ads == 0)
        {
            mstore = 0;
        }

        return metad + pstore + mstore;
    }
};