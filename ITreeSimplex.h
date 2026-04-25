#pragma once
#include <memory>
#include <vector>
#include <queue>
#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include "SimplexSolver.h"
#include "MerkleTree.h"

// Defines a half-space constraint: normal * x <= rhs
struct LinearConstraint {
    Eigen::VectorXd normal;
    double rhs;
};

struct LPTreeNode {
    int splitting_plane_id = -1; // Index into LPTreeBuilder::m_history
    std::unique_ptr<LPTreeNode> left;
    std::unique_ptr<LPTreeNode> right;
    std::unique_ptr<MerkleNode> hroot;
    bool is_leaf() const { return splitting_plane_id == -1; }
    std::vector<std::string> funcOrder;
};


class LPTreeBuilder {
private:
    std::vector<Eigen::VectorXd> m_history;
    int m_dim;

    // ⬇️ timing accumulator (nanoseconds)
    std::chrono::nanoseconds lp_cut_time{0};

public:
    std::unique_ptr<LPTreeNode> root;

    LPTreeBuilder(int dim) : m_dim(dim) {
        root = std::make_unique<LPTreeNode>();
    }

    // ---------------------------------------------------------
    // ITERATIVE INSERTION (BFS-Based, Work-Queue with plane_id)
    // ---------------------------------------------------------
    void insert(const Eigen::VectorXd& h_vec) {
        // 1. Register the new plane in history and get its id
        m_history.push_back(h_vec);
        const int plane_id = static_cast<int>(m_history.size()) - 1;

        // 2. Work item for BFS: node + its constraints
        struct WorkItem {
            LPTreeNode* node;
            std::vector<LinearConstraint> constraints;
        };

        // 3. Initialize queue with root node and empty constraint set
        std::queue<WorkItem> q;
        q.push(WorkItem{root.get(), {}});

        // 4. BFS loop (layer-by-layer)
        while (!q.empty()) {
            WorkItem item = q.front();
            q.pop();

            LPTreeNode* node = item.node;
            std::vector<LinearConstraint>& constraints = item.constraints;
            const Eigen::VectorXd& current_plane = m_history[plane_id];

            using clock = std::chrono::high_resolution_clock;

            auto t0 = clock::now();
            bool cuts = cuts_region_lp(constraints, current_plane);
            auto t1 = clock::now();

            lp_cut_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);

            if (!cuts) {
                continue;
            }

            // B. Logic for leaf vs internal node
            if (node->is_leaf()) {
                // CUTTING A LEAF -> SPLIT
                node->splitting_plane_id = plane_id;
                node->left  = std::make_unique<LPTreeNode>();
                node->right = std::make_unique<LPTreeNode>();
                // The new plane is established at this node. No further processing needed
                // for its children during this insertion of this plane.
                continue;
            }

            // CUTTING AN INTERNAL NODE -> enqueue children (next BFS layer)
            int existing_id = node->splitting_plane_id;
            const Eigen::VectorXd& h_existing = m_history[existing_id];

            // LEFT child: h_existing(x) <= 0
            std::vector<LinearConstraint> left_constraints = constraints;
            left_constraints.push_back({h_existing, 0.0});
            q.push(WorkItem{node->left.get(), std::move(left_constraints)});

            // RIGHT child: h_existing(x) >= 0  =>  -h_existing(x) <= 0
            std::vector<LinearConstraint> right_constraints = constraints;
            right_constraints.push_back({-h_existing, 0.0});
            q.push(WorkItem{node->right.get(), std::move(right_constraints)});
        }
    }

    // Statistics
    int count_nodes(const LPTreeNode* node = nullptr) {
        if (!node) node = root.get();
        int c = 1;
        if (node->left) c += count_nodes(node->left.get());
        if (node->right) c += count_nodes(node->right.get());
        return c;
    }

    int count_leaves(const LPTreeNode* node = nullptr) {
        if (!node) node = root.get();
        if (node->is_leaf()) return 1;
        return count_leaves(node->left.get()) + count_leaves(node->right.get());
    }

    double get_lp_cut_time_ms() const {
        return lp_cut_time.count() / 1e6;  // milliseconds
    }

    double get_lp_cut_time_sec() const {
        return lp_cut_time.count() / 1e9;  // seconds
    }

    int compute_depth(const LPTreeNode* node = nullptr) const {
        if (!node) node = root.get();
        if (!node) return 0;
        if (!node->left && !node->right) return 1;
        int left_depth = node->left ? compute_depth(node->left.get()) : 0;
        int right_depth = node->right ? compute_depth(node->right.get()) : 0;
        return 1 + std::max(left_depth, right_depth);
    }

    void populateLeaf(int n, int dim, bool firstIt, CompactDataset ds, LPTreeNode* node = nullptr)
    {
        if (!node && !firstIt)
        {
            return;
        }
        else if (!node && firstIt)
        {
            node = root.get();
        }
        if (node->left != nullptr) populateLeaf(n, dim, false, ds, node->left.get());
        if (node->right != nullptr) populateLeaf(n, dim, false, ds, node->right.get());
        
        if (node->is_leaf())
        {
            node->funcOrder.resize(n);
            std::queue<int8_t> q2(std::deque<int8_t>(ds.data.begin(), ds.data.end()));

            for (int i = 0; i < n; i++)
            {
                std::string func = "";
                for (int j = 0; j < dim; j++)
                {
                    func += std::to_string(q2.front());
                    q2.pop();
                }
                node->funcOrder[i] = func;
            }
        }
    }

    void createADSRecursive(int n, int dim, bool firstIt, const CompactDataset& ds, LPTreeNode* node = nullptr)
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

        if (node->left != nullptr) createADSRecursive(n, dim, false, ds, node->left.get());
        if (node->right != nullptr) createADSRecursive(n, dim, false, ds, node->right.get());
        node->hroot = std::make_unique<MerkleNode>();
        //if (node->splitting_plane_id == -1) return;

        if (node->is_leaf())
        {
            std::vector<std::string> dataStrings;
            dataStrings.reserve(n);

            for (int i = 0; i < n; i++)
            {
                const char* row_ptr = reinterpret_cast<const char*>(&ds.data[i * dim]);
                dataStrings.emplace_back(row_ptr, dim);
            }

            size_t dataIdx = 0;
            int depth = static_cast<int>(std::ceil(std::log2(n)));

            makeEmptyTree(depth, 0, node->hroot.get(), dataStrings, dataIdx);
            hashData(node->hroot.get());
        }
        else
        {
            if (node->left && node->right && node->left->hroot->is_set && node->right->hroot->is_set)
            {
                computeInternalHash(node->hroot.get(), node->left->hroot.get(), node->right->hroot.get());
            }
        }
    }

    //check to ensure that the ADS has been constructed
    //Only checks the root of the Merkle-tree of each node
    //If Merkle-root contains an empty hash, then all leaf nodes of Merkle-tree are also empty
    bool checkHashConsRec(LPTreeNode* node = nullptr)
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

        //if hash is empty always return false
        //if hash is not empty and child nodes are also not empty then return true
        if (node->hroot == nullptr) return false;
        if (node->left == nullptr && node->right == nullptr && !node->hroot->hash.empty()) return true;
        else if (!node->hroot->is_set) return false;
        else return (l && r);
    }

    size_t estimateStorage(int tnodes, int tleaf, int dim, int n, int ads)
    {
        size_t metad = (size_t)tnodes * sizeof(int);
        size_t mnodes = (2 * (size_t)n) - 1;
        size_t thashes = (size_t)tleaf * mnodes;
        size_t mstore = thashes * 32;

        size_t strPerLeaf = (size_t)n;
        size_t bytesPerString = (size_t)dim;
        size_t funcBytes = (size_t)tleaf * strPerLeaf * bytesPerString;

        if (ads == 0)
        {
            mstore = 0;
        }

        return metad + mstore + funcBytes;
    }

private:
    // Returns true if the objective cuts the region: min < 0 and max > 0
    bool cuts_region_lp(const std::vector<LinearConstraint>& consts, const Eigen::VectorXd& objective) {
        int m = (int)consts.size();
        Eigen::MatrixXd A(m, m_dim);
        Eigen::VectorXd b(m);

        for (int i = 0; i < m; ++i) {
            A.row(i) = consts[i].normal;
            b(i) = consts[i].rhs;
        }

        LPResult res_max = SimplexSolver::solve(A, b, objective);
        if (res_max.status != LPStatus::Optimal) return false;
        double max_val = res_max.objective_value;

        LPResult res_min = SimplexSolver::solve(A, b, -objective);
        if (res_min.status != LPStatus::Optimal) return false;
        double min_val = -res_min.objective_value;

        return (min_val < 0.0 && max_val > 0.0);
    }
};