#pragma once

#include <vector>
#include <unordered_map>
#include <Eigen/Dense>

struct Vertex {
    int id;
    Eigen::VectorXd position;
};

struct Edge {
    int id;
    int v1;
    int v2;
};

// ---------------------------------------------------------------------------
// Farkas Certificate for hyperplane-polytope classification.
//
// The polytope P = conv({v_0,...,v_{n-1}}) is in V-representation.
// For any x in P: x = Sum(lambda_i * v_i), lambda_i >= 0, Sum(lambda_i) = 1.
// Therefore H·x = Sum(lambda_i * H·v_i), so the extremum over P is attained
// at a vertex.  This lets us read off LP dual certificates directly.
//
// PARTITION  (type == Partition):
//   Certificate: two vertex indices pos_witness, neg_witness s.t.
//     H · v[pos_witness] > 0   and   H · v[neg_witness] < 0
//   Verifier: recompute both dot products, check opposite strict signs.
//
// NON-PARTITION, positive side  (type == NonPartitionPositive):
//   Certificate: scalar mu = min_i H·v_i >= 0, achieved at extremal_vertex.
//   This is the dual optimal value for min{H^T x : x in P}.
//   Verifier: confirm H·v[extremal_vertex] == mu, confirm mu >= 0.
//   (Full check: also verify H·v_i >= mu for every vertex i.)
//
// NON-PARTITION, negative side  (type == NonPartitionNegative):
//   Certificate: scalar mu = max_i H·v_i <= 0, achieved at extremal_vertex.
//   This is the dual optimal value for max{H^T x : x in P}.
//   Verifier: confirm H·v[extremal_vertex] == mu, confirm mu <= 0.
//   (Full check: also verify H·v_i <= mu for every vertex i.)
// ---------------------------------------------------------------------------
struct FarkasCertificate {
    enum class Type {
        Partition,              // hyperplane strictly partitions P
        NonPartitionPositive,   // all of P satisfies H·x >= 0
        NonPartitionNegative    // all of P satisfies H·x <= 0
    };

    Type type = Type::NonPartitionNegative;

    // ---- Partition fields (valid when type == Partition) ----
    int    pos_witness = -1;    // index into Polytope::vertices with H·v > 0
    double pos_value   = 0.0;   // H · v[pos_witness]
    int    neg_witness = -1;    // index into Polytope::vertices with H·v < 0
    double neg_value   = 0.0;   // H · v[neg_witness]

    // ---- Non-partition fields (valid when type != Partition) ----
    double mu              = 0.0;  // extremal dot-product (dual certificate scalar)
    int    extremal_vertex = -1;   // index into Polytope::vertices achieving mu
};

struct Polytope {
    int dim; // Dimension of the polytope

    std::vector<Vertex> vertices;
    std::vector<Edge> edges;
    std::unordered_map<int, std::vector<int>> constraints;

    void add_vertex(const Eigen::VectorXd& pos, const std::vector<int>& plane_ids) {
        int new_index = vertices.size();
        vertices.push_back({new_index, pos});
        constraints[new_index] = plane_ids;
    }
};