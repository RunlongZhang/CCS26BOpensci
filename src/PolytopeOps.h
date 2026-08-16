#pragma once

// 1. Include the Data Structures
#include "PolytopeStructs.h"

// 2. Include Standard Libraries needed for logic/printing
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <map>
#include <algorithm>

// Shared geometric epsilon for "on-plane" and side classification.
// Keep this aligned with the tolerance you use in the simplex/LP layer.
inline constexpr double GEOM_EPS = 1e-7;

// Note the 'inline' keyword!
inline Polytope create_hypercube(int dim) {
    Polytope cube;
    cube.dim = dim;
    size_t num_vertices = 1ULL << dim;

    for (size_t i = 0; i < num_vertices; ++i) {
        Eigen::VectorXd position(dim);
        std::vector<int> plane_ids;
        plane_ids.reserve(dim);

        for (int k = 0; k < dim; ++k) {
            bool is_upper_bound = (i >> k) & 1;
            if (is_upper_bound) {
                position[k] = 1.0;
                plane_ids.push_back(-(2 * k + 2));
            } else {
                position[k] = 0.0;
                plane_ids.push_back(-(2 * k + 1));
            }
        }
        std::sort(plane_ids.begin(), plane_ids.end());
        cube.add_vertex(position, plane_ids);
    }

    int edge_id_counter = 0;
    for (int i = 0; i < num_vertices; ++i) {
        for (int k = 0; k < dim; ++k) {
            int neighbor_idx = i ^ (1 << k);
            if (i < neighbor_idx) {
                Edge e{edge_id_counter++, static_cast<int>(i), neighbor_idx};
                cube.edges.push_back(e);
            }
        }
    }
    return cube;
}

// Create an axis-aligned hypercube [p, p+length]^dim
// - `p_low` is the low (min) corner, size must be `dim`.
// - `length` is the side length.
// - Uses the same constraint-id convention as create_hypercube(int dim).
inline Polytope create_input_domain_poly(int dim, const Eigen::VectorXd& p_low, double length) {
    Polytope cube;
    cube.dim = dim;
    if (p_low.size() != dim) {
        throw std::runtime_error("create_input_domain_poly: p_low.size() != dim");
    }
    size_t num_vertices = 1ULL << dim;

    for (size_t i = 0; i < num_vertices; ++i) {
        Eigen::VectorXd position(dim);
        std::vector<int> plane_ids;
        plane_ids.reserve(dim);

        for (int k = 0; k < dim; ++k) {
            bool is_upper_bound = (i >> k) & 1;
            if (is_upper_bound) {
                position[k] = p_low[k] + length;
                plane_ids.push_back(-(2 * k + 2));
            } else {
                position[k] = p_low[k];
                plane_ids.push_back(-(2 * k + 1));
            }
        }
        std::sort(plane_ids.begin(), plane_ids.end());
        cube.add_vertex(position, plane_ids);
    }

    int edge_id_counter = 0;
    for (size_t i = 0; i < num_vertices; ++i) {
        for (int k = 0; k < dim; ++k) {
            int neighbor_idx = i ^ (1 << k);
            if (i < static_cast<size_t>(neighbor_idx)) {
                Edge e{edge_id_counter++, static_cast<int>(i), neighbor_idx};
                cube.edges.push_back(e);
            }
        }
    }

    return cube;
}

inline void print_polytope(const Polytope& poly) {
    if (poly.vertices.empty()) {
        std::cout << "Empty Polytope\n";
        return;
    }

    std::cout << "=== Polytope Dump (Dim: " << poly.vertices[0].position.size() << ") ===\n";
    std::cout << "Vertices: " << poly.vertices.size() << " | Edges: " << poly.edges.size() << "\n\n";

    std::cout << "--- Vertices ---\n";
    for (const auto& v : poly.vertices) {
        std::cout << "V[" << std::setw(2) << v.id << "] Coords: (";
        for (int k = 0; k < v.position.size(); ++k) {
            std::cout << std::fixed << std::setprecision(2) << v.position[k];
            if (k < v.position.size() - 1) std::cout << ", ";
        }
        std::cout << ")";

        std::cout << " | Constraints: { ";
        if (poly.constraints.count(v.id)) {
            const auto& c_list = poly.constraints.at(v.id);
            for (size_t k = 0; k < c_list.size(); ++k) {
                std::cout << c_list[k];
                if (k < c_list.size() - 1) std::cout << ", ";
            }
        }
        std::cout << " }\n";
    }

    std::cout << "\n--- Edges ---\n";
    for (const auto& e : poly.edges) {
        std::cout << "E[" << std::setw(2) << e.id << "]: "
                  << "V" << e.v1 << " <---> V" << e.v2 << "\n";
    }
    std::cout << "========================================\n";
}


// --- Helper: Generate Combinations ---
// Generates all subsets of size k from the input vector
// We use this to generate the keys for the hashmap
inline void generate_combinations_recursive(
    const std::vector<int>& input,
    std::vector<int>& current,
    int start_index,
    int k,
    std::vector<std::vector<int>>& result)
{
    if (k == 0) {
        result.push_back(current);
        return;
    }
    for (int i = start_index; i <= (int)input.size() - k; ++i) {
        current.push_back(input[i]);
        generate_combinations_recursive(input, current, i + 1, k - 1, result);
        current.pop_back();
    }
}


inline std::vector<std::vector<int>> get_combinations(const std::vector<int>& input, int k) {
    std::vector<std::vector<int>> result;
    std::vector<int> current;
    current.reserve(k);
    generate_combinations_recursive(input, current, 0, k, result);
    return result;
}

// Classifies a polytope (vertex set) against a hyperplane H.
// Returns:
//   -1 : all vertices H(v) <= 0 (including all-on-plane)
//    1 : all vertices H(v) >= 0 (no negative, possibly some on-plane)
//    2 : polytope straddles plane (some vertices H(v)>0, some H(v)<0)
inline int classify_polytope_against_plane(const Polytope& P, const Eigen::VectorXd& H) {
    bool has_pos = false, has_neg = false;
    for (const auto& v : P.vertices) {
        double d = H.dot(v.position);
        if (d > GEOM_EPS) {
            has_pos = true;
        } else if (d < -GEOM_EPS) {
            has_neg = true;
        }
        if (has_pos && has_neg) return 2;
    }
    if (has_pos && !has_neg) return 1;
    return -1;
}

inline int classify_polytope_against_plane_v2(const Polytope& P, const Eigen::VectorXd& H, double eps = 1e-9) {
    bool has_pos = false, has_neg = false;
    for (const auto& v : P.vertices) {
        double d = H.dot(v.position);
        if (d > eps) {
            has_pos = true;
        } else if (d < -eps) {
            has_neg = true;
        }
        if (has_pos && has_neg) return 2;
    }
    if (has_pos && !has_neg) return 1;  // all on the positive side (or on a plane)
    if (has_neg && !has_pos) return -1; // all on the negative side (or on a plane)
    return 0;                            // all (approximately) on the plane
}

// ---------------------------------------------------------------------------
// classify_polytope_against_plane_v3
//
// Returns:
//   2  : hyperplane strictly partitions P (vertices on both strict sides)
//   1  : P lies entirely on the positive side  (all H·v >= 0, none < 0)
//  -1  : P lies entirely on the negative side  (all H·v <= 0, none > 0)
//        (includes the degenerate case where all vertices are on the plane)
//
// Optional output parameter `cert_out`:
//   When non-null, a FarkasCertificate is written that proves the result.
//   See FarkasCertificate in PolytopeStructs.h for the verification procedure.
//   When null, the early-exit optimisation for the partition case is preserved.
// ---------------------------------------------------------------------------
inline int classify_polytope_against_plane_v3(
    const Polytope&      P,
    const Eigen::VectorXd& H,
    FarkasCertificate*   cert_out = nullptr)
{
    // ---- Degenerate: empty polytope ----------------------------------------
    if (P.vertices.empty()) {
        if (cert_out) {
            *cert_out = FarkasCertificate{};
            cert_out->type             = FarkasCertificate::Type::NonPartitionNegative;
            cert_out->mu               = 0.0;
            cert_out->extremal_vertex  = -1;
        }
        return -1;
    }

    // ---- Single pass: classify + track extrema for certificate -------------
    bool   has_pos = false,  has_neg = false;
    int    pos_witness = -1, neg_witness = -1;
    double pos_val = 0.0,    neg_val = 0.0;

    // Initialise extrema from vertex 0
    double init_d  = H.dot(P.vertices[0].position);
    double max_val = init_d,  min_val = init_d;
    int    max_vtx = 0,       min_vtx = 0;

    for (int i = 0; i < static_cast<int>(P.vertices.size()); ++i) {
        double d = H.dot(P.vertices[i].position);

        // Update running extrema (needed for the non-partition certificate)
        if (d > max_val) { max_val = d; max_vtx = i; }
        if (d < min_val) { min_val = d; min_vtx = i; }

        // Classify vertex
        if (d > GEOM_EPS) {
            if (!has_pos) { has_pos = true; pos_witness = i; pos_val = d; }
        } else if (d < -GEOM_EPS) {
            if (!has_neg) { has_neg = true; neg_witness = i; neg_val = d; }
        }

        // Early exit only when no certificate is requested
        if (!cert_out && has_pos && has_neg) return 2;
    }

    // ---- Build certificate and return --------------------------------------

    // ---- Build certificate and return -------------------------------------

    if (has_pos && has_neg) {
        // --- PARTITION ---
        // Primal witnesses: one vertex strictly above, one strictly below.
        if (cert_out) {
            cert_out->type        = FarkasCertificate::Type::Partition;
            cert_out->pos_witness = pos_witness;
            cert_out->pos_value   = pos_val;
            cert_out->neg_witness = neg_witness;
            cert_out->neg_value   = neg_val;
        }
        return 2;
    }

    if (has_pos) {
        // --- NON-PARTITION, positive side ---
        // Dual certificate: mu = min_i H·v_i >= 0.
        // For all x in P: H·x >= mu >= 0  (since H·x is a convex combination
        // of vertex evaluations, each >= mu).
        if (cert_out) {
            cert_out->type             = FarkasCertificate::Type::NonPartitionPositive;
            cert_out->mu               = min_val;
            cert_out->extremal_vertex  = min_vtx;
        }
        return 1;
    }

    // --- NON-PARTITION, negative side (includes all-on-plane) ---
    // Dual certificate: mu = max_i H·v_i <= 0.
    // For all x in P: H·x <= mu <= 0.
    if (cert_out) {
        cert_out->type             = FarkasCertificate::Type::NonPartitionNegative;
        cert_out->mu               = max_val;
        cert_out->extremal_vertex  = max_vtx;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// verify_farkas_certificate
//
// Checks a FarkasCertificate produced by classify_polytope_against_plane_v3.
//
// `full_check`:
//   false (default) — lightweight O(d) verification:
//     Partition:     recomputes the two witness dot products.
//     Non-partition: recomputes the single extremal dot product and checks mu.
//     This is the intended use on the receiver side: the cloud does O(n*d)
//     work to find the certificate; the verifier does O(d) to confirm it.
//
//   true — full O(n*d) verification: additionally sweeps every vertex to
//     confirm no vertex violates the claimed bound.  Use when you do not
//     trust the extremal-vertex claim.
//
// Returns true iff the certificate is valid under the chosen check level.
// ---------------------------------------------------------------------------
inline bool verify_farkas_certificate(
    const Polytope&          P,
    const Eigen::VectorXd&   H,
    const FarkasCertificate& cert,
    double eps        = GEOM_EPS,
    bool   full_check = true)
{
    const int n = static_cast<int>(P.vertices.size());

    switch (cert.type) {

        case FarkasCertificate::Type::Partition: {
            if (cert.pos_witness < 0 || cert.pos_witness >= n) return false;
            if (cert.neg_witness < 0 || cert.neg_witness >= n) return false;
            double pv = H.dot(P.vertices[cert.pos_witness].position);
            double nv = H.dot(P.vertices[cert.neg_witness].position);
            return (pv > eps) && (nv < -eps);
        }

        case FarkasCertificate::Type::NonPartitionPositive: {
            if (cert.extremal_vertex < 0 || cert.extremal_vertex >= n) return false;
            if (cert.mu < -eps) return false;
            double ev = H.dot(P.vertices[cert.extremal_vertex].position);
            if (std::abs(ev - cert.mu) > eps) return false;
            if (full_check) {
                for (const auto& v : P.vertices)
                    if (H.dot(v.position) < cert.mu - eps) return false;
            }
            return true;
        }

        case FarkasCertificate::Type::NonPartitionNegative: {
            if (cert.extremal_vertex < 0 || cert.extremal_vertex >= n) return false;
            if (cert.mu > eps) return false;
            double ev = H.dot(P.vertices[cert.extremal_vertex].position);
            if (std::abs(ev - cert.mu) > eps) return false;
            if (full_check) {
                for (const auto& v : P.vertices)
                    if (H.dot(v.position) > cert.mu + eps) return false;
            }
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// print_farkas_certificate  —  diagnostic / logging helper
// ---------------------------------------------------------------------------
inline void print_farkas_certificate(
    const Polytope&          P,
    const Eigen::VectorXd&   H,
    const FarkasCertificate& cert)
{
    std::cout << "=== Farkas Certificate ===\n";
    switch (cert.type) {

        case FarkasCertificate::Type::Partition:
            std::cout << "Type        : PARTITION\n";
            std::cout << "Pos witness : V[" << cert.pos_witness << "]"
                      << "  H·v = " << std::fixed << std::setprecision(6)
                      << cert.pos_value << "  (> 0, above plane)\n";
            std::cout << "Neg witness : V[" << cert.neg_witness << "]"
                      << "  H·v = " << cert.neg_value << "  (< 0, below plane)\n";
            break;

        case FarkasCertificate::Type::NonPartitionPositive:
            std::cout << "Type           : NON-PARTITION  (P on positive side)\n";
            std::cout << "mu (min H·v)   : " << std::fixed << std::setprecision(6)
                      << cert.mu << "  >= 0\n";
            std::cout << "Extremal vertex: V[" << cert.extremal_vertex << "]"
                      << "  H·v = "
                      << H.dot(P.vertices[cert.extremal_vertex].position) << "\n";
            std::cout << "Proof          : for all x in P,  H·x >= mu >= 0\n";
            break;

        case FarkasCertificate::Type::NonPartitionNegative:
            std::cout << "Type           : NON-PARTITION  (P on negative side)\n";
            std::cout << "mu (max H·v)   : " << std::fixed << std::setprecision(6)
                      << cert.mu << "  <= 0\n";
            std::cout << "Extremal vertex: V[" << cert.extremal_vertex << "]"
                      << "  H·v = "
                      << H.dot(P.vertices[cert.extremal_vertex].position) << "\n";
            std::cout << "Proof          : for all x in P,  H·x <= mu <= 0\n";
            break;
    }
    std::cout << "==========================\n";
}

inline Polytope slice_polytope(const Polytope& P, const Eigen::VectorXd& H, int h_id) {
    Polytope result_poly;
    result_poly.dim = P.dim;

    // Helper to track which ORIGINAL vertices have been added to the result
    // Key: Old Vertex ID, Value: New Vertex ID in result_poly
    std::unordered_map<int, int> old_to_new_map;

    auto is_origin = [](const Eigen::VectorXd& pos) {
        return pos.squaredNorm() < GEOM_EPS * GEOM_EPS;
    };

    // use shared GEOM_EPS

    // --- PHASE 1: Geometry (Find Vertices) ---
    for (const auto& edge : P.edges) {
        const Vertex& v1 = P.vertices[edge.v1];
        const Vertex& v2 = P.vertices[edge.v2];

        double d1 = v1.position.dot(H); // Add bias here if needed
        double d2 = v2.position.dot(H);

        // CASE A: Vertex 1 lies exactly on the plane (but ignore the origin)
        if (std::abs(d1) < GEOM_EPS && !is_origin(v1.position)) {
            if (old_to_new_map.find(v1.id) == old_to_new_map.end()) {
                // Copy the vertex, but ADD the new hyperplane constraint
                std::vector<int> new_cons = P.constraints.at(v1.id);
                new_cons.push_back(h_id);
                std::sort(new_cons.begin(), new_cons.end());

                // Add to result
                result_poly.add_vertex(v1.position, new_cons);
                old_to_new_map[v1.id] = result_poly.vertices.back().id;
            }
        }

        // CASE B: Vertex 2 lies exactly on the plane (but ignore the origin)
        // (Note: We check both independently because an edge might lie entirely on the plane)
        if (std::abs(d2) < GEOM_EPS && !is_origin(v2.position)) {
            if (old_to_new_map.find(v2.id) == old_to_new_map.end()) {
                std::vector<int> new_cons = P.constraints.at(v2.id);
                new_cons.push_back(h_id);
                std::sort(new_cons.begin(), new_cons.end());

                result_poly.add_vertex(v2.position, new_cons);
                old_to_new_map[v2.id] = result_poly.vertices.back().id;
            }
        }

        // CASE C: Strict Crossing (One positive, one negative)
        // We only calculate intersection if NEITHER is on the plane (to avoid duplicates with A/B)
        if ((d1 > GEOM_EPS && d2 < -GEOM_EPS) || (d1 < -GEOM_EPS && d2 > GEOM_EPS)) {

            double t = d1 / (d1 - d2);
            Eigen::VectorXd new_pos = v1.position - t * (v1.position - v2.position);

            // Merge constraints from v1 and v2
            std::vector<int> new_constraints;
            const auto& c1 = P.constraints.at(v1.id);
            const auto& c2 = P.constraints.at(v2.id);

            std::set_intersection(
                c1.begin(), c1.end(),
                c2.begin(), c2.end(),
                std::back_inserter(new_constraints)
            );
            new_constraints.push_back(h_id);
            std::sort(new_constraints.begin(), new_constraints.end());

            result_poly.add_vertex(new_pos, new_constraints);
        }
    }

    // --- PHASE 2: Topology (Reconnect Edges) ---
    // This remains exactly the same as before. Your streaming logic handles
    // the reused vertices perfectly because they now include the 'h_id' constraint.

    std::map<std::vector<int>, int> pending_partners;
    int edge_id_counter = 0;
    int d = P.dim;

    for (const auto& v : result_poly.vertices) {
        const std::vector<int>& c_ids = result_poly.constraints.at(v.id);

        // Important: Old vertices on the plane might have > d constraints.
        // Generating all combinations of (d-1) ensures they connect
        // to both their "old" neighbors (if on plane) and "new" neighbors.
        std::vector<std::vector<int>> keys = get_combinations(c_ids, d - 1);

        for (const auto& key : keys) {
            auto it = pending_partners.find(key);
            if (it != pending_partners.end()) {
                int partner_id = it->second;
                // Avoid self-loops (rare degenerate case)
                if (partner_id != v.id) {
                    Edge new_edge{edge_id_counter++, partner_id, v.id};
                    result_poly.edges.push_back(new_edge);
                    pending_partners.erase(it);
                }
            } else {
                pending_partners[key] = v.id;
            }
        }
    }

    return result_poly;
}


// Returns {P_positive, P_negative}
// Returns {P_positive, P_negative}
inline std::pair<Polytope, Polytope>
split_polytope(const Polytope& P, const Eigen::VectorXd& H, int h_id) {
    Polytope p_pos, p_neg;
    p_pos.dim = P.dim;
    p_neg.dim = P.dim;


    // Map original vertex id -> new vertex id in each sub-polytope
    std::unordered_map<int, int> pos_vertex_map;
    std::unordered_map<int, int> neg_vertex_map;

    auto add_pos_vertex = [&](int orig_vid) -> int {
        auto it = pos_vertex_map.find(orig_vid);
        if (it != pos_vertex_map.end()) return it->second;
        const Vertex& v = P.vertices[orig_vid];
        p_pos.add_vertex(v.position, P.constraints.at(v.id));
        int new_id = p_pos.vertices.back().id;
        pos_vertex_map[orig_vid] = new_id;
        return new_id;
    };

    auto add_neg_vertex = [&](int orig_vid) -> int {
        auto it = neg_vertex_map.find(orig_vid);
        if (it != neg_vertex_map.end()) return it->second;
        const Vertex& v = P.vertices[orig_vid];
        p_neg.add_vertex(v.position, P.constraints.at(v.id));
        int new_id = p_neg.vertices.back().id;
        neg_vertex_map[orig_vid] = new_id;
        return new_id;
    };

    struct IntersectionInfo {
        Eigen::VectorXd pos;      // intersection coordinates
        std::vector<int> cons;    // full constraint set (includes h_id), sorted
        int pos_id;               // vertex id in p_pos (or -1 if not created)
        int neg_id;               // vertex id in p_neg (or -1 if not created)
    };

    std::vector<IntersectionInfo> intersections;
    // Keyed by full constraint set; assumes constraint sets uniquely identify vertices
    std::map<std::vector<int>, int> intersection_index_by_cons;

    int pos_edge_id = 0;
    int neg_edge_id = 0;

    // --- MAIN EDGE LOOP ---
    for (const auto& e : P.edges) {
        int i = e.v1;
        int j = e.v2;
        const Vertex& vi = P.vertices[i];
        const Vertex& vj = P.vertices[j];

        double hi = H.dot(vi.position);
        double hj = H.dot(vj.position);

        // Classify side with tolerance
        int si = 0, sj = 0;
        if (hi > GEOM_EPS)      si = 1;
        else if (hi < -GEOM_EPS) si = -1;
        else                     si = 0;

        if (hj > GEOM_EPS)      sj = 1;
        else if (hj < -GEOM_EPS) sj = -1;
        else                     sj = 0;

        // Corner case: both endpoints on the plane -> discard this edge completely
        if (si == 0 && sj == 0) {
            continue;
        }

        // ---- Case 1: H(vi) >= 0 and H(vj) >= 0 ----
        // (no strictly negative endpoint)
        if ((si != -1) && (sj != -1)) {
            int ni = add_pos_vertex(i);
            int nj = add_pos_vertex(j);
            p_pos.edges.push_back({pos_edge_id++, ni, nj});
            continue;
        }

        // ---- Case 2: H(vi) <= 0 and H(vj) <= 0 ----
        // (no strictly positive endpoint)
        if ((si != 1) && (sj != 1)) {
            int ni = add_neg_vertex(i);
            int nj = add_neg_vertex(j);
            p_neg.edges.push_back({neg_edge_id++, ni, nj});
            continue;
        }

        // ---- Case 3: strict crossing: H(vi) > 0, H(vj) < 0 or vice versa ----
        if ((si == 1 && sj == -1) || (si == -1 && sj == 1)) {
            // Compute intersection point P on this edge
            double d1 = hi;
            double d2 = hj;
            double t = d1 / (d1 - d2); // standard interpolation parameter
            Eigen::VectorXd new_pos = vi.position - t * (vi.position - vj.position);

            // Compute constraint set S for intersection: (c(vi) ∩ c(vj)) ∪ {h_id}
            std::vector<int> new_constraints;
            const auto& c1 = P.constraints.at(vi.id);
            const auto& c2 = P.constraints.at(vj.id);
            std::set_intersection(
                c1.begin(), c1.end(),
                c2.begin(), c2.end(),
                std::back_inserter(new_constraints)
            );
            new_constraints.push_back(h_id);
            std::sort(new_constraints.begin(), new_constraints.end());

            // Deduplicate intersection points by constraint set
            int idx;
            auto it_idx = intersection_index_by_cons.find(new_constraints);
            if (it_idx != intersection_index_by_cons.end()) {
                idx = it_idx->second;
            } else {
                IntersectionInfo info{new_pos, new_constraints, -1, -1};
                intersections.push_back(std::move(info));
                idx = static_cast<int>(intersections.size()) - 1;
                intersection_index_by_cons[new_constraints] = idx;
            }

            IntersectionInfo& I = intersections[idx];

            // Ensure this intersection exists as a vertex in both polytopes
            if (I.pos_id == -1) {
                p_pos.add_vertex(I.pos, I.cons);
                I.pos_id = p_pos.vertices.back().id;
            }
            if (I.neg_id == -1) {
                p_neg.add_vertex(I.pos, I.cons);
                I.neg_id = p_neg.vertices.back().id;
            }

            // Connect vi/P and P/vj on the correct sides
            if (si == 1 && sj == -1) {
                int vi_pos_id = add_pos_vertex(i);
                int vj_neg_id = add_neg_vertex(j);

                p_pos.edges.push_back({pos_edge_id++, vi_pos_id, I.pos_id});
                p_neg.edges.push_back({neg_edge_id++, I.neg_id, vj_neg_id});
            } else { // si == -1 && sj == 1
                int vj_pos_id = add_pos_vertex(j);
                int vi_neg_id = add_neg_vertex(i);

                p_pos.edges.push_back({pos_edge_id++, vj_pos_id, I.pos_id});
                p_neg.edges.push_back({neg_edge_id++, I.neg_id, vi_neg_id});
            }

            continue;
        }

        // Logically we should never reach here, but we keep it for safety
    }

    // --- POST-PROCESSING: connect intersection points among themselves ---
    // We build edges on the cutting plane between intersection points whose
    // constraint sets share any (d-1)-subset.
    if (!intersections.empty()) {
        std::map<std::vector<int>, int> pending;
        int d = P.dim;

        for (int idx = 0; idx < static_cast<int>(intersections.size()); ++idx) {
            const auto& cons = intersections[idx].cons; // sorted, includes h_id
            std::vector<std::vector<int>> keys = get_combinations(cons, d - 1);

            for (const auto& key : keys) {
                auto it = pending.find(key);
                if (it != pending.end()) {
                    int other_idx = it->second;
                    const IntersectionInfo& A = intersections[idx];
                    const IntersectionInfo& B = intersections[other_idx];

                    // Avoid degenerate self-edges (shouldn't happen if dedupe works)
                    if (A.pos_id != B.pos_id) {
                        p_pos.edges.push_back({pos_edge_id++, A.pos_id, B.pos_id});
                    }
                    if (A.neg_id != B.neg_id) {
                        p_neg.edges.push_back({neg_edge_id++, A.neg_id, B.neg_id});
                    }

                    pending.erase(it);
                } else {
                    pending[key] = idx;
                }
            }
        }
    }

    return {p_pos, p_neg};
}


// Given:
//   - root_poly: the initial domain, e.g. the unit hypercube [0,1]^d
//   - planes:    list of hyperplane normals H (all through origin)
//   - plane_ids: corresponding constraint IDs for each H
//   - q:         query point in R^d (should lie inside root_poly)
//
// Returns the polytope that contains q after applying all planes
// in order, always following the subdomain that contains q.
inline Polytope trace_subdomain_along_point(
    const Polytope& root_poly,
    const std::vector<Eigen::VectorXd>& planes,
    const std::vector<int>& plane_ids,
    const Eigen::VectorXd& q
) {
    if (planes.size() != plane_ids.size()) {
        throw std::runtime_error("trace_subdomain_along_point: planes.size() != plane_ids.size()");
    }

    // Start from the root domain
    Polytope current = root_poly;

    // Optional: quick sanity check that q is inside [0,1]^d
    // (you can remove or relax this if your root domain is different)
    for (int k = 0; k < q.size(); ++k) {
        if (q[k] < -GEOM_EPS || q[k] > 1.0 + GEOM_EPS) {
            std::cerr << "[trace_subdomain_along_point] WARNING: q[" << k
                      << "] = " << q[k] << " is outside [0,1] with tolerance.\n";
        }
    }

    for (std::size_t i = 0; i < planes.size(); ++i) {
        const Eigen::VectorXd& H = planes[i];
        int h_id = plane_ids[i];

        // Evaluate current plane at the query point
        double hq = H.dot(q);

        // If q lies *very* close to the plane, you need a tie-breaking rule.
        // Here: treat it as "no split" and just continue.
        if (std::abs(hq) <= GEOM_EPS) {
            // You could also choose to split and follow both, but that
            // defeats the purpose of a single-path trace.
            continue;
        }

        // Split the current polytope by H
        auto [p_pos, p_neg] = split_polytope(current, H, h_id);

        // If split_polytope produced empty polytopes (degenerate case),
        // just keep the current polytope and move on.
        if (p_pos.vertices.empty() && p_neg.vertices.empty()) {
            std::cerr << "[trace_subdomain_along_point] WARNING: split produced two empty polytopes at plane index "
                      << i << "\n";
            continue;
        }

        // Decide which side to keep based on the sign of H(q)
        if (hq > GEOM_EPS) {
            // q is on the "positive" side → keep p_pos if it is non-empty
            if (!p_pos.vertices.empty()) {
                current = std::move(p_pos);
            } else {
                // Numerical corner case: positive sign but empty p_pos
                std::cerr << "[trace_subdomain_along_point] WARNING: H(q)>0 but p_pos is empty at plane index "
                          << i << ", keeping p_neg as fallback.\n";
                current = std::move(p_neg);
            }
        } else { // hq < -GEOM_EPS
            // q is on the "negative" side → keep p_neg if it is non-empty
            if (!p_neg.vertices.empty()) {
                current = std::move(p_neg);
            } else {
                // Numerical corner case: negative sign but empty p_neg
                std::cerr << "[trace_subdomain_along_point] WARNING: H(q)<0 but p_neg is empty at plane index "
                          << i << ", keeping p_pos as fallback.\n";
                current = std::move(p_pos);
            }
        }
    }

    return current;
}