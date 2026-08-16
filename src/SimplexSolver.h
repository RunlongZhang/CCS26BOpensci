#pragma once
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Highs.h> 

enum class LPStatus { Optimal, Infeasible, Unbounded, Error };

struct LPResult {
    LPStatus status;
    double objective_value;
};

class SimplexSolver {
public:
    // ------------------------------------------------------------
    // Solves: Maximize c^T * x
    // Subject to: Ax <= b
    // AND GLOBAL BOUNDS: 0 <= x <= 1
    // ------------------------------------------------------------
    static LPResult solve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c) {
        Highs highs;
        
        // 1. Configure Solver (Silence Output)
        highs.setOptionValue("output_flag", false);
        highs.setOptionValue("presolve", "on"); // Crucial for speed

        // 2. Dimensions
        int num_cons = (int)A.rows();
        int num_vars = (int)A.cols();

        HighsModel model;
        model.lp_.num_col_ = num_vars;
        model.lp_.num_row_ = num_cons;
        model.lp_.sense_ = ObjSense::kMaximize; // We always Maximize
        model.lp_.offset_ = 0;

        // 3. Setup Variables (Columns) -> Bounds [0, 1]
        model.lp_.col_cost_.resize(num_vars);
        model.lp_.col_lower_.resize(num_vars, 0.0);
        model.lp_.col_upper_.resize(num_vars, 1.0);
        
        for(int i=0; i<num_vars; ++i) {
            model.lp_.col_cost_[i] = c(i);
        }

        // 4. Setup Constraints (Rows) -> Bounds [-inf, b]
        // HiGHS uses range constraints L <= Ax <= U
        model.lp_.row_lower_.resize(num_cons, -kHighsInf);
        model.lp_.row_upper_.resize(num_cons);
        
        for(int i=0; i<num_cons; ++i) {
            model.lp_.row_upper_[i] = b(i);
        }

        // 5. Setup Matrix (CSC Format)
        // HiGHS requires Compressed Sparse Column format
        std::vector<int>& a_start = model.lp_.a_matrix_.start_;
        std::vector<int>& a_index = model.lp_.a_matrix_.index_;
        std::vector<double>& a_value = model.lp_.a_matrix_.value_;

        a_start.reserve(num_vars + 1);
        a_start.push_back(0);

        // Iterate Column-by-Column (Eigen is ColMajor by default, but we loop manually for safety)
        for(int j=0; j<num_vars; ++j) {
            for(int i=0; i<num_cons; ++i) {
                double val = A(i, j);
                a_index.push_back(i);
                a_value.push_back(val);
            }
            a_start.push_back((int)a_index.size());
        }
        
        model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;

        // 6. Solve
        HighsStatus pass_status = highs.passModel(model);
        if (pass_status != HighsStatus::kOk) return {LPStatus::Error, 0.0};

        HighsStatus run_status = highs.run();
        
        // 7. Interpret Result
        HighsModelStatus model_status = highs.getModelStatus();
        double obj_val = highs.getObjectiveValue();

        if (model_status == HighsModelStatus::kOptimal) {
            return {LPStatus::Optimal, obj_val};
        } else if (model_status == HighsModelStatus::kInfeasible) {
            return {LPStatus::Infeasible, 0.0};
        } else if (model_status == HighsModelStatus::kUnbounded) {
            return {LPStatus::Unbounded, 0.0};
        } else {
            return {LPStatus::Error, 0.0};
        }
    }
};