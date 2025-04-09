#include <bio_ik/manipulability_goal.h>
#include <bio_ik/robot_info.h> // Include for RobotInfo object definition
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_model/joint_model_group.h>
#include <moveit/robot_model/robot_model.h> // Needed for RobotModelConstPtr
#include "utils.h" // For LOG if needed
#include <ros/console.h> // For ROS_DEBUG_NAMED etc.

#include <Eigen/Dense>

namespace bio_ik {

void ManipulabilityGoal::describe(GoalContext& context) const
{
    // Call base class describe first
    Goal::describe(context);

    // This goal depends on all active joints in the group
    const auto& jmg = context.getJointModelGroup();
    const auto& active_joint_models = jmg.getActiveJointModels(); // Get models directly

    // Ensure all relevant joints are marked as used by the problem context
    for (const moveit::core::JointModel* joint_model : active_joint_models) {
        if (joint_model) {
            for (const std::string& var_name : joint_model->getVariableNames()) {
                 context.addVariable(var_name);
            }
        }
    }
}


double ManipulabilityGoal::evaluate(const GoalContext& context) const
{
    FNPROFILER(); // Optional

    // Fix 1: Get RobotModelConstPtr via RobotInfo stored in GoalContext
    // const moveit::core::RobotModelConstPtr& robot_model_ptr = context.getRobotInfo().robot_model;
    const moveit::core::RobotModelConstPtr& robot_model_ptr = context.getRobotInfo().getRobotModelPtr(); // <--- CHANGE THIS LINE
    const auto& jmg = context.getJointModelGroup();

    if (!robot_model_ptr) {
         LOG("ManipulabilityGoal::evaluate - Error: RobotModelConstPtr is null!");
         return 1.0 / epsilon_;
    }

    moveit::core::RobotState temp_state(robot_model_ptr);

    // Fix 2: Initialize state with defaults, then overwrite active variables
    // Get default values into a vector sized for the whole model
    std::vector<double> current_joint_values;
    robot_model_ptr->getVariableDefaultPositions(current_joint_values);

    // Overwrite with current values for active variables being optimized
    for (size_t i = 0; i < context.getProblemVariableCount(); ++i) {
        size_t var_idx = context.getProblemVariableIndex(i); // Index in the full robot model
        if (var_idx < current_joint_values.size()) {
            current_joint_values[var_idx] = context.getProblemVariablePosition(i);
        } else {
             LOG("ManipulabilityGoal::evaluate - Error: Variable index %zu out of bounds (size %zu)!", var_idx, current_joint_values.size());
             return 1.0 / epsilon_; // Return high cost on error
        }
    }

    // Set the state and update FK, Jacobians etc.
    temp_state.setVariablePositions(current_joint_values);
    temp_state.update(true); // Force update

    // Fix 3: Correctly call getEndEffectorTips
    std::vector<std::string> tip_frame_names;
    jmg.getEndEffectorTips(tip_frame_names); // Pass vector by reference

    std::string jacobian_link_name;

    if (!tip_frame_names.empty()) {
        jacobian_link_name = tip_frame_names.front(); // Use the first defined EEF tip
    } else {
        // Fallback to the last link in the group if no EEF tip defined
        const std::vector<std::string>& group_link_names = jmg.getLinkModelNames();
        if (group_link_names.empty()) {
            LOG("ManipulabilityGoal::evaluate - Error: Joint Model Group '%s' has no links!", jmg.getName().c_str());
            return 1.0 / epsilon_; // Return high cost
        }
        jacobian_link_name = group_link_names.back();
    }

    const moveit::core::LinkModel* jacobian_link_model = robot_model_ptr->getLinkModel(jacobian_link_name);

    if (!jacobian_link_model) {
         LOG("ManipulabilityGoal::evaluate - Error: Could not find link model for '%s'", jacobian_link_name.c_str());
         return 1.0 / epsilon_; // Return high cost
    }


    // Compute the Jacobian (passing pointer to jmg is correct)
    Eigen::MatrixXd jacobian;
    if (!temp_state.getJacobian(&jmg, jacobian_link_model, Eigen::Vector3d::Zero(), jacobian)) {
        LOG("ManipulabilityGoal::evaluate - Warning: Failed to compute Jacobian for link '%s'.", jacobian_link_name.c_str());
        return 1.0 / epsilon_; // Return high cost if Jacobian fails
    }

    // Calculate Yoshikawa's manipulability measure: sqrt(det(J * J^T)) or sqrt(det(J^T * J))
    double manipulability = 0.0;
    if (jacobian.rows() > 0 && jacobian.cols() > 0) {
        if (jacobian.rows() <= jacobian.cols()) { // Typically rows <= cols for redundant manipulators
            Eigen::MatrixXd jjt = jacobian * jacobian.transpose();
            double determinant = jjt.determinant();
            manipulability = std::sqrt(std::max(0.0, determinant));
        } else { // Handle cases where task space dim > joint space dim
             Eigen::MatrixXd jtj = jacobian.transpose() * jacobian;
             double determinant = jtj.determinant();
             manipulability = std::sqrt(std::max(0.0, determinant));
        }
    } else {
         LOG("ManipulabilityGoal::evaluate - Warning: Jacobian is empty (rows=%ld, cols=%ld).", jacobian.rows(), jacobian.cols());
    }

    // Cost is inverse of manipulability
    double cost = 1.0 / (manipulability + epsilon_);
    ROS_DEBUG_NAMED("bio_ik.ManipulabilityGoal", "Manip: %.6f -> Cost: %.6f (Weight: %.3f)", manipulability, cost, weight_);

    return cost;
}

} // namespace bio_ik