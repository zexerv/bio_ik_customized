#pragma once

#include "goal.h"
#include <moveit/robot_state/robot_state.h> // For RobotState & Jacobian
#include <moveit/robot_model/robot_model.h> // For RobotModel
#include <Eigen/Dense>                      // For matrix operations

namespace bio_ik {

/**
 * @brief Goal to maximize manipulability (specifically Yoshikawa's measure).
 * Cost is calculated as 1 / (manipulability + epsilon) to be minimized.
 */
class ManipulabilityGoal : public Goal {
private:
    double epsilon_; // Small value to prevent division by zero

public:
    /**
     * @brief Construct a new Manipulability Goal object
     * @param weight Cost weight for this goal.
     * @param epsilon Small value added to manipulability before inversion (default 1e-6).
     * @param secondary Should this be treated as a secondary goal? (default true).
     */
    ManipulabilityGoal(double weight = 1.0, double epsilon = 1e-6, bool secondary = true)
        : epsilon_(epsilon)
    {
        weight_ = weight;
        secondary_ = secondary;
    }

    // Describe which resources are needed (all joints in the group)
    virtual void describe(GoalContext& context) const override;

    // Calculate the cost based on current state
    virtual double evaluate(const GoalContext& context) const override;
};

} // namespace bio_ik