#pragma once

#include <bio_ik/goal.h>
#include <moveit/planning_scene/planning_scene.h> // Correct header
#include <moveit/robot_state/robot_state.h>
#include <moveit/collision_detection/collision_common.h>
#include <ros/console.h> // For ROS logging

namespace bio_ik {

/**
 * @brief Goal to penalize configurations that are close to or in collision
 * with world obstacles. (Initially ignoring self-collisions).
 */
class CollisionAvoidanceGoal : public Goal {
private:
    double distance_threshold_; // Distance below which penalties start applying
    double penalty_scale_;      // Factor to scale the penalty cost
    double epsilon_;            // Small value for numerical stability if needed

public:
    /**
     * @brief Construct a new Collision Avoidance Goal object
     * @param weight Cost weight for this goal.
     * @param distance_threshold Distance (m) below which penalties start.
     * @param penalty_scale Scaling factor for the penalty cost.
     * @param secondary Should this be treated as a secondary goal? (default true).
     * @param epsilon Small value for potential division stability (default 1e-6).
     */
    CollisionAvoidanceGoal(double weight = 1.0,
                           double distance_threshold = 0.05, // e.g., 5cm
                           double penalty_scale = 10.0,      // Arbitrary starting value
                           bool secondary = true,
                           double epsilon = 1e-6)
        : distance_threshold_(distance_threshold)
        , penalty_scale_(penalty_scale)
        , epsilon_(epsilon)
    {
        weight_ = weight;
        secondary_ = secondary;
    }

    // Describe which resources are needed (all active joints)
    virtual void describe(GoalContext& context) const override;

    // Calculate the cost based on current state and collision checks
    virtual double evaluate(const GoalContext& context) const override;
};

} // namespace bio_ik