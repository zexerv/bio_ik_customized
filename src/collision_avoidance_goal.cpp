#include <bio_ik/collision_avoidance_goal.h>
#include <bio_ik/robot_info.h> // Access to robot info if needed
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene/planning_scene.h> // Include necessary headers
#include <moveit/collision_detection/collision_common.h>
#include <moveit/robot_model/joint_model_group.h>
#include <moveit/robot_model/robot_model.h>
#include "utils.h" // For FNPROFILER, LOG
#include <ros/console.h> // For ROS_DEBUG_NAMED etc.

namespace bio_ik {

void CollisionAvoidanceGoal::describe(GoalContext& context) const
{
    // Call base class describe first
    Goal::describe(context);

    // This goal depends on the state of all active joints in the group,
    // as collision checking requires the full robot configuration within the group.
    const auto& jmg = context.getJointModelGroup();
    const auto& active_joint_models = jmg.getActiveJointModels();

    // Ensure all relevant joints are marked as used by the problem context
    for (const moveit::core::JointModel* joint_model : active_joint_models) {
        if (joint_model) {
            for (const std::string& var_name : joint_model->getVariableNames()) {
                context.addVariable(var_name);
            }
        }
    }
    // Note: We don't explicitly add links here, as collision checking implicitly
    // involves the geometry of all links associated with the joints being checked.
}


double CollisionAvoidanceGoal::evaluate(const GoalContext& context) const
{
    FNPROFILER(); // Optional profiling macro

    // 1. Get PlanningScene and RobotModel
    const planning_scene::PlanningScene* planning_scene_ptr = context.getPlanningScene();
    const moveit::core::RobotModel& robot_model = context.getRobotModel(); // Get reference
    const auto& jmg = context.getJointModelGroup();

    // Check if PlanningScene is available
    if (!planning_scene_ptr) {
        ROS_WARN_THROTTLE_NAMED(5.0, "bio_ik.CollisionAvoidanceGoal", "PlanningScene pointer is null in GoalContext. Skipping collision check.");
        return 0.0; // Return zero cost if scene is unavailable
    }

    // 2. Create and populate RobotState
    // Use a RobotState from the PlanningScene to ensure it has the same context,
    // but we need to modify its joint values. Create a copy.
    moveit::core::RobotState temp_state = planning_scene_ptr->getCurrentState(); // Start with current scene state
    // Alternatively: moveit::core::RobotState temp_state(robot_model_ptr); // If using model directly

    // Get default values (needed if context_state wasn't used in plugin's searchPositionIK)
    // std::vector<double> current_joint_values;
    // robot_model.getVariableDefaultPositions(current_joint_values); // Using reference now

    // Overwrite active variables with the ones being evaluated by BioIK
    // Use the values directly from the context's active_variable_positions_
    // We need to map the context's active variable indices (0 to N-1) to the
    // full robot model's variable indices.
    for (size_t i = 0; i < context.getProblemVariableCount(); ++i) {
        size_t model_var_idx = context.getProblemVariableIndex(i); // Index in the full robot model
        temp_state.setVariablePosition(model_var_idx, context.getProblemVariablePosition(i));
    }

    // Enforce bounds and update transforms
    temp_state.enforceBounds(&jmg); // Enforce bounds for the group being checked
    temp_state.updateCollisionBodyTransforms(); // Update transforms needed for collision checking

    // 3. Prepare Collision Check Request
    collision_detection::CollisionRequest req;
    req.group_name = jmg.getName();
    req.distance = true;  // Request distance information
    req.contacts = false; // Don't need detailed contacts for now
    req.cost = false;     // Not using cost sources
    req.verbose = false;
    req.max_contacts = 1; // Only need to know if there is at least one collision/close contact

    // 4. Perform Collision Check
    collision_detection::CollisionResult res;
    // Use the PlanningScene's checkCollision method. This checks against
    // the world geometry AND performs self-collision checks based on the ACM.
    // Since we are initially focusing on world collisions, we will interpret
    // the result, prioritizing world collisions if details were available.
    // For now, we use the overall result (collision status and min distance).
    planning_scene_ptr->checkCollision(req, res, temp_state);

    // 5. Calculate Cost
    double cost = 0.0;
    if (res.collision) {
        // Robot is in collision (could be self or world). Assign a high penalty.
        // We could potentially use res.penetration_depth if needed, but a fixed
        // large penalty is simpler to start.
        cost = penalty_scale_ * 10.0; // Make collision cost significantly higher
        ROS_DEBUG_NAMED("bio_ik.CollisionAvoidanceGoal", "Collision detected! Assigning high cost: %.4f", cost);

        // To check *if* it was a world collision (more complex):
        // Iterate through res.contacts to see if any involve the CollisionWorld object.
        // collision_detection::ContactMap::const_iterator it;
        // bool world_collision = false;
        // for (it = res.contacts.begin(); it != res.contacts.end(); ++it) {
        //    // Check if contact involves the world object name or specific obstacle names
        // }
        // if (world_collision) cost = ... ; else cost = ... (for self)

    } else if (res.distance < distance_threshold_) {
        // Robot is close to collision (below threshold). Calculate penalty.
        // Cost increases linearly as distance decreases.
        cost = (distance_threshold_ - res.distance) * penalty_scale_;
        ROS_DEBUG_NAMED("bio_ik.CollisionAvoidanceGoal", "Close to collision (Dist: %.4f < Threshold: %.4f). Cost: %.4f", res.distance, distance_threshold_, cost);
    }
    // Else: res.distance >= distance_threshold_, cost remains 0.0

    // Apply the goal's weight
    return cost * weight_; // Apply the specific weight for this goal instance
}

} // namespace bio_ik