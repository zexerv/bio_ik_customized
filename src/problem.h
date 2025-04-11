
#pragma once

#include "utils.h"
#include <vector>

#include <bio_ik/robot_info.h>

#include <geometric_shapes/shapes.h>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/collision_detection_fcl/collision_common.h>

#include <bio_ik/goal.h>
#include <moveit/planning_scene/planning_scene.h>
namespace planning_scene { class PlanningScene; } 
namespace bio_ik
{

class Problem
{
private:
    bool ros_params_initrd;
    std::vector<int> joint_usage;
    std::vector<ssize_t> link_tip_indices;
    std::vector<double> minimal_displacement_factors;
    std::vector<double> joint_transmission_goal_temp, joint_transmission_goal_temp2;
    moveit::core::RobotModelConstPtr robot_model;
    const moveit::core::JointModelGroup* joint_model_group;
    IKParams params;
    RobotInfo modelInfo;
    double dpos, drot, dtwist;
    const planning_scene::PlanningScene* planning_scene_ptr_; // Optional: could just pass through
#if (MOVEIT_FCL_VERSION < FCL_VERSION_CHECK(0, 6, 0))
    struct CollisionShape
    {
        std::vector<Vector3> vertices;
        std::vector<fcl::Vec3f> points;
        std::vector<int> polygons;
        std::vector<fcl::Vec3f> plane_normals;
        std::vector<double> plane_dis;
        collision_detection::FCLGeometryConstPtr geometry;
        Frame frame;
        std::vector<std::vector<size_t>> edges;
    };
    struct CollisionLink
    {
        bool initialized;
        std::vector<std::shared_ptr<CollisionShape>> shapes;
        CollisionLink()
            : initialized(false)
        {
        }
    };
    std::vector<CollisionLink> collision_links;
#endif
    size_t addTipLink(const moveit::core::LinkModel* link_model);

public:
    /*enum class GoalType;
    struct BalanceGoalInfo
    {
        ssize_t tip_index;
        double mass;
        Vector3 center;
    };
    struct GoalInfo
    {
        const Goal* goal;
        GoalType goal_type;
        size_t tip_index;
        double weight;
        double weight_sq;
        double rotation_scale;
        double rotation_scale_sq;
        Frame frame;
        tf2::Vector3 target;
        tf2::Vector3 direction;
        tf2::Vector3 axis;
        double distance;
        ssize_t active_variable_index;
        double variable_position;
        std::vector<ssize_t> variable_indices;
        mutable size_t last_collision_vertex;
        std::vector<BalanceGoalInfo> balance_goal_infos;
    };*/
    enum class GoalType;
    // std::vector<const Frame*> temp_frames;
    // std::vector<double> temp_variables;
    struct GoalInfo
    {
        const Goal* goal;
        double weight_sq;
        double weight;
        GoalType goal_type;
        size_t tip_index;
        Frame frame;
        GoalContext goal_context;
    };
    double timeout;
    std::vector<double> initial_guess;
    std::vector<size_t> active_variables;
    std::vector<size_t> tip_link_indices;
    std::vector<GoalInfo> goals;
    std::vector<GoalInfo> secondary_goals;
    Problem();
    void initialize(moveit::core::RobotModelConstPtr robot_model, const moveit::core::JointModelGroup* joint_model_group, const IKParams& params, const std::vector<const Goal*>& goals2, const BioIKKinematicsQueryOptions* options, const planning_scene::PlanningScene* planning_scene_ptr);
    void initialize2();
    double computeGoalFitness(GoalInfo& goal, const Frame* tip_frames, const double* active_variable_positions);
    double computeGoalFitness(std::vector<GoalInfo>& goals, const Frame* tip_frames, const double* active_variable_positions);
    bool checkSolutionActiveVariables(const std::vector<Frame>& tip_frames, const double* active_variable_positions);
};
}
