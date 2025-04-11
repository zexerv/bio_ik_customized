
#pragma once

#include "frame.h"

#include <moveit/kinematics_base/kinematics_base.h>

#include <moveit/robot_model/joint_model_group.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/planning_scene/planning_scene.h>
namespace planning_scene { class PlanningScene; } 
namespace bio_ik
{

class RobotInfo;

class GoalContext
{
protected:
    const double* active_variable_positions_;
    const Frame* tip_link_frames_;
    std::vector<ssize_t> goal_variable_indices_;
    std::vector<size_t> goal_link_indices_;
    bool goal_secondary_;
    std::vector<std::string> goal_link_names_, goal_variable_names_;
    double goal_weight_;
    const moveit::core::JointModelGroup* joint_model_group_;
    std::vector<size_t> problem_active_variables_;
    std::vector<size_t> problem_tip_link_indices_;
    std::vector<double> initial_guess_;
    std::vector<double> velocity_weights_;
    const RobotInfo* robot_info_;
    mutable std::vector<double> temp_vector_;
    const planning_scene::PlanningScene* planning_scene_ptr_; // Added pointer

public:
    // Constructor initializing members (including the new pointer)
    GoalContext()
      : active_variable_positions_(nullptr)
      , tip_link_frames_(nullptr)
      , goal_secondary_(false)
      , goal_weight_(1.0) // Assuming 1.0 is a sensible default
      , joint_model_group_(nullptr)
      , robot_info_(nullptr)
      , planning_scene_ptr_(nullptr) // Initialize added pointer
    {}

    // Getters
    inline const Frame& getLinkFrame(size_t i = 0) const { return tip_link_frames_[goal_link_indices_[i]]; }
    inline const double getVariablePosition(size_t i = 0) const
    {
        auto j = goal_variable_indices_[i];
        if(j >= 0)
            return active_variable_positions_[j];
        else
            return initial_guess_[-1 - j];
    }
    inline const Frame& getProblemLinkFrame(size_t i) const { return tip_link_frames_[i]; }
    inline size_t getProblemLinkCount() const { return problem_tip_link_indices_.size(); }
    inline size_t getProblemLinkIndex(size_t i) const { return problem_tip_link_indices_[i]; }
    inline double getProblemVariablePosition(size_t i) const { return active_variable_positions_[i]; }
    inline size_t getProblemVariableCount() const { return problem_active_variables_.size(); }
    inline size_t getProblemVariableIndex(size_t i) const { return problem_active_variables_[i]; }
    inline double getProblemVariableInitialGuess(size_t i) const { return initial_guess_[problem_active_variables_[i]]; }
    inline double getProblemVariableWeight(size_t i) const { return velocity_weights_[i]; }
    inline const RobotInfo& getRobotInfo() const { return *robot_info_; }
    inline const planning_scene::PlanningScene* getPlanningScene() const { return planning_scene_ptr_; } // Added getter

    // Modifiers/Adders used during setup
    void addLink(const std::string& name) { goal_link_names_.push_back(name); }
    void addVariable(const std::string& name) { goal_variable_names_.push_back(name); }
    void setSecondary(bool secondary) { goal_secondary_ = secondary; }
    void setWeight(double weight) { goal_weight_ = weight; }
    const moveit::core::JointModelGroup& getJointModelGroup() const { return *joint_model_group_; }
    const moveit::core::RobotModel& getRobotModel() const { return joint_model_group_->getParentModel(); }
    std::vector<double>& getTempVector() const { return temp_vector_; }

    friend class Problem; // Allows Problem to set protected members

// Setter for PlanningScene pointer - public or protected
public:
// protected:
    inline void setPlanningScenePtr(const planning_scene::PlanningScene* ps) { planning_scene_ptr_ = ps; } // Added setter
};

class Goal
{
protected:
    bool secondary_;
    double weight_;

public:
    Goal()
        : weight_(1)
        , secondary_(false)
    {
    }
    virtual ~Goal() {}
    bool isSecondary() const { return secondary_; }
    double getWeight() const { return weight_; }
    void setWeight(double w) { weight_ = w; }
    virtual void describe(GoalContext& context) const
    {
        context.setSecondary(secondary_);
        context.setWeight(weight_);
    }
    virtual double evaluate(const GoalContext& context) const { return 0; }
};

struct BioIKKinematicsQueryOptions : kinematics::KinematicsQueryOptions
{
    std::vector<std::unique_ptr<Goal>> goals;
    std::vector<std::string> fixed_joints;
    bool replace;
    mutable double solution_fitness;
    BioIKKinematicsQueryOptions();
    ~BioIKKinematicsQueryOptions();
};
}
