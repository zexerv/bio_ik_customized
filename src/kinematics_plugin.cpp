// No license block or /*** comments as requested */

#include <bio_ik/goal.h> // Includes KinematicsBase definition via nested include

#include "forward_kinematics.h" // BioIK internal
#include "ik_base.h"            // BioIK internal
#include "ik_parallel.h"        // BioIK internal
#include "problem.h"            // BioIK internal
#include "utils.h"              // BioIK internal (check if needed, defines LOG?)

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <moveit/rdf_loader/rdf_loader.h> // For loading RobotModel
#include <pluginlib/class_list_macros.hpp> // For plugin export
#include <srdfdom/model.h> // For SRDF parsing (indirectly via rdf_loader)
#include <urdf/model.h>    // For URDF parsing (indirectly via rdf_loader)

#include <tf2_eigen/tf2_eigen.h>        // Use this instead of eigen_conversions
#include <moveit/kinematics_base/kinematics_base.h> // Already included via goal.h?
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>

#include <atomic> // Check if needed
#include <mutex>
#include <random>
#include <memory> // For std::unique_ptr, std::make_unique
#include <cstring> // For std::memcpy
#include <tuple> // Check if needed
#include <type_traits> // Check if needed

// Include goal types
#include <bio_ik/goal_types.h>
#include <bio_ik/manipulability_goal.h>
#include <bio_ik/collision_avoidance_goal.h> // Include the new goal header

// Include MoveIt/ROS extras
#include <moveit/robot_state/robot_state.h> // Already included?
#include <ros/console.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h> // Needed for tf2::fromMsg

using namespace bio_ik;

// implement BioIKKinematicsQueryOptions
namespace bio_ik {

std::mutex bioIKKinematicsQueryOptionsMutex;
std::unordered_set<const void *> bioIKKinematicsQueryOptionsList;

BioIKKinematicsQueryOptions::BioIKKinematicsQueryOptions()
    : replace(false), solution_fitness(0) {
  std::lock_guard<std::mutex> lock(bioIKKinematicsQueryOptionsMutex);
  bioIKKinematicsQueryOptionsList.insert(this);
}

BioIKKinematicsQueryOptions::~BioIKKinematicsQueryOptions() {
  std::lock_guard<std::mutex> lock(bioIKKinematicsQueryOptionsMutex);
  if (bioIKKinematicsQueryOptionsList.count(this)) {
      bioIKKinematicsQueryOptionsList.erase(this);
  }
}

bool isBioIKKinematicsQueryOptions(const void *ptr) {
  std::lock_guard<std::mutex> lock(bioIKKinematicsQueryOptionsMutex);
  return bioIKKinematicsQueryOptionsList.count(ptr);
}

const BioIKKinematicsQueryOptions *
toBioIKKinematicsQueryOptions(const void *ptr) {
  if (isBioIKKinematicsQueryOptions(ptr))
    return static_cast<const BioIKKinematicsQueryOptions *>(ptr);
  else
    return nullptr;
}

} // namespace bio_ik

// BioIK Kinematics Plugin
namespace bio_ik_kinematics_plugin {

// Fallback for older MoveIt versions which don't support lookupParam yet
template <class T>
static void lookupParam(const std::string &param, T &val,
                        const T &default_val) {
  // Use private node handle ~ for parameters associated with this plugin instance
  ros::NodeHandle nodeHandle("~");
  val = nodeHandle.param(param, default_val);
}

struct BioIKKinematicsPlugin : kinematics::KinematicsBase {
  // --- Member Variables ---
  // Use local members as originally designed in BioIK
  std::vector<std::string> joint_names; // Local member (no underscore)
  std::vector<std::string> link_names;  // Local member (no underscore)

  // Base class members are still inherited and used where appropriate
  // e.g., robot_model_, group_name_, robot_description_, base_frame_, tip_frames_

  // Other members
  const moveit::core::JointModelGroup *joint_model_group = nullptr;
  mutable std::unique_ptr<IKParallel> ik;
  mutable std::vector<double> state, temp;
  mutable std::unique_ptr<moveit::core::RobotState> temp_state;
  mutable std::vector<Frame> tipFrames;
  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_ptr_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_ptr_;
  RobotInfo robot_info;
  bool enable_profiler = false;

  mutable std::vector<std::unique_ptr<Goal>> default_goals;
  mutable std::vector<const bio_ik::Goal *> all_goals;
  IKParams ikparams;
  mutable Problem problem;

  // Constructor
  BioIKKinematicsPlugin() = default;

  // --- KinematicsBase Virtual Functions ---
  virtual const std::vector<std::string> &getJointNames() const override {
    return joint_names; // Use local member (no underscore)
  }

  virtual const std::vector<std::string> &getLinkNames() const override {
    return link_names; // Use local member (no underscore)
  }

  // Stubs for deprecated/unused FK/IK methods
  virtual bool getPositionFK(const std::vector<std::string> &link_names,
                             const std::vector<double> &joint_angles,
                             std::vector<geometry_msgs::Pose> &poses) const override {
    ROS_ERROR_NAMED("bio_ik", "BioIK::getPositionFK is not implemented");
    return false;
  }

  virtual bool getPositionIK(const geometry_msgs::Pose &ik_pose,
                             const std::vector<double> &ik_seed_state,
                             std::vector<double> &solution,
                             moveit_msgs::MoveItErrorCodes &error_code,
                             const kinematics::KinematicsQueryOptions &options =
                                 kinematics::KinematicsQueryOptions()) const override {
     ROS_ERROR_NAMED("bio_ik", "BioIK::getPositionIK is not implemented. Use searchPositionIK instead.");
     error_code.val = moveit_msgs::MoveItErrorCodes::NO_IK_SOLUTION;
     return false;
  }

  // Static helper to load RobotModel (cached) - Keep this internal detail
  static moveit::core::RobotModelConstPtr
  loadRobotModelInternal(const std::string &robot_description_param_name) {
    static std::map<std::string, moveit::core::RobotModelConstPtr>
        robot_model_cache;
    static std::mutex cache_mutex;
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (robot_model_cache.find(robot_description_param_name) == robot_model_cache.end()) {
      rdf_loader::RDFLoader rdf_loader(robot_description_param_name);
      auto srdf = rdf_loader.getSRDF();
      auto urdf_model = rdf_loader.getURDF();

      if (!urdf_model || !srdf) {
        ROS_ERROR_NAMED("bio_ik", "URDF and SRDF could not be loaded for kinematics solver from parameter '%s'.", robot_description_param_name.c_str());
        return nullptr;
      }
      robot_model_cache[robot_description_param_name] = moveit::core::RobotModelConstPtr(
          new robot_model::RobotModel(urdf_model, srdf));
      ROS_INFO_NAMED("bio_ik", "Loaded robot model from parameter: %s", robot_description_param_name.c_str());
    }
    return robot_model_cache[robot_description_param_name];
  }

  // Main initialization logic called by initialize() overrides
  // Receives the correct robot description *parameter name* from initialize overrides
  bool load(const moveit::core::RobotModelConstPtr &model,
            const std::string& robot_description_param_name, // Explicitly name this parameter
            const std::string& group_name) {
    ROS_INFO_NAMED("bio_ik", "Loading BioIK plugin for group '%s'", group_name.c_str());

    // --- 1. Load Robot Model (set base class robot_model_) ---
    if (model) {
      robot_model_ = model;
      ROS_INFO_NAMED("bio_ik", "Using provided RobotModel instance.");
    } else {
      if (robot_description_param_name.empty()) {
          ROS_FATAL_NAMED("bio_ik", "Cannot initialize BioIK: No RobotModel instance provided and robot_description parameter name is empty.");
          return false;
      }
      robot_model_ = loadRobotModelInternal(robot_description_param_name);
      if (!robot_model_) {
        ROS_ERROR_NAMED("bio_ik", "Failed to load RobotModel from description parameter '%s'.", robot_description_param_name.c_str());
        return false;
      }
    }
    if (!robot_model_) { // Double check
        ROS_FATAL_NAMED("bio_ik", "RobotModel is null after loading attempt.");
        return false;
    }

    // --- 2. Initialize TF Buffer and Planning Scene Monitor ---
    if (!tf_buffer_ptr_) {
      tf_buffer_ptr_ = std::make_shared<tf2_ros::Buffer>();
      ROS_INFO_NAMED("bio_ik", "Initialized TF Buffer for PlanningSceneMonitor.");
    }
if (robot_description_param_name.empty()) {
            ROS_ERROR_NAMED("bio_ik", "Cannot initialize PlanningSceneMonitor: robot_description parameter name is empty.");
       } else {
            ROS_INFO_NAMED("bio_ik", "[LOAD] Initializing PlanningSceneMonitor using description parameter name '%s'...", robot_description_param_name.c_str());
            try {
                 planning_scene_monitor_ptr_ =
                     std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
                         robot_description_param_name, tf_buffer_ptr_); // Use function argument

                 if (planning_scene_monitor_ptr_ && planning_scene_monitor_ptr_->getPlanningScene()) {
                   // PSM object created, now try starting monitors
                   ROS_INFO_NAMED("bio_ik", "[LOAD] PSM object created. Starting monitors...");

                   ROS_INFO_NAMED("bio_ik", "[LOAD] Starting Scene Monitor...");
                   planning_scene_monitor_ptr_->startSceneMonitor();
                   ROS_INFO_NAMED("bio_ik", "[LOAD] Scene Monitor Start Requested.");

                   ROS_INFO_NAMED("bio_ik", "[LOAD] Starting World Geometry Monitor...");
                   planning_scene_monitor_ptr_->startWorldGeometryMonitor();
                   ROS_INFO_NAMED("bio_ik", "[LOAD] World Geometry Monitor Start Requested.");

                   ROS_INFO_NAMED("bio_ik", "[LOAD] Starting State Monitor...");
                   planning_scene_monitor_ptr_->startStateMonitor();
                   ROS_INFO_NAMED("bio_ik", "[LOAD] State Monitor Start Requested.");

                   // ros::Duration(0.5).sleep(); // <<< Temporarily comment out sleep

                   // Check if monitors seem active (basic check, might not be fully reliable)
                   bool monitors_ok = planning_scene_monitor_ptr_->getStateMonitor() && planning_scene_monitor_ptr_->getStateMonitor()->isActive();
                   if(monitors_ok) {
                       ROS_INFO_NAMED("bio_ik", "[LOAD] PlanningSceneMonitor initialization appears successful.");
                   } else {
                       ROS_WARN_NAMED("bio_ik", "[LOAD] PlanningSceneMonitor monitors did not seem to start correctly immediately after request.");
                       // Continue anyway for now, maybe they connect later?
                   }

                 } else {
                   ROS_ERROR_NAMED("bio_ik", "[LOAD] PlanningSceneMonitor failed to get PlanningScene after construction.");
                   planning_scene_monitor_ptr_.reset(); // Reset if failed
                 }
            } catch (const std::exception& e) {
                ROS_ERROR("[LOAD] Exception caught during PlanningSceneMonitor setup using '%s': %s", robot_description_param_name.c_str(), e.what());
                planning_scene_monitor_ptr_.reset();
            }
        }
    } // End PSM init block

    // ... (Rest of the load function: JMG setup, joint/link names, params, goals etc.) ...

    ROS_INFO_NAMED("bio_ik", "[LOAD] Reached end of BioIK load function for group '%s'.", group_name_.c_str()); // Add log at the end
    return true; // Ensure it returns true if no fatal error occurred



if (robot_description_param_name.empty()) {
            ROS_ERROR_NAMED("bio_ik", "Cannot initialize PlanningSceneMonitor: robot_description parameter name is empty.");
       } else {
            ROS_INFO_NAMED("bio_ik", "[LOAD] Initializing PlanningSceneMonitor using description parameter name '%s'...", robot_description_param_name.c_str());
            try {
                 planning_scene_monitor_ptr_ =
                     std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
                         robot_description_param_name, tf_buffer_ptr_); // Use function argument

                 if (planning_scene_monitor_ptr_ && planning_scene_monitor_ptr_->getPlanningScene()) {
                   // PSM object created, now try starting monitors
                   ROS_INFO_NAMED("bio_ik", "[LOAD] PSM object created. Starting monitors...");

                   ROS_INFO_NAMED("bio_ik", "[LOAD] Starting Scene Monitor...");
                   planning_scene_monitor_ptr_->startSceneMonitor();
                   ROS_INFO_NAMED("bio_ik", "[LOAD] Scene Monitor Start Requested.");

                   ROS_INFO_NAMED("bio_ik", "[LOAD] Starting World Geometry Monitor...");
                   planning_scene_monitor_ptr_->startWorldGeometryMonitor();
                   ROS_INFO_NAMED("bio_ik", "[LOAD] World Geometry Monitor Start Requested.");

                   ROS_INFO_NAMED("bio_ik", "[LOAD] Starting State Monitor...");
                   planning_scene_monitor_ptr_->startStateMonitor();
                   ROS_INFO_NAMED("bio_ik", "[LOAD] State Monitor Start Requested.");

                   // ros::Duration(0.5).sleep(); // <<< Temporarily comment out sleep

                   // Check if monitors seem active (basic check, might not be fully reliable)
                   bool monitors_ok = planning_scene_monitor_ptr_->getStateMonitor() && planning_scene_monitor_ptr_->getStateMonitor()->isActive();
                   if(monitors_ok) {
                       ROS_INFO_NAMED("bio_ik", "[LOAD] PlanningSceneMonitor initialization appears successful.");
                   } else {
                       ROS_WARN_NAMED("bio_ik", "[LOAD] PlanningSceneMonitor monitors did not seem to start correctly immediately after request.");
                       // Continue anyway for now, maybe they connect later?
                   }

                 } else {
                   ROS_ERROR_NAMED("bio_ik", "[LOAD] PlanningSceneMonitor failed to get PlanningScene after construction.");
                   planning_scene_monitor_ptr_.reset(); // Reset if failed
                 }
            } catch (const std::exception& e) {
                ROS_ERROR("[LOAD] Exception caught during PlanningSceneMonitor setup using '%s': %s", robot_description_param_name.c_str(), e.what());
                planning_scene_monitor_ptr_.reset();
            }
        }
    } 




    // if (!planning_scene_monitor_ptr_) {
    //    if (robot_description_param_name.empty()) {
    //         ROS_ERROR_NAMED("bio_ik", "Cannot initialize PlanningSceneMonitor: robot_description parameter name is empty.");
    //    } else {
    //         ROS_INFO_NAMED("bio_ik", "Initializing PlanningSceneMonitor using description parameter name '%s'...", robot_description_param_name.c_str());
    //         try {
    //              planning_scene_monitor_ptr_ =
    //                  std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
    //                      robot_description_param_name, tf_buffer_ptr_); // Use function argument

    //              if (planning_scene_monitor_ptr_->getPlanningScene()) {
    //                ROS_INFO_NAMED("bio_ik", "PlanningSceneMonitor: Starting scene monitor...");
    //                planning_scene_monitor_ptr_->startSceneMonitor();
    //                ROS_INFO_NAMED("bio_ik", "PlanningSceneMonitor: Starting world geometry monitor...");
    //                planning_scene_monitor_ptr_->startWorldGeometryMonitor();
    //                ROS_INFO_NAMED("bio_ik", "PlanningSceneMonitor: Starting state monitor...");
    //                planning_scene_monitor_ptr_->startStateMonitor();
    //                ros::Duration(0.5).sleep();
    //                ROS_INFO_NAMED("bio_ik", "PlanningSceneMonitor started successfully.");
    //              } else {
    //                ROS_ERROR_NAMED("bio_ik", "PlanningSceneMonitor failed to initialize PlanningScene after construction.");
    //                planning_scene_monitor_ptr_.reset();
    //              }
    //         } catch (const std::exception& e) {
    //             ROS_ERROR("Exception caught during PlanningSceneMonitor setup using '%s': %s", robot_description_param_name.c_str(), e.what());
    //             planning_scene_monitor_ptr_.reset();
    //         }
    //     }
    // }

    // --- 3. Get Joint Model Group ---
    joint_model_group = robot_model_->getJointModelGroup(group_name);
    if (!joint_model_group) {
      ROS_ERROR_NAMED("bio_ik", "Failed to find Joint Model Group '%s' in the RobotModel.", group_name.c_str());
      return false;
    }

    // --- 4. Setup Joint and Link Names (Populate LOCAL members) --- <<< CORRECTED BLOCK
    // Populate the local joint_names member declared in this struct
    joint_names.clear(); // Use local member (no underscore)
    for (const auto *joint_model : joint_model_group->getActiveJointModels()) {
        joint_names.push_back(joint_model->getName()); // Use local member (no underscore)
    }
    // Populate the local link_names member using base class tip_frames_
    // (tip_frames_ should be populated by setValues called from initialize)
    link_names = tip_frames_; // Use local member (no underscore) = base class member

    ROS_INFO_NAMED("bio_ik", "Plugin configured with %zu joints and %zu tip links (link_names).", joint_names.size(), link_names.size());


    // --- 5. Initialize IK Parameters & Robot Info ---
    robot_info = RobotInfo(robot_model_);
    ikparams.robot_model = robot_model_;
    ikparams.joint_model_group = joint_model_group;

    lookupParam("mode", ikparams.solver_class_name, std::string("bio2_memetic"));
    lookupParam("enable_counter", ikparams.enable_counter, false);
    lookupParam("num_threads", ikparams.thread_count, 0);
    lookupParam("random_seed", ikparams.random_seed, static_cast<int>(std::random_device()()));
    lookupParam("dpos", ikparams.dpos, DBL_MAX);
    lookupParam("drot", ikparams.drot, DBL_MAX);
    lookupParam("dtwist", ikparams.dtwist, 1e-5);
    lookupParam("evolution_no_wipeout", ikparams.opt_no_wipeout, false);
    lookupParam("evolution_population_size", ikparams.population_size, 8);
    lookupParam("evolution_elite_count", ikparams.elite_count, 4);
    lookupParam("evolution_linear_fitness", ikparams.linear_fitness, false);
    lookupParam("profiler", enable_profiler, false);

    // --- 6. Initialize Solver Backend ---
    temp_state.reset(new moveit::core::RobotState(robot_model_));
    ik.reset(new IKParallel(ikparams));

    // --- 7. Setup Default Goals ---
    { // Scope for goal setup
      default_goals.clear();

      // 7.1 Default Pose Goals for Tip Frames
      double rotation_scale = 0.5;
      lookupParam("goal_rotation_scale", rotation_scale, rotation_scale);
      bool position_only_ik_local = false; // Use local variable (Fix #2)
      lookupParam("position_only_ik", position_only_ik_local, position_only_ik_local);
      if (position_only_ik_local) { // Check local variable (Fix #2)
          rotation_scale = 0.0;
          ROS_INFO_NAMED("bio_ik", "Position Only IK enabled via parameter. Setting PoseGoal rotation scale to 0.");
      }

      if (tip_frames_.empty()) {
          ROS_WARN_NAMED("bio_ik", "No tip frames defined for group '%s'. Default Pose Goals cannot be added.", group_name.c_str());
      }
      // Use local link_names which was populated from tip_frames_
      for (size_t i = 0; i < link_names.size(); i++) {
        auto goal = std::make_unique<bio_ik::PoseGoal>();
        goal->setLinkName(link_names[i]); // Use local member
        goal->setRotationScale(rotation_scale);
        default_goals.emplace_back(std::move(goal));
        ROS_INFO_NAMED("bio_ik", "Added default PoseGoal for tip frame '%s' with rotation scale %.2f", link_names[i].c_str(), rotation_scale);
      }

      // 7.2 Center Joints Goal
      {
        double weight = 0.0;
        lookupParam("center_joints_weight", weight, weight);
        if (weight > 0.0) {
          auto goal = std::make_unique<bio_ik::CenterJointsGoal>(weight, true);
          default_goals.emplace_back(std::move(goal));
          ROS_INFO_NAMED("bio_ik", "Added default CenterJointsGoal with weight %.3f", weight);
        }
      }

      // 7.3 Avoid Joint Limits Goal
      {
        double weight = 0.0;
        lookupParam("avoid_joint_limits_weight", weight, weight);
        if (weight > 0.0) {
          auto goal = std::make_unique<bio_ik::AvoidJointLimitsGoal>(weight, true);
          default_goals.emplace_back(std::move(goal));
          ROS_INFO_NAMED("bio_ik", "Added default AvoidJointLimitsGoal with weight %.3f", weight);
        }
      }

      // 7.4 Minimal Displacement Goal
      {
        double weight = 0.0;
        lookupParam("minimal_displacement_weight", weight, weight);
        if (weight > 0.0) {
          auto goal = std::make_unique<bio_ik::MinimalDisplacementGoal>(weight, true);
          default_goals.emplace_back(std::move(goal));
          ROS_INFO_NAMED("bio_ik", "Added default MinimalDisplacementGoal with weight %.3f", weight);
        }
      }

      // 7.5 Manipulability Goal
      {
        double weight = 0.0;
        double epsilon = 1e-6;
        lookupParam("manipulability_weight", weight, weight);
        lookupParam("manipulability_epsilon", epsilon, epsilon);
        if (weight > 0.0) {
          auto goal = std::make_unique<bio_ik::ManipulabilityGoal>(weight, epsilon, true);
          default_goals.emplace_back(std::move(goal));
          ROS_INFO_NAMED("bio_ik", "Added default ManipulabilityGoal with weight %.3f, epsilon %.3e", weight, epsilon);
        }
      }

      // 7.6 Collision Avoidance Goal (NEW)
      {
          double weight = 0.0;
          double distance_threshold = 0.05;
          double penalty_scale = 10.0;
          bool secondary = true;

          lookupParam("collision_avoidance_weight", weight, weight);
          lookupParam("collision_distance_threshold", distance_threshold, distance_threshold);
          lookupParam("collision_penalty_scale", penalty_scale, penalty_scale);
          // lookupParam("collision_is_secondary", secondary, secondary); // Optional

          if (weight > 0.0) {
              if (!planning_scene_monitor_ptr_) {
                   ROS_WARN_NAMED("bio_ik", "CollisionAvoidanceGoal requested (weight=%.3f) but PlanningSceneMonitor is not available. Goal NOT added.", weight);
              } else {
                  auto goal = std::make_unique<bio_ik::CollisionAvoidanceGoal>(
                      weight, distance_threshold, penalty_scale, secondary);
                  ROS_INFO_NAMED("bio_ik", "Added default CollisionAvoidanceGoal (Weight=%.3f, Threshold=%.3fm, Scale=%.2f)",
                                  weight, distance_threshold, penalty_scale);
                  default_goals.emplace_back(std::move(goal));
              }
          }
      }

    } // End scope for default goal setup

    ROS_INFO_NAMED("bio_ik", "BioIK plugin initialization complete for group '%s'.", group_name_.c_str());
    return true;
  } // End load function


  // --- Standard KinematicsBase initialize() overrides ---
  virtual bool initialize(const std::string &robot_description,
                          const std::string &group_name,
                          const std::string &base_frame,
                          const std::string &tip_frame,
                          double search_discretization) override {
    setValues(robot_description, group_name, base_frame, {tip_frame}, search_discretization);
    return load(moveit::core::RobotModelConstPtr(), robot_description_, group_name_);
  }

  virtual bool initialize(const std::string &robot_description,
                          const std::string &group_name,
                          const std::string &base_frame,
                          const std::vector<std::string> &tip_frames,
                          double search_discretization) override {
    setValues(robot_description, group_name, base_frame, tip_frames, search_discretization);
    return load(moveit::core::RobotModelConstPtr(), robot_description_, group_name_);
  }

  virtual bool initialize(const moveit::core::RobotModel &robot_model_ref,
                          const std::string &group_name,
                          const std::string &base_frame,
                          const std::vector<std::string> &tip_frames,
                          double search_discretization) override {
    // Create a non-owning ConstPtr to pass to load IF needed (setValues sets base robot_model_)
    moveit::core::RobotModelConstPtr model_ptr(&robot_model_ref, [](const moveit::core::RobotModel *ptr) {});
    std::string standard_description_param_name = "/robot_description";
    setValues(standard_description_param_name, group_name, base_frame, tip_frames, search_discretization);
    // Pass the model_ptr AND the standard param name explicitly to load
    return load(model_ptr, standard_description_param_name, group_name_);
  }


  // --- IK Search Methods ---
  // Override variants just call the main searchPositionIK below
  virtual bool
  searchPositionIK(const geometry_msgs::Pose &ik_pose,
                   const std::vector<double> &ik_seed_state, double timeout,
                   std::vector<double> &solution,
                   moveit_msgs::MoveItErrorCodes &error_code,
                   const kinematics::KinematicsQueryOptions &options =
                       kinematics::KinematicsQueryOptions()) const override {
    return searchPositionIK(std::vector<geometry_msgs::Pose>{ik_pose},
                            ik_seed_state, timeout, std::vector<double>(),
                            solution, IKCallbackFn(), error_code, options);
  }

  virtual bool
  searchPositionIK(const geometry_msgs::Pose &ik_pose,
                   const std::vector<double> &ik_seed_state, double timeout,
                   const std::vector<double> &consistency_limits,
                   std::vector<double> &solution,
                   moveit_msgs::MoveItErrorCodes &error_code,
                   const kinematics::KinematicsQueryOptions &options =
                       kinematics::KinematicsQueryOptions()) const override {
    return searchPositionIK(std::vector<geometry_msgs::Pose>{ik_pose},
                            ik_seed_state, timeout, consistency_limits,
                            solution, IKCallbackFn(), error_code, options);
  }

  virtual bool
  searchPositionIK(const geometry_msgs::Pose &ik_pose,
                   const std::vector<double> &ik_seed_state, double timeout,
                   std::vector<double> &solution,
                   const IKCallbackFn &solution_callback,
                   moveit_msgs::MoveItErrorCodes &error_code,
                   const kinematics::KinematicsQueryOptions &options =
                       kinematics::KinematicsQueryOptions()) const override {
    return searchPositionIK(std::vector<geometry_msgs::Pose>{ik_pose},
                            ik_seed_state, timeout, std::vector<double>(),
                            solution, solution_callback, error_code, options);
  }

  virtual bool
  searchPositionIK(const geometry_msgs::Pose &ik_pose,
                   const std::vector<double> &ik_seed_state, double timeout,
                   const std::vector<double> &consistency_limits,
                   std::vector<double> &solution,
                   const IKCallbackFn &solution_callback,
                   moveit_msgs::MoveItErrorCodes &error_code,
                   const kinematics::KinematicsQueryOptions &options =
                       kinematics::KinematicsQueryOptions()) const override {
    return searchPositionIK(std::vector<geometry_msgs::Pose>{ik_pose},
                            ik_seed_state, timeout, consistency_limits,
                            solution, solution_callback, error_code, options);
  }

  // Main IK Solver Implementation
  virtual bool
  searchPositionIK(const std::vector<geometry_msgs::Pose> &ik_poses,
                   const std::vector<double> &ik_seed_state, double timeout,
                   const std::vector<double> &consistency_limits, // Note: BioIK doesn't directly use consistency_limits
                   std::vector<double> &solution,
                   const IKCallbackFn &solution_callback,
                   moveit_msgs::MoveItErrorCodes &error_code,
                   const kinematics::KinematicsQueryOptions &options =
                       kinematics::KinematicsQueryOptions(),
                   const moveit::core::RobotState *context_state = NULL) const override {

    double t0 = ros::WallTime::now().toSec();

    // Check if initialized (using robot_model_ as proxy)
    if (!robot_model_ || !joint_model_group) {
        ROS_ERROR_NAMED("bio_ik", "Kinematics solver not initialized (model or group missing). Call initialize first.");
        error_code.val = moveit_msgs::MoveItErrorCodes::FAILURE;
        return false;
    }

    // Use local joint_names member for checks
    if (ik_poses.empty() || ik_poses.size() != link_names.size()) { // Check against local link_names size
        ROS_ERROR_NAMED("bio_ik", "Number of poses (%zu) does not match number of tip frames/links (%zu)", ik_poses.size(), link_names.size());
        error_code.val = moveit_msgs::MoveItErrorCodes::FAILURE;
        return false;
    }
    if (ik_seed_state.size() != joint_names.size()) { // Check against local joint_names size
        ROS_ERROR_NAMED("bio_ik", "Seed state size (%zu) does not match number of joints in group (%zu)", ik_seed_state.size(), joint_names.size());
        error_code.val = moveit_msgs::MoveItErrorCodes::FAILURE;
        return false;
    }

    // --- Prepare IK Problem ---
    auto *bio_ik_options = toBioIKKinematicsQueryOptions(&options);

    // Initialize full robot state vector
    state.resize(robot_model_->getVariableCount());
    if (context_state) {
      std::memcpy(state.data(), context_state->getVariablePositions(), robot_model_->getVariableCount() * sizeof(double));
    } else {
      robot_model_->getVariableDefaultPositions(state);
    }

    // Overwrite state with seed for the active group joints
    size_t seed_idx = 0;
    for (const std::string &joint_name : joint_names) { // Use local member
        const moveit::core::JointModel* jm = robot_model_->getJointModel(joint_name);
        if (!jm) {
             ROS_ERROR_NAMED("bio_ik", "Could not find joint model for joint '%s'", joint_name.c_str());
             error_code.val = moveit_msgs::MoveItErrorCodes::PLANNING_FAILED;
             return false;
        }
        // TODO: Check if variable count matches seed state partitioning? BioIK assumes seed matches joint_names order.
        for (size_t vi = 0; vi < jm->getVariableCount(); ++vi) {
             if (seed_idx < ik_seed_state.size()) {
                  state.at(jm->getFirstVariableIndex() + vi) = ik_seed_state[seed_idx++];
             } else {
                  ROS_ERROR_NAMED("bio_ik", "Internal error: Seed state size mismatch during variable assignment.");
                  error_code.val = moveit_msgs::MoveItErrorCodes::FAILURE;
                  return false;
             }
        }
    }
    // Enforce bounds on the initial seed state for the group
    moveit::core::RobotState seed_state(robot_model_);
    seed_state.setVariablePositions(state);
    seed_state.enforceBounds(joint_model_group);
    std::memcpy(state.data(), seed_state.getVariablePositions(), robot_model_->getVariableCount() * sizeof(double));

    problem.initial_guess = state;

    // Transform goal poses to model frame
    tipFrames.clear();
    if (!bio_ik_options || !bio_ik_options->replace) {
        for (size_t i = 0; i < ik_poses.size(); i++) {
            Eigen::Isometry3d pose_eigen;
            tf2::fromMsg(ik_poses[i], pose_eigen);
            tipFrames.emplace_back(pose_eigen);
        }

        // Update the target poses in the default PoseGoal objects
        size_t pose_goal_idx = 0;
        for (const auto& goal_ptr : default_goals) {
            if (auto* pose_goal = dynamic_cast<bio_ik::PoseGoal*>(goal_ptr.get())) {
                if (pose_goal_idx < tipFrames.size()) {
                    pose_goal->setPosition(tipFrames[pose_goal_idx].getPosition());
                    pose_goal->setOrientation(tipFrames[pose_goal_idx].getOrientation());
                    pose_goal_idx++;
                }
            }
        }
    }

    // Collect all goals
    all_goals.clear();
    if (!bio_ik_options || !bio_ik_options->replace) {
      for (const auto &goal_ptr : default_goals) {
        all_goals.push_back(goal_ptr.get());
      }
    }
    if (bio_ik_options) {
      for (const auto &goal_ptr : bio_ik_options->goals) {
        all_goals.push_back(goal_ptr.get());
      }
    }

    // --- Initialize the IK Problem Solver ---
    // CRITICAL SAFETY NOTE REITERATED: Passing nullptr for the planning scene here.
    // The collision goal evaluate() method CANNOT safely use a pointer passed
    // from here because the lock is released. This MUST be refactored to pass
    // the planning_scene_monitor_ptr_ down and acquire a lock within evaluate().
    {
        problem.timeout = t0 + timeout;
        // TODO: Refactor to pass planning_scene_monitor_ptr_
        problem.initialize(robot_model_, joint_model_group,
                           ikparams, all_goals, bio_ik_options,
                           nullptr); // Passing nullptr until refactored
    }

    {
        ik->initialize(problem);
    }

    // --- Run the IK Solver ---
    {
        ik->solve();
    }

    // --- Process Solution ---
    state = ik->getSolution();

    // Post-process solution
    moveit::core::RobotState result_state(robot_model_);
    result_state.setVariablePositions(state);
    result_state.enforceBounds(joint_model_group);
    result_state.update(); // Update FK for the result state

    // Copy final solution for the group joints into the output vector
    solution.clear();
    solution.resize(joint_names.size()); // Use local member
    result_state.copyJointGroupPositions(joint_model_group, solution);

    // Check if a valid solution was found
    bool success = ik->getSuccess();

    if (bio_ik_options) {
      bio_ik_options->solution_fitness = ik->getSolutionFitness();
    }

    // --- Handle Results ---
    if (!success && !options.return_approximate_solution) {
      ROS_DEBUG_NAMED("bio_ik", "IK solution not found (requested exact solution).");
      error_code.val = moveit_msgs::MoveItErrorCodes::NO_IK_SOLUTION;
      return false;
    }

    if (success) {
        error_code.val = moveit_msgs::MoveItErrorCodes::SUCCESS;
        ROS_DEBUG_NAMED("bio_ik", "IK solution found.");
    } else {
        // Approximate solution is acceptable
        error_code.val = moveit_msgs::MoveItErrorCodes::SUCCESS; // Report success even if approximate
        ROS_DEBUG_NAMED("bio_ik", "IK approximate solution found and accepted.");
    }

    // Execute solution callback if provided
    if (!solution_callback.empty()) {
      // Provide the solved pose(s) if the callback needs them? Requires FK on result_state
      // geometry_msgs::Pose solved_pose; // Example for single tip
      // tf2::toMsg(result_state.getGlobalLinkTransform(link_names.front()), solved_pose); // Requires #include <tf2_geometry_msgs/tf2_geometry_msgs.h>
      solution_callback(ik_poses.front(), solution, error_code); // Pass original target pose for now
      return error_code.val == moveit_msgs::MoveItErrorCodes::SUCCESS;
    } else {
      return true; // Return true if exact OR approximate solution is acceptable
    }
  } // End searchPositionIK


  virtual bool supportsGroup(const moveit::core::JointModelGroup *jmg,
                             std::string *error_text_out = 0) const override {
    if (!jmg) return false;
    // Could check if jmg->getName() == group_name_ if desired
    return true;
  }

}; // End struct BioIKKinematicsPlugin
} // namespace bio_ik_kinematics_plugin

// Register plugin
#undef LOG // Undefine local macros if defined in utils.h
#undef ERROR
PLUGINLIB_EXPORT_CLASS(bio_ik_kinematics_plugin::BioIKKinematicsPlugin,
                       kinematics::KinematicsBase);