/*
This is the JACO-specific version of the robot plugin.
*/
#pragma once

#include <Eigen/Dense>
#include <pluginlib/class_list_macros.h>

#include "gps_agent_pkg/armplugin.h"
#include "gps_agent_pkg/controller.h"
#include "gps_agent_pkg/positioncontroller.h"
#include "gps_agent_pkg/encodersensor.h"
#include "robot_state_publisher/robot_state_publisher.h"
#include "gps/proto/gps.pb.h"

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>

#include <hardware_interface/joint_command_interface.h>
#include <controller_interface/controller.h>

namespace gps_control
{

class JACOArmPlugin: public ArmPlugin,
		     public controller_interface::Controller<hardware_interface::EffortJointInterface>
{
private:

    std::vector<hardware_interface::JointHandle*> active_arm_joint_state_;

    std::vector<std::string> active_arm_joint_names_;

    ros::Time last_update_time_;

    int controller_counter_;

    int controller_step_length_;

    //robot_state_publisher::RobotStatePublisher jaco_state_publisher;

public:

    JACOArmPlugin();

    virtual ~JACOArmPlugin();

    virtual void initialize_position_controllers(ros::NodeHandle& n);

    virtual bool init(hardware_interface::EffortJointInterface* hw, ros::NodeHandle &n);

    virtual void starting(const ros::Time& time);

    virtual void stopping(const ros::Time& time);

    virtual void update(const ros::Time& time, const ros::Duration& period);

    virtual ros::Time get_current_time() const;

    virtual void get_joint_encoder_readings(Eigen::VectorXd &angles, gps::ActuatorType arm) const;
};

}
