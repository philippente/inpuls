#include "gps_agent_pkg/jacoarmplugin.h"
#include "gps_agent_pkg/positioncontroller.h"
#include "gps_agent_pkg/trialcontroller.h"
#include "gps_agent_pkg/encodersensor.h"
#include "gps_agent_pkg/util.h"

#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>

#include <urdf/model.h>

namespace gps_control {

JACOArmPlugin::JACOArmPlugin()
{
    controller_counter_ = 0;
    controller_step_length_ = 50;
}

JACOArmPlugin::~JACOArmPlugin()
{
    // Nothing to do here
}

void JACOArmPlugin::initialize_position_controllers(ros::NodeHandle& n)
{
    // Create active arm position controller.
    active_arm_controller_.reset(new PositionController(n, gps::TRIAL_ARM, 6));
}

bool JACOArmPlugin::init(hardware_interface::EffortJointInterface* hw, ros::NodeHandle& n)
{
    int control_frequency;
    if (!n.getParam("control_frequency", control_frequency))
    {
	ROS_DEBUG("No ROS Parameter set for control frequency; assuming 1000Hz");
    }
    else
    {
	// 20Hz is the desired rate of the controller
	controller_step_length_ = control_frequency / 20;
	ROS_DEBUG("Read control_frequency: %d", control_frequency);
	ROS_DEBUG("Set controller_step_length_ to: %d", controller_step_length_);
    }


    std::string active_root_name, active_tip_name;

    if (!n.getParam("active_root_name", active_root_name))
    {
	ROS_ERROR("Property active_root_name not found in namespace: '%s'", n.getNamespace().c_str());
	return false;
    }

    if (!n.getParam("active_tip_name", active_tip_name))
    {
	ROS_ERROR("Property active_tip_name not found in namespace: '%s'", n.getNamespace().c_str());
	return false;
    }


	
    urdf::Model urdf_model;
    if (!urdf_model.initParam("/robot_description"))
    {
	ROS_ERROR("Could not load urdf Model from param server");
	return false;
    }
	
    KDL::Tree kdl_tree;
    if (!kdl_parser::treeFromUrdfModel(urdf_model, kdl_tree))
    {
	ROS_ERROR("Could not convert urdf into kdl tree");
	return false;
    }
	
    bool res;
    try
    {
	res = kdl_tree.getChain(active_root_name, active_tip_name, active_arm_fk_chain_);
    }
    catch(...)
    {
	res = false;
    }
    if (!res)
    {
	ROS_ERROR("Could not extract chain between %s and %s from kdl tree",
		  active_root_name.c_str(), active_tip_name.c_str());
	return false;
    }

    active_arm_fk_solver_.reset(new KDL::ChainFkSolverPos_recursive(active_arm_fk_chain_));
	
    active_arm_jac_solver_.reset(new KDL::ChainJntToJacSolver(active_arm_fk_chain_));
	

    int joint_index;

    joint_index = 1;
    while (true)
    {
	std::string joint_name;
	std::string param_name = std::string("/active_arm_joint_name_" + to_string(joint_index));
	if (!n.getParam(param_name.c_str(), joint_name))
	    break;
			
	hardware_interface::JointHandle* jointHandle = new hardware_interface::JointHandle;
	*jointHandle = hw->getHandle(joint_name);
	active_arm_joint_state_.push_back(jointHandle);
	if (jointHandle == NULL)
	    ROS_INFO_STREAM("jointState: " + joint_name + " is null");

	active_arm_joint_names_.push_back(joint_name);
		
	joint_index++;
    }
	
    if (active_arm_fk_chain_.getNrOfJoints() != active_arm_joint_state_.size())
    {
	ROS_INFO_STREAM("num_fk_chain: " + to_string(active_arm_fk_chain_.getNrOfJoints()));
	ROS_INFO_STREAM("num_joint_state: " + to_string(active_arm_joint_state_.size()));
	ROS_ERROR("Number of joints in th active arm FK chain does not match the number of joints in the active arm joint state!");
	return false;
    }
	

    active_arm_torques_.resize(active_arm_fk_chain_.getNrOfJoints());

    initialize(n);

    return true;
}

void JACOArmPlugin::starting(const ros::Time& time)
{
    last_update_time_ = time;
    controller_counter_ = 0;
	
    for (int sensor = 0; sensor < 1; sensor++)
    {
	sensors_[sensor]->reset(this, last_update_time_);
    }

    active_arm_controller_->reset(last_update_time_);

    if (trial_controller_ != NULL) trial_controller_->reset(last_update_time_);
}

void JACOArmPlugin::stopping(const ros::Time& time)
{
    // Nothing to do here
}

void JACOArmPlugin::update(const ros::Time& time, const ros::Duration& period)
{
    last_update_time_ = time;
	
    controller_counter_++;
    if (controller_counter_ >= controller_step_length_) controller_counter_ = 0;
    bool is_controller_step = (controller_counter_ == 0);
	
    update_sensors(last_update_time_, is_controller_step);
	
    update_controllers(last_update_time_, is_controller_step);

    //ROS_INFO_STREAM("active_arm_torques: " + to_string(active_arm_torques_));

    for(unsigned i = 0; i < active_arm_joint_state_.size(); i++)
	active_arm_joint_state_[i]->setCommand(active_arm_torques_[i]);
}

ros::Time JACOArmPlugin::get_current_time() const
{
    return last_update_time_;
}

void JACOArmPlugin::get_joint_encoder_readings(Eigen::VectorXd &angles, gps::ActuatorType arm) const
{
    if (arm == gps::TRIAL_ARM)
    {
	if (angles.rows() != (int) active_arm_joint_state_.size())
	    angles.resize(active_arm_joint_state_.size());
	for (unsigned i = 0; i < angles.size(); i++)
	    angles(i) = active_arm_joint_state_[i]->getPosition();
    }
    else
    {
	ROS_ERROR("Unknown Armtype %i requested for joint encoder readings!", arm);
    }
}

} // namespace gps_control

PLUGINLIB_EXPORT_CLASS(gps_control::JACOArmPlugin, controller_interface::ControllerBase)
