import os
import numpy as np
from gym.envs.robotics.robot_env import RobotEnv
from gym.envs.robotics import utils
from gym.envs.robotics.fetch_env import goal_distance


PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])

def vec_to_qpos(vec):
    return {
            'r_shoulder_pan_joint': vec[0],
            'r_shoulder_lift_joint': vec[1],
            'r_upper_arm_roll_joint': vec[2],
            'r_elbow_flex_joint': vec[3],
            'r_forearm_roll_joint': vec[4],
            'r_wrist_flex_joint': vec[5],
            'r_wrist_roll_joint': vec[6],
        }


class PegInsertionEnv(RobotEnv):
    def __init__(self):
        initial_qpos = vec_to_qpos([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0])
        RobotEnv.__init__(
            self,
            model_path=os.path.join(os.path.dirname(__file__), 'assets/pr2_arm3d.xml'),
            n_substeps=5,
            n_actions=7,
            initial_qpos=initial_qpos
        )
        self.distance_threshold = 0.05

        # Find seeds for good initial states
        # results = np.empty(1000)
        # for i in range(results.shape[0]):
        #     self.seed(i)
        #     self._reset_sim()
        #     obs = self._get_obs()
        #     dist = goal_distance(obs['achieved_goal'], obs['desired_goal'])
        #     if obs['achieved_goal'][2] < -0.4:  # Peg below/inside table
        #         dist = 10
        #     results[i] = dist
        # print(list(np.sort(np.argsort(results)[:8])))
        # print(np.sort(results)[:8])
        # exit()

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        return -goal_distance(achieved_goal, goal)

    def _reset_sim(self):
        self._env_setup(vec_to_qpos(self.np_random.uniform(
                # Limited range
                low=np.asarray([-2.0, 0.0, -2.0, -2.0, 1.0, -0.5, 0]), 
                high=np.asarray([1.0, 1.0, -1.0, -1.0, 2.0,  0.0, 0]))))
                # Full range
                #low=np.asarray([-2.2854,  -0.5236, -3.9, -2.3213, -np.pi, -2.094, -np.pi]),
                #high=np.asarray([1.714602, 1.3963,  0.8,  0.0,     np.pi,  0.0,    np.pi]))))
        return True
 
    def _set_action(self, action):
        assert action.shape == (7, )
        action = np.clip(action, -1, +1) / 10
        utils.ctrl_set_action(self.sim, action / PR2_GAINS)

    def _get_obs(self):
        obs = np.concatenate(
            [
                self.sim.data.qpos,
                self.sim.data.qvel,
                self.sim.data.get_site_xpos('leg_bottom'),
                self.sim.data.get_site_xpos('leg_top'),
                #self.sim.data.get_site_xvelp('leg_bottom'),
                #self.sim.data.get_site_xvelp('leg_top'),
                #np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )
        achieved_goal = np.concatenate([
            self.sim.data.get_site_xpos('leg_bottom'),
            self.sim.data.get_site_xpos('leg_top')])
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        lookat = [0.0, 0.0, -0.2]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.75
        self.viewer.cam.azimuth = -90.
        self.viewer.cam.elevation = -30.

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal(self):
        return np.asarray([0.0, 0.3, -0.5, 0.0, 0.3, -0.2])

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
            self.sim.data.set_joint_qvel(name, 0)
        self.sim.forward()
