import numpy as np
from physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None, task_type='agent'):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities,
                              init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.task_type = task_type

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([
                                                                             0., 0., 10.])

    def get_reward_basic(self):
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        done = False
        return reward, done

    def get_reward_agent(self):
        """Uses current pose of sim to return reward."""
        # discount for accumulated l1 distance from target
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        # reward for velocities in z (based on target task)
        reward += 20.0 * abs(self.sim.v[2])
        # penalities for x,y directions
        pen_xy = [-0.5 * abs(self.target_pos[d] - self.sim.pose[d])
                  for d in range(2)]
        pen_v = [reward] + pen_xy
        reward = np.dot(pen_v, np.ones_like(pen_v))
        # reward += -0.5 * abs(self.target_pos[0] - self.sim.pose[0])
        # reward += -0.5 * abs(self.target_pos[1] - self.sim.pose[1])

        # set final reward if agent crossed the target
        done = False
        if self.sim.pose[2] >= self.target_pos[2]:
            reward += 40.0
            done = True

        return reward, done

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        return self.get_reward_basic() if self.task_type == 'basic' else self.get_reward_agent()

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            # update get_reward to give more reward if the agent crossed the target
            delta_reward, done_height = self.get_reward()
            reward += delta_reward
            if done_height:
                done = done_height

            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
