import numpy as np
import math
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None, initial_pos=None):
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
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 12
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 100.])
        self.init_pose = init_pose[:3] if init_pose is not None else np.array([0., 0., 10.])
        self.distance_to_cover = self.distanceBetweenPoints(self.init_pose, self.target_pos)

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # reward = 1.-.003*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        # labels = ['time', 'pose', 'v', 'angular_v', 'linear_accel', 'angular_accels']
        # print('time', "--", self.sim.time)
        # print('pose', "--", self.sim.pose)
        # print('v', "--", self.sim.v)
        # # print('angular_v', "--", self.angular_v)
        # print('linear_accel', "--", self.sim.linear_accel)
        # print('angular_accels', "--", self.sim.angular_accels)
        # print('target_pos', "--", self.target_pos)
        # print("---------------------------\n\n")
        #print(self.init_pose, self.sim.pose[:3], self.target_pos, self.distance_to_cover)
        #print(self.distanceBetweenPoints(self.initial_pos, self.sim.pose[:3]))
        #print(self.distanceBetweenPoints(self.initial_pos, self.target_pos))
        # if( self.distanceBetweenPoints(self.target_pos, self.sim.pose[:3]) < 0.1*self.distance_to_cover):
        #     reward += 50
        #     print("wwwooooowwwwww")
        # print(self.distanceBetweenPoints(self.target_pos, self.sim.pose[:3]))
        d = self.distanceBetweenPoints(self.target_pos, self.sim.pose[:3])
        reward = self.distance_to_cover/d
        if(d < 0.1):
            reward += 1.6*reward# + reward/np.abs(self.sim.v).sum()
        elif(d < 0.2):
            reward += 0.8*reward# + reward/np.abs(self.sim.v).sum()
        elif(d < 0.3):
            reward += 0.4*reward# + reward/np.abs(self.sim.v).sum()
        else:
            reward += 0.2*reward# + np.abs(self.sim.v).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
            pose_all.append(self.sim.v)
            pose_all.append(self.sim.angular_accels)
            # if done:
            #     reward += 1
        next_state = np.concatenate(pose_all)
        info = {}
        if done:
            info = {"target_pos": self.target_pos, "final_pose": self.sim.pose[:3], "init_pose": self.init_pose }
        return next_state, reward, done, info

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose, self.sim.v, self.sim.angular_accels] * self.action_repeat)
        return state

    def distanceBetweenPoints(self, p1, p2):
        return math.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 +(p1[2] - p2[2])**2 )
