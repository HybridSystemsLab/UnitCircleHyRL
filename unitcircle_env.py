import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math

class UnitCircle(gym.Env):
    def __init__(self, random_init=True, state_init=np.array([-1,0]), 
                 steps=150, backwards=False, hybrid=False, 
                 hybridsim_index=None, Z_i=None):
        self.random_init = random_init
        self.state_init = state_init
        self.steps = steps
        self.backwards = backwards
        self.hybrid = hybrid
        self.hybridsim_index = hybridsim_index
        self.Z_i = Z_i
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_x = -1
        self.max_x = 1
        self.min_y = -1
        self.max_y = 1
        self.t_sampling = 0.05
        
        self.low_state = np.array(
            [self.min_x, self.min_y], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_x, self.max_y], dtype=np.float32
        )

        
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
        
    def step(self, action):
        x = self.state[0]
        y = self.state[1]
        angle = np.arctan2(y, x)
        # map angle between 0 and 2pi
        angle = (angle + 2*np.pi) % (2*np.pi)
        force = min(max(action, self.min_action), self.max_action)
        
        if self.backwards==False:
            angle += force*self.t_sampling
        else:
            angle -= force*self.t_sampling
        
        # Check bounds, ensure theta is in [0,2pi)
        if angle < 0:
            angle = angle + 2*math.pi
        if angle >= 2*math.pi:
            angle = angle - 2*math.pi
        
        # update states
        self.state[0] = np.cos(angle)
        self.state[1] = np.sin(angle)
        # Update steps left
        self.steps_left -= 1
        
        # hybrid learning
        if self.hybrid:
            if self.Z_i.check_in_Z_i(self.hybridsim_index, self.state)==False:
                # undo update on the states to remain in Z_i
                self.state[0] = x
                self.state[1] = y

        # Calculate reward
        reward = -abs(np.arctan2(self.state[1], self.state[0]))/np.pi
        
        # Check if simulation is done
        if self.steps_left <= 0:
            done = True
        else:
            done = False 
        # Set placeholder for info
        info = {}
        return self.state, reward, done, info
    
    def reset(self):
        if self.random_init==True:
            # initialize system around theta = pi
            angle = np.pi + 0.2*np.pi*np.random.uniform(-1,1)
            self.state = np.array([np.cos(angle), np.sin(angle)])
            if self.hybrid:
                inside_Z_i = self.Z_i.check_in_Z_i(self.hybridsim_index, self.state)
                while inside_Z_i == False:
                    angle = np.pi + 0.2*np.pi*np.random.uniform(-1,1)
                    self.state = np.array([np.cos(angle), np.sin(angle)])
                    inside_Z_i = self.Z_i.check_in_Z_i(self.hybridsim_index, self.state)
                    
        else:
            self.state = self.state_init
        # set the total number of episode steps
        self.steps_left = self.steps
        return self.state
