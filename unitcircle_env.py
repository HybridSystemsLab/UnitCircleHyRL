import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class UnitCircle(gym.Env):
    def __init__(self, steps=150, random_init=True, 
                 state_init=np.array([-1,0], dtype=np.float32), 
                 spread=0.2, backwards=False, hybridlearning=False, 
                 M_ext=None):
        self.steps = steps
        self.random_init = random_init
        self.state_init = state_init
        self.spread = spread
        self.backwards = backwards
        self.hybridlearning = hybridlearning
        self.M_ext = M_ext
        
        self.min_action, self.max_action = -1., 1.
        self.min_x, self.max_x = -1., 1.
        self.min_y, self.max_y = -1., 1.
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
            shape=(2,),
            dtype=np.float32
        )

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def state_to_angle(self):
        angle = np.arctan2(self.state[1], self.state[0])
        # map angle between 0 and 2pi
        angle = np.float32((angle + 2*np.pi) % (2*np.pi))
        return angle
    
    def angle_to_state(self, angle):
        self.state = np.array([np.cos(angle), np.sin(angle)], 
                              dtype=np.float32)
        return self.state
    
    def step(self, action):
        angle = self.state_to_angle()

        force = np.float32(*min(max(action, self.min_action), self.max_action))

        if self.backwards == True:
            sign = -1
        else:
            sign = 1
        
        if self.hybridlearning == True:
            if self.M_ext.in_M_ext(self.state) == False:
                # this is to prevet leaving the extended set during training 
                sign = 0
        
        angle += sign * force * self.t_sampling

        # Check bounds, ensure theta is in [0,2pi)
        if angle < 0:
            angle = angle + 2*np.pi
        if angle >= 2*np.pi:
            angle = angle - 2*np.pi

        # update state
        self.angle_to_state(angle)
        # Update steps left
        self.steps_left -= 1

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
        self.state = self.state_init
        if self.random_init==True:
            # initialize system around the initial state
            angle = np.float32(self.state_to_angle() + \
                    self.spread*np.random.uniform(-1,1))
            self.angle_to_state(angle)
            if self.hybridlearning:
                # ensure that the initial state is in the extended set
                while self.M_ext.in_M_ext(self.state) == False:
                    angle = np.float32(self.state_to_angle() + \
                            self.spread*np.random.uniform(-1,1))
                    self.angle_to_state(angle)
                    
        # set the total number of episode steps
        self.steps_left = self.steps
        return self.state