import os
import numpy as np
import torch as th
import matplotlib
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.cluster import KMeans
from unitcircle_env import UnitCircle
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, \
    StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
matplotlib.rcParams['text.usetex'] = True

def find_critical_points(initial_points, state_difference, model, Env,
                         min_state_difference, steps, threshold,
                         n_clusters=8, custom_state_init=None,
                         custom_state_to_observation=None,
                         get_state_from_env=None, verbose=False):
    
    def generate_rod(center_point, dimension, state_difference):
        ndims = center_point.ndim+1
        if ndims > 1:
            displacement = np.zeros(ndims)
            displacement[dimension] = state_difference
        else:
            displacement = state_difference
        rod = [center_point + displacement, center_point - displacement]
        return rod

    def get_rod_length(rod_points):
        return LA.norm(rod_points[0]-rod_points[1])
    # initialize the set of points to consider
    next_points = initial_points
    while state_difference > min_state_difference:
        new_points = []
        if verbose:
            print('number of points: ', len(next_points))
        for center_point in next_points: 
            for dim in range(center_point.ndim+1):
                # along each dimension of the point, do the following:
                    # 1) create a "rod", i.e., consider two points close to the
                    # original point. 
                    # 2) compute the length of the rod before simulation, i.e.,
                    # the distance between the two points
                    # 3) simulate the system for n_steps
                    # 4) compute the distance between the rod after simulation
                    # 5) if the ending length of the rod is greater than the 
                    # starting length, the points diverged from each other.
                    # Then the original center point is potentially a critical
                    # point.
                # creating the rod
                start_points = generate_rod(center_point, dim, state_difference)
                # compute the length of the starting rod
                rod_length_start = get_rod_length(start_points)
                end_points = []
                steps_left_total = 0
                # simulate the system for both points in the rod
                for start in start_points:
                    if custom_state_init is not None:
                        start = custom_state_init(start)
                    env = Env(steps=steps, random_init=False, 
                                            state_init=start)
                    if custom_state_to_observation is None:
                        obs = np.copy(start)
                    else:
                        obs = custom_state_to_observation(np.copy(start))
                    done = False
                    while done == False:
                        action, _ = model.predict(obs)
                        obs, _, done, _ = env.step(action)
                    steps_left = env.steps_left
                    if get_state_from_env is None:
                        end_points.append(env.state)
                    else:
                        end_points.append(get_state_from_env(env))
                    steps_left_total += steps_left
                if steps_left_total == 0:
                    # compute the final length of the rod (after simulation)
                    rod_length_end = get_rod_length(end_points)
                    # if the length of the rod increased, create new points for
                    # the next loop
                    if (rod_length_end-rod_length_start) > threshold:
                        # the new points for the next loop are taken slightly
                        # spaced from the orignal point
                        new_rod = generate_rod(center_point, dim, 
                                               state_difference/4)
                        new_points.extend(new_rod)
        # finally, K-means clustering is used to find n_disconnected sets of
        # critical points
        next_points = new_points
        state_difference = state_difference/2
        cluster_array = np.array(next_points)
        if center_point.ndim+1 == 1:
            cluster_array = cluster_array.reshape(-1,1)
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=0).fit(cluster_array)
        cluster_centers = kmeans.cluster_centers_
        if center_point.ndim+1 == 1:
            cluster_centers = cluster_centers[0]
    return cluster_centers

def angle_to_observation_UC(angle):
    return np.array([np.cos(angle), np.sin(angle)],dtype=np.float32).reshape(2)

def get_angle_from_env_UC(env):
    angle = np.arctan2(env.state[1], env.state[0])
    # map angle between 0 and 2pi
    angle = np.float32((angle + 2*np.pi) % (2*np.pi))  
    return angle

def angle_to_state(angle):
    state = np.array([np.cos(angle), np.sin(angle)],dtype=np.float32)
    return state

def state_to_angle(state):
    angle = np.arctan2(state[1], state[0])
    # map angle between 0 and 2pi
    angle = np.float32((angle + 2*np.pi) % (2*np.pi))
    return angle

def find_X_i(M_i, model, horizon=0.5, n_sims=200, t_sampling=0.05):
    steps = int(horizon/t_sampling)
    if M_i.index == 0:
        sign = -1
    elif M_i.index == 1:
        sign = 1  
    angle_star = state_to_angle(M_i.M_star)
    angles = np.linspace(angle_star+sign*0.1, angle_star, n_sims, 
                         dtype=np.float32).tolist()
    X_i = [M_i.M_star]
    for angle in angles:
        # building the backwards environment
        obs = angle_to_state(angle)
        env_bw = UnitCircle(steps=steps, random_init=False, 
                            state_init = obs, 
                            backwards=True)
        done = False
        while done == False:
            if M_i.in_M(obs) == False:
                X_i.append(obs)
            else:
                action, _ =  model.predict(obs)
            obs, _, done, _ = env_bw.step(action)
    distance = []
    for entry in X_i:
        distance.append(LA.norm(M_i.M_star-entry))
    indx = distance.index(max(distance))
    extension = X_i[indx]
    return extension

def train_hybrid_agent(env, save_name, M_exti, load_agent=None, timesteps=200000):
    # Create log dir
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    
    # wrap the environment
    env = Monitor(env, log_dir)
    
    # Separate evaluation env
    eval_env = UnitCircle(hybridlearning=True, M_ext=M_exti)
    
    if load_agent is not None:
        # loading the existing model
        model = PPO.load(load_agent, env)
    else:
        # building the model
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=[dict(pi=[112, 112], vf=[112,112])])
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
                learning_rate=0.0003450521546798998, gamma= 0.963741783329832)
    
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-28, 
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env, 
                                 callback_on_new_best=callback_on_best, 
                                 verbose=1)
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    model.save(save_name)
    return model

class M_i():
    def __init__(self, M_star, index):
        self.M_star = M_star
        self.index = index
        
    def in_M(self, state):
        if self.index == 0:
            x, y = state[0], state[1]
            if (x <= 0 and y >= self.M_star[1]) or (x > 0 and y >= 0):
                return True
            else:
                return False 
        elif self.index == 1:
            x, y = state[0], state[1]
            if (x <= 0 and y <= self.M_star[1]) or (x > 0 and y < 0):
                return True
            else:
                return False
        else:
            print('Warning! Index out of bounds!')

class M_ext():
    def __init__(self, M_i, X_i):
        self.M_i = M_i
        self.X_i = X_i
        
    def in_M_ext(self, state):
        angle = state_to_angle(state)
        if self.M_i.index == 0:
            if angle <= state_to_angle(self.X_i) or self.M_i.in_M(state):
                return True
            else:
                return False
        if self.M_i.index == 1:
            if angle >= state_to_angle(self.X_i) or self.M_i.in_M(state):
                return True
            else:
                return False 

class HyRL_agent():
    def __init__(self, agent_0, agent_1, M_ext0, M_ext1, q_init=0):
        self.agent_0 = agent_0
        self.agent_1 = agent_1
        self.M_ext0 = M_ext0
        self.M_ext1 = M_ext1
        self.q = q_init
        
    def predict(self, observation):
        switch = -10
        if self.q == 0:
            if self.M_ext0.in_M_ext(observation):
                active_agent = self.agent_0
            else:
                switch = 1
                self.q = 1
                active_agent = self.agent_1
                
        elif self.q == 1:
            if self.M_ext1.in_M_ext(observation):
                active_agent = self.agent_1
            else:
                switch = 1
                self.q = 0
                active_agent = self.agent_0
        action, _ = active_agent.predict(observation)
        return action, switch

def simulate_unitcircle(hybrid_agent, original_agent, state_init, 
                        noise_mag=0.1, figure_number=3, show_switches=False,
                        show_noise_bounds=False):
    env_or = UnitCircle(state_init=state_init, random_init=False, steps=50)
    env_hyb = UnitCircle(state_init=state_init, random_init=False, steps=50)
    done = False
    obs_or, obs_hyb = state_init, state_init
    angles_or = [state_to_angle(obs_or)]
    angles_hyb = [state_to_angle(obs_hyb)]
    _, switch = hybrid_agent.predict(obs_or)
    switches = []
    score_or, score_hyb = 0, 0
    sign = -1
    while done == False:
        disturbance = noise_mag*sign
        angle_or, angle_hyb = state_to_angle(obs_or), state_to_angle(obs_hyb)
        obs_or = angle_to_state(angle_or+disturbance)
        obs_hyb = angle_to_state(angle_hyb+disturbance)
        
        action_or, _ = original_agent.predict(obs_or)
        action_hyb, switch = hybrid_agent.predict(obs_hyb)
        
        obs_or, reward_or, done, _ = env_or.step(action_or)
        obs_hyb, reward_hyb, done, _ = env_hyb.step(action_hyb)
        
        score_or += reward_or
        score_hyb += reward_hyb
        
        angles_or.append(state_to_angle(obs_or))
        angles_hyb.append(state_to_angle(obs_hyb))
        
        switches.append(switch*angle_hyb)
        
        sign *= -1
    plt.figure(figure_number)
    plt.plot(np.unwrap(angles_hyb)/(np.pi), 'blue', linewidth=3)
    plt.plot(np.unwrap(angles_or)/(np.pi), 'red', linestyle='--', linewidth=3)
    if show_switches:
        plt.plot(switches, 'x', markersize=15, color='black', linewidth=2,
                  fillstyle='none')
    if show_noise_bounds:
        ub = [x+noise_mag for x in angles_hyb]
        lb = [x-noise_mag for x in angles_hyb]
        plt.fill_between(np.linspace(0,50,51), np.unwrap(lb), 
                          np.unwrap(ub))
    plt.grid(visible=True)
    plt.ylim(0, 2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('$t$', fontsize=22)
    plt.ylabel(r'$\angle\xi/\pi$', fontsize=22)
    plt.tight_layout()
    print('reward original', score_or, ' reward hybrid', score_hyb)    
    
def visualize_M_ext(M_ext, figure_number, resolution=100):
    plt.figure(figure_number)
    theta = np.linspace(0, 2*np.pi, resolution)
    in_M = []
    for entry in theta:
        obs = np.array((np.cos(entry), np.sin(entry)), dtype=np.float32)
        in_M.append(M_ext.in_M_ext(obs))
    plt.scatter(np.cos(theta), np.sin(theta), s=15, c=in_M)
    cbar = plt.colorbar(ticks=[-1, 0, 1])
    plt.clim(-1,1)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('$u$', rotation=270, fontsize=22, labelpad=22)
    plt.grid()
    plt.xticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.yticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$y$', fontsize=22)