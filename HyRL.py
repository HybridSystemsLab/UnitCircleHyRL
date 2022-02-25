import numpy as np
import matplotlib.pyplot as plt
from unitcircle_env import UnitCircle
from stable_baselines3 import PPO
from utils import find_critical_points, angle_to_observation_UC, \
    get_angle_from_env_UC, angle_to_state, find_X_i, \
        train_hybrid_agent, M_i, M_ext, HyRL_agent, simulate_unitcircle, \
            visualize_M_ext

if __name__ == '__main__':
    # Loading in the trained agent
    model = PPO.load("ppo_unitcircle")
    
    # finding the set of critical points
    resolution = 100
    theta = np.linspace(0,2*np.pi, resolution)
    state_difference = theta[1]-theta[0]
    initial_points = []
    for idx in range(resolution):
        initial_points.append(np.array(theta[idx]))
    
    M_star = find_critical_points(initial_points, state_difference, model, 
                                  UnitCircle, min_state_difference=1e-3, 
                                  steps=5, threshold=1e-3, n_clusters=8, 
                                  custom_state_init=angle_to_observation_UC,
                                  get_state_from_env=get_angle_from_env_UC, 
                                  verbose=False)
    M_star = angle_to_state(M_star[0])
    
    # building sets M_0 and M_1
    M_0 = M_i(M_star, index=0)
    M_1 = M_i(M_star, index=1)
    
    # finding the extension sets
    X_0 = find_X_i(M_0, model)
    X_1 = find_X_i(M_1, model)
    M_ext0 = M_ext(M_0, X_0)
    M_ext1 = M_ext(M_1, X_1)
    
    # visualizing the extended sets
    visualize_M_ext(M_ext0, figure_number=1)
    visualize_M_ext(M_ext1, figure_number=2)
    
    # building the environment for hybrid learning
    env_0 = UnitCircle(hybridlearning=True, M_ext=M_ext0)
    env_1 = UnitCircle(hybridlearning=True, M_ext=M_ext1)
    
    # training the new agents
    training2 = False
    if training2:
        agent_0 = train_hybrid_agent(env_0, #load_agent='ppo_unitcircle', 
                                     save_name='ppo_unitcircle_0',
                                     M_exti=M_ext0, timesteps=100000)
        agent_1 = train_hybrid_agent(env_1, #load_agent='ppo_unitcircle', 
                                     save_name='ppo_unitcircle_1',
                                     M_exti=M_ext1, timesteps=100000)
    else:
        agent_0 = PPO.load('ppo_unitcircle_0')
        agent_1 = PPO.load('ppo_unitcircle_1')
    
    # simulation the hybrid agent compared to the original agent    
    angles_init = [0.75*np.pi, 0.9*np.pi, np.pi, 1.1*np.pi, 1.25*np.pi]
    for q in range(2):
        for angle in angles_init:
            hybrid_agent = HyRL_agent(agent_0, agent_1, M_ext0, M_ext1, 
                                      q_init=q)
            state_init = angle_to_state(angle)
            simulate_unitcircle(hybrid_agent, model, state_init,
                                figure_number=3+q)
        save_name = 'HybridSim_q'+str(q)+'_UC.eps'
        plt.savefig(save_name, format='eps')