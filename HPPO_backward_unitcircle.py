import numpy as np
from unitcircle_env import UnitCircle
import matplotlib.pyplot as plt 
import torch as T
from PPO_unitcircle import training_PPO, plot_policy_circle
from parula_cm import parula_map
import pickle

# Step 1: train the agent
env_fw = UnitCircle()
agent = training_PPO(env_fw, 'normal', load_checkpoint=True)

# Step 2: Define the critical points
set_M_star = np.array([-0.9968, -0.08])

# Step 3: Define sets M_0 and M_1      
def check_in_M_i(index, set_M_star, observation):
    x = observation[0]
    y = observation[1]
    
    if index == 0:
        if (x >= 0 and y >= 0) or (x < 0 and y >=set_M_star[1]):
            in_M_0 = True
        else:
            in_M_0 = False
        return in_M_0
    
    if index == 1:
        if (x >= 0 and y < 0) or (x < 0 and y <=set_M_star[1]):
            in_M_1 = True
        else:
            in_M_1 = False
        return in_M_1       

# Step 4: Define policies pi_0 and pi_1
def pi_i(agent, index, in_M_0, in_M_1, observation):
    done = False
    action = get_action(agent, observation)
    if index == 0:
        if not in_M_0:
            done = True
    if index == 1:
        if not in_M_1:
            done = True
    return action, done

def get_action(agent, observation):
    action, _, __ = agent.choose_action(observation)
    return action

# Step 5: backwards propagate the system and extend sets M_0 and M_1
n_sims = 50
# perform this operation for both M_0 and M_1
for index in [0,1]:
    set_X_i_x = []
    set_X_i_y = []
    for ii in range(n_sims):
        # initializing backwards simulation
        done = False
        angle = np.arctan2(set_M_star[1], set_M_star[0])
        angle = (angle + 2*np.pi) % (2*np.pi)
        if index == 0:
            angle -= np.random.uniform(0, 0.05)*2*np.pi
            in_M_i = np.array([1, 0])
        if index == 1:
            angle += np.random.uniform(0, 0.05)*2*np.pi
            in_M_i = np.array([0, 1])
        state_init_bw = np.array([np.cos(angle), np.sin(angle)])
        env_bw = UnitCircle(random_init=False, state_init=state_init_bw, 
                backwards=True, steps=10)
        observation = env_bw.reset()
        
        # Running the simulation until outside of M_i
        done_steps = False
        # while in_M_i[index]==1 and done_steps==False:
        while done_steps==False:
            if in_M_i[index]==1:
                action, done = pi_i(agent, index, in_M_i[0], in_M_i[1], observation)
            observation, _, done_steps, _ = env_bw.step(action)
            in_M_i[index] = check_in_M_i(index, set_M_star, observation)

        set_X_i_x.append(observation[0])
        set_X_i_y.append(observation[1])
    # extend sets M_i
    if index == 0:
        # find furthest element from set M_0
        min_y = min(set_X_i_y)
    if index == 1:
        # find furthest element from set M_1
        max_y = max(set_X_i_y)

# Create new sets Z_i
class Class_sets_Z():
    def __init__(self, min_y, max_y):
        self.Z_0_lb = min_y
        self.Z_1_ub = max_y
        
    def check_in_Z_i(self, index, observation):
        x = observation[0]
        y = observation[1]
        
        if index == 0:
            if (x >= 0 and y >= 0) or (x < 0 and y >= self.Z_0_lb):
                return True
            else:
                return False
        
        if index == 1:
            if (x >= 0 and y < 0) or (x < 0 and y <= self.Z_1_ub):
                return True
            else:
                return False

Sets_Z = Class_sets_Z(min_y,max_y)

# Step 6: Run PPO again for the two new sets Z_0 and Z_1 to find two new policies
run_hybrid_PPO = False
if run_hybrid_PPO:
    hybrid_agents = []
    for index in [0,1]:
        env_hb = UnitCircle(hybrid=True, hybridsim_index=index, Z_i=Sets_Z)
        agent_hb = training_PPO(env_hb, 'hybridtest1', load_checkpoint=True, 
                                alpha = 0.01, beta=0.01, min_steps=100)
        hybrid_agents.append(agent_hb)
    # plot_policy_circle(agent_hb)
    
    # saving hybrid agents
    with open('hybridagents_circle', 'wb') as f:
        pickle.dump(hybrid_agents, f)
else:
    # load in hybrid agents
    with open('hybridagents_circle', 'rb') as f:
        hybrid_agents = pickle.load(f)
        
def plot_new_policy_circle(agent, Z_i, index, f):
    plt.figure(f)
    resolution = 500
    theta = np.linspace(0, 2*np.pi, resolution)
    mean = []
    for entry in theta:
        if Z_i.check_in_Z_i(index, np.array([np.cos(entry), np.sin(entry)]))==True:
            mu, _ = agent.actor.forward(T.tensor([np.cos(entry), np.sin(entry)],dtype=T.float32).to(agent.actor.device))
            mu = mu.cpu().detach().numpy()[0].item()
        else:
            mu = np.NaN
        mean.append(mu)
    plt.scatter(np.cos(theta), np.sin(theta), s=15, c=np.tanh(mean), 
                cmap=parula_map)
    cbar = plt.colorbar(ticks=[-1, 0, 1])
    plt.clim(-1,1)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('$u$', rotation=270, fontsize=16, labelpad=16)
    plt.grid(b=True)
    plt.xticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=12)
    plt.yticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=12)
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$y$', fontsize=16)

# Step 7: build the hybrid system and simulate with a disturbance on the state
# environment is same as used for the original training i.e. env_fw

def plot_M_i(set_M_star, f):
    resolution = 500
    theta = np.linspace(0, 2*np.pi, resolution)
    val_0_x = []
    val_0_y = []
    val_1_x = []
    val_1_y = []
    for entry in theta:
        observation = np.array([np.cos(entry), np.sin(entry)])
        if check_in_M_i(0, set_M_star, observation):
            val_0_x.append(observation[0])
            val_0_y.append(observation[1])
        if check_in_M_i(1, set_M_star, observation):
            val_1_x.append(observation[0])
            val_1_y.append(observation[1])
    plt.figure(f)
    plt.plot(np.cos(theta), np.sin(theta), '--', linewidth=1, color='black')
    plt.plot(val_0_x, val_0_y, linewidth=3, color='green')
    plt.plot(set_M_star[0], set_M_star[1], 'x', color='red', markersize=15)
    plt.text(set_M_star[0]+0.08, set_M_star[1]-0.075, r'$\xi_c$', fontsize=22)
    plt.grid(b=True)
    plt.xticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.yticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$y$', fontsize=22)
    plt.tight_layout()
    
    plt.figure(f+1)
    plt.plot(np.cos(theta), np.sin(theta), '--', linewidth=1, color='black')
    plt.plot(val_1_x, val_1_y, linewidth=3, color='blue')
    plt.plot(1, 0, 'o', color='blue', markersize=15, markerfacecolor='none', markeredgewidth=3 )
    plt.plot(set_M_star[0], set_M_star[1], 'x', color='red', markersize=15)
    plt.text(set_M_star[0]+0.08, set_M_star[1]-0.075, r'$\xi_c$', fontsize=22)
    plt.grid(b=True)
    plt.xticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.yticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$y$', fontsize=22)
    plt.tight_layout()
    
def plot_M_i_ext(set_M_star, f):
    resolution = 500
    theta = np.linspace(0, 2*np.pi, resolution)
    val_0_x = []
    val_0_y = []
    val_1_x = []
    val_1_y = []
    for entry in theta:
        observation = np.array([np.cos(entry), np.sin(entry)])
        if Sets_Z.check_in_Z_i(0, observation):
            val_0_x.append(observation[0])
            val_0_y.append(observation[1])
        if Sets_Z.check_in_Z_i(1, observation):
            val_1_x.append(observation[0])
            val_1_y.append(observation[1])
    plt.figure(f)
    plt.plot(np.cos(theta), np.sin(theta), '--', linewidth=1, color='black')
    plt.plot(val_0_x, val_0_y, linewidth=3, color='green')
    plt.plot(set_M_star[0], set_M_star[1], 'x', color='red', markersize=15)
    plt.text(set_M_star[0]+0.08, set_M_star[1]-0.075, r'$\xi_c$', fontsize=22)
    plt.grid(b=True)
    plt.xticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.yticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$y$', fontsize=22)
    plt.tight_layout()
    
    plt.figure(f+1)
    plt.plot(np.cos(theta), np.sin(theta), '--', linewidth=1, color='black')
    plt.plot(val_1_x, val_1_y, linewidth=3, color='blue')
    plt.plot(1, 0, 'o', color='blue', markersize=15, markerfacecolor='none', markeredgewidth=3 )
    plt.plot(set_M_star[0], set_M_star[1], 'x', color='red', markersize=15)
    plt.text(set_M_star[0]+0.08, set_M_star[1]-0.075, r'$\xi_c$', fontsize=22)
    plt.grid(b=True)
    plt.xticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.yticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$y$', fontsize=22)
    plt.tight_layout()
# plot_M_i(set_M_star, 5)

def plot_trajectory(x,y,f, switch_x=[], switch_y=[], lcolor='red'):
    plt.figure(f)
    p = plt.plot(x,y, ':', linewidth=3, color='red')
    plt.plot(x[0], y[0], 'o', markerfacecolor='none', color=lcolor, markersize=15)
    plt.plot(x[-1], y[-1], 'x', color=lcolor, markersize=20)
    plt.plot(switch_x, switch_y, '*', color=lcolor)
    plt.grid(b=True)
    plt.xticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.yticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$y$', fontsize=22)
    plt.tight_layout()

def plot_trajectory_angle(angle,f,lcolor, lstyle, dt=0.05):
    plt.figure(f)
    angle = np.unwrap(angle)
    plt.plot(np.linspace(0,dt*(len(angle)-1),len(angle)), angle,  lstyle, 
             linewidth=3, color=lcolor)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(b=True)
    plt.xlabel('$t$', fontsize=22)
    plt.ylabel(r'$\angle\xi$', fontsize=22)
    plt.tight_layout()

def plot_av_error(av, av_hyb, f):
    plt.figure(f)
    plt.plot(av)
    plt.plot(av_hyb)
    plt.grid(b=True)
    plt.xlabel('Steps', fontsize=16)
    plt.ylabel('Average error', fontsize=16)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(['Normal policy', 'Hybrid policy'])
    
# initialize for simulation
plt.close('all')
np.random.seed(2)
episode_steps = 80
theta_0 = [2.5120626915906676, 3.4570141785702333, np.pi]
epsilon = 0.12 # magnitude of the noise
for entry in theta_0:
    observation = np.array([np.cos(entry), np.sin(entry)])
    env_hb_sim = UnitCircle(random_init=False, state_init=observation, 
                            steps=episode_steps)
    observation = env_hb_sim.reset()
    
    average_diff_hyb = np.zeros(episode_steps)
    q = 1 
    done = False
    hyb_x = [observation[0]]
    hyb_y = [observation[1]]
    hyb_diff = []
    hyb_angles = []
    switch_x = []
    switch_y = []
    active_agent = hybrid_agents[q]
    signdist = 1
    while not done:
        if q == 0:
            if Sets_Z.check_in_Z_i(q, observation)==False:
                q = 1
                active_agent = hybrid_agents[q]
        if q == 1:
            if Sets_Z.check_in_Z_i(q, observation)==False:
                q = 0
                active_agent = hybrid_agents[q]
        disturbance = signdist * epsilon
        angle = np.arctan2(observation[1], observation[0])
          # map angle between 0 and 2pi
        angle = (angle + 2*np.pi) % (2*np.pi)
        hyb_angles.append(angle)
        angle += disturbance
        # Check bounds, ensure theta is in [0,2pi)
        if angle < 0:
            angle = angle + 2*np.pi
        if angle >= 2*np.pi:
            angle = angle - 2*np.pi
        observation = np.array([np.cos(angle), np.sin(angle)])
        
        action, _, _ = active_agent.choose_action(observation)
        observation, reward, done, info = env_hb_sim.step(action)         
        hyb_x.append(observation[0])
        hyb_y.append(observation[1])
        
        angle = np.arctan2(observation[1], observation[0])
        hyb_diff.append(abs(angle))
        
        signdist *= -1
        
    average_diff_hyb += np.array(hyb_diff)/len(theta_0) 
        
    plot_trajectory_angle(hyb_angles, 3, 'blue', '-')

np.random.seed(2)
for entry in theta_0:
    observation = np.array([np.cos(entry), np.sin(entry)])
    env_hb_sim = UnitCircle(random_init=False, state_init=observation,
                            steps = episode_steps)
    observation = env_hb_sim.reset()
    
    average_diff = np.zeros(episode_steps)
    done = False
    norm_x = [observation[0]]
    norm_y = [observation[1]]
    norm_angles = []
    diff = []
    signdist = 1
    while not done:
        disturbance = signdist * epsilon
        angle = np.arctan2(observation[1], observation[0])
          # map angle between 0 and 2pi
        angle = (angle + 2*np.pi) % (2*np.pi)
        norm_angles.append(angle)
        angle += disturbance
        # Check bounds, ensure theta is in [0,2pi)
        if angle < 0:
            angle = angle + 2*np.pi
        if angle >= 2*np.pi:
            angle = angle - 2*np.pi
        observation = np.array([np.cos(angle), np.sin(angle)])
        
        action, _, _ = agent.choose_action(observation)
        observation, reward, done, info = env_hb_sim.step(action)         
        norm_x.append(observation[0])
        norm_y.append(observation[1])
        
        angle = np.arctan2(observation[1], observation[0])
        diff.append(abs(angle))
        
        signdist *= -1
        
    average_diff += np.array(diff)/len(theta_0)
    plot_trajectory(norm_x, norm_y, 1, switch_x=switch_x, switch_y=switch_y)
    theta_bg = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta_bg), np.sin(theta_bg), linewidth=0.5, color='black', zorder = 0)
    plt.text(np.cos(theta_0[0])+0.1, np.sin(theta_0[0])-.15, r'$\xi_0^1$', fontsize=22)
    plt.text(np.cos(theta_0[1])+0.1, np.sin(theta_0[1])+0, r'$\xi_0^2$', fontsize=22)
    plt.text(np.cos(theta_0[2])+0.1, -.05, r'$\xi_0^3$', fontsize=22)
    plot_trajectory_angle(norm_angles, 3, 'red', '--')
# plot_av_error(average_diff, average_diff_hyb, 2)    
# plot_new_policy_circle(hybrid_agents[0], Sets_Z, 0, 5)
# plot_new_policy_circle(hybrid_agents[1], Sets_Z, 1, 6)