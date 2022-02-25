import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from unitcircle_env import UnitCircle
matplotlib.rcParams['text.usetex'] = True

def angle_to_state(angle):
    state = np.array([np.cos(angle), np.sin(angle)],dtype=np.float32)
    return state

def state_to_angle(state):
    angle = np.arctan2(state[1], state[0])
    # map angle between 0 and 2pi
    angle = np.float32((angle + 2*np.pi) % (2*np.pi))
    return angle

def plot_policy(model, resolution=100, figure_number=1):
    plt.figure(figure_number)
    theta = np.linspace(0, 2*np.pi, resolution)
    actions = []
    for entry in theta:
        obs = np.array((np.cos(entry), np.sin(entry)), dtype=np.float32)
        action, _ = model.predict(obs)
        actions.append(action)
    plt.plot(np.linspace(0,2, resolution), actions,)
    plt.grid()
    plt.xlabel('$S * \pi$')
    plt.ylabel('$\mu $')    
    plt.ylim(-1.05, 1.05)
    plt.xlim(0.9, 1.1)
    plt.tight_layout()

def plot_policy_circle(model, resolution=5000, figure_number=2):
    plt.figure(figure_number)
    theta = np.linspace(0, 2*np.pi, resolution)
    actions = []
    for entry in theta:
        obs = np.array((np.cos(entry), np.sin(entry)), dtype=np.float32)
        action, _ = model.predict(obs)
        actions.append(action)
    plt.scatter(np.cos(theta), np.sin(theta), s=15, c=actions)
    cbar = plt.colorbar(ticks=[-1, 0, 1])
    plt.clim(-1,1)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('$u$', rotation=270, fontsize=22, labelpad=22)
    plt.grid()
    plt.xticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.yticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$y$', fontsize=22)
    plt.tight_layout()

def simulate_unitcircle(agent, state_init, noise_mag=0.1, figure_number=3):
    theta = np.linspace(0, 2*np.pi, 200)
    env = UnitCircle(state_init=state_init, random_init=False, steps=80)
    done = False
    obs = state_init
    angles = [state_to_angle(obs)]
    score = 0
    sign = -1
    while done == False:
        disturbance = noise_mag*sign
        angle = state_to_angle(obs)
        obs = angle_to_state(angle+disturbance)
        action, _ = agent.predict(obs)
        obs, reward, done, _ = env.step(action)
        score += reward
        angles.append(state_to_angle(obs))
        sign *= -1
    plt.figure(figure_number)
    plt.plot(np.cos(theta), np.sin(theta), '--', color='black',zorder=-1)
    plt.plot(np.cos(angles), np.sin(angles), '--', linewidth=2, color='red')
    plt.plot(np.cos(angles)[0], np.sin(angles)[0], 'o', markerfacecolor='none', color='red', markersize=15)
    plt.plot(np.cos(angles)[-1], np.sin(angles)[-1], 'x', color='red', markersize=20)
    plt.grid()
    plt.xticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.yticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$y$', fontsize=22)      
    plt.tight_layout()

plt.close('all')
env = UnitCircle()
model = PPO.load("ppo_unitcircle", env)

plot_policy(model)
plot_policy_circle(model)
plt.savefig('policy_circle_unit_circle_pi_star.eps',  format='eps')

# simulation the hybrid agent compared to the original agent    
angles_init = [0.8*np.pi, 1.1*np.pi, 1.*np.pi]
for angle in angles_init:
    state_init = angle_to_state(angle)
    simulate_unitcircle(model, state_init)
plt.text(np.cos(angles_init[0])+0.1, np.sin(angles_init[0])-.1, r'$\xi_0^1$', fontsize=22)
plt.text(np.cos(angles_init[1])+0.1, np.sin(angles_init[1])-0.07, r'$\xi_0^2$', fontsize=22)
plt.text(np.cos(angles_init[2])+0.1, -.05, r'$\xi_0^3$', fontsize=22)
plt.savefig('PolicySimUnitCircle.eps',  format='eps')