# Based on the code by Phil Tabor 
# https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch

import numpy as np
from PPO_agent_continuousActions import Agent
from unitcircle_env import UnitCircle
import matplotlib.pyplot as plt 
import torch as T
import time
from parula_cm import parula_map

plt.close('all')
def plot_policy(agent):
    plt.figure(1)
    resolution = 100
    theta = np.linspace(0, 2*np.pi, resolution)
    mean = []
    stdev = []
    for entry in theta:
        mu, sigma = agent.actor.forward(T.tensor([np.cos(entry), 
                    np.sin(entry)],dtype=T.float32).to(agent.actor.device))
        mu = mu.cpu().detach().numpy()[0].item()
        sigma = sigma.cpu().detach().numpy()[0].item()
        mean.append(mu)
        stdev.append(sigma)
    plt.plot(np.linspace(0,2, resolution), np.tanh(mean))#,  '--bo')
    plt.fill_between(np.linspace(0,2,resolution), 
                     np.tanh(np.array(mean)-np.array(stdev)),
                     np.tanh(np.array(mean)+np.array(stdev)) ,alpha=0.1)
    plt.grid(b=True)
    plt.xlabel('$S * \pi$')
    plt.ylabel('$\mu $')    
    plt.ylim(-1.05, 1.05)
    plt.savefig('plots/policy_unit_circle')
    
def plot_policy_circle(agent):
    plt.figure(3)
    resolution = 500
    theta = np.linspace(0, 2*np.pi, resolution)
    mean = []
    for entry in theta:
        mu, _ = agent.actor.forward(T.tensor([np.cos(entry), 
                    np.sin(entry)],dtype=T.float32).to(agent.actor.device))
        mu = mu.cpu().detach().numpy()[0].item()
        mean.append(mu)
    plt.scatter(np.cos(theta), np.sin(theta), s=15, c=np.tanh(mean), 
                cmap=parula_map)
    cbar = plt.colorbar(ticks=[-1, 0, 1])
    plt.clim(-1,1)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('$u$', rotation=270, fontsize=22, labelpad=22)
    plt.grid(b=True)
    plt.xticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.yticks(np.linspace(-1, 1, num=5, endpoint=True), fontsize=18)
    plt.xlabel('$x$', fontsize=22)
    plt.ylabel('$y$', fontsize=22)
    plt.tight_layout()

    plt.savefig('plots/policy_circle_unit_circle')
    
def plot_critic(agent):
    plt.figure(2)
    resolution = 100
    theta = np.linspace(0, 2*np.pi, resolution)
    vals = []
    for entry in theta:
        value = agent.critic.forward(T.tensor([np.cos(entry), 
                    np.sin(entry)],dtype=T.float32).to(agent.actor.device))
        value = value.cpu().detach().numpy()[0].item()
        vals.append(value)
    plt.plot(np.linspace(0,2, resolution), vals)
    plt.grid(b=True)
    plt.xlabel('$S * \pi$')
    plt.ylabel('Value')    
    plt.savefig('plots/value_unit_circle')


def plot_learning_curve(x, scores, figure_file):
    plt.figure(0)
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, scores)
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    plt.grid()

# if __name__ == '__main__':
def training_PPO(env, name, load_checkpoint=False, alpha=0.0001, beta=0.00005,
                 N=5, min_steps=1):
    # env = UnitCircle()
    # N =  5 
    batch_size =  40
    n_epochs = 5
    # alpha = 0.0001
    # beta  = 0.00005
    agent = Agent(n_actions=env.action_space.shape, max_actions=env.action_space.high,
                  batch_size=batch_size, alpha=alpha, beta=beta, n_epochs=n_epochs, 
                  input_dims=env.observation_space.shape, fc1_dims=128, 
                  fc2_dims=128, policy_clip=0.2, gamma=0.95)

    figure_file = 'plots/unitcircle'+name+'.png'
    
    best_score = env.reward_range[0]
    score_history = []
    
    learn_iters = 0
    avg_score = -100
    n_steps = 0
    
    #load_checkpoint = True
    
    if load_checkpoint:
        agent.load_models()
    i = 0
    t = time.time()
    while avg_score < -31 or i <min_steps:
        observation = env.reset()
        done = False
        score = 0
        if i % 200 == 0:
            print(i)
            plt.close('all')
            plot_policy(agent)
            plot_critic(agent)
            plot_policy_circle(agent)
            plt.pause(0.05)
        while not done:
            action, log_prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            # if i > 400:
            #     print('reward = ' + str(reward), 'x,y = ' + str(observation_))
            agent.remember(observation, action, log_prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            # agent.save_models()
            
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
        i += 1
    print('elapsed time =' + str(time.time()-t))
    x =[i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    plot_policy(agent)
    # plot_policy_circle(agent)
    return agent