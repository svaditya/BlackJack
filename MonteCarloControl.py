import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle


def get_action(state, Q):
    if state not in Q.keys():
        action = np.random.choice(range(env.action_space.n))
        Q[state] = [0,0]
    else:
        action = np.argmax(Q[state])
    return action, Q

def get_episode(env, Q):
    episode = []
    state = env.reset()
    # To ensure exploring starts
    action = np.random.choice(range(env.action_space.n)) 
    if state not in Q.keys():
        Q[state] = [0,0]
    

    while True:
        
        next_state, reward, done, info = env.step(action) # OpenAI gym gives feedback in this tuple form : state,reward,if_done?,other relevant info
        episode.append((state, action, reward))
        
        if done:
            break
        
        state = next_state
        action, Q = get_action(state, Q)

    return episode, Q

def mc_control_algorithm(iters, env):

    # Initialize
    Q = {}
    Returns = {}
    gamma = 1

    for e in range(iters):
        episode, Q = get_episode(env, Q)
        G = 0
        for i in range(len(episode) - 1, -1, -1):

            state = episode[i][0]
            episode_states = [s[0] for s in episode]
            #print(episode_states)
            action = episode[i][1]
            reward = episode[i][2]

            R_tplus1 = reward
            G += gamma*R_tplus1
            #print(G)
            if state not in episode_states[0:i-1]:
                if state not in Returns.keys():
                    Returns[state] = {}
                    Returns[state][action] = []
                    Returns[state][action].append(G)
                elif action not in Returns[state].keys():
                    Returns[state][action] = []
                    Returns[state][action].append(G)
                else:
                    Returns[state][action].append(G)
                #print(Returns)
                Q[state][action] = np.mean(Returns[state][action])

    return Q, Returns 

def plot_blackjack_values(V, filename):
    
    def get_Z(x, y, usable_ace):
        if (x,y,usable_ace) in V:
            return V[x,y,usable_ace]
        else:
            return 0

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)
        
        Z = np.array([get_Z(x,y,usable_ace) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.savefig(filename)
    plt.show()
    
if __name__ == "__main__":
    env = gym.make('Blackjack-v0')
    Q, Returns = mc_control_algorithm(500000, env)
    # Save the policy
    pickle.dump(Q, open( "Q_action_values.p", "wb" ) )
    # obtain the corresponding state-value function
    V = dict((k,np.max(v)) for k, v in Q.items())
    # plot the state-value function
    plot_blackjack_values(V, "MC_Control_StateValueFunction_Viz.png")

    
