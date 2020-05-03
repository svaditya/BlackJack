from MonteCarloControl import *

def get_action_epsilon_soft(state, Q, epsilon):
    if state not in Q.keys():
        action_star = np.random.choice(range(env.action_space.n))
        Q[state] = [0,0]
    else:
        action_star = np.argmax(Q[state])
    prob = [epsilon/((env.action_space.n))]*(env.action_space.n)
    prob[action_star] += 1 - epsilon 
    action = np.random.choice(range(env.action_space.n), p = prob)
    return action, Q

def get_episode_epsilon_soft(env, Q, epsilon):
    episode = []
    state = env.reset()

    while True:
        action, Q = get_action_epsilon_soft(state, Q, epsilon)
        next_state, reward, done, info = env.step(action) # OpenAI gym gives feedback in this tuple form : state,reward,if_done?,other relevant info
        episode.append((state, action, reward))
        
        if done:
            break
        
        state = next_state
        

    return episode, Q

def mc_control_epsilon_soft_algorithm(iters, env, epsilon):
    
    # Initialize
    Q = {}
    Returns = {}
    gamma = 1

    for e in range(iters):
        episode, Q = get_episode_epsilon_soft(env, Q, epsilon)
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

if __name__ == "__main__":
    env = gym.make('Blackjack-v0')
    Q, Returns = mc_control_epsilon_soft_algorithm(500000, env, epsilon = 0.05)
    # Save the policy
    pickle.dump(Q, open( "Q_action_values_epsilon_soft.p", "wb" ) )
    # obtain the corresponding state-value function
    V = dict((k,np.max(v)) for k, v in Q.items())
    # plot the state-value function
    plot_blackjack_values(V, "MC_Control_epsilon_soft_StateValueFunction_Viz.png")
