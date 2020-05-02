from MonteCarloControl import *

Q = pickle.load( open( "Q_action_values.p", "rb" ) )

def simulate_episodes(episodes, env, policy = 'random'):
    
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        # print(state)
        while True:
            if policy == "random":
                action = env.action_space.sample() # takes random action from environment's action space
                # print(action)
            else:
                action, _ = get_action(state, Q)
            state, reward, done, info = env.step(action) # OpenAI gym gives feedback in this tuple form : state,reward,if_done?,other relevant info
            # print(state)
            if done:
                # print('Game has ended! Your Reward: ', reward)
                # print('You won :)\n') if reward > 0 else print('You lost :(\n')
                rewards.append(reward)
                break
    
    return sum(rewards)

if __name__ == "__main__":
    env = gym.make('Blackjack-v0')
    games = 1000
    random_policy_rewards = []
    learnt_policy_rewards = []
    for _ in range(games):
        random_policy_rewards.append(simulate_episodes(episodes = 100, env = env, policy = 'random'))
        learnt_policy_rewards.append(simulate_episodes(episodes = 100, env = env, policy = 'learnt'))

    plt.plot(range(games), random_policy_rewards, label = 'random')
    plt.plot(range(games), learnt_policy_rewards, label = 'learnt')
    plt.xlabel('Game')
    plt.legend(loc='upper right')
    plt.savefig('policy_comparison.png')
    plt.show()