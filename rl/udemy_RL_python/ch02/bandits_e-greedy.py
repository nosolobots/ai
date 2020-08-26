import numpy as np 
import matplotlib.pyplot as plt

BANDIT_MEANS = [0.2, 0.5, 0.8]
NUM_TRIALS = 10000
EPS = [0.05, 0.1, 0.2, 0.5, 0.8]

class Bandit:
    def __init__(self, mu):
        self.mu = mu
        self.mu_estimate = 0.
        self.n = 0
    
    def pull(self):
        # draw a random sample from a normal distribution of mean=self.mu and variance=1
        return np.random.randn() + self.mu

    def update(self, x):
        self.n += 1
        self.mu_estimate = (1/self.n)*((self.n - 1)*self.mu_estimate + x)

def experiment(epsilon):
    bandits = [Bandit(mu) for mu in BANDIT_MEANS]
    rewards = np.zeros(NUM_TRIALS)
    total_reward = 0.0
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    optimal_j = np.argmax([b.mu for b in bandits])
    print("optimal j:", optimal_j)

    for i in range(NUM_TRIALS):
        # select bandit
        if np.random.random() < epsilon:
            # randomly
            num_times_explored += 1
            j = int(np.random.random()*len(bandits))
        else:
            # greedy
            num_times_exploited += 1
            j = np.argmax([b.mu_estimate for b in bandits])

        if j==optimal_j:
            num_optimal +=1

        # pull bandit
        x = bandits[j].pull()

        # update rewards
        rewards[i] = x

        # update pulled bandit distribution
        bandits[j].update(x)

    return rewards, num_optimal, num_times_explored, num_times_exploited, (b.mu_estimate for b in bandits)

if __name__ == "__main__":
    for e in EPS:
        rewards, num_optimal, num_times_explored, num_times_exploited, bandit_estimates = experiment(e)

        # Results
        print("-"*30)
        print("eps:", e)
        for b in bandit_estimates:
            print("mean estimate:", b)
        print("total reward:", rewards.sum())
        print("overall win rate:", rewards.sum()/NUM_TRIALS)
        print("num times explored:", num_times_explored)
        print("num times exploited:", num_times_exploited)
        print("num times selected optimal bandit:", num_optimal)

        # plot results
        cumulative_rewards = np.cumsum(rewards)
        win_rates = cumulative_rewards/(np.arange(NUM_TRIALS) + 1)  # (total reward) vs (num trials) ratio 
                                                                    # should must improve over time!!
        plt.plot(win_rates, label="eps:" + str(e))
    
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_MEANS), label="optimal")
    
    plt.title("Multi-Armed Bandit epsilon-greedy")
    plt.ylabel("win rate")
    plt.xlabel("tries")
    plt.xscale("log")
    plt.legend()
    
    plt.show()


    
