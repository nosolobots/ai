import numpy as np 
import matplotlib.pyplot as plt

BANDIT_MEANS = [0.2, 0.5, 0.75]
NUM_TRIALS = 10000

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

def experiment():
    bandits = [Bandit(mu) for mu in BANDIT_MEANS]
    rewards = np.zeros(NUM_TRIALS)
    total_reward = 0.0
    num_optimal = 0
    optimal_j = np.argmax([b.mu for b in bandits])
    print("optimal j:", optimal_j)
    total_tries = 0

    # pull one initial time each bandit
    for b in bandits:
        b.update(b.pull())
        total_tries += 1

    for i in range(NUM_TRIALS):
        total_tries += 1
        
        # select bandit (greedy)
        j = np.argmax([(b.mu_estimate + np.sqrt(2*np.log(total_tries))/b.n) for b in bandits])

        if j==optimal_j:
            num_optimal +=1

        # pull bandit
        x = bandits[j].pull()

        # update rewards
        rewards[i] = x

        # update pulled bandit distribution
        bandits[j].update(x)

    return rewards, num_optimal, (b.mu_estimate for b in bandits)

if __name__ == "__main__":
    rewards, num_optimal, bandit_estimates = experiment()

    # Results
    print("-"*30)
    for b in bandit_estimates:
        print("mean estimate:", b)
    print("total reward:", rewards.sum())
    print("overall win rate:", rewards.sum()/NUM_TRIALS)
    print("num times selected optimal bandit:", num_optimal)

    # plot results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards/(np.arange(NUM_TRIALS) + 1)  # (total reward) vs (num trials) ratio 
                                                                # should must improve over time!!
    plt.plot(win_rates)
    
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_MEANS), label="optimal")
    
    plt.title("Multi-Armed Bandit Upper-Bound Confidence")
    plt.ylabel("win rate")
    plt.xlabel("tries")
    plt.xscale("log")
    plt.legend()
    
    plt.show()


    
