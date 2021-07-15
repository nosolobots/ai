import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import beta

BANDIT_MEANS = [0.2, 0.5, 0.8]
NUM_TRIALS = 2000

class Bandit:
    def __init__(self, mu):
        self.mu = mu
        self.mu_estimate = 0.
        self.a = 1.
        self.b = 1.
        self.n = 0
    
    def sample(self):
        # draw a random sample from beta distribution
        return np.random.beta(self.a, self.b)

    def pull(self):
        return np.random.random()<self.mu

    def update(self, x):
        self.n += 1
        self.mu_estimate = (1/self.n)*((self.n - 1)*self.mu_estimate + x)
        # update beta distribution parameters
        self.a += x
        self.b += (1-x)

def plot(bandits, trial):
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label=f"real p: {b.mu:.4f}, win rate = {b.a - 1}/{b.n}")
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
    plt.show()

def experiment():
    bandits = [Bandit(mu) for mu in BANDIT_MEANS]
    rewards = np.zeros(NUM_TRIALS)
    total_reward = 0.0
    num_optimal = 0
    optimal_j = np.argmax([b.mu for b in bandits])
    print("optimal j:", optimal_j)

    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]

    for i in range(NUM_TRIALS):
        # select bandit
        j = np.argmax([b.sample() for b in bandits])

        if j==optimal_j:
            num_optimal +=1

        # pull bandit
        x = bandits[j].pull()

        # update rewards
        rewards[i] = x

        # update pulled bandit distribution
        bandits[j].update(x)

        # plot distributions
        if i in sample_points:
            plot(bandits, i)

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
    
    plt.title("Multi-Armed Bandit Thompson sampling")
    plt.ylabel("win rate")
    plt.xlabel("tries")
    plt.xscale("log")
    plt.legend()
    
    plt.show()


    
