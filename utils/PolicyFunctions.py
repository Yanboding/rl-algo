import numpy as np

def create_greedy_policy(Q):
    """
    Creates a greedy policy based on Q values.

    Args:
        Q: A dictionary that maps from state -> action values

    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities.
    """

    def policy_fn(state):
        # All actions that available in the given state
        actions = np.arange(len(Q[state]))
        best_action = np.random.choice(actions[Q[state] == np.max(Q[state])])
        A = np.where(actions == best_action, 1.0, 0.0)
        return A

    return policy_fn


def create_epsilon_greedy_policy(Q, epsilon):
    """
    Creates a epsilon greedy policy based on Q values.

    Args:
        Q: A dictionary that maps from state -> action values
        epsilon: A probability of selecting an action randomly.

    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities.
    """

    def policy_fn(state):
        # All actions that available in the given state
        actions = np.arange(len(Q[state]))
        best_action = np.random.choice(actions[Q[state] == np.max(Q[state])])
        ramdomProb = epsilon / len(Q[state])
        A = np.where(actions == best_action, 1 - epsilon + ramdomProb, ramdomProb)
        return A

    return policy_fn


def create_random_policy(nA):
    """
    Creates a random policy function.

    Args:
        nA: Number of actions in the environment.

    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA

    def policy_fn(observation):
        return A

    return policy_fn