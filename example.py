import acme
from acme.wrappers import GymnasiumAtariAdapter, AtariWrapper
import gymnasium as gym
import numpy as np

env = gym.make("PongNoFrameskip-v4", repeat_action_probability=0.25)
env = AtariWrapper(GymnasiumAtariAdapter(env))

class RandomAgent(acme.Actor):
    """A random agent."""
    
    def __init__(self):
        
        # specify the behavior policy
        self.behavior_policy = lambda: np.random.choice(6)
        
        # store timestep, action, next_timestep
        self.timestep = None
        self.action = None
        self.next_timestep = None
        
    def select_action(self, observation):
        "Choose an action according to the behavior policy."
        action = self.behavior_policy()
        return action

    def observe_first(self, timestep):
        "Observe the first timestep." 
        self.timestep = timestep

    def observe(self, action, next_timestep):
        "Observe the next timestep."
        self.action = action
        self.next_timestep = next_timestep
        
    def update(self, wait = False):
        "Update the policy."
        # no updates occur here, it's just a random policy
        self.timestep = self.next_timestep 

actor = RandomAgent()

env_loop = acme.EnvironmentLoop(env, actor)

data = env_loop.run(num_steps=5_000)
print(data)

