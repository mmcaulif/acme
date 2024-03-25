from typing import List, Optional
import dm_env

from acme import types
from acme import specs
from acme.wrappers.gym_wrapper import GymWrapper
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tree

class GymnasiumWrapper(GymWrapper):
    """Environment wrapper for OpenAI Gym environments."""

    # Note: we don't inherit from base.EnvironmentWrapper because that class
    # assumes that the wrapped environment is a dm_env.Environment.

    def __init__(self, environment: gym.Env):

        self._environment = environment
        self._reset_next_step = True
        self._last_info = None

        # Convert action and observation specs.
        obs_space = self._environment.observation_space
        act_space = self._environment.action_space
        self._observation_spec = _convert_to_spec(obs_space, name='observation')
        self._action_spec = _convert_to_spec(act_space, name='action')

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        observation, info = self._environment.reset()
        # Reset the diagnostic information.
        self._last_info = info
        return dm_env.restart(observation)

    def step(self, action: types.NestedArray) -> dm_env.TimeStep:
        """Steps the environment."""
        if self._reset_next_step:
            return self.reset()

        observation, reward, terminated, truncated, info = self._environment.step(action)
        self._reset_next_step = terminated or truncated
        self._last_info = info

        # Convert the type of the reward based on the spec, respecting the scalar or
        # array property.
        reward = tree.map_structure(
            lambda x, t: (  # pylint: disable=g-long-lambda
                t.dtype.type(x)
                if np.isscalar(x) else np.asarray(x, dtype=t.dtype)),
            reward,
            self.reward_spec())

        if terminated:
            return dm_env.termination(reward, observation)
        if truncated:
            return dm_env.truncation(reward, observation)
        return dm_env.transition(reward, observation)
  

def _convert_to_spec(space: gym.Space,
                     name: Optional[str] = None) -> types.NestedSpec:
    """Converts an OpenAI Gym space to a dm_env spec or nested structure of specs.

    Box, MultiBinary and MultiDiscrete Gym spaces are converted to BoundedArray
    specs. Discrete OpenAI spaces are converted to DiscreteArray specs. Tuple and
    Dict spaces are recursively converted to tuples and dictionaries of specs.

    Args:
        space: The Gym space to convert.
        name: Optional name to apply to all return spec(s).

    Returns:
        A dm_env spec or nested structure of specs, corresponding to the input
        space.
    """
    if isinstance(space, spaces.Discrete):
        return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

    elif isinstance(space, spaces.Box):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=space.dtype,
            minimum=space.low,
            maximum=space.high,
            name=name)

    elif isinstance(space, spaces.MultiBinary):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=space.dtype,
            minimum=0.0,
            maximum=1.0,
            name=name)

    elif isinstance(space, spaces.MultiDiscrete):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=space.dtype,
            minimum=np.zeros(space.shape),
            maximum=space.nvec - 1,
            name=name)

    elif isinstance(space, spaces.Tuple):
        return tuple(_convert_to_spec(s, name) for s in space.spaces)

    elif isinstance(space, spaces.Dict):
        return {
            key: _convert_to_spec(value, key)
            for key, value in space.spaces.items()
        }

    else:
        raise ValueError('Unexpected gym space: {}'.format(space))
  

class GymnasiumAtariAdapter(GymnasiumWrapper):
    """Specialized wrapper exposing a Gym Atari environment.

    This wraps the Gym Atari environment in the same way as GymWrapper, but also
    exposes the lives count as an observation. The resuling observations are
    a tuple whose first element is the RGB observations and the second is the
    lives count.
    """

    def _wrap_observation(self, observation: types.NestedArray) -> types.NestedArray:
        return observation, self._environment.unwrapped.ale.lives()

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        observation, info = self._environment.reset()
        observation = self._wrap_observation(observation)
        # Reset the diagnostic information.
        self._last_info = info
        return dm_env.restart(observation)

    def step(self, action: List[np.ndarray]) -> dm_env.TimeStep:
        """Steps the environment."""
        if self._reset_next_step:
            return self.reset()

        observation, reward, terminated, truncated, info = self._environment.step(action)
        observation = self._wrap_observation(observation)
        self._reset_next_step = terminated or truncated
        self._last_info = info

        # Convert the type of the reward based on the spec, respecting the scalar or
        # array property.
        reward = tree.map_structure(
            lambda x, t: (  # pylint: disable=g-long-lambda
                t.dtype.type(x)
                if np.isscalar(x) else np.asarray(x, dtype=t.dtype)),
            reward,
            self.reward_spec())

        if terminated:
            return dm_env.termination(reward, observation)
        if truncated:
            return dm_env.truncation(reward, observation)
        return dm_env.transition(reward, observation)

    def observation_spec(self) -> types.NestedSpec:
        return (self._observation_spec,
                specs.Array(shape=(), dtype=np.dtype('float64'), name='lives'))

    def action_spec(self) -> List[specs.BoundedArray]:
        return [self._action_spec]  # pytype: disable=bad-return-type
