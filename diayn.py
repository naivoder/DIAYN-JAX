import jax 
import jax.numpy as jnp
import optax
import flax.linen as nn 
from typing import NamedTuple

class ReplayBufferState(NamedTuple):
    obs: jnp.ndarray                # (capacity, obs_dim)
    action: jnp.ndarray             # (capacity, action_dim)
    reward: jnp.ndarray             # (capacity,)
    next_obs: jnp.ndarray           # (capacity, obs_dim)
    done: jnp.ndarray               # (capacity,)
    skill: jnp.ndarray              # (capacity,) integer skill index
    episode_step: jnp.ndarray       # (capacity,) for skipping states
    size: jnp.ndarray               # scalar - current size of the buffer
    ptr: jnp.ndarray                # scalar - write pointer

def init_replay_buffer(capacity: int, obs_dim: int, act_dim: int) -> ReplayBufferState:
    return ReplayBufferState(
        obs=jnp.zeros((capacity, obs_dim)),
        action=jnp.zeros((capacity, act_dim)),
        reward=jnp.zeros((capacity,)),
        next_obs=jnp.zeros((capacity, obs_dim)),
        done=jnp.zeros((capacity,)),
        skill=jnp.zeros((capacity,), dtype=jnp.int32),
        episode_step=jnp.zeros((capacity,), dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
        ptr=jnp.array(0, dtype=jnp.int32)
    )

def buffer_add_batch(buf: ReplayBufferState, obs, action, reward, next_obs, done, skill, episode_step) -> ReplayBufferState:
    capacity = buf.obs.shape[0]
    batch_size = obs.shape[0]

    ids = (buf.ptr + jnp.arange(batch_size)) % capacity 

    return ReplayBufferState(
        obs=buf.obs.at[ids].set(obs),
        action=buf.action.at[ids].set(action),
        reward=buf.reward.at[ids].set(reward),
        next_obs=buf.next_obs.at[ids].set(next_obs),
        done=buf.done.at[ids].set(done),
        skill=buf.skill.at[ids].set(skill),
        episode_step=buf.episode_step.at[ids].set(episode_step),
        size=jnp.minimum(buf.size + batch_size, capacity),
        ptr=(buf.ptr + batch_size) % capacity
    )

def buffer_sample(buf: ReplayBufferState, key: jnp.ndarray, batch_size: int):
    # use jnp.maximum to avoid randint(0, 0) edge case 
    safe_size = jnp.maximum(buf.size, 1)
    ids = jax.random.randint(key, (batch_size,), 0, safe_size)
    return (
        buf.obs[ids],
        buf.action[ids],
        buf.reward[ids],
        buf.next_obs[ids],
        buf.done[ids],
        buf.skill[ids],
        buf.episode_step[ids]
    )

