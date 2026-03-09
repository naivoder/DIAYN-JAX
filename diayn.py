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

class QNetwork(nn.Module):
    hidden_dim: int = 300 

    @nn.compact
    def __call__(self, obs, skill_onehot, action):
        x = jnp.concatenate([obs, skill_onehot, action], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)
    
class TwinQ(nn.Module):
    hidden_dim: int = 300

    @nn.compact
    def __call__(self, obs, skill_onehot, action):
        q1 = QNetwork(self.hidden_dim, name="q1")(obs, skill_onehot, action)
        q2 = QNetwork(self.hidden_dim, name="q2")(obs, skill_onehot, action)
        return q1, q2
    
class GaussianPolicy(nn.Module):
    hidden_dim: int = 300
    action_dim: int = 1
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, obs, skill_onehot, key):
        x = jnp.concatenate([obs, skill_onehot], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        std = jnp.exp(log_std)

        # Reparameterization trick
        noise = jax.random.normal(key, mean.shape)
        raw_action = mean + std * noise 

        # Clamp raw_action to prevent tanh saturation 
        # tanh(20) ~ 1.0, log(cosh(20))) ~ 20, so this is a safe range 
        # Outside of this there are issues with Jacobian correction
        raw_action = jnp.clip(raw_action, -20.0, 20.0)
        action = jnp.tanh(raw_action)

        log_prob = -0.5 * (
            ((raw_action - mean) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)
        )
        log_prob = log_prob.sum(axis=-1)

        # Correction for Tanh squashing
        # log(1 - tanh^2) is unstable for large actions 
        log_prob -= jnp.sum(2.0 * (
            jnp.abs(raw_action) + jnp.log(1 + jnp.exp(-2.0 * jnp.abs(raw_action))) - jnp.log(2.0)),
            axis=-1)
        
        return action, log_prob, mean 
    
class Discriminator(nn.Module):
    hidden_dim: int = 300
    n_skills: int = 10 

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.n_skills)(x)
        return logits