import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
from typing import Any 

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
            ((raw_action - mean) / (std + 1e-8)) ** 2
            + 2 * log_std
            + jnp.log(2 * jnp.pi)
        )
        log_prob = log_prob.sum(axis=-1)

        # Correction for Tanh squashing
        # log(1 - tanh^2) is unstable for large actions
        log_prob -= jnp.sum(
            2.0
            * (
                jnp.abs(raw_action)
                + jnp.log(1 + jnp.exp(-2.0 * jnp.abs(raw_action)))
                - jnp.log(2.0)
            ),
            axis=-1,
        )

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
    
def compute_diayn_reward(disc_params, discriminator, obs, skill_ids, n_skills):
    logits = discriminator.apply(disc_params, obs)
    logits = jnp.clip(logits, -20.0, 20.0)  # prevent extreme values
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    log_qz_s = log_probs[jnp.arange(log_probs.shape[0]), skill_ids]
    log_pz = -jnp.log(n_skills + 0.0)  # 0.0 to ensure floats
    reward = log_qz_s - log_pz

    # Clamp reward to prevent q-value divergence
    return jnp.clip(reward, -10.0, 10.0)


def update_critic(
    critic_state: TrainState,
    target_critic_params: Any,
    policy_state: TrainState,
    obs,
    action,
    reward,
    next_obs,
    done,
    skill_onehot,
    gamma: float,
    alpha: float,
    key: jnp.ndarray,
):
    next_action, next_log_prob, _ = policy_state.apply_fn(
        policy_state.params, next_obs, skill_onehot, key
    )
    next_q1, next_q2 = critic_state.apply_fn(
        target_critic_params, next_obs, skill_onehot, next_action
    )
    next_q = jnp.minimum(next_q1, next_q2) - alpha * jnp.clip(
        next_log_prob, -20.0, 20.0
    )
    next_q = jnp.clip(next_q, -100.0, 100.0)  # TO DO: test sensitivity to this value
    target_q = jax.lax.stop_gradient(reward + gamma * (1.0 - done) * next_q)

    def critic_loss_fn(params):
        q1, q2 = critic_state.apply_fn(params, obs, skill_onehot, action)
        # Huber loss is more robust to outliers than MSE
        loss_q1 = jnp.mean(optax.huber_loss(q1, target_q, delta=1.0))
        loss_q2 = jnp.mean(optax.huber_loss(q2, target_q, delta=1.0))
        return loss_q1 + loss_q2

    loss, grads = jax.value_and_grad(critic_loss_fn)(critic_state.params)
    return critic_state.apply_gradients(grads=grads), loss


def update_actor(
    policy_state: TrainState,
    critic_state: TrainState,
    obs,
    skill_onehot,
    alpha: float,
    key: jnp.ndarray,
):
    def actor_loss_fn(params):
        action, log_prob, _ = policy_state.apply_fn(params, obs, skill_onehot, key)
        q1, q2 = critic_state.apply_fn(critic_state.params, obs, skill_onehot, action)
        q_min = jnp.clip(jnp.minimum(q1, q2), -100, 100)
        log_prob = jnp.clip(log_prob, -20.0, 20.0)
        return jnp.mean(alpha * log_prob - q_min)

    loss, grads = jax.value_and_grad(actor_loss_fn)(policy_state.params)
    return policy_state.apply_gradients(grads=grads), loss


def update_discriminator(
    disc_state: TrainState,
    obs,
    skill_ids,
    mask=None,
    episode_step=None,
    skip_initial_steps: int = 0,
):
    # Mask terminal states and optional initial steps
    batch_size = obs.shape[0]
    combined_mask = jnp.ones(batch_size)
    if mask is not None:
        combined_mask = combined_mask * mask
    if episode_step is not None:
        step_mask = combined_mask * (episode_step >= skip_initial_steps).astype(
            jnp.float32
        )
        combined_mask = combined_mask * step_mask

    def disc_loss_fn(params):
        logits = disc_state.apply_fn(params, obs)
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, skill_ids)
        masked_loss = ce_loss * combined_mask
        return jnp.sum(masked_loss) / (jnp.sum(combined_mask) + 1e-8)

    loss, grads = jax.value_and_grad(disc_loss_fn)(disc_state.params)
    return disc_state.apply_gradients(grads=grads), loss


def soft_update(target_params, online_params, tau: float = 0.005):
    # Polyak averaging for target network update
    return jax.tree.map(
        lambda t, o: tau * o + (1 - tau) * t, target_params, online_params
    )