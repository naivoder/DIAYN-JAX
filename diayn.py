import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from typing import NamedTuple, Any, Tuple
from functools import partial

class ReplayBufferState(NamedTuple):
    obs: jnp.ndarray  # (capacity, obs_dim)
    action: jnp.ndarray  # (capacity, action_dim)
    reward: jnp.ndarray  # (capacity,)
    next_obs: jnp.ndarray  # (capacity, obs_dim)
    done: jnp.ndarray  # (capacity,)
    skill: jnp.ndarray  # (capacity,) integer skill index
    episode_step: jnp.ndarray  # (capacity,) for skipping states
    size: jnp.ndarray  # scalar - current size of the buffer
    ptr: jnp.ndarray  # scalar - write pointer


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
        ptr=jnp.array(0, dtype=jnp.int32),
    )


def buffer_add_batch(
    buf: ReplayBufferState, obs, action, reward, next_obs, done, skill, episode_step
) -> ReplayBufferState:
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
        ptr=(buf.ptr + batch_size) % capacity,
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
        buf.episode_step[ids],
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


class DIAYNTrainingState(NamedTuple):
    # Flax TrainState bundles params and opt
    policy_state: TrainState  # policy state
    critic_state: TrainState  # critic state
    disc_state: TrainState  # discriminator state
    target_critic_params: Any  # slow moving copy of critic
    replay_buf_state: ReplayBufferState  # replay buffer state
    env_state: Any  # brax env states
    obs: jnp.ndarray  # current observations
    env_skills: jnp.ndarray  # skill assigned to each env
    env_ep_rewards: jnp.ndarray  # cumulative rewards per env
    key: jnp.ndarray  # PRNG key
    step: jnp.ndarray  # global step count
    episode_count: jnp.ndarray  # total episodes completed


class Metrics(NamedTuple):
    critic_loss: jnp.ndarray
    actor_loss: jnp.ndarray
    disc_loss: jnp.ndarray
    avg_step_reward: jnp.ndarray  # average DIAYN reward per env step
    num_episodes: jnp.ndarray


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


def make_training_step(
    env,
    policy_net,
    critic_net,
    disc_net,
    n_skills: int,
    num_envs: int,
    batch_size: int,
    gamma: float,
    alpha: float,
    tau: float,
    action_scale: float,
    warmup_steps: int,
    utd_ratio: int = 1,
    disc_utd_ratio: int = 1,
    skip_initial_steps: int = 0,
):
    compute_reward = partial(
        compute_diayn_reward, discriminator=disc_net, n_skills=n_skills
    )
    do_update_critic = partial(update_critic, gamma=gamma, alpha=alpha)
    do_update_actor = partial(update_actor, alpha=alpha)
    do_soft_update = partial(soft_update, tau=tau)
    do_update_disc = partial(
        update_discriminator, skip_initial_steps=skip_initial_steps
    )

    def training_step(state: DIAYNTrainingState) -> Tuple[DIAYNTrainingState, Metrics]:
        key = state.key

        # 1. Select actions for all envs
        key, key_act, key_random = jax.random.split(key, 3)
        skill_onehot = jax.nn.one_hot(state.env_skills, n_skills)

        policy_actions = policy_net.apply(
            state.policy_state.params, state.obs, skill_onehot, key_act
        )[0]
        random_actions = jax.random.uniform(
            key_random, (num_envs, policy_actions.shape[-1]), minval=-1.0, maxval=1.0
        )

        # During warmup use random, else use policy
        actions = jax.lax.cond(
            state.step < warmup_steps, lambda: random_actions, lambda: policy_actions
        )
        env_actions = actions * action_scale

        # 2. Step all envs in parallel
        next_env_state = env.step(state.env_state, env_actions)
        next_obs = next_env_state.obs
        dones = next_env_state.done

        # Sanitize obs to remove potential NaNs
        next_obs = jnp.nan_to_num(next_obs, nan=0.0, posinf=1e6, neginf=-1e6)
        next_obs = jnp.clip(next_obs, -1e6, 1e6)

        # 3. Compute DIAYN Rewards
        diayn_rewards = compute_reward(
            state.disc_state.params, obs=next_obs, skill_ids=state.env_skills
        )

        # Zero rewards for terminal transitions
        diayn_rewards = diayn_rewards * (1.0 - dones)

        # 4. Add transitions to replay buffer
        new_replay_buf = buffer_add_batch(
            state.replay_buf,
            state.obs,
            actions,
            diayn_rewards,
            next_obs,
            dones.astype(jnp.float32),
            state.env_skills,
            state.env_episode_steps,
        )

        # 5. Update networks after warmup
        key, key_updates = jax.random.split(key)

        should_update = (state.step >= warmup_steps) & (
            new_replay_buf.size >= batch_size
        )

        update_inputs = (
            state.policy_state,
            state.critic_state,
            state.disc_state,
            state.target_critic_params,
            new_replay_buf,
            key_updates,
        )

        def skip_updates(inputs):
            policy_state, critic_state, disc_state, target_params, _, _ = inputs
            return policy_state, critic_state, disc_state, target_params, 0.0, 0.0, 0.0

        def do_updates(inputs):
            # Perform multiple gradient updates (UTD ratio)
            (
                policy_state,
                critic_state,
                disc_state,
                target_params,
                replay_buf,
                update_key,
            ) = inputs

            def single_update(carry, _):
                policy_state, critic_state, disc_state, target_params, key = carry
                key, key_sample, key_critic, key_actor, key_disc = jax.random.split(
                    key, 5
                )

                batch = buffer_sample(replay_buf, key_sample, batch_size)
                (
                    batch_obs,
                    batch_act,
                    batch_rew,
                    batch_next,
                    batch_done,
                    batch_skill,
                    batch_ep_step,
                ) = batch

                batch_obs = jnp.nan_to_num(batch_obs, nan=0.0)
                batch_act = jnp.nan_to_num(batch_act, nan=0.0)
                batch_rew = jnp.nan_to_num(batch_rew, nan=0.0)
                batch_next = jnp.nan_to_num(batch_next, nan=0.0)

                batch_skill_onehot = jax.nn.one_hot(batch_skill, n_skills)

                critic_state, c_loss = do_update_critic(
                    critic_state,
                    target_params,
                    policy_state,
                    batch_obs,
                    batch_act,
                    batch_rew,
                    batch_next,
                    batch_done,
                    batch_skill_onehot,
                    key=key_critic,
                )

                policy_state, a_loss = do_update_actor(
                    policy_state,
                    critic_state,
                    batch_obs,
                    batch_skill_onehot,
                    key=key_actor,
                )

                def disc_update_step(disc_carry, _):
                    disc_state, d_key = disc_carry
                    d_key, d_sample_key = jax.random.split(d_key)
                    d_batch = buffer_sample(replay_buf, d_sample_key, batch_size)
                    d_next = jnp.nan_to_num(d_batch[3], nan=0.0)
                    d_done = d_batch[4]
                    d_skill = d_batch[5]
                    d_ep_step = d_batch[6]
                    non_terminal_mask = 1.0 - d_done
                    disc_state, d_loss = do_update_disc(
                        disc_state,
                        d_next,
                        d_skill,
                        mask=non_terminal_mask,
                        episode_step=d_ep_step,
                    )
                    return (disc_state, d_key), d_loss

                # Update disc_utd_ratio number of times
                (disc_state, _), d_losses = jax.lax.scan(
                    disc_update_step,
                    (disc_state, key_disc),
                    None,
                    length=disc_utd_ratio,
                )
                d_loss = jnp.mean(d_losses)

                target_params = do_soft_update(target_params, critic_state.params)

                return (policy_state, critic_state, disc_state, target_params, key), (
                    c_loss,
                    a_loss,
                    d_loss,
                )

            # Update utd_ratio number of times
            init_carry = (
                policy_state,
                critic_state,
                disc_state,
                target_params,
                update_key,
            )
            (policy_state, critic_state, disc_state, target_params, _), losses = (
                jax.lax.scan(single_update, init_carry, None, length=utd_ratio)
            )

            c_loss, a_loss, d_loss = (
                jnp.mean(losses[0]),
                jnp.mean(losses[1]),
                jnp.mean(losses[2]),
            )
            return (
                policy_state,
                critic_state,
                disc_state,
                target_params,
                c_loss,
                a_loss,
                d_loss,
            )

        (
            policy_state,
            critic_state,
            disc_state,
            target_params,
            c_loss,
            a_loss,
            d_loss,
        ) = jax.lax.cond(
            should_update,
            do_updates,
            skip_updates,
            update_inputs,
        )

        # 6. Handle episode resets
        num_done = jnp.sum(dones)
        new_ep_rewards = state.env_ep_rewards + diayn_rewards
        new_ep_rewards = new_ep_rewards * (1 - dones)

        key, key_new_skills = jax.random.split(key)
        new_skills = jax.random.randint(key_new_skills, (num_envs,), 0, n_skills)
        env_skills = jnp.where(dones, new_skills, state.env_skills)

        new_episode_steps = state.env_episode_steps + 1
        new_episode_steps = jnp.where(dones, 0, new_episode_steps)

        # 7. Build new state and metrics
        new_state = DIAYNTrainingState(
            policy_state=policy_state,
            critic_state=critic_state,
            disc_state=disc_state,
            target_critic_params=target_params,
            replay_buf=new_replay_buf,
            env_state=next_env_state,
            obs=next_obs,
            env_skills=env_skills,
            env_ep_rewards=new_ep_rewards,
            env_episode_steps=new_episode_steps,
            key=key,
            step=state.step + num_envs,
            episode_count=state.episode_count + num_done.astype(jnp.int32),
        )

        metrics = Metrics(
            critic_loss=c_loss,
            actor_loss=a_loss,
            disc_loss=d_loss,
            avg_step_reward=jnp.mean(diayn_rewards),
            num_episodes=num_done,
        )

        return new_state, metrics

    return jax.jit(training_step)
