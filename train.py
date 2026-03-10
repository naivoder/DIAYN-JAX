from flax.training.train_state import TrainState
from typing import NamedTuple, Any, Tuple
from functools import partial
import jax
import jax.numpy as jnp
import optax
from buffer import ReplayBufferState, buffer_add_batch, buffer_sample
from networks import compute_diayn_reward, update_actor, update_critic, update_discriminator, soft_update

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
    env_ep_steps: jnp.ndarray  # current step in each env
    key: jnp.ndarray  # PRNG key
    step: jnp.ndarray  # global step count
    episode_count: jnp.ndarray  # total episodes completed


class Metrics(NamedTuple):
    critic_loss: jnp.ndarray
    actor_loss: jnp.ndarray
    disc_loss: jnp.ndarray
    avg_step_reward: jnp.ndarray  # average DIAYN reward per env step
    num_episodes: jnp.ndarray

def create_train_state(model, key, *input_shapes_and_args, lr=3e-4, max_grad_norm=1.0):
    params = model.init(key, *input_shapes_and_args)
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr))
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

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
            state.replay_buf_state,
            state.obs,
            actions,
            diayn_rewards,
            next_obs,
            dones.astype(jnp.float32),
            state.env_skills,
            state.env_ep_steps,
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

        new_episode_steps = state.env_ep_steps + 1
        new_episode_steps = jnp.where(dones, 0, new_episode_steps)

        # 7. Build new state and metrics
        new_state = DIAYNTrainingState(
            policy_state=policy_state,
            critic_state=critic_state,
            disc_state=disc_state,
            target_critic_params=target_params,
            replay_buf_state=new_replay_buf,
            env_state=next_env_state,
            obs=next_obs,
            env_skills=env_skills,
            env_ep_rewards=new_ep_rewards,
            env_ep_steps=new_episode_steps,
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


def make_full_training(
    training_step_fn, total_iterations: int, log_every: int, warmup_steps: int
):
    def full_training(state: DIAYNTrainingState) -> Tuple[DIAYNTrainingState, Metrics]:
        def scan_fn(carry, iteration):
            state, running_metrics = carry
            new_state, metrics = training_step_fn(state)

            safe_critic = jnp.where(
                jnp.isnan(metrics.critic_loss), 0.0, metrics.critic_loss
            )
            safe_actor = jnp.where(
                jnp.isnan(metrics.actor_loss), 0.0, metrics.actor_loss
            )
            safe_disc = jnp.where(jnp.isnan(metrics.disc_loss), 0.0, metrics.disc_loss)
            safe_reward = jnp.where(
                jnp.isnan(metrics.avg_step_reward), 0.0, metrics.avg_step_reward
            )

            new_running = Metrics(
                critic_loss=running_metrics.critic_loss + safe_critic,
                actor_loss=running_metrics.actor_loss + safe_actor,
                disc_loss=running_metrics.disc_loss + safe_disc,
                avg_step_reward=running_metrics.avg_step_reward + safe_reward,
                num_episodes=running_metrics.num_episodes + metrics.num_episodes,
            )

            def do_log(_):
                jax.debug.print(
                    "  Step {step:>7d} | Ep {ep:>4d} | Reward/Step: {reward:>+6.3f} | "
                    "Disc Loss: {disc:.4f} | Critic Loss: {crit:.4f}",
                    step=new_state.step,
                    ep=new_state.episode_count,
                    reward=metrics.avg_step_reward,
                    disc=metrics.disc_loss,
                    crit=metrics.critic_loss,
                )

            def no_log(_):
                pass

            should_log = (
                (new_state.step >= warmup_steps)
                & (iteration % log_every == 0)
                & (iteration > 0)
            )
            jax.lax.cond(should_log, do_log, no_log, None)

            return (new_state, new_running), metrics

        init_metrics = Metrics(
            critic_loss=0.0,
            actor_loss=0.0,
            disc_loss=0.0,
            avg_step_reward=0.0,
            num_episodes=0.0,
        )

        (final_state, final_running), all_metrics = jax.lax.scan(
            scan_fn, (state, init_metrics), jnp.arange(total_iterations)
        )

        aggregated = Metrics(
            critic_loss=final_running.critic_loss / total_iterations,
            actor_loss=final_running.actor_loss / total_iterations,
            disc_loss=final_running.disc_loss / total_iterations,
            avg_step_reward=final_running.avg_step_reward / total_iterations,
            num_episodes=final_running.num_episodes,
        )

        return final_state, aggregated

    return jax.jit(full_training)