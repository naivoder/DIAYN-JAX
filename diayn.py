import jax
import jax.numpy as jnp
from brax import envs as brax_envs
from networks import GaussianPolicy, TwinQ, Discriminator
from train import (
    create_train_state,
    make_training_step,
    make_full_training,
    DIAYNTrainingState,
)
from buffer import init_replay_buffer
from eval import evaluate_skills
from utils import save_policies


def run_diayn(
    env_name: str = "halfcheetah",
    n_skills: int = 50,
    total_steps: int = 100_000,
    batch_size: int = 128,
    utd_ratio: int = 1,
    disc_utd_ratio: int = 1,
    buffer_capacity: int = 1_000_000,
    gamma: float = 0.99,
    alpha: float = 0.1,
    tau: float = 0.01,
    learning_rate: float = 3e-4,
    warmup_steps: int = 1000,
    action_scale: float = 1.0,
    seed: int = 1,
    log_interval: int = 10000,
    hidden_dim: int = 300,
    num_envs: int = 1,
    eval_episode_len: int = 1000,
    skip_initial_steps: int = 10,
):
    print(f"{'='*70}")
    print(f"  DIAYN Training on {env_name}")
    print(f"  Skills: {n_skills} | Steps: {total_steps} | Envs: {num_envs}")
    print(f"{'='*70}")

    env = brax_envs.create(env_name, auto_reset=True, batch_size=num_envs)
    obs_dim = env.observation_size
    act_dim = env.action_size
    print(f"  Obs dim: {obs_dim} | Act dim: {act_dim}")

    key = jax.random.PRNGKey(seed)
    key, key_policy, key_critic, key_disc = jax.random.split(key, 4)

    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_skill = jnp.zeros((1, n_skills))
    dummy_action = jnp.zeros((1, act_dim))
    dummy_key = jax.random.PRNGKey(0)

    print(f"  Compiling networks and training functions...", flush=True)

    policy_net = GaussianPolicy(hidden_dim=hidden_dim, action_dim=act_dim)
    policy_state = create_train_state(
        policy_net, key_policy, dummy_obs, dummy_skill, dummy_key, lr=learning_rate
    )

    critic_net = TwinQ(hidden_dim=hidden_dim)
    critic_state = create_train_state(
        critic_net, key_critic, dummy_obs, dummy_skill, dummy_action, lr=learning_rate
    )
    target_critic_params = critic_state.params

    disc_net = Discriminator(hidden_dim=hidden_dim, n_skills=n_skills)
    disc_state = create_train_state(disc_net, key_disc, dummy_obs, lr=learning_rate)

    replay_buf = init_replay_buffer(buffer_capacity, obs_dim, act_dim)

    key, key_reset, key_skills = jax.random.split(key, 3)
    env_state = env.reset(key_reset)
    env_skills = jax.random.randint(key_skills, (num_envs,), 0, n_skills)

    training_step_fn = make_training_step(
        env=env,
        policy_net=policy_net,
        critic_net=critic_net,
        disc_net=disc_net,
        n_skills=n_skills,
        num_envs=num_envs,
        batch_size=batch_size,
        gamma=gamma,
        alpha=alpha,
        tau=tau,
        action_scale=action_scale,
        warmup_steps=warmup_steps,
        utd_ratio=utd_ratio,
        disc_utd_ratio=disc_utd_ratio,
        skip_initial_steps=skip_initial_steps,
    )

    total_iterations = total_steps // num_envs
    log_every = log_interval // num_envs

    full_training_fn = make_full_training(
        training_step_fn, total_iterations, log_every, warmup_steps
    )

    state = DIAYNTrainingState(
        policy_state=policy_state,
        critic_state=critic_state,
        disc_state=disc_state,
        target_critic_params=target_critic_params,
        replay_buf_state=replay_buf,
        env_state=env_state,
        obs=env_state.obs,
        env_skills=env_skills,
        env_ep_rewards=jnp.zeros(num_envs),
        env_ep_steps=jnp.zeros(num_envs, dtype=jnp.int32),
        key=key,
        step=jnp.array(0, dtype=jnp.int32),
        episode_count=jnp.array(0, dtype=jnp.int32),
    )

    print(
        f"  Running {total_iterations} iterations ({total_steps} env steps)...",
        flush=True,
    )

    state, metrics = full_training_fn(state)
    jax.block_until_ready(state.obs)

    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Total steps: {int(state.step)} | Episodes: {int(state.episode_count)}")
    print(f"  Avg Reward/Step: {float(metrics.avg_step_reward):+.4f}")
    print(f"{'='*70}")

    print(f"\n  Evaluating {n_skills} learned skills...\n")
    eval_results = evaluate_skills(
        env_name,
        policy_net,
        state.policy_state.params,
        disc_net,
        state.disc_state.params,
        n_skills,
        state.key,
        action_scale=action_scale,
        episode_len=eval_episode_len,
    )

    config = {
        "env_name": env_name,
        "n_skills": n_skills,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "gamma": gamma,
        "alpha": alpha,
        "tau": tau,
        "learning_rate": learning_rate,
        "action_scale": action_scale,
        "seed": seed,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
    }
    save_dir = save_policies(state, env_name, config)

    return {
        "policy_state": state.policy_state,
        "critic_state": state.critic_state,
        "disc_state": state.disc_state,
        "eval_results": eval_results,
        "save_dir": save_dir,
    }


if __name__ == "__main__":
    import warnings
    import logging

    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)

    results = run_diayn(
        env_name="halfcheetah",
        n_skills=50,
        total_steps=1_000_000,
        batch_size=512,
        utd_ratio=32,
        disc_utd_ratio=1,
        buffer_capacity=1_000_000,
        gamma=0.99,
        alpha=0.1,
        tau=0.01,
        learning_rate=3e-4,
        warmup_steps=10000,
        action_scale=1.0,
        seed=42,
        log_interval=50000,
        hidden_dim=300,
        num_envs=32,
        skip_initial_steps=10,
    )
