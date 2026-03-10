import jax
import jax.numpy as jnp
from brax import envs as brax_envs

def make_parallel_eval(
    env_name: str,
    policy_net,
    disc_net,
    n_skills: int,
    n_eval_episodes: int,
    episode_len: int,
    action_scale: float,
):
    total_evals = n_skills * n_eval_episodes
    eval_env = brax_envs.create(env_name, auto_reset=True, batch_size=total_evals)

    def parallel_eval(policy_params, disc_params, key):
        skill_ids = jnp.repeat(jnp.arange(n_skills), n_eval_episodes)
        skill_onehots = jax.nn.one_hot(skill_ids, n_skills)

        key, key_reset = jax.random.split(key)
        env_state = eval_env.reset(key_reset)

        def step_fn(carry, _):
            env_state, key, total_rewards, correct_counts = carry
            key, key_act = jax.random.split(key)
            obs = env_state.obs

            # Mean for deterministic actions during eval
            _, _, mean = policy_net.apply(policy_params, obs, skill_onehots, key_act)
            actions = jnp.tanh(mean) * action_scale

            next_env_state = eval_env.step(env_state, actions)
            next_obs = next_env_state.obs
            rewards = next_env_state.reward

            # Check discriminator predictions
            logits = disc_net.apply(disc_params, next_obs)
            pred_skills = jnp.argmax(logits, axis=-1)
            is_correct = (pred_skills == skill_ids).astype(jnp.float32)

            return (
                next_env_state,
                key,
                total_rewards + rewards,
                correct_counts + is_correct,
            ), None

        init_carry = (env_state, key, jnp.zeros(total_evals), jnp.zeros(total_evals))
        (_, _, total_rewards, correct_counts), _ = jax.lax.scan(
            step_fn, init_carry, None, length=episode_len
        )

        accuracies = correct_counts / episode_len
        rewards_per_skill = total_rewards.reshape(n_skills, n_eval_episodes).mean(
            axis=1
        )
        accuracy_per_skill = accuracies.reshape(n_skills, n_eval_episodes).mean(axis=1)

        return rewards_per_skill, accuracy_per_skill

    return jax.jit(parallel_eval)


def evaluate_skills(
    env_name,
    policy_net,
    policy_params,
    disc_net,
    disc_params,
    n_skills,
    key,
    n_eval_episodes=5,
    episode_len=1000,
    action_scale=1.0,
):
    parallel_eval_fn = make_parallel_eval(
        env_name,
        policy_net,
        disc_net,
        n_skills,
        n_eval_episodes,
        episode_len,
        action_scale,
    )

    rewards_per_skill, accuracy_per_skill = parallel_eval_fn(
        policy_params, disc_params, key
    )
    jax.block_until_ready(rewards_per_skill)

    results = {}
    for z in range(n_skills):
        results[z] = {
            "env_reward": float(rewards_per_skill[z]),
            "disc_accuracy": float(accuracy_per_skill[z]),
        }
        print(
            f"  Skill {z:>2d}: "
            f"Env Reward = {results[z]['env_reward']:>8.1f} | "
            f"Disc Accuracy = {results[z]['disc_accuracy']:.2%}"
        )

    avg_accuracy = float(accuracy_per_skill.mean())
    print(f"\n  Overall Discriminator Accuracy: {avg_accuracy:.2%}")
    print(f"  (Random baseline: {1/n_skills:.2%})")

    return results