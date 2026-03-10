import os
import json 
from flax import serialization
import jax
import jax.numpy as jnp

def save_policies(state, env_name, config):
    save_dir = os.path.join("policies", env_name)
    os.makedirs(save_dir, exist_ok=True)

    policy_bytes = serialization.to_bytes(state.policy_state.params)
    with open(os.path.join(save_dir, "policy_params.bin"), "wb") as f:
        f.write(policy_bytes)

    disc_bytes = serialization.to_bytes(state.disc_state.params)
    with open(os.path.join(save_dir, "discriminator_params.bin"), "wb") as f:
        f.write(disc_bytes)

    critic_bytes = serialization.to_bytes(state.critic_state.params)
    with open(os.path.join(save_dir, "critic_params.bin"), "wb") as f:
        f.write(critic_bytes)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved trained policies to {save_dir}/")
    return save_dir


def load_policies(env_name, policy_net, disc_net, critic_net, key):
    load_dir = os.path.join("policies", env_name)

    with open(os.path.join(load_dir, "config.json"), "r") as f:
        config = json.load(f)

    obs_dim = config["obs_dim"]
    act_dim = config["act_dim"]
    n_skills = config["n_skills"]

    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_skill = jnp.zeros((1, n_skills))
    dummy_action = jnp.zeros((1, act_dim))
    dummy_key = jax.random.PRNGKey(0)

    key, k1, k2, k3 = jax.random.split(key, 4)
    policy_params_structure = policy_net.init(k1, dummy_obs, dummy_skill, dummy_key)
    disc_params_structure = disc_net.init(k2, dummy_obs)
    critic_params_structure = critic_net.init(k3, dummy_obs, dummy_skill, dummy_action)

    with open(os.path.join(load_dir, "policy_params.bin"), "rb") as f:
        policy_params = serialization.from_bytes(policy_params_structure, f.read())

    with open(os.path.join(load_dir, "discriminator_params.bin"), "rb") as f:
        disc_params = serialization.from_bytes(disc_params_structure, f.read())

    with open(os.path.join(load_dir, "critic_params.bin"), "rb") as f:
        critic_params = serialization.from_bytes(critic_params_structure, f.read())

    print(f"  Loaded policies from {load_dir}/")
    return policy_params, disc_params, critic_params, config