from gym.envs.registration import register
register(
    id = 'ur5_env-v0',
    entry_point = 'env_gym.env:grasp'
)