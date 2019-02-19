from gym.envs.registration import register

register(
    id='PegInsertion-v0',
    entry_point='gps.envs.peg_insertion:PegInsertionEnv',
    max_episode_steps=1000,
)
