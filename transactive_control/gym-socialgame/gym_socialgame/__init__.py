from gym.envs.registration import register

register(
    id='socialgame-v0',
    entry_point='gym_socialgame.envs:SocialGameEnv',
)

register(
    id='socialgame_hourly-v0',
    entry_point='gym_socialgame.envs:SocialGameEnvHourly',
)

register(
	id = "socialgame_planning-v0",
	entry_point = "gym_socialgame.envs:SocialGamePlanningEnv",
)

register(
    id='socialgame_dr-v0',
    entry_point='gym_socialgame.envs:SocialGameEnvDR',
)
