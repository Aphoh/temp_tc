from gym.envs.registration import register

register(
    id='microgrid-v0',
    entry_point='gym_microgrid.envs:MicrogridEnv',
)

register(
    id='microgrid-rllib-v0',
    entry_point='gym_microgrid.envs:MicrogridEnvRLLib',
)

register(
    id='microgrid-counterfactual-v0',
    entry_point='gym_microgrid.envs:CounterfactualMicrogridEnvRLLib',
)

# register(
#     id='socialgame_hourly-v0',
#     entry_point='gym_socialgame.envs:SocialGameEnvHourly',
# )

# register(
#     id='socialgame_monthly-v0',
#     entry_point='gym_socialgame.envs:SocialGameEnvMonthly',
# )

# register(
# 	id = "socialgame_planning-v0",
# 	entry_point = "gym_socialgame.envs:SocialGamePlanningEnv")