from stable_baselines3.common.env_checker import check_env

from ..rcjs_environment import Environment

env = Environment()

check_env(env)

