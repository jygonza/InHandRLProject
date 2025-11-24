from inhand_env import CanRotateEnv

env = CanRotateEnv(render_mode="human")
obs, info = env.reset()
print("fingers:", obs[:16])
print("xyz:", obs[16:19])
print("rpy:", obs[19:22])
print("distance:", obs[22])