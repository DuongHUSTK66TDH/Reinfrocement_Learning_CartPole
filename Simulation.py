import gymnasium as gym
import torch
import numpy as np
from Environment_Agent import StudentModel,Environment
from pkg_resources import resource_stream, resource_exists

#Environment init
FILE_MODEL = "RL_only result\\LR=0.001\\RL_CartPole-v1.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
env_test = Environment(env_name="CartPole-v1",render_mode="rgb_array",low_bounder=-0.2,up_bounder=0.2)
model_test = StudentModel(no_of_obs=env_test.state_size,no_of_action=env_test.action_size,drop_out=0.5)
model_test_state_dic = torch.load(FILE_MODEL)
model_test.load_state_dict(model_test_state_dic)
model_test.eval()
res,avg_res,goal,fail=env_test.simulate_agent(model=model_test,num_episodes=1000)
print(avg_res,goal,fail)
#env_test.video_simulation(agent_model=model_test,model_name="RL only",video_path="RL_only result\\LR=0.001\\RL_Video_Simulation.mp4")



env_test.close()