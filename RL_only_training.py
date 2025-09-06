import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import deque
import random
from tqdm import tqdm

from RL_only import soft_update,optimize_model
from Environment_Agent import StudentModel,Environment
from BC_only import EarlyStopping
# Tham số
ENV_NAME = 'CartPole-v1'
REN_MODE = "rgb_array" #Human or rgb_array
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.99
LR = 0.001
BUFFER_SIZE = 1000000
NO_OF_EPISODE = 10000
BEST_REWARD = -999999
LOWER_POS = -1.2
UPPER_POS = -0.8
TAU = 0.001
# Khai bao cac bien theo doi
train_reward = []
avg_reward = 0
rate_fail = 0
# Khởi tạo môi trường
env_train = Environment(ENV_NAME,low_bounder=-0.2,up_bounder=0.2)
env_eval = Environment(ENV_NAME,render_mode=REN_MODE,low_bounder=-0.2,up_bounder=0.2)
state_size = env_train.state_size
action_size = env_train.action_size

# Khởi tạo mô hình và bộ tối ưu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = StudentModel(no_of_obs=state_size,no_of_action=action_size,drop_out=0.5).to(device)
target_net = StudentModel(no_of_obs=state_size,no_of_action=action_size,drop_out=0.5).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
policy_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=BUFFER_SIZE)
progress = tqdm(total = NO_OF_EPISODE, desc=f"Episode", unit="sample", ncols=100,leave=True)
#Khởi tạo EarlyStopper
early_stopper = EarlyStopping(patience=5,restore_best_weights=True,mode="maintain")
# Vòng lặp huấn luyện
epsilon = EPSILON_START
time_step = 0
for i_episode in range(NO_OF_EPISODE):  # Huấn luyện trong 1000 episode
    state = env_train.reset()
    total_reward = 0
    truncated = False
    terminated = False
    while not terminated and not truncated:
        time_step += 1
        action = env_train.select_action(state, epsilon,device,policy_net)
        next_state, reward, terminated, truncated , info = env_train.step(action)
        total_reward += reward
        memory.append((state, action, reward, next_state, terminated))
        state = next_state
        optimize_model(memory, BATCH_SIZE, policy_net=policy_net, target_net=target_net, GAMMA=GAMMA,
                       optimizer=optimizer)
        soft_update(target_net, policy_net, TAU)
        if truncated or terminated:
            break
    # Cập nhật epsilon và target network
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    #if i_episode % 10 == 0:
        #target_net.load_state_dict(policy_net.state_dict())
    _, avg_reward, _, rate_fail = env_eval.simulate_agent(policy_net, num_episodes=100)
    train_reward.append([i_episode,time_step,avg_reward,total_reward])
            # Kiểm tra điều kiện dừng sớm
    if early_stopper(val_loss=avg_reward, model=target_net, maintaince_score=500):
        break

    # Cập nhật thanh tiến độ
    progress.update(1)
    progress.set_postfix({"Avg_Reward": avg_reward,"Epsilon": epsilon,"Memory":len(memory)})

progress.close()

# Lưu trọng số mô hình
torch.save(early_stopper.best_weights, f'RL_{ENV_NAME}.pth')
print("Mô hình đã được lưu thành công!")

# Lưu biến theo dõi quá trình training
#print(train_reward)
df  = pd.DataFrame(data = train_reward,columns = ["Episode", "Time Step","Average Reward", "Episode Reward"])
df.to_csv(f"RL_Training_reward_{ENV_NAME}.txt")
env_train.close()
env_eval.close()
