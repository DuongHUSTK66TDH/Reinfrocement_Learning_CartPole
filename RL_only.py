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

def soft_update(target_net, main_net, tau):
    """
    Cập nhật trọng số của target network bằng soft update.
    """
    for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

# Hàm tối ưu hóa mô hình
def optimize_model(memory,BATCH_SIZE,policy_net,target_net,GAMMA,optimizer,device="cpu"):
    if len(memory) < BATCH_SIZE:
        return
    transitions = random.sample(memory, BATCH_SIZE)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    # Chuyển đổi sang Tensor
    batch_state = torch.tensor(np.array(batch_state), dtype=torch.float32).to(device)
    batch_action = torch.tensor(batch_action, dtype=torch.int64).unsqueeze(-1).to(device)
    batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(-1).to(device)
    batch_next_state = torch.tensor(np.array(batch_next_state), dtype=torch.float32).to(device)
    batch_done = torch.tensor(batch_done, dtype=torch.bool).unsqueeze(-1).to(device)

    # Tính giá trị Q
    q_values = policy_net(batch_state).gather(1, batch_action)
    next_q_values = target_net(batch_next_state).max(1)[0].unsqueeze(1).detach()
    target_q_values = batch_reward + (GAMMA * next_q_values) * (~batch_done)

    # Tính mất mát và tối ưu
    loss = F.mse_loss(q_values, target_q_values)
    #print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




