import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from torch import nn
from collections import deque
from BC_only import BC_Dataset,EarlyStopping
from Environment_Agent import Environment, StudentModel, Critic
import torch.nn.functional as F


class MyCombinedLoss(nn.Module):
    def __init__(self, bc_weight, actor_weight, critic_weight, L2_weight):
        super().__init__()
        self.BC_weight = bc_weight
        self.actor_weight = actor_weight
        self.critic_weight = critic_weight
        self.L2_weight = L2_weight

    def forward(self, predict_action_actor, current_q_value, R1, batch_action,
                q_value_for_predicted_actions, loss_L2_critic, loss_L2_actor):
        # 1. Behavior Cloning Loss
        bc_loss = 0.5 * F.mse_loss(predict_action_actor, batch_action)
        # 2. Q-learning Loss
        q_loss = 0.5 * F.mse_loss(q_value_for_predicted_actions, R1)
        # 3. Actor Loss
        actor_loss = -torch.mean(q_value_for_predicted_actions)
        # 4. Total Loss
        combined_loss = (self.BC_weight * bc_loss +
                         self.critic_weight * q_loss +
                         self.actor_weight * actor_loss +
                         self.L2_weight * (loss_L2_actor + loss_L2_critic))
        return combined_loss


class Train_Update:
    def __init__(self, actor_model, actor_optimizer, actor_target,
                 critic_model, critic_optimizer, critic_target,
                 batch, expert_rate, bc_weight, actor_weight, critic_weight, L2_weight,
                 Tau_soft_update,GAMMA):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.actor_target = actor_target
        self.critic_target = critic_target
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.batch = batch
        self.expert_nums = int(expert_rate * batch)
        self.agent_nums = batch - self.expert_nums
        self.combine_loss = MyCombinedLoss(bc_weight, actor_weight, critic_weight, L2_weight)
        self.Tau = Tau_soft_update
        self.GAMMA = GAMMA
    def soft_update(self, target_net, main_net):
        """Cập nhật trọng số của target network bằng soft update."""
        for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
            target_param.data.copy_(self.Tau * main_param.data + (1.0 - self.Tau) * target_param.data)

    def __call__(self, expert_relay, agent_relay, pretrain=True):
        if pretrain or len(agent_relay) < self.agent_nums:
            transitions = random.sample(expert_relay, self.batch)
        else:
            transitions = random.sample(expert_relay, self.expert_nums)
            transitions.extend(random.sample(agent_relay, self.agent_nums))

        # Chuyển đổi sang Tensor
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)
        batch_state = torch.tensor(np.array(batch_state), dtype=torch.float32).to(device)
        # Sử dụng dtype torch.float32 cho action nếu không gian hành động liên tục
        batch_action = F.one_hot(torch.tensor(batch_action, dtype=torch.int64).to(device), num_classes=2).float()

        batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(-1).to(device)
        batch_next_state = torch.tensor(np.array(batch_next_state), dtype=torch.float32).to(device)
        #batch_done = torch.tensor(batch_done, dtype=torch.bool).unsqueeze(-1).to(device)

        # Tính toán Q-value cho hành động được dự đoán bởi Actor
        predicted_actions_actor = self.actor_model(batch_state)
        q_value_for_predicted_actions = self.critic_model(batch_state, predicted_actions_actor)
        # Tính toán Q-value hiện tại từ batch dữ liệu
        current_q_value = self.critic_model(batch_state, batch_action)
        # Tính toán Target Q-value (R1)
        with torch.no_grad():
            next_actions = self.actor_target(batch_next_state)
            next_q_values = self.critic_target(batch_next_state, next_actions)
            R1 = batch_reward + self.GAMMA * next_q_values

        # Tính toán L2 Regularization Loss
        loss_L2_actor = sum(p.pow(2.0).sum() for p in self.actor_model.parameters())
        loss_L2_critic = sum(p.pow(2.0).sum() for p in self.critic_model.parameters())

        # Tính toán total loss
        total_loss = self.combine_loss(predict_action_actor=predicted_actions_actor,
                                       current_q_value=current_q_value,R1=R1,
            batch_action=batch_action,
            q_value_for_predicted_actions=q_value_for_predicted_actions,
            loss_L2_critic=loss_L2_critic,
            loss_L2_actor=loss_L2_actor)

        # Zero_grad cho cả hai optimizer trước khi gọi backward
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # Thực hiện backward cho hàm loss tổng hợp
        total_loss.backward()

        # Cập nhật trọng số của cả hai mạng
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Cập nhật soft update cho các mạng mục tiêu
        self.soft_update(self.actor_target, self.actor_model)
        self.soft_update(self.critic_target, self.critic_model)
        return total_loss

# Khởi tạo thông số
L = 2000 #pre-training steps
T = 10000 #training steps
M = 500 #data collection steps
N = 64 #batch size
LEARNING_RATE_ACTOR = 0.001
LEARNING_RATE_CRITIC= 0.001
EXPERT_BUFFER_SIZE = 500000
AGENT_BUFFER_SIZE = 1000000
# Đường link dữ liệu trên Huggingface
link = "hf://datasets/NathanGavenski/CartPole-v1/teacher.jsonl"
#Xác định cấu hình phần cứng
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Khởi tạo môi trường
ENV_NAME = 'CartPole-v1'
REN_MODE = "rgb_array" #Human or rgb_array
env = Environment(env_name=ENV_NAME,render_mode=REN_MODE,low_bounder=-0.2,up_bounder=0.2)
env_test = Environment(env_name=ENV_NAME,render_mode=REN_MODE,low_bounder=-0.2,up_bounder=0.2)

# Trích xuất tập dữ liệu chuyên gia
expert_data = BC_Dataset(file_path=link,sample_rate=0.1) #[state,action,reward,next_state]
for exp in expert_data.list:
    exp[3] = env.next_state(state=exp[0],action=exp[1])


# Khởi tạo actor và critic
actor = StudentModel(no_of_obs=env.state_size,no_of_action=env.action_size,drop_out=0.5)
target_actor = StudentModel(no_of_obs=env.state_size,no_of_action=env.action_size,drop_out=0.5)
target_actor.load_state_dict(actor.state_dict())
optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE_ACTOR)

critic = Critic(no_of_obs=env.state_size,no_of_action=env.action_size,drop_out=0.5)
target_critic = Critic(no_of_obs=env.state_size,no_of_action=env.action_size,drop_out=0.5)
target_critic.load_state_dict(critic.state_dict())
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE_CRITIC)

# Khởi tạo bộ nhớ đệm actor, critic
memory_actor = deque(list(expert_data.list),maxlen=EXPERT_BUFFER_SIZE)
memory_critic = deque(maxlen=AGENT_BUFFER_SIZE)

train = Train_Update(actor_model=actor,actor_optimizer=optimizer_actor,actor_target=target_actor,
                     critic_model=critic,critic_optimizer=optimizer_critic,critic_target=target_critic,
                     batch=N,expert_rate=0.25,bc_weight=1,actor_weight=1,critic_weight=1,L2_weight=1e-4,Tau_soft_update=0.001,GAMMA=0.99)
early_stopper = EarlyStopping(patience=5,mode="maintain")
avg_reward =[]
timestep = 0
best_pretrain = 0
# Pretrain
for i in range(L):
    train(expert_relay=memory_actor,agent_relay=memory_critic,pretrain=True)
    timestep += 1
    if i % 10 ==0:
        _,agv,_,_ = env_test.simulate_agent(actor,num_episodes=100)
        avg_reward.append([timestep, agv, 0])
        if agv > best_pretrain:
            best_pretrain = agv
            best_weights = actor.state_dict()
        if i % 100 ==0:
            print(f"Pretrain: Timestep {i}, Avg_reward: {agv}")

Out_training = False
actor.load_state_dict(best_weights)
for i in range(T):
    state = env.reset()
    episode_reward = 0
    for j in range(M):
        with torch.no_grad():
            actor.eval()
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action = actor(state_tensor).argmax().item()
            next_state, reward, terminated, truncated, info = env.step(action)
            memory_critic.append([state,action,reward,next_state])
            episode_reward += reward
            state = next_state
            timestep += 1
            if terminated or truncated:
                state = env.reset()
                break
        train(expert_relay=memory_actor, agent_relay=memory_critic, pretrain=False)
        if timestep % 10 == 0:
            _, agv, _, _ = env_test.simulate_agent(actor, num_episodes=100)
            avg_reward.append([timestep, agv, episode_reward])
            if timestep % 100 == 0:
                print(f"Train: Timestep {timestep}, Avg_reward: {agv}")
        if early_stopper(val_loss=agv, model=actor, maintaince_score=500):
            Out_training = True
            print(f"Mô hình đã đạt hiệu suất yêu cầu ở Timestep {timestep},episode {L+i}")
            break
    if Out_training:
        break

# Lưu các thông số theo dõi
df  = pd.DataFrame(data = avg_reward,columns = ["Timestep", "Average Reward","Episode Reward"])
episodes = df["Timestep"].tolist()
avg_rewards = df["Average Reward"].tolist()
eps_rewards = df["Episode Reward"].tolist()

# Lưu model
torch.save(early_stopper.best_weights, f'RL_BC_only_{ENV_NAME}.pth')
print("Mô hình đã được lưu thành công!")

#Lưu tham số theo dõi
df.to_csv(f'RL_BC_training_{ENV_NAME}.csv', index=False)

# Vẽ biểu đồ
plt.plot(episodes, avg_rewards, label='Average Reward', color='blue')
plt.title('Learning Curve: Average Reward over Timestep')
plt.xlabel('Timestep')
plt.ylabel('Average Reward')
plt.grid(True)
plt.legend()
plt.savefig("RL_BC Average Reward")
plt.show(block=False)
plt.close()

plt.plot(episodes, eps_rewards, label='Episode Reward', color='blue')
plt.title('Learning Curve: Episode Reward')
plt.xlabel('Episodes')
plt.ylabel('Total Episode Reward')
plt.grid(True)
plt.legend()
plt.savefig("RL_BC Episode Reward")
plt.show(block=False)
plt.close()

env.close()
