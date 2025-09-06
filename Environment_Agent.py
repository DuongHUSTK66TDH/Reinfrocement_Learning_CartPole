import torch
import numpy as np
from torch import nn
import gymnasium as gym
import cv2
import os
from gymnasium.wrappers import RecordVideo


# Hàm để thêm văn bản lên một frame bằng OpenCV
def add_text_on_frame(frame, text, position=(10, 30), font_scale=0.7, color=(0, 0, 0)):
    frame = np.array(frame)
    return cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

class Environment:
    def __init__(self,env_name,render_mode="rgb_array",low_bounder = -0.05, up_bounder = 0.05):
        self.env = gym.make(env_name,render_mode=render_mode)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.low = low_bounder
        self.up = up_bounder

    def reset(self):
        return self.env.reset(options={"low": self.low, "high": self.up})[0]

    def next_state(self, state, action):
        self.env.reset()
        self.env.unwrapped.state = np.array(state, dtype=np.float32)
        next, reward, terminated, truncated, _ = self.env.step(action)
        return next.tolist()

    def step(self,action):
        return self.env.step(action)
    def simulate_agent(self,model,device="cpu", num_episodes=1,video_mode = False):
        """
        Chạy mô phỏng agent trong một môi trường nhất định.
        Args:
            model: Mạng DQN đã được huấn luyện.
            env: Môi trường có chế độ hiển thị "human".
            num_episodes: Số lượng episode muốn mô phỏng.
        """
        res_reward = []
        fail_goal = 0
        model.eval()
        for i in range(num_episodes):
            state = self.env.reset()[0]
            truncated = False
            terminated = False
            total_reward = 0
            frame_episode = []
            while not truncated and not terminated:
                # env.render() # Lệnh này không cần thiết nếu đã có render_mode="human"
                # Chọn hành động bằng mô hình (không cần exploration)
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    action = model(state_tensor).argmax().item()

                # Thực hiện hành động
                next_state, reward, terminated, truncated , info = self.env.step(action)
                state = next_state
                total_reward += reward
                if truncated or terminated:
                    break
            if total_reward < 500:
                fail_goal = fail_goal + 1
            res_reward.append(total_reward)
        return res_reward, np.mean(res_reward,dtype=float), fail_goal, float(fail_goal/num_episodes)

    def close(self):
        self.env.close()

    def select_action(self,state, epsilon,device,policy_net):
        if np.random.random_sample() < epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values = policy_net(state)
                return q_values.argmax().item()

    ## Hàm ghi lại video của một episode bằng video writer riêng
    def video_simulation(self,agent_model,model_name, video_path="videos/cartpole_run.mp4",device="cpu"):
        # Tạo thư mục nếu nó chưa tồn tại
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        # Reset môi trường
        state, info = self.env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        # Lấy kích thước khung hình và FPS từ môi trường
        frame = self.env.render()
        height, width, _ = frame.shape
        fps = self.env.metadata.get('render_fps') or 30  # Mặc định là 30 FPS

        # Khởi tạo VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec cho file mp4
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        total_reward = 0
        try:
            while True:
                # 1. Chọn hành động từ mô hình
                with torch.no_grad():
                    logits = agent_model(state_tensor)
                    action = torch.argmax(logits, dim=1).item()

                # 2. Thực hiện bước trong môi trường
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                #3. Cập nhật trạng thái và phần thưởng
                state = next_state
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                total_reward += reward

                # 4. Lấy frame từ môi trường
                frame = self.env.render()

                # 5. Thêm chú thích vào frame
                comment = f"Reward: {total_reward} | {model_name}"
                frame_with_text = add_text_on_frame(frame, comment)

                # 6. Ghi frame vào file video
                out.write(frame_with_text)

                if terminated or truncated:
                    break
        finally:
            # 6. Đóng VideoWriter và môi trường
            out.release()
            self.env.close()

        print(f"Video đã được lưu vào: {video_path}")
        print(f"Tổng phần thưởng (reward) trong episode này là: {total_reward}")

class StudentModel(nn.Module):
    def __init__(self,no_of_obs,no_of_action,drop_out=0.5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(no_of_obs, 32),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(16, no_of_action),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Critic(nn.Module):
    def __init__(self,no_of_obs,no_of_action,drop_out=0.5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(no_of_obs + no_of_action, 32),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(16, 1)
        )


    def forward(self, state,action):
        x = torch.cat([state, action], 1)
        logits = self.linear_relu_stack(x)
        return logits