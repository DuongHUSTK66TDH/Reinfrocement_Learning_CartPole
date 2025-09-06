import matplotlib.pyplot as plt
import pandas as pd
from sympy.printing.pretty.pretty_symbology import line_width

# Đường dẫn đến tệp của bạn
RL_FILE_PATH = "RL_only result\\LR=0.0001\\RL_Training_reward_CartPole-v1.txt"
RL_BC_FILE_PATH = "RL_BC_training_CartPole-v1.csv"
# Đọc tệp tin và tạo DataFrame
LR_df = pd.read_csv(RL_FILE_PATH)
LR_BC_df = pd.read_csv(RL_BC_FILE_PATH)

#Sử dụng khi dùng RL:
# Trục X: các episode
RL_timestep= LR_df["Time Step"].tolist()
# Trục Y: các giá trị phần thưởng
RL_avg_reward = LR_df["Average Reward"].tolist()
RL_episode_reward = LR_df["Episode Reward"].tolist()
#RL_episodes = LR_df["Epoch"].tolist()

#Sử dụng khi dùng RL_BC:
RL_BC_timestep = LR_BC_df["Timestep"].tolist()
RL_BC_reward = LR_BC_df["Average Reward"].tolist()

# Thêm nhãn và tiêu đề để biểu đồ rõ ràng hơn
plt.plot(RL_timestep,RL_avg_reward , linestyle='-',label="Average Reward",linewidth=2,color="red")
plt.plot(RL_timestep,RL_episode_reward  , linestyle='--',label="Episode Reward",linewidth=0.5,color="blue")
plt.title("Episode Reward")
plt.ylabel("Episode Reward")
plt.xlabel("Timestep")
plt.legend(loc='lower right')
plt.savefig("RL_0,001")
plt.show()
plt.close()

# # Sử dụng khi dùng BC:
# # Trục X: các epoch
# epochs = df["Epoch"].tolist()
# # Trục Y: các tiêu chí theo dõi
# train_loss = df["Train_loss"].tolist()
# train_acc = df["Train_acc"].tolist()
# val_loss = df["Val_loss"].tolist()
# val_acc = df["Val_acc"].tolist()
# avg_reward = df["Val_reward"].tolist()
#
# plt.plot(epochs, train_acc, color="blue", label="train")
# plt.plot(epochs, val_acc, color="orange", label="valid")
# plt.title("Training curve - Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend()
# plt.savefig("BC_only Training Curve - Accuracy")
# plt.show()
# plt.close()