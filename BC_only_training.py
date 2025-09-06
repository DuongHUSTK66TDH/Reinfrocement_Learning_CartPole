import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from BC_only import BC_Dataset,train_loop,test_loop,EarlyStopping
from Environment_Agent import StudentModel,Environment
from torch.utils.data import Dataset, DataLoader

# Đường link dữ liệu trên Huggingface
link = "hf://datasets/NathanGavenski/CartPole-v1/teacher.jsonl"

#Khởi tạo, định nghĩa thông số cơ bản của Agent và môi trường
ENV_NAME = 'CartPole-v1'
REN_MODE = "rgb_array" #Human or rgb_array
env = Environment(env_name=ENV_NAME,render_mode=REN_MODE)
NO_OF_OBS = env.state_size
NO_OF_ACTION = env.action_size

#Hyper parameter
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 50
DROP_OUT = 0.5
WEIGHT_DECAY = 1e-4
# XÁc định cấu hình training
device = "cuda" if torch.cuda.is_available() else "cpu"
# Khởi tạo dataset
train_dataset = BC_Dataset(file_path=link, is_train=True,sample_rate=0.005)
test_dataset = BC_Dataset(file_path=link, is_train=False,sample_rate=0.005)
print(len(train_dataset),len(test_dataset))
# Sử dụng DataLoader để tải dữ liệu theo từng batch
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Khởi tạo model mới
model = StudentModel(no_of_action=NO_OF_ACTION,no_of_obs=NO_OF_OBS,drop_out=DROP_OUT).to(device)
#Khởi tạo hàm lỗi
loss_fn = nn.CrossEntropyLoss()
#Khởi tạo bộ tối ưu
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
# Khởi tạo lịch trình giảm tỉ lệ học
scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
#Khởi tạo early_stopper
early_stopper = EarlyStopping(patience=5)
keep_track = [0,0,0,0,0,0]
obser = []
# Bắt đầu huấn luyện
for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer,keep_track=keep_track,batch_size=BATCH_SIZE,device=device)
    print(
        f"Train Error: \n Avg loss: {keep_track[0]}, Accuracy: {keep_track[1]}% ")
    test_loop(test_dataloader, model, loss_fn,env_test=env,num_episode=100,device=device,keep_track=keep_track)
    print(
        f"Test Error: \n Avg loss: {keep_track[2]}, Accuracy: {keep_track[3]}%, Avg reward:{keep_track[4]},Fail rate: {keep_track[5]}%")
    obser.append([t+1,keep_track[0],keep_track[1],keep_track[2],keep_track[3],keep_track[4],keep_track[5]])
    if early_stopper(val_loss=keep_track[2], model=model):
        break
    scheduler.step()

# Đóng môi trường giả lập
env.close()

# Lưu trọng số mô hình
torch.save(early_stopper.best_weights, f'BC_only_{ENV_NAME}.pth')
print("Mô hình đã được lưu thành công!")

# Lưu biến theo dõi quá trình training
df  = pd.DataFrame(data = obser,columns = ["Epoch", "Train_loss", "Train_acc","Val_loss","Val_acc","Val_reward","Val_fail_rate"])
df.to_csv(f"BC_training_{ENV_NAME}.txt")