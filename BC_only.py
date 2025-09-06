import pandas as pd
import huggingface_hub
import torch
import numpy as np
import math
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from Environment_Agent import Environment


class ObservationTransform:
    def __call__(self,data):
        return torch.tensor(data, dtype=torch.float32)

class LabelTransform:
    def __init__(self,no_of_label):
        self.no_of_label = no_of_label

    def __call__(self,y):
        return torch.zeros(self.no_of_label, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)


class BC_Dataset(Dataset):
    def __init__(self, file_path, test_size=0.3, seed=123, is_train=True,sample_rate = 1):
        self.data_set = pd.read_json(file_path, lines=True)

        if sample_rate != 1:
            self.data_set = self.data_set.sample(frac=sample_rate,random_state = 123)
        self.list = self.data_set.values.tolist()
        # Trích xuất dữ liệu bằng tên cột thay vì chỉ mục để an toàn hơn
        self.input = self.data_set['obs'].apply(lambda x: np.array(x, dtype=np.float32))
        self.label = self.data_set['actions']

        # Chia dữ liệu thành tập huấn luyện và kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(
            self.input, self.label, test_size=test_size, random_state=seed, shuffle=True,stratify=self.label)

        if is_train:
            self.features = X_train
            self.labels = y_train
        else:
            self.features = X_test
            self.labels = y_test

        self.transform = ObservationTransform()
        self.target_transform = LabelTransform(no_of_label=2)  # CartPole có 2 hành động

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features.iloc[idx]
        label = self.labels.iloc[idx]

        # Áp dụng các biến đổi
        feature_tensor = self.transform(feature)
        label_tensor = self.target_transform(label)

        return feature_tensor, label_tensor

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True,mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.patience_counter = 0
        self.best_weights = None
        self.mode = mode
    def __call__(self, val_loss, model,maintaince_score = 0):
        match self.mode:
            case "min":
                if self.best_loss is None:
                    self.best_loss = val_loss
                    self.best_weights = model.state_dict()
                elif val_loss < self.best_loss - self.min_delta:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                    self.best_weights = model.state_dict()
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print("\nĐã đạt đủ kiên nhẫn. Dừng huấn luyện sớm.")
                        if self.restore_best_weights:
                            model.load_state_dict(self.best_weights)
                        return True
                return False
            case "max":
                if self.best_loss is None:
                    self.best_loss = val_loss
                    self.best_weights = model.state_dict()
                elif val_loss > self.best_loss + self.min_delta:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                    self.best_weights = model.state_dict()
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print("\nĐã đạt đủ kiên nhẫn. Dừng huấn luyện sớm.")
                        if self.restore_best_weights:
                            model.load_state_dict(self.best_weights)
                        return True
                return False
            case "maintain":
                if self.best_loss is None:
                    self.best_loss = maintaince_score
                    self.best_weights = model.state_dict()
                elif val_loss >= self.best_loss:
                    self.best_loss = val_loss
                    self.patience_counter +=1
                    self.best_weights = model.state_dict()
                    if self.patience_counter >= self.patience:
                        #print("\nĐã đạt đủ kiên nhẫn. Dừng huấn luyện sớm.")
                        if self.restore_best_weights:
                            model.load_state_dict(self.best_weights)
                        return True
                else:
                    self.patience_counter = 0
                return False

def train_loop(dataloader, model, loss_fn, optimizer,batch_size,keep_track,device="cpu"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    total_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        # Đẩy dữ liệu lên cấu hình phần cứng
        X,y = X.to(device),y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    keep_track[0] = round(total_loss/num_batches,7)
    keep_track[1] = round(100*(correct/size),2)

def test_loop(dataloader, model, loss_fn,keep_track,env_test = None,num_episode=1,device="gpu"):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            #Đẩy dữ liệu lên phần cứng
            X,y = X.to(device),y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    val_loss = round(test_loss / num_batches,7)
    val_acc = round(100*(correct/size),2)
    keep_track[2],keep_track[3] = val_loss,val_acc
    # Giả lập chơi game
    if env_test :
        _,avg_reward,_,fail_rate = env_test.simulate_agent(model,device="cpu", num_episodes=num_episode)
        keep_track[4],keep_track[5] = avg_reward,fail_rate

