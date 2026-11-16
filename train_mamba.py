import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm

# 引入刚才定义的模型
from model_mamba import BiMambaSmoother

# ==========================================
# 1. 复用之前的数据处理工具
# ==========================================
def axis_angle_to_6d(p):
    batch_size = p.shape[0]
    angle = torch.norm(p, p=2, dim=1, keepdim=True)
    axis = p / (angle + 1e-8)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    t = 1 - cos
    x, y, z = axis[:, 0:1], axis[:, 1:2], axis[:, 2:3]
    r1 = torch.cat([t*x*x + cos, t*x*y - sin*z, t*x*z + sin*y], dim=1)
    r2 = torch.cat([t*x*y + sin*z, t*y*y + cos, t*y*z - sin*x], dim=1)
    return torch.cat([r1, r2], dim=1)

class AMASSDataset(Dataset):
    def __init__(self, data_root, seq_len=64):
        self.seq_len = seq_len
        self.files = []
        for root, _, filenames in os.walk(data_root):
            for f in filenames:
                if f.endswith(".npz"):
                    self.files.append(os.path.join(root, f))
        print(f"📚 Found {len(self.files)} files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            data = np.load(self.files[idx])
            poses = torch.from_numpy(data['poses']).float()
            # 简化处理：只取前24关节
            frames = poses.shape[0]
            poses_subset = poses[:, :72].reshape(-1, 3)
            poses_6d = axis_angle_to_6d(poses_subset).reshape(frames, 144)
            
            target_dim = 266
            padding = torch.zeros((frames, target_dim - 144))
            final_poses = torch.cat([poses_6d, padding], dim=1)

            if frames >= self.seq_len:
                start = np.random.randint(0, frames - self.seq_len + 1)
                sequence = final_poses[start : start + self.seq_len]
            else:
                res = self.seq_len - frames
                sequence = torch.cat([final_poses, final_poses[:res]], dim=0)
            return sequence
        except:
            # 容错处理：如果文件坏了返回零
            return torch.zeros((self.seq_len, 266))

# ==========================================
# 2. 训练主循环
# ==========================================
def resolve_training_data_path():
    data_path = os.environ.get("DYNAMESH4D_AMASS_DATASET")
    if not data_path:
        raise RuntimeError(
            "Set DYNAMESH4D_AMASS_DATASET to the external AMASS subset directory."
        )
    return data_path


def train():
    # ⚠️ Mamba 必须跑在 CUDA 上
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("❌ Mamba requires CUDA!")

    # External training data is configured separately from the repository.
    data_path = resolve_training_data_path()
    dataset = AMASSDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    
    # 初始化 Bi-Mamba
    print("🐍 Initializing Bi-Mamba Model...")
    model = BiMambaSmoother(input_dim=266, d_model=256, n_layers=4).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    model.train()
    print("🚀 Start Training Mamba...")
    
    for epoch in range(15): # Mamba 收敛快，15轮足够
        total_loss = 0
        pbar = tqdm(dataloader)
        for batch in pbar:
            clean_motion = batch.to(device)
            
            # 加噪
            noise = torch.randn_like(clean_motion) * 0.05
            noisy_motion = clean_motion + noise
            
            optimizer.zero_grad()
            output = model(noisy_motion)
            loss = criterion(output, clean_motion)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1} Loss: {loss.item():.5f}")
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
        
        # 每 5 轮保存一次
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"mamba_smoother_epoch_{epoch+1}.pth")
            print("💾 Model saved.")

if __name__ == "__main__":
    train()