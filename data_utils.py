import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from glob import glob

if sys.platform == 'win32':
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except:
        pass


class TPUSelfPlayDataset(Dataset):
    """
    极简、内存优化的 TPU 自我对弈数据集
    专门用于读取由 Pgx 产生的 (19, 19, 17) 特征矩阵
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

        obs_list = []
        policy_list = []
        value_list = []

        # 读取目录下所有的 npz 文件
        npz_files = sorted(glob(os.path.join(data_dir, "*.npz")))
        print(f"[Dataset] 正在加载数据，找到 {len(npz_files)} 个数据包...")

        for npz_file in npz_files:
            try:
                data = np.load(npz_file)
                obs = data["obs"]  # 形状: (T, B, 19, 19, 17)
                policy = data["policy"]  # 形状: (T, B, 361)
                value = data["value"]  # 形状: (T, B)
                mask = data["mask"]  # 形状: (T, B)

                T, B = obs.shape[:2]

                # 将时间和批次维度展平
                obs_flat = obs.reshape(T * B, 19, 19, 17)
                policy_flat = policy.reshape(T * B, 362)
                value_flat = value.reshape(T * B)
                mask_flat = mask.reshape(T * B)

                # 利用 mask 过滤掉无效步数（避免无效内存占用）
                valid_indices = np.where(mask_flat)[0]

                obs_list.append(obs_flat[valid_indices])
                policy_list.append(policy_flat[valid_indices])
                value_list.append(value_flat[valid_indices])
            except Exception as e:
                print(f"读取文件 {npz_file} 时出错: {e}")

        # 将所有合法对局拼接成巨大的连续内存块，极大提升读取速度和避免内存碎片
        if obs_list:
            self.obs = np.concatenate(obs_list, axis=0)
            self.policy = np.concatenate(policy_list, axis=0)
            self.value = np.concatenate(value_list, axis=0)
            print(f"[Dataset] 加载完毕！总计可用训练样本数: {len(self.obs)}")
        else:
            self.obs = np.array([])
            self.policy = np.array([])
            self.value = np.array([])
            print(f"[Dataset] 警告：没有找到任何训练数据！")

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回 PyTorch Tensor，以便借助 DataLoader 的多进程预读取能力。
        在 TPU/JAX 训练循环中，直接将返回的 Tensor 转为 jnp.array 即可。
        """
        return (
            torch.tensor(self.obs[idx], dtype=torch.float32),
            torch.tensor(self.policy[idx], dtype=torch.float32),
            torch.tensor(self.value[idx], dtype=torch.float32)
        )


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True,
                      num_workers: int = 4, pin_memory: bool = True) -> DataLoader:
    """
    创建适用于 TPU 数据流的高效 DataLoader
    """
    if sys.platform == 'win32':
        num_workers = 0  # Windows 下多进程容易报错，强制设为 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory if num_workers > 0 else False,
        drop_last=True  # 确保 batch size 始终一致，这对于 TPU pmap 非常重要
    )