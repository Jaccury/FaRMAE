import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class TransCGDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=224):
        """
        TransCG 数据集加载器 (已适配微调阶段)
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.samples = []

        all_scenes = sorted([d for d in os.listdir(root_dir) if d.startswith('scene')])
        split_idx = int(len(all_scenes) * 0.8)
        target_scenes = all_scenes[:split_idx] if split == 'train' else all_scenes[split_idx:]

        for scene in target_scenes:
            scene_path = os.path.join(root_dir, scene)
            for root, dirs, files in os.walk(scene_path):
                rgb_files = [f for f in files if f.startswith('rgb') and f.endswith('.png')]

                for f in rgb_files:
                    idx_str = f.replace('rgb', '').replace('.png', '')
                    rgb_file = os.path.join(root, f)

                    # TransCG 提供的一般是带有 gt 后缀的真值深度
                    depth_file = os.path.join(root, f"depth{idx_str}-gt.png")

                    if os.path.exists(depth_file):
                        self.samples.append({
                            'rgb': rgb_file,
                            'depth': depth_file,
                            'filename': f"{scene}_{os.path.basename(root)}_{f}"
                        })

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 1. RGB
        rgb = cv2.imread(s['rgb'])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size))
        rgb_tensor = self.rgb_transform(rgb)

        # 2. 深度图
        depth = cv2.imread(s['depth'], cv2.IMREAD_ANYDEPTH)
        depth = cv2.resize(depth, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32) / 1000.0
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)

        # ❗ 核心修改：适配 finetune.py
        # 由于这里我们直接使用 TransCG 进行微调，而你手头只有 depth-gt 文件，
        # 我们暂时将输入 depth 和监督标签 depth_gt 指向同一个张量，确保代码逻辑能跑通。
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,  # 作为网络输入的观测深度
            'depth_gt': depth_tensor,  # 作为计算 Loss 的真值深度
            'filename': s['filename']
        }