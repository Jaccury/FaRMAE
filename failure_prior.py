import torch
import torch.nn as nn
import torch.nn.functional as F


class FailurePrior(nn.Module):
    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2, eps=1e-6):
        """
        根据论文 Eq.1 - Eq.4 计算失效概率图 Sf
        alpha: 纹理衰减先验 T 的权重，默认 0.4
        beta:  几何不稳定性先验 G 的权重，默认 0.4
        gamma: 传感器掉队证据 M 的权重，默认 0.2
        eps:   防止除零的微小量，默认 1e-6
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

        # 构造 Sobel 卷积核用于计算空间梯度 (X 和 Y 方向)
        # 尺寸为 1x1x3x3，方便通过 F.conv2d 处理单通道或多通道特征
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        # 将卷积核注册为不可训练的 buffer，这样它们会自动跟随模型移动到 GPU/CPU
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_gradient_magnitude(self, x):
        """
        计算输入张量 x 的空间梯度幅值 (L2范数)
        x: shape (B, C, H, W)
        返回: shape (B, 1, H, W)
        """
        B, C, H, W = x.shape

        # 将 batch 内所有通道展开到 batch 维度，方便利用 1x1x3x3 的卷积核统一计算
        # (B*C, 1, H, W)
        x_reshaped = x.view(B * C, 1, H, W)

        # 为了保持输出尺寸与输入相同，需要 padding=1
        grad_x = F.conv2d(x_reshaped, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_reshaped, self.sobel_y, padding=1)

        # 计算梯度幅值: sqrt(dx^2 + dy^2)
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)

        # 恢复形状 (B, C, H, W)
        magnitude = magnitude.view(B, C, H, W)

        # 如果是多通道 (如 RGB)，则在通道维度求平均或求 L2 范数得到单一的特征梯度图
        # 论文中 ||\nabla I|| 是单一映射，这里我们取通道的平均值 (B, 1, H, W)
        if C > 1:
            magnitude = magnitude.mean(dim=1, keepdim=True)

        return magnitude

    def get_max_spatial(self, x):
        """
        获取张量在空间维度上的最大值，用于公式中的归一化分母
        x: (B, 1, H, W)
        返回: (B, 1, 1, 1) 用于广播机制
        """
        B, C, H, W = x.shape
        # 展平 H 和 W 维度，求最大值
        max_val = x.view(B, C, -1).max(dim=2)[0]
        return max_val.view(B, C, 1, 1)

    def get_min_spatial(self, x):
        """
        获取张量在空间维度上的最小值，用于公式4的全局归一化
        """
        B, C, H, W = x.shape
        min_val = x.view(B, C, -1).min(dim=2)[0]
        return min_val.view(B, C, 1, 1)

    def forward(self, rgb, depth):
        """
        rgb: shape (B, 3, H, W)，取值范围应已归一化 (如 0~1 或标准正态)
        depth: shape (B, 1, H, W)，深度图，单位为米 (0 表示缺失/失效)
        返回 Sf: shape (B, 1, H, W)，取值范围 [0, 1]
        """
        # 1. 计算纹理衰减先验 T(I) (Eq. 1)
        grad_rgb = self.compute_gradient_magnitude(rgb)
        max_grad_rgb = self.get_max_spatial(grad_rgb)
        T = 1.0 - (grad_rgb / (max_grad_rgb + self.eps))

        # 2. 计算几何不稳定性先验 G(D_obs) (Eq. 2)
        grad_depth = self.compute_gradient_magnitude(depth)
        max_grad_depth = self.get_max_spatial(grad_depth)
        G = grad_depth / (max_grad_depth + self.eps)

        # 3. 计算传感器掉队证据 M(D_obs) (Eq. 3)
        # 深度图上数值为 0 的像素就是直接的缺失位置
        M = (depth == 0).float()

        # 4. 融合生成 Sf_raw
        Sf_raw = self.alpha * T + self.beta * G + self.gamma * M

        # 5. 最终 Min-Max 归一化 (Eq. 4)
        min_Sf = self.get_min_spatial(Sf_raw)
        max_Sf = self.get_max_spatial(Sf_raw)

        Sf = (Sf_raw - min_Sf) / (max_Sf - min_Sf + self.eps)

        return Sf


# ==========================================
# 测试代码 (可直接运行验证逻辑和形状对不对)
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    # 实例化模块
    failure_prior_module = FailurePrior().to(device)

    # 模拟一个 Batch 的数据 (B=2, RGB=3, H=224, W=224)
    # 假设图片是 224x224
    dummy_rgb = torch.rand(2, 3, 224, 224).to(device)
    dummy_depth = torch.rand(2, 1, 224, 224).to(device)

    # 手动制造一些 depth 为 0 的“传感器掉队”区域，以测试 M(D_obs) 的计算
    dummy_depth[:, :, 100:150, 100:150] = 0.0

    # 前向传播计算 Sf
    Sf = failure_prior_module(dummy_rgb, dummy_depth)

    print(f"RGB input shape: {dummy_rgb.shape}")
    print(f"Depth input shape: {dummy_depth.shape}")
    print(f"Sf output shape: {Sf.shape}")
    print(f"Sf max value: {Sf.max().item():.4f}")
    print(f"Sf min value: {Sf.min().item():.4f}")