from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


PROBLEMS: dict[str, dict] = {
    "scaled-dot-product-attention": {
        "id": "scaled-dot-product-attention",
        "title": "1. 手写 Scaled Dot-Product Attention",
        "difficulty": "中等",
        "tags": ["Attention", "Transformer", "基础"],
        "description": """实现缩放点积注意力函数 `scaled_dot_product_attention`。

要求：
1. 计算 `scores = q @ k^T / sqrt(d_k)`。
2. 若 `mask` 不为 `None`，其中 `True` 表示可见，`False` 表示不可见；不可见位置在 softmax 前应被屏蔽。
3. 对最后一维做 softmax，得到注意力权重后与 `v` 相乘。

输入：
- `q`: `[B, H, Lq, D]`
- `k`: `[B, H, Lk, D]`
- `v`: `[B, H, Lk, D]`
- `mask`: `None` 或可广播到 `[B, H, Lq, Lk]` 的 `bool` 张量

输出：
- `[B, H, Lq, D]`

注意：
- 不要直接调用 `torch.nn.functional.scaled_dot_product_attention`。
""",
        "starter_code": """import math
from typing import Optional

import torch


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    \"\"\"
    q: [B, H, Lq, D]
    k: [B, H, Lk, D]
    v: [B, H, Lk, D]
    mask: None or bool tensor broadcastable to [B, H, Lq, Lk]
    \"\"\"
    # TODO: write your code here
    raise NotImplementedError
""",
        "entry_name": "scaled_dot_product_attention",
        "entry_type": "function",
        "reference_code": """import math
from typing import Optional

import torch


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, -1e9)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out
""",
        "visible_cases": [
            {
                "name": "基础形状-无mask",
                "spec": {
                    "seed": 7,
                    "batch": 2,
                    "heads": 2,
                    "q_len": 4,
                    "k_len": 4,
                    "head_dim": 8,
                    "use_mask": False,
                },
            },
            {
                "name": "有mask",
                "spec": {
                    "seed": 11,
                    "batch": 1,
                    "heads": 4,
                    "q_len": 6,
                    "k_len": 6,
                    "head_dim": 8,
                    "use_mask": True,
                },
            },
            {
                "name": "Lq和Lk不同",
                "spec": {
                    "seed": 23,
                    "batch": 2,
                    "heads": 3,
                    "q_len": 5,
                    "k_len": 7,
                    "head_dim": 4,
                    "use_mask": True,
                },
            },
        ],
        "hidden_cases": [
            {
                "name": "hidden-1",
                "spec": {
                    "seed": 101,
                    "batch": 1,
                    "heads": 8,
                    "q_len": 16,
                    "k_len": 16,
                    "head_dim": 8,
                    "use_mask": False,
                },
            },
            {
                "name": "hidden-2",
                "spec": {
                    "seed": 223,
                    "batch": 2,
                    "heads": 4,
                    "q_len": 12,
                    "k_len": 10,
                    "head_dim": 16,
                    "use_mask": True,
                },
            },
            {
                "name": "hidden-3",
                "spec": {
                    "seed": 409,
                    "batch": 3,
                    "heads": 2,
                    "q_len": 9,
                    "k_len": 9,
                    "head_dim": 32,
                    "use_mask": True,
                },
            },
            {
                "name": "hidden-4",
                "spec": {
                    "seed": 733,
                    "batch": 1,
                    "heads": 1,
                    "q_len": 2,
                    "k_len": 11,
                    "head_dim": 64,
                    "use_mask": True,
                },
            },
        ],
        "custom_case_schema": {
            "batch": "int, default=1",
            "heads": "int, default=4",
            "q_len": "int, default=8",
            "k_len": "int, default=8",
            "head_dim": "int, default=16",
            "use_mask": "bool, default=true",
            "seed": "int, default=0",
        },
        "custom_case_example": [
            {
                "batch": 2,
                "heads": 4,
                "q_len": 8,
                "k_len": 8,
                "head_dim": 16,
                "use_mask": True,
                "seed": 0,
            }
        ],
    },
    "multi-head-self-attention-forward": {
        "id": "multi-head-self-attention-forward",
        "title": "2. 手写 Multi-Head Self-Attention.forward",
        "difficulty": "中等",
        "tags": ["Attention", "Module", "面试高频"],
        "description": """补全 `MultiHeadSelfAttention` 的 `forward`。

要求：
1. 使用 `q_proj/k_proj/v_proj` 做线性映射。
2. 切分为多头后计算注意力：`softmax(QK^T / sqrt(head_dim))V`。
3. 支持可选 `mask`（`True` 可见）。
4. 合并多头并通过 `o_proj` 输出。
5. 对注意力权重应用 `self.dropout`。

输入：
- `x`: `[B, L, D]`
- `mask`: `None` 或可广播到 `[B, 1, L, L]` 的 `bool` 张量
输出：
- `[B, L, D]`

提示：
- 评测时会复用 `__init__` 中定义的参数并注入固定权重，请保持字段语义一致。
""",
        "starter_code": """import math
from typing import Optional

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        \"\"\"
        x: [B, L, D]
        mask: None or bool tensor broadcastable to [B, 1, L, L]
        \"\"\"
        # TODO: write your code here
        raise NotImplementedError
""",
        "entry_name": "MultiHeadSelfAttention",
        "entry_type": "class",
        "reference_code": """import math
from typing import Optional

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.o_proj(out)
        return out
""",
        "visible_cases": [
            {
                "name": "基础形状",
                "spec": {
                    "seed": 5,
                    "batch": 2,
                    "seq_len": 6,
                    "d_model": 32,
                    "num_heads": 4,
                    "dropout": 0.0,
                    "use_mask": False,
                },
            },
            {
                "name": "带mask",
                "spec": {
                    "seed": 17,
                    "batch": 1,
                    "seq_len": 8,
                    "d_model": 64,
                    "num_heads": 8,
                    "dropout": 0.0,
                    "use_mask": True,
                },
            },
            {
                "name": "短序列",
                "spec": {
                    "seed": 29,
                    "batch": 4,
                    "seq_len": 3,
                    "d_model": 16,
                    "num_heads": 4,
                    "dropout": 0.0,
                    "use_mask": True,
                },
            },
        ],
        "hidden_cases": [
            {
                "name": "hidden-1",
                "spec": {
                    "seed": 137,
                    "batch": 2,
                    "seq_len": 12,
                    "d_model": 96,
                    "num_heads": 8,
                    "dropout": 0.0,
                    "use_mask": False,
                },
            },
            {
                "name": "hidden-2",
                "spec": {
                    "seed": 211,
                    "batch": 3,
                    "seq_len": 9,
                    "d_model": 48,
                    "num_heads": 6,
                    "dropout": 0.0,
                    "use_mask": True,
                },
            },
            {
                "name": "hidden-3",
                "spec": {
                    "seed": 503,
                    "batch": 1,
                    "seq_len": 15,
                    "d_model": 80,
                    "num_heads": 10,
                    "dropout": 0.0,
                    "use_mask": True,
                },
            },
        ],
        "custom_case_schema": {
            "batch": "int, default=1",
            "seq_len": "int, default=8",
            "d_model": "int, default=32",
            "num_heads": "int, default=4, must divide d_model",
            "dropout": "float, default=0.0",
            "use_mask": "bool, default=true",
            "seed": "int, default=0",
        },
        "custom_case_example": [
            {
                "batch": 2,
                "seq_len": 10,
                "d_model": 64,
                "num_heads": 8,
                "dropout": 0.0,
                "use_mask": True,
                "seed": 0,
            }
        ],
    },
    "transformer-encoder-block-forward": {
        "id": "transformer-encoder-block-forward",
        "title": "3. 手写 Transformer Encoder Block.forward",
        "difficulty": "中等",
        "tags": ["Transformer", "Residual", "LayerNorm"],
        "description": """补全 `TransformerEncoderBlock` 的 `forward`（Pre-Norm 结构）。

结构要求：
1. `x1 = norm1(x)`
2. 自注意力：`attn(x1)`，并做残差：`x = x + dropout(attn_out)`
3. `x2 = norm2(x)`
4. MLP：`fc2(gelu(fc1(x2)))`
5. 第二个残差：`x = x + dropout(mlp_out)`

输入：
- `x`: `[B, L, D]`
- `mask`: `None` 或可广播到 `[B, 1, L, L]` 的 `bool` 张量
输出：
- `[B, L, D]`
""",
        "starter_code": """import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.fc1 = nn.Linear(d_model, d_model * mlp_ratio)
        self.fc2 = nn.Linear(d_model * mlp_ratio, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        \"\"\"
        x: [B, L, D]
        mask: None or bool tensor broadcastable to [B, 1, L, L]
        \"\"\"
        # TODO: write your code here
        raise NotImplementedError
""",
        "entry_name": "TransformerEncoderBlock",
        "entry_type": "class",
        "reference_code": """import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.fc1 = nn.Linear(d_model, d_model * mlp_ratio)
        self.fc2 = nn.Linear(d_model * mlp_ratio, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        x1 = self.norm1(x)
        q = self.q_proj(x1).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x1).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x1).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        attn_out = self.o_proj(attn_out)
        x = x + self.dropout(attn_out)

        x2 = self.norm2(x)
        mlp_out = self.fc2(F.gelu(self.fc1(x2)))
        x = x + self.dropout(mlp_out)
        return x
""",
        "visible_cases": [
            {
                "name": "基础形状",
                "spec": {
                    "seed": 31,
                    "batch": 2,
                    "seq_len": 6,
                    "d_model": 32,
                    "num_heads": 4,
                    "mlp_ratio": 4,
                    "dropout": 0.0,
                    "use_mask": False,
                },
            },
            {
                "name": "带mask",
                "spec": {
                    "seed": 41,
                    "batch": 1,
                    "seq_len": 10,
                    "d_model": 64,
                    "num_heads": 8,
                    "mlp_ratio": 2,
                    "dropout": 0.0,
                    "use_mask": True,
                },
            },
            {
                "name": "小维度",
                "spec": {
                    "seed": 59,
                    "batch": 3,
                    "seq_len": 5,
                    "d_model": 24,
                    "num_heads": 3,
                    "mlp_ratio": 3,
                    "dropout": 0.0,
                    "use_mask": True,
                },
            },
        ],
        "hidden_cases": [
            {
                "name": "hidden-1",
                "spec": {
                    "seed": 271,
                    "batch": 2,
                    "seq_len": 12,
                    "d_model": 96,
                    "num_heads": 8,
                    "mlp_ratio": 4,
                    "dropout": 0.0,
                    "use_mask": True,
                },
            },
            {
                "name": "hidden-2",
                "spec": {
                    "seed": 313,
                    "batch": 1,
                    "seq_len": 7,
                    "d_model": 48,
                    "num_heads": 6,
                    "mlp_ratio": 2,
                    "dropout": 0.0,
                    "use_mask": False,
                },
            },
            {
                "name": "hidden-3",
                "spec": {
                    "seed": 367,
                    "batch": 4,
                    "seq_len": 4,
                    "d_model": 40,
                    "num_heads": 5,
                    "mlp_ratio": 3,
                    "dropout": 0.0,
                    "use_mask": True,
                },
            },
        ],
        "custom_case_schema": {
            "batch": "int, default=1",
            "seq_len": "int, default=8",
            "d_model": "int, default=32",
            "num_heads": "int, default=4, must divide d_model",
            "mlp_ratio": "int, default=4",
            "dropout": "float, default=0.0",
            "use_mask": "bool, default=true",
            "seed": "int, default=0",
        },
        "custom_case_example": [
            {
                "batch": 2,
                "seq_len": 8,
                "d_model": 64,
                "num_heads": 8,
                "mlp_ratio": 4,
                "dropout": 0.0,
                "use_mask": True,
                "seed": 0,
            }
        ],
    },
    "lenet5-forward": {
        "id": "lenet5-forward",
        "title": "4. 手写 CNN LeNet-5.forward",
        "difficulty": "简单",
        "tags": ["CNN", "LeNet", "经典网络"],
        "description": """补全 LeNet-5 的 `forward`。

网络结构（已在 `__init__` 定义）：
1. `conv1(1->6, kernel=5)` + ReLU + AvgPool2d(2)
2. `conv2(6->16, kernel=5)` + ReLU + AvgPool2d(2)
3. Flatten
4. `fc1(16*5*5 -> 120)` + ReLU
5. `fc2(120 -> 84)` + ReLU
6. `fc3(84 -> num_classes)`

输入：
- `x`: `[B, 1, 32, 32]`
输出：
- `[B, num_classes]`
""",
        "starter_code": """import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: write your code here
        raise NotImplementedError
""",
        "entry_name": "LeNet5",
        "entry_type": "class",
        "reference_code": """import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
""",
        "visible_cases": [
            {
                "name": "标准分类10类",
                "spec": {"seed": 3, "batch": 2, "num_classes": 10},
            },
            {
                "name": "batch=1",
                "spec": {"seed": 13, "batch": 1, "num_classes": 10},
            },
            {
                "name": "类别数变化",
                "spec": {"seed": 19, "batch": 4, "num_classes": 7},
            },
        ],
        "hidden_cases": [
            {
                "name": "hidden-1",
                "spec": {"seed": 101, "batch": 8, "num_classes": 10},
            },
            {
                "name": "hidden-2",
                "spec": {"seed": 203, "batch": 3, "num_classes": 12},
            },
            {
                "name": "hidden-3",
                "spec": {"seed": 307, "batch": 5, "num_classes": 5},
            },
        ],
        "custom_case_schema": {
            "batch": "int, default=2",
            "num_classes": "int, default=10",
            "seed": "int, default=0",
        },
        "custom_case_example": [{"batch": 3, "num_classes": 10, "seed": 0}],
    },
    "rmsnorm-forward": {
        "id": "rmsnorm-forward",
        "title": "5. 手写 RMSNorm.forward",
        "difficulty": "简单",
        "tags": ["Normalization", "LLM", "面试高频"],
        "description": """补全 `RMSNorm` 的 `forward`。

定义：
`RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + eps) * weight`

输入：
- `x`: `[..., dim]`
输出：
- 同形状张量

说明：
- 仅做 RMS 归一化，不减均值。
""",
        "starter_code": """import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: write your code here
        raise NotImplementedError
""",
        "entry_name": "RMSNorm",
        "entry_type": "class",
        "reference_code": """import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(rms + self.eps)
        return x_norm * self.weight
""",
        "visible_cases": [
            {
                "name": "二维输入",
                "spec": {"seed": 2, "batch": 3, "seq_len": 5, "dim": 16, "eps": 1e-6},
            },
            {
                "name": "三维输入",
                "spec": {"seed": 17, "batch": 2, "seq_len": 7, "dim": 32, "eps": 1e-5},
            },
            {
                "name": "小维度",
                "spec": {"seed": 23, "batch": 4, "seq_len": 3, "dim": 4, "eps": 1e-6},
            },
        ],
        "hidden_cases": [
            {
                "name": "hidden-1",
                "spec": {"seed": 101, "batch": 8, "seq_len": 9, "dim": 64, "eps": 1e-6},
            },
            {
                "name": "hidden-2",
                "spec": {"seed": 211, "batch": 1, "seq_len": 13, "dim": 128, "eps": 1e-5},
            },
            {
                "name": "hidden-3",
                "spec": {"seed": 337, "batch": 2, "seq_len": 2, "dim": 10, "eps": 1e-6},
            },
        ],
        "custom_case_schema": {
            "batch": "int, default=2",
            "seq_len": "int, default=8",
            "dim": "int, default=32",
            "eps": "float, default=1e-6",
            "seed": "int, default=0",
        },
        "custom_case_example": [{"batch": 2, "seq_len": 8, "dim": 32, "eps": 1e-6, "seed": 0}],
    },
    "apply-rope": {
        "id": "apply-rope",
        "title": "6. 手写 RoPE 旋转位置编码",
        "difficulty": "中等",
        "tags": ["RoPE", "LLM", "位置编码"],
        "description": """实现 `apply_rope`，将旋转位置编码应用到输入 `x`。

约定：
- `x` 形状为 `[B, H, L, D]`，且 `D` 为偶数。
- 取偶数位为 `x_even`，奇数位为 `x_odd`。
- `cos/sin` 形状可为 `[L, D/2]` 或可广播到 `[B, H, L, D/2]`。

公式：
- `out_even = x_even * cos - x_odd * sin`
- `out_odd  = x_even * sin + x_odd * cos`
- 再将 even/odd 交错合并回 `[B, H, L, D]`
""",
        "starter_code": """import torch


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    \"\"\"
    x:   [B, H, L, D] (D must be even)
    cos: [L, D/2] or broadcastable to [B, H, L, D/2]
    sin: [L, D/2] or broadcastable to [B, H, L, D/2]
    \"\"\"
    # TODO: write your code here
    raise NotImplementedError
""",
        "entry_name": "apply_rope",
        "entry_type": "function",
        "reference_code": """import torch


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0).unsqueeze(0)

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out
""",
        "visible_cases": [
            {
                "name": "二维cos/sin",
                "spec": {
                    "seed": 7,
                    "batch": 2,
                    "heads": 4,
                    "seq_len": 8,
                    "dim": 16,
                    "broadcast_style": "2d",
                },
            },
            {
                "name": "可广播四维cos/sin",
                "spec": {
                    "seed": 19,
                    "batch": 1,
                    "heads": 2,
                    "seq_len": 5,
                    "dim": 8,
                    "broadcast_style": "4d",
                },
            },
            {
                "name": "小尺寸",
                "spec": {
                    "seed": 29,
                    "batch": 3,
                    "heads": 1,
                    "seq_len": 3,
                    "dim": 4,
                    "broadcast_style": "2d",
                },
            },
        ],
        "hidden_cases": [
            {
                "name": "hidden-1",
                "spec": {
                    "seed": 103,
                    "batch": 2,
                    "heads": 8,
                    "seq_len": 12,
                    "dim": 32,
                    "broadcast_style": "2d",
                },
            },
            {
                "name": "hidden-2",
                "spec": {
                    "seed": 227,
                    "batch": 1,
                    "heads": 4,
                    "seq_len": 9,
                    "dim": 64,
                    "broadcast_style": "4d",
                },
            },
            {
                "name": "hidden-3",
                "spec": {
                    "seed": 401,
                    "batch": 5,
                    "heads": 2,
                    "seq_len": 6,
                    "dim": 10,
                    "broadcast_style": "2d",
                },
            },
        ],
        "custom_case_schema": {
            "batch": "int, default=1",
            "heads": "int, default=4",
            "seq_len": "int, default=8",
            "dim": "int, default=16, must be even",
            "broadcast_style": "string('2d'|'4d'), default='2d'",
            "seed": "int, default=0",
        },
        "custom_case_example": [
            {
                "batch": 2,
                "heads": 4,
                "seq_len": 8,
                "dim": 16,
                "broadcast_style": "2d",
                "seed": 0,
            }
        ],
    },
}

PROBLEMS.update(
    {
        "layernorm-forward": {
            "id": "layernorm-forward",
            "title": "7. 手写 LayerNorm.forward",
            "difficulty": "简单",
            "tags": ["Normalization", "Transformer", "基础"],
            "description": """实现 `LayerNormManual` 的 `forward`，要求不调用 `nn.LayerNorm`。

定义：
- 沿最后一维做归一化。
- `mean = x.mean(dim=-1, keepdim=True)`
- `var = x.var(dim=-1, unbiased=False, keepdim=True)`
- `y = (x - mean) / sqrt(var + eps)`
- 最终输出 `y * weight + bias`

输入：
- `x`: `[..., dim]`
输出：
- 同形状张量
""",
            "starter_code": """import torch
import torch.nn as nn


class LayerNormManual(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: write your code here
        raise NotImplementedError
""",
            "entry_name": "LayerNormManual",
            "entry_type": "class",
            "reference_code": """import torch
import torch.nn as nn


class LayerNormManual(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        y = (x - mean) * torch.rsqrt(var + self.eps)
        return y * self.weight + self.bias
""",
            "visible_cases": [
                {"name": "二维输入", "spec": {"seed": 12, "batch": 3, "seq_len": 5, "dim": 16, "eps": 1e-5}},
                {"name": "小维度", "spec": {"seed": 29, "batch": 2, "seq_len": 7, "dim": 4, "eps": 1e-6}},
                {"name": "较大维度", "spec": {"seed": 37, "batch": 1, "seq_len": 8, "dim": 64, "eps": 1e-5}},
            ],
            "hidden_cases": [
                {"name": "hidden-1", "spec": {"seed": 111, "batch": 8, "seq_len": 10, "dim": 32, "eps": 1e-5}},
                {"name": "hidden-2", "spec": {"seed": 251, "batch": 2, "seq_len": 3, "dim": 128, "eps": 1e-6}},
                {"name": "hidden-3", "spec": {"seed": 401, "batch": 4, "seq_len": 11, "dim": 24, "eps": 1e-5}},
            ],
            "custom_case_schema": {
                "batch": "int, default=2",
                "seq_len": "int, default=8",
                "dim": "int, default=32",
                "eps": "float, default=1e-5",
                "seed": "int, default=0",
            },
            "custom_case_example": [{"batch": 2, "seq_len": 8, "dim": 32, "eps": 1e-5, "seed": 0}],
        },
        "swiglu-ffn-forward": {
            "id": "swiglu-ffn-forward",
            "title": "8. 手写 SwiGLU FFN.forward",
            "difficulty": "中等",
            "tags": ["MLP", "LLM", "SwiGLU"],
            "description": """补全 `SwiGLUFFN` 的 `forward`。

模块定义：
- `w1: Linear(d_model -> hidden_dim)`
- `w2: Linear(d_model -> hidden_dim)`
- `w3: Linear(hidden_dim -> d_model)`

前向公式：
- `gate = silu(w1(x))`
- `value = w2(x)`
- `out = w3(gate * value)`

输入：
- `x`: `[B, L, D]`
输出：
- `[B, L, D]`
""",
            "starter_code": """import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: write your code here
        raise NotImplementedError
""",
            "entry_name": "SwiGLUFFN",
            "entry_type": "class",
            "reference_code": """import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        value = self.w2(x)
        return self.w3(gate * value)
""",
            "visible_cases": [
                {"name": "基础维度", "spec": {"seed": 5, "batch": 2, "seq_len": 6, "d_model": 32, "hidden_dim": 64}},
                {"name": "短序列", "spec": {"seed": 17, "batch": 1, "seq_len": 3, "d_model": 16, "hidden_dim": 48}},
                {"name": "较大hidden", "spec": {"seed": 26, "batch": 3, "seq_len": 4, "d_model": 24, "hidden_dim": 96}},
            ],
            "hidden_cases": [
                {"name": "hidden-1", "spec": {"seed": 131, "batch": 4, "seq_len": 10, "d_model": 64, "hidden_dim": 128}},
                {"name": "hidden-2", "spec": {"seed": 233, "batch": 2, "seq_len": 5, "d_model": 80, "hidden_dim": 160}},
                {"name": "hidden-3", "spec": {"seed": 317, "batch": 1, "seq_len": 9, "d_model": 48, "hidden_dim": 192}},
            ],
            "custom_case_schema": {
                "batch": "int, default=2",
                "seq_len": "int, default=8",
                "d_model": "int, default=32",
                "hidden_dim": "int, default=128",
                "seed": "int, default=0",
            },
            "custom_case_example": [
                {"batch": 2, "seq_len": 8, "d_model": 32, "hidden_dim": 128, "seed": 0}
            ],
        },
        "lora-linear-forward": {
            "id": "lora-linear-forward",
            "title": "9. 手写 LoRA Linear.forward",
            "difficulty": "中等",
            "tags": ["LoRA", "PEFT", "LLM"],
            "description": """补全 `LoRALinear` 的 `forward`。

给定：
- 基础线性层：`base`
- 低秩分支：`lora_A(in->rank)` 和 `lora_B(rank->out)`
- 缩放系数：`scaling = alpha / rank`

前向：
- `base_out = base(x)`
- `lora_out = lora_B(lora_A(x)) * scaling`
- 返回 `base_out + lora_out`

输入：
- `x`: `[B, L, in_features]`
输出：
- `[B, L, out_features]`
""",
            "starter_code": """import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 8.0,
        bias: bool = True,
    ):
        super().__init__()
        assert rank > 0, "rank must be > 0"
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: write your code here
        raise NotImplementedError
""",
            "entry_name": "LoRALinear",
            "entry_type": "class",
            "reference_code": """import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 8.0,
        bias: bool = True,
    ):
        super().__init__()
        assert rank > 0, "rank must be > 0"
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return base_out + lora_out
""",
            "visible_cases": [
                {
                    "name": "基础配置",
                    "spec": {
                        "seed": 9,
                        "batch": 2,
                        "seq_len": 6,
                        "in_features": 32,
                        "out_features": 24,
                        "rank": 4,
                        "alpha": 8.0,
                        "bias": True,
                    },
                },
                {
                    "name": "无bias",
                    "spec": {
                        "seed": 21,
                        "batch": 1,
                        "seq_len": 8,
                        "in_features": 16,
                        "out_features": 16,
                        "rank": 2,
                        "alpha": 4.0,
                        "bias": False,
                    },
                },
                {
                    "name": "高rank",
                    "spec": {
                        "seed": 33,
                        "batch": 3,
                        "seq_len": 3,
                        "in_features": 48,
                        "out_features": 40,
                        "rank": 8,
                        "alpha": 16.0,
                        "bias": True,
                    },
                },
            ],
            "hidden_cases": [
                {
                    "name": "hidden-1",
                    "spec": {
                        "seed": 107,
                        "batch": 4,
                        "seq_len": 5,
                        "in_features": 64,
                        "out_features": 32,
                        "rank": 8,
                        "alpha": 16.0,
                        "bias": True,
                    },
                },
                {
                    "name": "hidden-2",
                    "spec": {
                        "seed": 211,
                        "batch": 2,
                        "seq_len": 11,
                        "in_features": 24,
                        "out_features": 72,
                        "rank": 6,
                        "alpha": 12.0,
                        "bias": False,
                    },
                },
                {
                    "name": "hidden-3",
                    "spec": {
                        "seed": 409,
                        "batch": 1,
                        "seq_len": 13,
                        "in_features": 80,
                        "out_features": 64,
                        "rank": 10,
                        "alpha": 20.0,
                        "bias": True,
                    },
                },
            ],
            "custom_case_schema": {
                "batch": "int, default=2",
                "seq_len": "int, default=8",
                "in_features": "int, default=32",
                "out_features": "int, default=32",
                "rank": "int, default=4, must be > 0",
                "alpha": "float, default=8.0",
                "bias": "bool, default=true",
                "seed": "int, default=0",
            },
            "custom_case_example": [
                {
                    "batch": 2,
                    "seq_len": 8,
                    "in_features": 32,
                    "out_features": 32,
                    "rank": 4,
                    "alpha": 8.0,
                    "bias": True,
                    "seed": 0,
                }
            ],
        },
        "decode-step-attention-with-kv-cache": {
            "id": "decode-step-attention-with-kv-cache",
            "title": "10. 手写单步解码 Attention + KV Cache",
            "difficulty": "中等",
            "tags": ["KV Cache", "Attention", "推理"],
            "description": """实现 `decode_step_attention`，用于增量解码单步。

输入：
- `q_t`: `[B, H, 1, D]`，当前时刻 query
- `k_cache`: `[B, H, T, D]`，历史 key 缓存（可为 `T=0`）
- `v_cache`: `[B, H, T, D]`，历史 value 缓存
- `k_t`: `[B, H, 1, D]`，当前时刻 key
- `v_t`: `[B, H, 1, D]`，当前时刻 value

要求：
1. 将 `k_t/v_t` 追加到缓存得到新缓存。
2. 计算 `scores = q_t @ new_k^T / sqrt(D)`。
3. `softmax` 后与 `new_v` 相乘得到输出。
4. 返回 `(out, new_k_cache, new_v_cache)`。
""",
            "starter_code": """import math
from typing import Tuple

import torch


def decode_step_attention(
    q_t: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO: write your code here
    raise NotImplementedError
""",
            "entry_name": "decode_step_attention",
            "entry_type": "function",
            "reference_code": """import math
from typing import Tuple

import torch


def decode_step_attention(
    q_t: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    new_k = torch.cat([k_cache, k_t], dim=2)
    new_v = torch.cat([v_cache, v_t], dim=2)
    d = q_t.size(-1)
    scores = torch.matmul(q_t, new_k.transpose(-2, -1)) / math.sqrt(d)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, new_v)
    return out, new_k, new_v
""",
            "visible_cases": [
                {
                    "name": "空缓存起步",
                    "spec": {"seed": 7, "batch": 2, "heads": 4, "cache_len": 0, "head_dim": 16},
                },
                {
                    "name": "有历史缓存",
                    "spec": {"seed": 19, "batch": 1, "heads": 8, "cache_len": 6, "head_dim": 8},
                },
                {
                    "name": "中等长度缓存",
                    "spec": {"seed": 29, "batch": 3, "heads": 2, "cache_len": 12, "head_dim": 32},
                },
            ],
            "hidden_cases": [
                {
                    "name": "hidden-1",
                    "spec": {"seed": 109, "batch": 2, "heads": 8, "cache_len": 24, "head_dim": 8},
                },
                {
                    "name": "hidden-2",
                    "spec": {"seed": 271, "batch": 1, "heads": 16, "cache_len": 11, "head_dim": 16},
                },
                {
                    "name": "hidden-3",
                    "spec": {"seed": 503, "batch": 4, "heads": 4, "cache_len": 3, "head_dim": 64},
                },
            ],
            "custom_case_schema": {
                "batch": "int, default=1",
                "heads": "int, default=4",
                "cache_len": "int, default=8",
                "head_dim": "int, default=16",
                "seed": "int, default=0",
            },
            "custom_case_example": [
                {"batch": 1, "heads": 4, "cache_len": 8, "head_dim": 16, "seed": 0}
            ],
        },
    }
)

SOLUTION_EXPLANATIONS: dict[str, str] = {
    "scaled-dot-product-attention": """## 思路
1. 用 `QK^T / sqrt(d_k)` 得到分数。
2. 有 mask 时，在 softmax 前把不可见位置置为极小值。
3. 对最后一维 softmax，再与 `V` 相乘得到输出。

## 易错点
- mask 语义是 `True=可见`，不要反了。
- softmax 维度必须是 key 维（最后一维）。
""",
    "multi-head-self-attention-forward": """## 思路
1. 线性映射得到 Q/K/V。
2. `view + transpose` 拆成多头。
3. 计算注意力并做 dropout。
4. 合并多头后过输出投影。

## 易错点
- 合并前要 `transpose(1, 2).contiguous()`。
- `d_model` 必须能整除 `num_heads`。
""",
    "transformer-encoder-block-forward": """## 思路
使用 Pre-Norm 结构：
1. `x1 = norm1(x)` 后做注意力并残差。
2. `x2 = norm2(x)` 后做 MLP 并残差。

## 易错点
- 两次残差都要加到主分支 `x` 上。
- `gelu` 放在 `fc1` 和 `fc2` 中间。
""",
    "lenet5-forward": """## 思路
LeNet-5 是经典卷积网络：
1. 两层 `conv + relu + pool`
2. flatten
3. 三层全连接得到 logits

## 易错点
- 输入尺寸默认 `[B,1,32,32]`，flatten 维度是 `16*5*5`。
""",
    "rmsnorm-forward": """## 思路
RMSNorm 只归一化均方根，不减均值：
1. `rms = mean(x^2)`
2. `x * rsqrt(rms + eps)`
3. 乘可学习参数 `weight`

## 易错点
- 不要减去 `mean(x)`，那是 LayerNorm 的操作。
""",
    "apply-rope": """## 思路
把最后一维按偶/奇拆开：
1. `x_even = x[...,0::2]`, `x_odd = x[...,1::2]`
2. 按旋转公式计算新 even/odd
3. 再交错写回原维度

## 易错点
- `D` 必须是偶数。
- `cos/sin` 可能是 `[L, D/2]`，要扩维到可广播形状。
""",
    "layernorm-forward": """## 思路
1. 在最后一维算 mean/var（`unbiased=False`）。
2. 标准化：`(x-mean)/sqrt(var+eps)`。
3. 仿射变换：`* weight + bias`。

## 易错点
- 方差要用总体方差（`unbiased=False`），和 PyTorch LayerNorm 对齐。
""",
    "swiglu-ffn-forward": """## 思路
SwiGLU 本质是带门控的 FFN：
1. 一路过 `w1` 后做 `silu` 作为门值。
2. 一路过 `w2` 作为内容值。
3. 两路逐元素相乘后，经 `w3` 投回 `d_model`。

## 易错点
- 是逐元素乘法，不是拼接。
""",
    "lora-linear-forward": """## 思路
LoRA 把增量更新写成低秩分解：
1. 主分支：`base(x)`。
2. LoRA 分支：`lora_B(lora_A(x)) * (alpha/rank)`。
3. 两支相加得到最终输出。

## 易错点
- 缩放系数不能漏。
- `lora_A/lora_B` 默认无偏置。
""",
    "decode-step-attention-with-kv-cache": """## 思路
单步解码要先扩充缓存，再算注意力：
1. `new_k = cat(k_cache, k_t)`，`new_v = cat(v_cache, v_t)`。
2. `scores = q_t @ new_k^T / sqrt(D)`。
3. softmax 后与 `new_v` 相乘得到当前步输出。

## 易错点
- 拼接维度是时间维（`dim=2`）。
- 返回值要同时包含输出和新缓存。
""",
}


OVERRIDES_PATH = Path(__file__).resolve().parent.parent / "data" / "problem_overrides.json"
EDITABLE_PROBLEM_FIELDS = {"description", "starter_code"}
EDITABLE_SOLUTION_FIELDS = {"explanation", "code"}


def _load_overrides() -> dict[str, dict]:
    if not OVERRIDES_PATH.exists():
        return {}
    try:
        raw = OVERRIDES_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _save_overrides(overrides: dict[str, dict]) -> None:
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES_PATH.write_text(json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8")


def _apply_problem_overrides(problem_id: str, problem: dict) -> tuple[dict, dict]:
    overrides = _load_overrides()
    override_entry = overrides.get(problem_id, {})
    if not isinstance(override_entry, dict):
        return problem, {}

    problem_part = override_entry.get("problem", {})
    solution_part = override_entry.get("solution", {})

    if isinstance(problem_part, dict):
        if isinstance(problem_part.get("description"), str):
            problem["description"] = problem_part["description"]
        if isinstance(problem_part.get("starter_code"), str):
            problem["starter_code"] = problem_part["starter_code"]

    if isinstance(solution_part, dict) and isinstance(solution_part.get("code"), str):
        # Keep solution display and judge reference code aligned.
        problem["reference_code"] = solution_part["code"]

    return problem, override_entry


def get_problem_effective(problem_id: str) -> dict | None:
    base_problem = PROBLEMS.get(problem_id)
    if base_problem is None:
        return None
    problem, _ = _apply_problem_overrides(problem_id, deepcopy(base_problem))
    return problem


def list_problem_briefs() -> list[dict]:
    briefs: list[dict] = []
    for problem in PROBLEMS.values():
        briefs.append(
            {
                "id": problem["id"],
                "title": problem["title"],
                "difficulty": problem["difficulty"],
                "tags": problem["tags"],
            }
        )
    return briefs


def get_problem_public(problem_id: str) -> dict | None:
    problem = get_problem_effective(problem_id)
    if problem is None:
        return None

    return {
        "id": problem["id"],
        "title": problem["title"],
        "difficulty": problem["difficulty"],
        "tags": problem["tags"],
        "description": problem["description"],
        "starter_code": problem["starter_code"],
        "entry_name": problem["entry_name"],
        "entry_type": problem["entry_type"],
        "visible_cases": deepcopy(problem["visible_cases"]),
        "custom_case_schema": deepcopy(problem["custom_case_schema"]),
        "custom_case_example": deepcopy(problem["custom_case_example"]),
    }


def get_problem_solution(problem_id: str) -> dict | None:
    problem = get_problem_effective(problem_id)
    if problem is None:
        return None

    _, override_entry = _apply_problem_overrides(problem_id, problem)
    explanation = SOLUTION_EXPLANATIONS.get(problem_id, "暂无讲解。")
    if isinstance(override_entry, dict):
        solution_override = override_entry.get("solution", {})
        if isinstance(solution_override, dict) and isinstance(solution_override.get("explanation"), str):
            explanation = solution_override["explanation"]

    return {
        "id": problem["id"],
        "title": problem["title"],
        "explanation": explanation,
        "code": problem["reference_code"],
    }


def get_problem_feedback(problem_id: str) -> dict | None:
    problem = get_problem_public(problem_id)
    solution = get_problem_solution(problem_id)
    if problem is None or solution is None:
        return None

    return {
        "problem": {
            "description": problem["description"],
            "starter_code": problem["starter_code"],
        },
        "solution": {
            "explanation": solution["explanation"],
            "code": solution["code"],
        },
    }


def save_problem_feedback(problem_id: str, payload: dict) -> dict:
    if problem_id not in PROBLEMS:
        raise ValueError(f"Unknown problem_id: {problem_id}")
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")

    problem_updates = payload.get("problem", {})
    solution_updates = payload.get("solution", {})
    if problem_updates is None:
        problem_updates = {}
    if solution_updates is None:
        solution_updates = {}

    if not isinstance(problem_updates, dict):
        raise ValueError("payload.problem must be an object")
    if not isinstance(solution_updates, dict):
        raise ValueError("payload.solution must be an object")

    for key, value in problem_updates.items():
        if key not in EDITABLE_PROBLEM_FIELDS:
            raise ValueError(f"unsupported problem field: {key}")
        if not isinstance(value, str):
            raise ValueError(f"problem.{key} must be a string")

    for key, value in solution_updates.items():
        if key not in EDITABLE_SOLUTION_FIELDS:
            raise ValueError(f"unsupported solution field: {key}")
        if not isinstance(value, str):
            raise ValueError(f"solution.{key} must be a string")

    overrides = _load_overrides()
    entry = overrides.get(problem_id, {})
    if not isinstance(entry, dict):
        entry = {}

    if problem_updates:
        problem_entry = entry.get("problem", {})
        if not isinstance(problem_entry, dict):
            problem_entry = {}
        problem_entry.update(problem_updates)
        entry["problem"] = problem_entry

    if solution_updates:
        solution_entry = entry.get("solution", {})
        if not isinstance(solution_entry, dict):
            solution_entry = {}
        solution_entry.update(solution_updates)
        entry["solution"] = solution_entry

    overrides[problem_id] = entry
    _save_overrides(overrides)

    feedback = get_problem_feedback(problem_id)
    if feedback is None:
        raise ValueError(f"Unknown problem_id: {problem_id}")
    return feedback
