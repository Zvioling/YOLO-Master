# 🐧 ES-MoE Cross-Scale Expert Sharing (端侧轻量化)
"""SharedExpertESMoE: 跨尺度专家共享的 ES-MoE 模块（端侧优化版）。

针对 YOLO-Master 的 ES_MOE（v0）改造的跨尺度专家共享变体。

设计动机
--------
- 原始 ES_MOE 在 4 个 ES_MOE 层（256/512/512/1024 通道）各有 4 个独立 expert
  → 总共 16 个 expert（每个 ~50KB weights），累计 ~800KB
- 移动端/嵌入式场景下显存和带宽受限，需要**显著削减 expert 数量**

核心创新
--------
- 多个 ES_MOE 块通过 `pool_id` 共享同一组 expert pool
- 浅层（P3, 256 通道）和深层浅层（P4, 256 通道）共享同一组 4 个 experts
  → 总 expert 数：16 → 12（-25%）
- 中层两个（P5/P6, 512 通道）共享同一组 4 个 experts
  → 总 expert 数：12 → 8（-33%）
- 最深层（P7, 1024 通道）独立 expert pool（深度特征差异大）
  → 总 expert 数：8 + 4 = 12（vs 原 16，**-25%**）

约束
----
- 共享 pool 的 ES_MOE 块必须有相同 in_channels, num_experts, top_k
- 不同 channel size 必须使用不同 pool_id
- pool 注册表是**模块级全局变量**，跨实例共享（与 SharedExpertMoE 一致）

论文价值
--------
- 讨论"ES-MoE 参数共享 vs 性能"的权衡
- 为端侧部署提供参数效率优化路径
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .modules import ES_MOE


# Build-time expert pool registry
# Key: pool_id, Value: dict{experts, num_experts, in_channels, kernel_sizes}
_SHARED_ESMOE_POOLS: Dict[str, dict] = {}


def reset_shared_esmoe_pools() -> None:
    """重置所有共享 expert pool（用于测试或重新构建模型）。"""
    _SHARED_ESMOE_POOLS.clear()


class SharedExpertESMoE(ES_MOE):
    """Cross-Scale Expert Sharing ES-MoE（v0 ES-MoE 端侧轻量化变体）。

    与基类 ES_MOE 的差异：
        - 多个 SharedExpertESMoE 实例通过 pool_id 共享同一个 expert pool
        - pool_id 相同的实例有相同的 expert 参数（节省内存）
        - 每个实例保留独立的 routing（用于不同尺度的路由决策）

    使用示例：
        >>> # YAML 中：
        >>> # - [-1, 1, SharedExpertESMoE, [256, 4, 2, "shared_p3_p4"]]
        >>> # - [-1, 1, SharedExpertESMoE, [256, 4, 2, "shared_p3_p4"]]
        >>>
        >>> # Python 中：
        >>> block_a = SharedExpertESMoE(256, num_experts=4, top_k=2, pool_id="shared")
        >>> block_b = SharedExpertESMoE(256, num_experts=4, top_k=2, pool_id="shared")
        >>> # block_a.experts is block_b.experts  → True（共享 expert pool）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        num_experts: int = 4,
        top_k: int = 2,
        pool_id: str = "shared",
        **kwargs,
    ):
        # 先初始化基类（创建本地 experts / routing 等）
        # out_channels 必须显式透传给 ES_MOE（默认 = in_channels），
        # 否则与 parse_model 的 [c1, c2, ...] 展开顺序错位。
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_experts=num_experts,
            top_k=top_k,
            **kwargs,
        )
        self.pool_id = pool_id
        self._is_pool_owner = False

        # 设置共享 pool（可能替换 self.experts）
        self._setup_shared_pool()

    def _setup_shared_pool(self):
        """设置共享 expert pool。

        第一个该 pool_id 的块是 owner（保留 self.experts 作为 pool）。
        后续块复用：替换 self.experts 为 pool 的 experts 引用。
        """
        pool_key = self.pool_id
        pool_signature = {
            "in_channels": self.in_channels if hasattr(self, "in_channels") else None,
            "out_channels": self.out_channels if hasattr(self, "out_channels") else None,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
        }

        if pool_key in _SHARED_ESMOE_POOLS:
            existing = _SHARED_ESMOE_POOLS[pool_key]
            # 验证参数兼容性
            for k, v in pool_signature.items():
                if existing.get(k) != v:
                    raise ValueError(
                        f"SharedExpertESMoE pool '{pool_key}' 参数不一致: "
                        f"{k} 现有={existing.get(k)} 新建={v}"
                    )
            # 替换 self.experts 为共享引用（参数共享！）
            self.experts = existing["experts"]
            self._is_pool_owner = False
        else:
            # 第一个该 pool_id 的块是 owner
            _SHARED_ESMOE_POOLS[pool_key] = {
                "experts": self.experts,  # 保存引用
                **pool_signature,
            }
            self._is_pool_owner = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward 沿用基类 ES_MOE 逻辑（routing 独立，experts 共享）。"""
        return super().forward(x)

    def __repr__(self) -> str:
        role = "OWNER" if self._is_pool_owner else "REUSER"
        return (
            f"SharedExpertESMoE(in_channels={self.in_channels if hasattr(self, 'in_channels') else '?'}, "
            f"num_experts={self.num_experts}, top_k={self.top_k}, "
            f"pool_id='{self.pool_id}', role={role})"
        )


# 在类定义后挂载：让 tasks.py 可用 SharedExpertESMoE.reset_shared_esmoe_pools() 调用
SharedExpertESMoE.reset_shared_esmoe_pools = staticmethod(reset_shared_esmoe_pools)


__all__ = ["SharedExpertESMoE", "_SHARED_ESMOE_POOLS", "reset_shared_esmoe_pools"]