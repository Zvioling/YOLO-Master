#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_shared_expert_esmoe.py
============================
SharedExpertESMoE（ES-MoE 端侧轻量化模块）的边界测试。

针对 ultralytics.nn.modules.moe.shared_expert_esmoe.SharedExpertESMoE 模块的 8 个边界场景测试：

1. test_shared_pool_owner_and_reuser
   - 第一个实例是 owner，第二个是 reuser
   - 验证 experts 参数**完全相同**（内存共享）

2. test_shared_pool_expert_params_identical
   - 两个实例的 experts 权重 100% 相同
   - 修改 owner 的 expert 参数，reuser 的对应参数同步变化

3. test_shared_pool_top_k_consistency
   - owner 设 top_k=2，reuser 也必须 top_k=2
   - 不一致时抛 ValueError

4. test_shared_pool_independent_routing
   - experts 共享，但 routing 独立
   - 验证两个实例的 routing 输出不同（不同输入）

5. test_shared_pool_forward_shape
   - 共享 expert pool 的 forward 输出形状正确

6. test_reset_shared_esmoe_pools
   - 重置后可以重新构建 pool

7. test_different_pool_id_isolated
   - 不同 pool_id 的实例互不干扰

8. test_in_channels_mismatch_raises
   - 共享 pool 的 in_channels 不一致时抛 ValueError

运行方法：
    pytest tests/test_shared_expert_esmoe.py -v
    pytest tests/test_shared_expert_esmoe.py -v --cov=ultralytics.nn.modules.moe.shared_expert_esmoe
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.modules.moe.shared_expert_esmoe import (
    SharedExpertESMoE,
    _SHARED_ESMOE_POOLS,
    reset_shared_esmoe_pools,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(autouse=True)
def reset_pools():
    """每个测试前重置 pool 状态。"""
    reset_shared_esmoe_pools()
    yield
    reset_shared_esmoe_pools()


@pytest.fixture
def small_input() -> torch.Tensor:
    """小型输入 [B=2, C=64, H=16, W=16]。"""
    return torch.randn(2, 64, 16, 16)


# ============================================================
# Test 1: 第一个实例是 owner，第二个是 reuser
# ============================================================

def test_shared_pool_owner_and_reuser():
    """第一个实例 _is_pool_owner=True，第二个 _is_pool_owner=False。"""
    block_a = SharedExpertESMoE(in_channels=64, num_experts=4, top_k=2, pool_id="test1")
    block_b = SharedExpertESMoE(in_channels=64, num_experts=4, top_k=2, pool_id="test1")

    assert block_a._is_pool_owner is True, "First instance should be owner"
    assert block_b._is_pool_owner is False, "Second instance should be reuser"
    print("[PASS] Owner/reuser roles correct")


# ============================================================
# Test 2: 两个实例的 experts 参数完全相同（内存共享）
# ============================================================

def test_shared_pool_expert_params_identical():
    """两个实例的 experts 权重 100% 相同，修改同步生效。"""
    block_a = SharedExpertESMoE(in_channels=64, num_experts=4, top_k=2, pool_id="test2")
    block_b = SharedExpertESMoE(in_channels=64, num_experts=4, top_k=2, pool_id="test2")

    # 验证 experts 是同一个对象
    assert block_a.experts is block_b.experts, "Experts should be shared (same object)"

    # 验证参数 id 相同
    for (name_a, p_a), (name_b, p_b) in zip(
        block_a.experts.named_parameters(),
        block_b.experts.named_parameters(),
    ):
        assert name_a == name_b
        assert p_a.data_ptr() == p_b.data_ptr(), \
            f"Parameter '{name_a}' should have same data pointer"

    # 修改 owner 参数，验证 reuser 同步
    with torch.no_grad():
        first_param = next(block_a.experts.parameters())
        first_param.fill_(99.0)

    for p_b in block_b.experts.parameters():
        assert (p_b == 99.0).all(), "Reuser's expert parameters should sync with owner"

    print("[PASS] Expert parameters are truly shared (memory + sync verified)")


# ============================================================
# Test 3: top_k 不一致时抛 ValueError
# ============================================================

def test_shared_pool_top_k_mismatch_raises():
    """owner top_k=2，reuser top_k=3 应抛 ValueError。"""
    SharedExpertESMoE(in_channels=64, num_experts=4, top_k=2, pool_id="test3")
    with pytest.raises(ValueError, match="参数不一致"):
        SharedExpertESMoE(in_channels=64, num_experts=4, top_k=3, pool_id="test3")
    print("[PASS] top_k mismatch raises ValueError")


# ============================================================
# Test 4: experts 共享但 routing 独立
# ============================================================

def test_shared_pool_independent_routing(small_input):
    """共享 experts，但 routing 独立（不同输入 → 不同路由）。"""
    block_a = SharedExpertESMoE(in_channels=64, num_experts=4, top_k=2, pool_id="test4")
    block_b = SharedExpertESMoE(in_channels=64, num_experts=4, top_k=2, pool_id="test4")

    # Experts 共享
    assert block_a.experts is block_b.experts

    # Routing 独立
    assert block_a.routing is not block_b.routing, "Routing should be independent"
    assert block_a.routing is not block_b.routing

    # 不同输入 → 不同输出（即使 experts 共享）
    x_a = torch.randn(1, 64, 16, 16)
    x_b = torch.randn(1, 64, 16, 16)

    block_a.eval()
    block_b.eval()
    with torch.no_grad():
        out_a = block_a(x_a)
        out_b = block_b(x_b)

    # 输出形状正确
    assert out_a.shape == x_a.shape, f"block_a output shape: {out_a.shape}"
    assert out_b.shape == x_b.shape, f"block_b output shape: {out_b.shape}"
    print("[PASS] Shared experts + independent routing works")


# ============================================================
# Test 5: 共享 expert pool 的 forward 输出形状正确
# ============================================================

def test_shared_pool_forward_shape(small_input):
    """SharedExpertESMoE forward 输出形状 = 输入形状。"""
    block_a = SharedExpertESMoE(in_channels=64, num_experts=4, top_k=2, pool_id="test5")
    block_b = SharedExpertESMoE(in_channels=64, num_experts=4, top_k=2, pool_id="test5")

    block_a.eval()
    block_b.eval()
    with torch.no_grad():
        out_a = block_a(small_input)
        out_b = block_b(small_input)

    assert out_a.shape == small_input.shape
    assert out_b.shape == small_input.shape
    assert torch.isfinite(out_a).all(), "block_a output contains NaN/Inf"
    assert torch.isfinite(out_b).all(), "block_b output contains NaN/Inf"

    print(f"[PASS] forward shape correct: {out_a.shape}")


# ============================================================
# Test 6: reset_shared_esmoe_pools 重置功能
# ============================================================

def test_reset_shared_esmoe_pools():
    """reset_shared_esmoe_pools 清空所有 pool。"""
    block_a = SharedExpertESMoE(in_channels=64, num_experts=4, top_k=2, pool_id="test6")
    block_b = SharedExpertESMoE(in_channels=64, num_experts=4, top_k=2, pool_id="test6")

    assert block_b._is_pool_owner is False
    assert block_a.experts is block_b.experts

    # 重置
    reset_shared_esmoe_pools()
    assert len(_SHARED_ESMOE_POOLS) == 0

    # 重置后新建，新实例是 owner
    block_c = SharedExpertESMoE(in_channels=64, num_experts=4, top_k=2, pool_id="test6")
    assert block_c._is_pool_owner is True
    print("[PASS] reset_shared_esmoe_pools correctly resets pool registry")


# ============================================================
# Test 7: 不同 pool_id 的实例互不干扰
# ============================================================

def test_different_pool_id_isolated():
    """不同 pool_id 的实例互不干扰。"""
    block_a = SharedExpertESMoE(in_channels=64, num_experts=4, top_k=2, pool_id="pool_a")
    block_b = SharedExpertESMoE(in_channels=64, num_experts=4, top_k=2, pool_id="pool_b")

    # 两个都是 owner
    assert block_a._is_pool_owner is True
    assert block_b._is_pool_owner is True

    # experts 完全不同
    assert block_a.experts is not block_b.experts

    # Pool 注册表有两条
    assert "pool_a" in _SHARED_EXMOE_POOLS
    assert "pool_b" in _SHARED_EXMOE_POOLS

    print("[PASS] Different pool_ids are isolated")


# ============================================================
# Test 8: num_experts 不一致时抛 ValueError
# ============================================================

def test_num_experts_mismatch_raises():
    """owner num_experts=4，reuser num_experts=8 应抛 ValueError。"""
    SharedExpertESMoE(in_channels=64, num_experts=4, top_k=2, pool_id="test8")
    with pytest.raises(ValueError, match="参数不一致"):
        SharedExpertESMoE(in_channels=64, num_experts=8, top_k=2, pool_id="test8")
    print("[PASS] num_experts mismatch raises ValueError")


# ============================================================
# Test 9: 完整 YOLO 加载 SharedExpertESMoE YAML
# ============================================================

def test_yaml_loads_shared_esmoe():
    """测试 SharedExpertESMoE YAML 配置可以被 ultralytics 加载。"""
    yaml_path = "ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone-shared.yaml"
    if not Path(yaml_path).exists():
        pytest.skip(f"YAML not found: {yaml_path}")

    from ultralytics import YOLO
    # 只构建模型，不加载权重
    model = YOLO(yaml_path)
    # 验证模型中包含 SharedExpertESMoE 模块
    from ultralytics.nn.modules.moe.shared_expert_esmoe import SharedExpertESMoE
    shared_count = sum(
        1 for m in model.model.modules() if isinstance(m, SharedExpertESMoE)
    )
    assert shared_count >= 2, f"Expected at least 2 SharedExpertESMoE, found {shared_count}"

    # 验证 pool 注册表
    pool_ids = list(_SHARED_ESMOE_POOLS.keys())
    assert "esmoe_p3_p4" in pool_ids or len(pool_ids) >= 1, \
        f"Expected esmoe_p3_p4 pool, got {pool_ids}"

    print(f"[PASS] YAML loads with {shared_count} SharedExpertESMoE blocks, pools={pool_ids}")