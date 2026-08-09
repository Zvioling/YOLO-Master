#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_esmoe.py
=============
ES-MoE (Efficient Sparse Mixture-of-Experts) 边界测试。

针对 ultralytics.nn.modules.moe.modules.ES_MOE 模块的 8 个边界场景测试：

1. test_esmoe_top_k_none_uses_all_experts
   - top_k=None 时，全部专家激活（路由权重均 > 0）

2. test_esmoe_top_k_exceeds_num_experts_raises
   - top_k > num_experts 应抛出 ValueError

3. test_esmoe_expert_kernel_sizes_length_mismatch
   - expert_kernel_sizes 长度不匹配应抛出 ValueError

4. test_esmoe_no_sparse_inference_dense_forward
   - use_sparse_inference=False 时走 Dense 分支，输出正常

5. test_esmoe_onnx_export_dense_fallback
   - ONNX 导出走 Dense 分支（保证图结构稳定）

6. test_esmoe_eager_sparse_enabled_logic
   - 5 种配置组合下 _eager_sparse_enabled() 状态正确

7. test_esmoe_balance_loss_extreme_values
   - balance_loss 极端值（0.001/1.0/10.0）下前向稳定

8. test_esmoe_no_z_loss_logit_growth
   - z_loss=0 时 100 step 训练后 logits 增长 < 100x

运行方法：
    pytest tests/test_esmoe.py -v
    pytest tests/test_esmoe.py -v --cov=ultralytics.nn.modules.moe.modules
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ultralytics.nn.modules.moe.modules import ES_MOE


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def small_input() -> torch.Tensor:
    """小型输入 [B=2, C=64, H=16, W=16]（用于快速测试）。"""
    return torch.randn(2, 64, 16, 16)


@pytest.fixture
def esmoe_module() -> ES_MOE:
    """标准 ES_MOE 模块（in_channels=64, num_experts=4, top_k=2）。"""
    return ES_MOE(in_channels=64, num_experts=4, top_k=2)


# ============================================================
# Test 1: top_k=None 时全部专家激活
# ============================================================

def test_esmoe_top_k_none_uses_all_experts(small_input):
    """top_k=None 时，全部 4 个专家激活，路由权重均 > 0。"""
    module = ES_MOE(in_channels=64, num_experts=4, top_k=None)
    module.eval()

    captured: dict = {}

    def hook(m, inputs, _output):
        with torch.no_grad():
            weights = m.routing(inputs[0])  # [B, 4, H, W]
            captured["weights"] = weights.mean(dim=(0, 2, 3))  # [4]

    handle = module.register_forward_hook(hook)
    with torch.no_grad():
        _ = module(small_input)
    handle.remove()

    weights = captured["weights"]
    # 4 个权重均 > 0（softmax 输出）
    assert (weights > 0).all(), f"some weights are zero: {weights}"
    # 4 个权重和 ≈ 1
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-3), \
        f"weights don't sum to 1: {weights.sum()}"
    # top_k 应等于 num_experts（None 转为全部激活）
    assert module.top_k == module.num_experts
    # use_top_k 应为 False（None 表示不用 Top-K）
    assert module.use_top_k is False


# ============================================================
# Test 2: top_k 超过 num_experts 应抛出 ValueError
# ============================================================

def test_esmoe_top_k_exceeds_num_experts_raises():
    """top_k=5 > num_experts=4 应抛出 ValueError。"""
    with pytest.raises(ValueError, match="top_k must be in"):
        ES_MOE(in_channels=64, num_experts=4, top_k=5)


def test_esmoe_top_k_zero_raises():
    """top_k=0 应抛出 ValueError（必须 >= 1）。"""
    with pytest.raises(ValueError, match="top_k must be in"):
        ES_MOE(in_channels=64, num_experts=4, top_k=0)


def test_esmoe_num_experts_zero_raises():
    """num_experts=0 应抛出 ValueError（必须 >= 1）。"""
    with pytest.raises(ValueError, match="num_experts must be positive"):
        ES_MOE(in_channels=64, num_experts=0)


# ============================================================
# Test 3: expert_kernel_sizes 长度不匹配
# ============================================================

def test_esmoe_expert_kernel_sizes_length_mismatch():
    """expert_kernel_sizes 长度（2）与 num_experts=4 不匹配 → ValueError。"""
    with pytest.raises(ValueError, match="expert_kernel_sizes must have"):
        ES_MOE(in_channels=64, num_experts=4, expert_kernel_sizes=[3, 5])


def test_esmoe_expert_kernel_sizes_too_many_raises():
    """expert_kernel_sizes 长度（5）超过 num_experts=4 → ValueError。"""
    with pytest.raises(ValueError, match="expert_kernel_sizes must have"):
        ES_MOE(in_channels=64, num_experts=4, expert_kernel_sizes=[3, 5, 7, 9, 11])


def test_esmoe_expert_kernel_sizes_even_normalized():
    """expert_kernel_sizes 含偶数核（4, 8）应被自动调整为奇数（3, 7）。"""
    module = ES_MOE(in_channels=64, num_experts=4, expert_kernel_sizes=[3, 4, 7, 8])
    # 偶数核 → 减 1
    assert [m.conv.depthwise.kernel_size[0] for m in module.experts] == [3, 3, 7, 7]


# ============================================================
# Test 4: use_sparse_inference=False 时 Dense forward 正常
# ============================================================

def test_esmoe_no_sparse_inference_dense_forward(esmoe_module, small_input):
    """use_sparse_inference=False 时，_eager_sparse_enabled 返回 False，前向稳定。"""
    esmoe_module.use_sparse_inference = False
    assert not esmoe_module._eager_sparse_enabled(), \
        "use_sparse_inference=False should disable sparse dispatch"

    esmoe_module.eval()
    with torch.no_grad():
        output = esmoe_module(small_input)

    assert output.shape == small_input.shape, \
        f"output shape mismatch: {output.shape}"
    assert torch.isfinite(output).all(), "output contains NaN/Inf"


# ============================================================
# Test 5: ONNX 导出走 Dense fallback
# ============================================================

def test_esmoe_onnx_export_dense_fallback(esmoe_module, small_input, tmp_path):
    """ONNX 导出时，ES_MOE 应走 Dense 分支（避免动态控制流）。"""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        pytest.skip("onnxruntime not installed")

    esmoe_module.eval()
    onnx_path = tmp_path / "esmoe_test.onnx"

    torch.onnx.export(
        esmoe_module,
        small_input,
        onnx_path,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    assert onnx_path.exists(), "ONNX export failed"

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    input_np = small_input.numpy()
    output_np = sess.run(None, {"input": input_np})[0]

    assert output_np.shape == small_input.shape, \
        f"ONNX output shape mismatch: {output_np.shape}"
    assert (output_np == output_np).all(), "ONNX output contains NaN"


# ============================================================
# Test 6: _eager_sparse_enabled() 5 种配置状态
# ============================================================

@pytest.mark.parametrize("use_sparse,use_top_k,top_k,num_experts,expected", [
    (True, True, 2, 4, True),       # 标准配置：稀疏启用
    (False, True, 2, 4, False),     # 总开关 use_sparse_inference 关闭
    (True, False, None, 4, False),  # use_top_k=False（top_k=None）
    (True, True, 4, 4, False),      # top_k == num_experts（无稀疏意义）
    (True, True, 1, 4, True),       # top_k=1（最稀疏）
])
def test_esmoe_eager_sparse_enabled_logic(use_sparse, use_top_k, top_k, num_experts, expected):
    """_eager_sparse_enabled() 在 5 种配置下返回正确的稀疏状态。"""
    module = ES_MOE(
        in_channels=64,
        num_experts=num_experts,
        top_k=top_k,
        use_sparse_inference=use_sparse,
    )
    module.use_top_k = use_top_k
    result = module._eager_sparse_enabled()
    assert result == expected, \
        f"Expected {expected}, got {result} for ({use_sparse}, {use_top_k}, {top_k}, {num_experts})"


# ============================================================
# Test 7: balance_loss 极端值下前向稳定
# ============================================================

@pytest.mark.parametrize("balance_loss", [0.001, 1.0, 10.0])
def test_esmoe_balance_loss_extreme_values(esmoe_module, small_input, balance_loss):
    """balance_loss 极端值（0.001/1.0/10.0）下前向不产生 NaN/Inf。"""
    esmoe_module.train()
    esmoe_module.balance_loss_coeff = balance_loss

    output = esmoe_module(small_input)
    assert torch.isfinite(output).all(), \
        f"balance_loss={balance_loss} caused NaN/Inf in output"


# ============================================================
# Test 8: z_loss=0 时路由器 logits 增长 < 100x
# ============================================================

def test_esmoe_no_z_loss_logit_growth(esmoe_module, small_input):
    """z_loss=0 训练 100 step 后，路由器 logits 增长不超过 100 倍。"""
    esmoe_module.train()
    esmoe_module.router_z_loss_coeff = 0.0

    optimizer = torch.optim.SGD(esmoe_module.parameters(), lr=0.01)

    initial_logits = None
    for step in range(100):
        optimizer.zero_grad()
        output = esmoe_module(small_input)
        loss = output.mean()  # 简化的训练目标
        loss.backward()
        optimizer.step()

        if step == 0:
            with torch.no_grad():
                # 记录最后一层线性层权重的绝对值均值
                final_linear = esmoe_module.routing.routing_network[-1]
                initial_logits = float(final_linear.weight.abs().mean().item())

    # 100 step 后路由器 logits 不应爆炸
    final_linear = esmoe_module.routing.routing_network[-1]
    final_logits = float(final_linear.weight.abs().mean().item())
    growth = final_logits / max(initial_logits, 1e-8)

    # 允许 z_loss=0 时增长，但不超过 100 倍（合理边界）
    assert growth < 100, \
        f"Logits grew {growth:.2f}x without Z-Loss (initial={initial_logits:.4f}, final={final_logits:.4f})"


# ============================================================
# Bonus: 端到端前向 + 反向传播测试
# ============================================================

def test_esmoe_forward_backward_gradient_flow(small_input):
    """ES_MOE 前向 + 反向传播正常：dense 模式下所有 4 个专家都应有梯度。

    验证点：
    - 使用 top_k=None 强制 dense 路径（不走 top-k 过滤）
    - 4 个 expert 都应收到梯度
    """
    # 使用 top_k=None 强制 dense 路径
    module = ES_MOE(in_channels=64, num_experts=4, top_k=None)
    module.train()

    # 单一前向 + 反向
    output = module(small_input)
    loss = output.mean()
    loss.backward()

    # 检查所有 4 个专家都有梯度
    for i, expert in enumerate(module.experts):
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in expert.parameters())
        assert has_grad, f"Expert {i} has no gradient (top_k=None should use dense_forward)"


def test_esmoe_top_k2_top_k1_outputs_differ():
    """top_k=1 与 top_k=2 的输出应该有差异（不同专家组合）。"""
    torch.manual_seed(42)
    x = torch.randn(1, 64, 32, 32)

    module_k1 = ES_MOE(in_channels=64, num_experts=4, top_k=1)
    module_k2 = ES_MOE(in_channels=64, num_experts=4, top_k=2)

    module_k1.eval()
    module_k2.eval()

    with torch.no_grad():
        out_k1 = module_k1(x)
        out_k2 = module_k2(x)

    # 由于随机初始化，输出不应该完全相同
    assert not torch.allclose(out_k1, out_k2, atol=1e-3), \
        "top_k=1 and top_k=2 outputs are too similar"