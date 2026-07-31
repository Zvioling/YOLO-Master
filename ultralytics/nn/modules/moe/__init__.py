# 🐧Please note that this file has been modified by Tencent on 2026/01/16. All Tencent Modifications are Copyright (C) 2026 Tencent.# 🐧Please note that this file has been modified by Tencent on 2026/01/09. All Tencent Modifications are Copyright (C) 2026 Tencent.
"""
Mixture-of-Experts (MoE) modules, routing layers, and compatibility shims.

This module provides several MoE variants and routers optimized for inference efficiency,
plus backward-compatibility aliases so legacy checkpoints can be loaded without changes.
"""

from .modules import (
    UltraOptimizedMoE,
    AdaptiveCapacityMoE,
    ES_MOE,
    OptimizedMOE,
    OptimizedMOEImproved,
    MOE,
    EfficientSpatialRouterMoE,
    ModularRouterExpertMoE,
    HyperSplitMoE,
    HyperFusedMoE,
    HyperUltimateMoE,
    UltimateOptimizedMoE,
    AdaptiveGateMoE,
    DualStreamGateRouter,
    FusedAdaptiveGateMoE,
    HybridAdaptiveGateMoE,
    LowRankHybridAdaptiveGateMoE,
    RefinedLowRankHybridAdaptiveGateMoE,
    VisualDetailGate,
    PyramidContextMixer,
    DetailAwareLowRankHybridAdaptiveGateMoE,
    ContextRefinedLowRankHybridAdaptiveGateMoE,
    VisualEnhancedAdaptiveGateMoE,
    A2C2fMoE,
    ABlockMoE,
)

from .shared_expert_moe import SharedExpertMoE, _SHARED_EXPERT_POOLS

from .experts import (
    OptimizedSimpleExpert,
    FusedGhostExpert,
    SimpleExpert,
    GhostExpert,
    InvertedResidualExpert,
    SharedInvertedExpertGroup,
    EfficientExpertGroup,
    DepthwiseSeparableConv
)

from .routers import (
    UltraEfficientRouter,
    BaseRouter,
    EfficientSpatialRouter,
    AdaptiveRoutingLayer,
    LocalRoutingLayer,
    AdvancedRoutingLayer,
    DynamicRoutingLayer
)

from .utils import (
    FlopsUtils,
    get_safe_groups,
    BatchedExpertComputation,
)

# 兼容旧版缺失符号（YOLO-Master Fork 未实现以下函数）
try:
    from .utils import is_core_moe_block  # type: ignore
except ImportError:
    is_core_moe_block = None  # type: ignore
try:
    from .utils import model_has_core_moe  # type: ignore
except ImportError:
    model_has_core_moe = None  # type: ignore
try:
    from .utils import iter_core_moe_expert_params  # type: ignore
except ImportError:
    iter_core_moe_expert_params = None  # type: ignore

from .analysis import ExpertUsageTracker, diagnose_model, RoutingCollapseDetector
from .diagnostics import MoELayerDiagnostic, collect_moe_diagnostics, diagnostics_to_dict, format_moe_diagnostics

# 兼容 YOLO-Master Fork 缺失的模块
try:
    from .history import MoEDiagnosticsRecorder, export_moe_history_plots  # type: ignore
except ImportError:
    MoEDiagnosticsRecorder = None  # type: ignore
    export_moe_history_plots = None  # type: ignore
try:
    from .pruning import prune_moe_model  # type: ignore
except ImportError:
    prune_moe_model = None  # type: ignore
try:
    from .scheduler import (  # type: ignore
        MoEDynamicScheduler, MoEDynamicSchedulerConfig,
        MoEDynamicScheduleState, compute_gini,
    )
except ImportError:
    MoEDynamicScheduler = None  # type: ignore
    MoEDynamicSchedulerConfig = None  # type: ignore
    MoEDynamicScheduleState = None  # type: ignore
    compute_gini = None  # type: ignore

__all__ = [
    "UltraOptimizedMoE",
    "AdaptiveCapacityMoE",
    "ES_MOE",
    "OptimizedMOE",
    "OptimizedMOEImproved",
    "MOE",
    "EfficientSpatialRouterMoE",
    "ModularRouterExpertMoE",
    "HyperSplitMoE",
    "HyperFusedMoE",
    "HyperUltimateMoE",
    "UltimateOptimizedMoE",
    "AdaptiveGateMoE",
    "DualStreamGateRouter",
    "FusedAdaptiveGateMoE",
    "HybridAdaptiveGateMoE",
    "LowRankHybridAdaptiveGateMoE",
    "RefinedLowRankHybridAdaptiveGateMoE",
    "VisualDetailGate",
    "PyramidContextMixer",
    "DetailAwareLowRankHybridAdaptiveGateMoE",
    "ContextRefinedLowRankHybridAdaptiveGateMoE",
    "VisualEnhancedAdaptiveGateMoE",
    "A2C2fMoE",
    "ABlockMoE",
    "SharedExpertMoE",
    "_SHARED_EXPERT_POOLS",
    "OptimizedSimpleExpert",
    "FusedGhostExpert",
    "SimpleExpert",
    "GhostExpert",
    "InvertedResidualExpert",
    "SharedInvertedExpertGroup",
    "EfficientExpertGroup",
    "DepthwiseSeparableConv",
    "UltraEfficientRouter",
    "BaseRouter",
    "EfficientSpatialRouter",
    "AdaptiveRoutingLayer",
    "LocalRoutingLayer",
    "AdvancedRoutingLayer",
    "DynamicRoutingLayer",
    "FlopsUtils",
    "get_safe_groups",
    "BatchedExpertComputation",
    "is_core_moe_block",
    "model_has_core_moe",
    "iter_core_moe_expert_params",
    "ExpertUsageTracker",
    "RoutingCollapseDetector",
    "diagnose_model",
    "MoELayerDiagnostic",
    "collect_moe_diagnostics",
    "diagnostics_to_dict",
    "format_moe_diagnostics",
    "MoEDiagnosticsRecorder",
    "export_moe_history_plots",
    "prune_moe_model",
    "MoEDynamicScheduler",
    "MoEDynamicSchedulerConfig",
    "MoEDynamicScheduleState",
    "compute_gini",
]
