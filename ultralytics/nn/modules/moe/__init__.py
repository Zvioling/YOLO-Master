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
<<<<<<< HEAD
    FusedAdaptiveGateMoE,
    HybridAdaptiveGateMoE,
=======
    DualStreamGateRouterV2,
    FusedAdaptiveGateMoE,
    HybridAdaptiveGateMoE,
    HybridAdaptiveGateMoEv2,
    OptimalHybridGateMoE,
    MultiHeadRouterMoE,
    DiversifiedExpertMoE,
    GatedFusionMoE,
>>>>>>> origin/main
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

<<<<<<< HEAD
from .shared_expert_moe import SharedExpertMoE, _SHARED_EXPERT_POOLS

=======
>>>>>>> origin/main
from .experts import (
    OptimizedSimpleExpert,
    FusedGhostExpert,
    SimpleExpert,
    GhostExpert,
    InvertedResidualExpert,
    SharedInvertedExpertGroup,
    EfficientExpertGroup,
<<<<<<< HEAD
    DepthwiseSeparableConv
=======
    DepthwiseSeparableConv,
>>>>>>> origin/main
)

from .routers import (
    UltraEfficientRouter,
    BaseRouter,
    EfficientSpatialRouter,
    AdaptiveRoutingLayer,
    LocalRoutingLayer,
    AdvancedRoutingLayer,
<<<<<<< HEAD
    DynamicRoutingLayer
=======
    DynamicRoutingLayer,
>>>>>>> origin/main
)

from .utils import (
    FlopsUtils,
    get_safe_groups,
    BatchedExpertComputation,
<<<<<<< HEAD
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
=======
    index_add_aligned_,
    cast_like,
    is_core_moe_block,
    model_has_core_moe,
    iter_core_moe_expert_params,
)

from .analysis import ExpertUsageTracker, diagnose_model, RoutingCollapseDetector
from .diagnostics import MoELayerDiagnostic, collect_moe_diagnostics, diagnostics_to_dict, format_moe_diagnostics
from .history import MoEDiagnosticsRecorder, export_moe_history_plots
from .protocol import RoutingMetrics, normalize_routing_snapshot, routing_metrics, usage_gini
from .pruning import prune_moe_model, prune_moe_module
from .scheduler import (
    MoEDynamicScheduler,
    MoEDynamicSchedulerConfig,
    MoEDynamicScheduleState,
    MapSaturationScheduler,
    MapSaturationSchedulerConfig,
    MapSaturationScheduleState,
    compute_gini,
)
from .config import (
    MIXTURE_DEFAULTS,
    CLI_FIELDS,
    ResolvedMixtureConfig,
    annotate_mixture_yaml_config,
    resolve_mixture_config,
    apply_mixture_config,
)
from .hooks import (
    RouterHook,
    DetailGateHook,
    ContextMixerHook,
    FeatureRefinementHook,
    register_router_hook,
    build_router_hook,
    resolve_router_hooks,
    apply_router_hooks,
    registered_router_hooks,
)


# ── API Stability Tiers ──────────────────────────────────────────────
# `__all__` remains compatibility-complete; use these tier manifests for discovery.
# STABLE: production-ready, well-tested, backward-compatible API.
STABLE_MOE_CLASSES = frozenset(
    {
        "UltraOptimizedMoE",
        "ES_MOE",
        "MOE",
        "AdaptiveGateMoE",
        "OptimalHybridGateMoE",
        "UltimateOptimizedMoE",
    }
)

# EXPERIMENTAL: functional but not yet benchmarked at scale; API may change.
EXPERIMENTAL_MOE_CLASSES = frozenset(
    {
        "AdaptiveCapacityMoE",
        "OptimizedMOE",
        "OptimizedMOEImproved",
        "EfficientSpatialRouterMoE",
        "ModularRouterExpertMoE",
        "HyperSplitMoE",
        "HyperFusedMoE",
        "HyperUltimateMoE",
        "FusedAdaptiveGateMoE",
        "HybridAdaptiveGateMoE",
        "HybridAdaptiveGateMoEv2",
        "MultiHeadRouterMoE",
        "DiversifiedExpertMoE",
        "GatedFusionMoE",
        "LowRankHybridAdaptiveGateMoE",
        "RefinedLowRankHybridAdaptiveGateMoE",
        "DetailAwareLowRankHybridAdaptiveGateMoE",
        "ContextRefinedLowRankHybridAdaptiveGateMoE",
        "VisualEnhancedAdaptiveGateMoE",
    }
)

# DEPRECATED: retained during a release window after two consecutive recorded
# versions without YAML usage. Removal still requires checkpoint/API review.
DEPRECATED_MOE_CLASSES = frozenset()

# LEGACY: kept for checkpoint/YAML compatibility while their public contract is
# migrated to the canonical routing protocol.
LEGACY_MOE_CLASSES = frozenset(
    {
        "A2C2fMoE",
        "ABlockMoE",
    }
)

_MOE_TIER_SETS = (
    STABLE_MOE_CLASSES,
    EXPERIMENTAL_MOE_CLASSES,
    DEPRECATED_MOE_CLASSES,
    LEGACY_MOE_CLASSES,
)
_MOE_TIER_OVERLAP = set().union(
    *(
        _MOE_TIER_SETS[i] & _MOE_TIER_SETS[j]
        for i in range(len(_MOE_TIER_SETS))
        for j in range(i + 1, len(_MOE_TIER_SETS))
    )
)
if _MOE_TIER_OVERLAP:
    raise RuntimeError(f"MoE API tier overlap detected: {_MOE_TIER_OVERLAP}")


def is_stable_moe(class_name: str) -> bool:
    """Check if a MoE class is in the stable (production-ready) tier."""
    return class_name in STABLE_MOE_CLASSES


def is_experimental_moe(class_name: str) -> bool:
    """Check if a MoE class is experimental (API may change)."""
    return class_name in EXPERIMENTAL_MOE_CLASSES


def is_deprecated_moe(class_name: str) -> bool:
    """Check if a MoE class is in its release-window deprecation period."""
    return class_name in DEPRECATED_MOE_CLASSES


def is_legacy_moe(class_name: str) -> bool:
    """Check if a MoE class is retained for compatibility only."""
    return class_name in LEGACY_MOE_CLASSES

>>>>>>> origin/main

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
<<<<<<< HEAD
    "FusedAdaptiveGateMoE",
    "HybridAdaptiveGateMoE",
=======
    "DualStreamGateRouterV2",
    "FusedAdaptiveGateMoE",
    "HybridAdaptiveGateMoE",
    "HybridAdaptiveGateMoEv2",
    "OptimalHybridGateMoE",
    "MultiHeadRouterMoE",
    "DiversifiedExpertMoE",
    "GatedFusionMoE",
>>>>>>> origin/main
    "LowRankHybridAdaptiveGateMoE",
    "RefinedLowRankHybridAdaptiveGateMoE",
    "VisualDetailGate",
    "PyramidContextMixer",
    "DetailAwareLowRankHybridAdaptiveGateMoE",
    "ContextRefinedLowRankHybridAdaptiveGateMoE",
    "VisualEnhancedAdaptiveGateMoE",
    "A2C2fMoE",
    "ABlockMoE",
<<<<<<< HEAD
    "SharedExpertMoE",
    "_SHARED_EXPERT_POOLS",
=======
>>>>>>> origin/main
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
<<<<<<< HEAD
=======
    "index_add_aligned_",
    "cast_like",
>>>>>>> origin/main
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
<<<<<<< HEAD
    "MoEDynamicScheduler",
    "MoEDynamicSchedulerConfig",
    "MoEDynamicScheduleState",
    "compute_gini",
=======
    "prune_moe_module",
    "MoEDynamicScheduler",
    "MoEDynamicSchedulerConfig",
    "MoEDynamicScheduleState",
    "MapSaturationScheduler",
    "MapSaturationSchedulerConfig",
    "MapSaturationScheduleState",
    "compute_gini",
    "MIXTURE_DEFAULTS",
    "CLI_FIELDS",
    "ResolvedMixtureConfig",
    "annotate_mixture_yaml_config",
    "resolve_mixture_config",
    "apply_mixture_config",
    "RouterHook",
    "DetailGateHook",
    "ContextMixerHook",
    "FeatureRefinementHook",
    "register_router_hook",
    "build_router_hook",
    "resolve_router_hooks",
    "apply_router_hooks",
    "registered_router_hooks",
    "STABLE_MOE_CLASSES",
    "EXPERIMENTAL_MOE_CLASSES",
    "DEPRECATED_MOE_CLASSES",
    "LEGACY_MOE_CLASSES",
    "is_stable_moe",
    "is_experimental_moe",
    "is_deprecated_moe",
    "is_legacy_moe",
    "RoutingMetrics",
    "normalize_routing_snapshot",
    "routing_metrics",
    "usage_gini",
>>>>>>> origin/main
]
