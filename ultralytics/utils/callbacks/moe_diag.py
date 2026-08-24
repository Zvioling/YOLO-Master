<<<<<<< HEAD
"""Custom callbacks for lightweight MoE routing diagnostics."""
=======
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Optional MoE diagnostic callbacks.

The callbacks module imports these helpers unconditionally, so they must remain
lightweight and dependency-safe even when plotting packages are unavailable.
"""
>>>>>>> origin/main

from __future__ import annotations

from pathlib import Path

<<<<<<< HEAD
from ultralytics.nn.modules.moe.diagnostics import collect_moe_diagnostics, format_moe_diagnostics
from ultralytics.nn.modules.moe.history import MoEDiagnosticsRecorder, export_moe_history_plots
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model


def _format_alert(alert: dict) -> str:
    """Render a concise one-line alert for training logs."""
    return (
        f"[MoE][alert] {alert['alert_type']} | {alert['layer_name']} | "
        f"E{alert['expert_id']} | step={alert['step']} | epoch={alert['epoch']} | "
        f"threshold={alert['threshold']:.3f} | window={alert['window']}"
    )


def create_moe_diagnostic_callback(
    interval: int = 10,
    collapse_threshold: float = 0.8,
    include_empty: bool = False,
    history_subdir: str = "moe_diagnostics",
    dead_threshold: float = 0.01,
    dead_window: int = 5,
    collapse_window: int = 3,
):
    """Create a train-batch-end callback that logs and persists MoE diagnostics."""
    state = {"step": 0, "recorder": None}
    interval = max(int(interval), 1)

    def _callback(trainer):
        state["step"] += 1
        if state["step"] % interval != 0:
            return

        model = unwrap_model(trainer.model)
        diagnostics = collect_moe_diagnostics(model, collapse_threshold=collapse_threshold)
        if not diagnostics and not include_empty:
            return

        if state["recorder"] is None:
            state["recorder"] = MoEDiagnosticsRecorder(
                Path(trainer.save_dir) / history_subdir,
                dead_threshold=dead_threshold,
                dead_window=dead_window,
                collapse_threshold=collapse_threshold,
                collapse_window=collapse_window,
            )

        alerts = state["recorder"].record(
            step=state["step"], epoch=trainer.epoch + 1, diagnostics=diagnostics, stage="train"
        )
        summary = format_moe_diagnostics(
            diagnostics,
            title=f"Train Step {state['step']} (epoch {trainer.epoch + 1})",
        )
        LOGGER.info(summary)
        for alert in alerts:
            LOGGER.warning(_format_alert(alert))

    return _callback


def create_moe_diagnostic_train_end_callback(history_subdir: str = "moe_diagnostics"):
    """Create a train-end callback that exports routing history plots."""

    def _callback(trainer):
        history_dir = Path(trainer.save_dir) / history_subdir
        written = export_moe_history_plots(history_dir)
        if written:
            LOGGER.info(f"[MoE] Exported {len(written)} diagnostic plots to {history_dir / 'plots'}")

    return _callback
=======

def create_moe_diagnostic_callback(interval: int = 1, output_dir: str | Path | None = None):
    """Create an epoch-end callback that can run MoE diagnostics when enabled.

    The returned callback is intentionally defensive: if the model has no MoE
    router modules or optional plotting dependencies are missing, training
    should continue without failing.
    """

    def on_train_epoch_end(trainer):
        epoch = int(getattr(trainer, "epoch", 0))
        if interval <= 0 or (epoch + 1) % interval:
            return
        try:
            from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker

            tracker = ExpertUsageTracker(trainer.model)
            setattr(trainer, "_moe_diagnostic_tracker", tracker)
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            setattr(trainer, "_moe_diagnostic_error", str(exc))

    return on_train_epoch_end


def create_moe_diagnostic_train_end_callback(output_dir: str | Path | None = None):
    """Create a train-end callback that prints a best-effort MoE diagnostic report."""

    def on_train_end(trainer):
        tracker = getattr(trainer, "_moe_diagnostic_tracker", None)
        if tracker is None:
            return
        try:
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            tracker.print_report()
            tracker.remove_hooks()
        except Exception as exc:
            setattr(trainer, "_moe_diagnostic_error", str(exc))

    return on_train_end
>>>>>>> origin/main
