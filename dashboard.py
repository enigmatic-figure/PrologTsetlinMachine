#!/usr/bin/env python3
"""Dear PyGUI dashboard for exploring the Prolog Tsetlin Machine.

This dashboard provides an interactive frontend for:
- Training and visualizing the scalar binary Tsetlin Machine
- Exploring hyperparameters (clauses, features, states_per_action, specificity, threshold)
- Viewing literal catalogs and feature schemas
- Running inference and comparing predictions to targets
- Exporting trained models as portable .ptm artifacts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg

# Add the python package to the path
sys.path.insert(0, str(Path(__file__).parent / "python"))

from prolog_tsetlin import (
    FeatureSchema,
    FieldKind,
    LiteralCatalog,
    ScalarBinaryTsetlinMachine,
    TMSnapshot,
    export_packed_tm,
)


# =============================================================================
# Application State
# =============================================================================


class DashboardState:
    """Holds all mutable state for the dashboard."""

    def __init__(self) -> None:
        # Hyperparameters
        self.number_of_clauses = 20
        self.number_of_features = 2
        self.states_per_action = 100
        self.specificity = 3.0
        self.threshold = 10
        self.seed = 7
        self.epochs = 150

        # Training data (XOR example by default)
        self.training_data = [
            {"x0": False, "x1": False},
            {"x0": False, "x1": True},
            {"x0": True, "x1": False},
            {"x0": True, "x1": True},
        ]
        self.targets = [0, 1, 1, 0]

        # Model state
        self.catalog: LiteralCatalog | None = None
        self.machine: ScalarBinaryTsetlinMachine | None = None
        self.is_trained = False
        self.predictions: list[int] | None = None
        self.last_training_log: list[str] = []

        # Export settings
        self.export_path = "out/artifacts/dashboard_model.ptm"
        self.model_name = "dashboard_xor_model"
        self.model_description = "XOR model trained from dashboard"

    def create_catalog(self) -> None:
        """Create the literal catalog from current training data."""
        schema = FeatureSchema.from_fields(
            **{f"x{i}": FieldKind.BOOLEAN for i in range(self.number_of_features)}
        )
        self.catalog = LiteralCatalog(schema)
        for i in range(self.number_of_features):
            self.catalog.category_eq(f"x{i}", True)

    def create_machine(self) -> None:
        """Create a new TM with current hyperparameters."""
        if self.catalog is None:
            self.create_catalog()
        assert self.catalog is not None
        literal_count = len(self.catalog.literals)
        self.machine = ScalarBinaryTsetlinMachine(
            number_of_clauses=self.number_of_clauses,
            number_of_features=literal_count,
            states_per_action=self.states_per_action,
            specificity=self.specificity,
            threshold=self.threshold,
            seed=self.seed,
        )

    def train(self) -> list[str]:
        """Train the model and return log messages."""
        if self.catalog is None:
            self.create_catalog()
        if self.machine is None:
            self.create_machine()

        assert self.catalog is not None
        assert self.machine is not None

        log = []
        log.append(f"Training with {self.number_of_clauses} clauses, {self.epochs} epochs")
        log.append(f"Hyperparameters: specificity={self.specificity}, threshold={self.threshold}")

        batch = self.catalog.encode(self.training_data, row_ids=[f"row_{i}" for i in range(len(self.training_data))])
        log.append(f"Encoded {batch.ta.row_count} rows with {batch.ta.literal_count} literals")

        self.machine.fit_literal_batch(batch.ta, self.targets, epochs=self.epochs)
        self.is_trained = True
        self.predictions = self.machine.predict(
            [batch.ta.row_values(i) for i in range(batch.ta.row_count)]
        )

        accuracy = sum(p == t for p, t in zip(self.predictions, self.targets)) / len(self.targets)
        log.append(f"Training complete. Accuracy: {accuracy * 100:.1f}%")

        return log

    def get_clause_state_table(self) -> list[list[Any]]:
        """Get clause states as a table for visualization."""
        if self.machine is None:
            return []

        table_data = []
        for clause_idx in range(min(self.machine.number_of_clauses, 10)):  # Limit display
            row = [clause_idx]
            for lit_idx in range(min(self.machine.number_of_features, 4)):
                state = self.machine.state(clause_idx, lit_idx)
                include = "✓" if self.machine.action_include(clause_idx, lit_idx) else "✗"
                row.append(f"{state} ({include})")
            table_data.append(row)
        return table_data


# Global state instance
STATE = DashboardState()


# =============================================================================
# UI Callbacks
# =============================================================================


def on_train_clicked(sender: Any, app_data: Any) -> None:
    """Handle train button click."""
    try:
        STATE.number_of_clauses = dpg.get_value("##num_clauses")
        STATE.states_per_action = dpg.get_value("##states_per_action")
        STATE.specificity = dpg.get_value("##specificity")
        STATE.threshold = dpg.get_value("##threshold")
        STATE.seed = dpg.get_value("##seed")
        STATE.epochs = dpg.get_value("##epochs")

        log = STATE.train()
        STATE.last_training_log = log

        # Update log display
        dpg.set_value("##training_log", "\n".join(log))

        # Update results table
        update_results_table()

        # Enable export button
        dpg.configure_item("##export_btn", enabled=STATE.is_trained)

    except Exception as e:
        dpg.set_value("##training_log", f"Error: {e}")


def on_reset_clicked(sender: Any, app_data: Any) -> None:
    """Reset the model state."""
    STATE.is_trained = False
    STATE.machine = None
    STATE.catalog = None
    STATE.predictions = None
    STATE.last_training_log = []

    dpg.set_value("##training_log", "")
    dpg.set_value("##results_table", [])
    dpg.configure_item("##export_btn", enabled=False)


def on_export_clicked(sender: Any, app_data: Any) -> None:
    """Export the trained model."""
    if STATE.machine is None or not STATE.is_trained:
        dpg.set_value("##export_status", "Error: No trained model to export")
        return

    try:
        snapshot = STATE.machine.snapshot()
        output_path = Path(STATE.export_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        artifact = export_packed_tm(
            snapshot,
            name=STATE.model_name,
            path=output_path,
            description=STATE.model_description,
            authors=["Dashboard User"],
            license="research",
            intended_use="exploration",
            limitations="prototype",
        )

        dpg.set_value(
            "##export_status",
            f"Exported: {output_path} (ID: {artifact.artifact_id[:16]}...)",
        )
    except Exception as e:
        dpg.set_value("##export_status", f"Export error: {e}")


def update_results_table() -> None:
    """Update the results table with predictions."""
    if STATE.predictions is None:
        return

    table_data = []
    for i, (pred, target) in enumerate(zip(STATE.predictions, STATE.targets)):
        match = "✓" if pred == target else "✗"
        input_str = ", ".join(str(int(v)) for v in [STATE.training_data[i]["x0"], STATE.training_data[i]["x1"]])
        table_data.append([i, input_str, target, pred, match])

    dpg.set_value("##results_table", table_data)


# =============================================================================
# UI Layout
# =============================================================================


def create_hyperparameter_window() -> None:
    """Create the hyperparameter configuration window."""
    with dpg.window(
        label="Hyperparameters",
        tag="##hyperparams_window",
        pos=(10, 10),
        size=(300, 500),
        no_collapse=False,
    ):
        dpg.add_text("Model Configuration", color=(0, 200, 255))
        dpg.add_separator()

        dpg.add_input_int(
            label="Number of Clauses",
            tag="##num_clauses",
            default_value=STATE.number_of_clauses,
            min_value=1,
            max_value=1000,
        )
        dpg.add_input_int(
            label="States per Action",
            tag="##states_per_action",
            default_value=STATE.states_per_action,
            min_value=10,
            max_value=1000,
        )
        dpg.add_input_float(
            label="Specificity",
            tag="##specificity",
            default_value=STATE.specificity,
            min_value=1.1,
            max_value=100.0,
            step=0.1,
        )
        dpg.add_input_int(
            label="Threshold",
            tag="##threshold",
            default_value=STATE.threshold,
            min_value=1,
            max_value=1000,
        )
        dpg.add_input_int(
            label="Random Seed",
            tag="##seed",
            default_value=STATE.seed,
            min_value=0,
        )
        dpg.add_input_int(
            label="Epochs",
            tag="##epochs",
            default_value=STATE.epochs,
            min_value=1,
            max_value=10000,
        )

        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Train",
                callback=on_train_clicked,
                width=100,
                color=(0, 200, 0),
            )
            dpg.add_button(
                label="Reset",
                callback=on_reset_clicked,
                width=100,
                color=(200, 0, 0),
            )


def create_training_log_window() -> None:
    """Create the training log window."""
    with dpg.window(
        label="Training Log",
        tag="##log_window",
        pos=(320, 10),
        size=(400, 300),
    ):
        dpg.add_input_text(
            tag="##training_log",
            multiline=True,
            readonly=True,
            height=200,
            width=-1,
        )


def create_results_window() -> None:
    """Create the inference results window."""
    with dpg.window(
        label="Inference Results",
        tag="##results_window",
        pos=(320, 320),
        size=(400, 300),
    ):
        dpg.add_text("Prediction vs Target", color=(0, 200, 255))
        dpg.add_separator()

        with dpg.table(
            header_row=True,
            borders_innerH=True,
            borders_outerH=True,
            borders_innerV=True,
            borders_outerV=True,
            tag="##results_table_container",
            width=-1,
            height=200,
        ):
            dpg.add_table_column(label="Row", width_fixed=True, init_width_or_weight=50)
            dpg.add_table_column(label="Input", width_fixed=True, init_width_or_weight=100)
            dpg.add_table_column(label="Target", width_fixed=True, init_width_or_weight=60)
            dpg.add_table_column(label="Predict", width_fixed=True, init_width_or_weight=60)
            dpg.add_table_column(label="Match", width_fixed=True, init_width_or_weight=50)

            # Add placeholder rows
            for _ in range(4):
                with dpg.table_row():
                    dpg.add_text("")
                    dpg.add_text("")
                    dpg.add_text("")
                    dpg.add_text("")
                    dpg.add_text("")


def create_export_window() -> None:
    """Create the model export window."""
    with dpg.window(
        label="Export Model",
        tag="##export_window",
        pos=(10, 520),
        size=(300, 200),
    ):
        dpg.add_text("Export Trained Model", color=(0, 200, 255))
        dpg.add_separator()

        dpg.add_input_text(
            label="Output Path",
            tag="##export_path",
            default_value=STATE.export_path,
            width=-1,
        )
        dpg.add_input_text(
            label="Model Name",
            tag="##model_name",
            default_value=STATE.model_name,
            width=-1,
        )
        dpg.add_input_text(
            label="Description",
            tag="##model_desc",
            default_value=STATE.model_description,
            width=-1,
            multiline=True,
            height=50,
        )

        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Export .ptm",
                tag="##export_btn",
                callback=on_export_clicked,
                enabled=False,
                width=120,
            )

        dpg.add_text("", tag="##export_status", color=(255, 200, 0))


def create_clause_visualization_window() -> None:
    """Create a window showing clause states."""
    with dpg.window(
        label="Clause States (Preview)",
        tag="##clause_window",
        pos=(730, 10),
        size=(350, 400),
    ):
        dpg.add_text("First 10 clauses, first 4 literals", color=(0, 200, 255))
        dpg.add_separator()
        dpg.add_text("(Trains to show detailed states)", color=(150, 150, 150))


# =============================================================================
# Main Application
# =============================================================================


def setup_viewport() -> None:
    """Configure the main viewport."""
    dpg.create_context()
    
    # Cross-platform font handling
    import platform
    system = platform.system()
    
    with dpg.font_registry():
        if system == "Windows":
            with dpg.font("C:\\Windows\\Fonts\\consola.ttf", 14) as mono_font:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Mono)
            default_font = dpg.load_font("C:\\Windows\\Fonts\\arial.ttf", 16)
        elif system == "Darwin":
            default_font = dpg.load_font("/System/Library/Fonts/Menlo.ttc", 14)
        else:
            # Linux - try common monospace fonts
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
                "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
            ]
            font_loaded = False
            for font_path in font_paths:
                if Path(font_path).exists():
                    default_font = dpg.load_font(font_path, 14)
                    font_loaded = True
                    break
            if not font_loaded:
                default_font = None  # Use default Dear PyGUI font
    
    dpg.setup_dearpygui()
    dpg.show_viewport(width=1100, height=750)


def main() -> None:
    """Main entry point for the dashboard."""
    # Create windows
    create_hyperparameter_window()
    create_training_log_window()
    create_results_window()
    create_export_window()
    create_clause_visualization_window()

    setup_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
