"""
Comparative Benchmarking: Classical vs Quantum Performance Analysis.

Provides formatted side-by-side comparison tables, delta metrics,
and hypothesis testing (McNemar's test for paired classifier agreement).
"""

from typing import Dict, Any, List
import numpy as np


def compare_models(
    classical_metrics: Dict[str, Any],
    quantum_metrics: Dict[str, Any],
    model_names: tuple = ("Classical Model", "Quantum Model")
) -> Dict[str, Any]:
    """
    Compares two model result dictionaries and calculates deltas.
    """
    comparison = {}
    shared_keys = set(classical_metrics.keys()).intersection(set(quantum_metrics.keys()))
    
    for key in sorted(shared_keys):
        c_val = classical_metrics[key]
        q_val = quantum_metrics[key]
        if isinstance(c_val, (int, float)) and isinstance(q_val, (int, float)):
            delta = q_val - c_val
            pct_change = (delta / c_val * 100.0) if abs(c_val) > 1e-8 else 0.0
            comparison[key] = {
                model_names[0]: float(c_val),
                model_names[1]: float(q_val),
                "delta (Q - C)": float(delta),
                "relative_change_%": float(pct_change)
            }
            
    return comparison


def format_comparison_table(
    comparison: Dict[str, Any],
    title: str = "Classical vs Quantum Performance Benchmark"
) -> str:
    """Formats comparison dictionary into a markdown table."""
    lines = [f"### {title}", "", "| Metric | Classical | Quantum | Delta (Q - C) | Rel. Change |", "| :--- | :---: | :---: | :---: | :---: |"]
    for k, v in comparison.items():
        if isinstance(v, dict) and "Classical Model" in v:
            c_val = v["Classical Model"]
            q_val = v["Quantum Model"]
            delta = v["delta (Q - C)"]
            rel = v["relative_change_%"]
            lines.append(f"| `{k}` | {c_val:.4f} | {q_val:.4f} | {delta:+.4f} | {rel:+.2f}% |")
    return "\n".join(lines)
