import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def load_run_scalars(events_path: str) -> Dict[str, pd.DataFrame]:
    size_guidance = {
        event_accumulator.SCALARS: 0,
        event_accumulator.HISTOGRAMS: 0,
        event_accumulator.IMAGES: 0,
        event_accumulator.AUDIO: 0,
        event_accumulator.COMPRESSED_HISTOGRAMS: 0,
        event_accumulator.TENSORS: 0,
    }
    acc = event_accumulator.EventAccumulator(events_path, size_guidance=size_guidance)
    acc.Reload()
    result = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        steps = [e.step for e in events]
        wall_times = [e.wall_time for e in events]
        values = [e.value for e in events]
        df = pd.DataFrame(
            {
                "step": np.array(steps, dtype=np.int64),
                "wall_time": np.array(wall_times, dtype=np.float64),
                "value": np.array(values, dtype=np.float32),
            }
        )
        result[tag] = df
    return result


def find_event_file(run_dir: str) -> str:
    for name in os.listdir(run_dir):
        if name.startswith("events.out.tfevents"):
            return os.path.join(run_dir, name)
    raise FileNotFoundError(f"No tfevents file found in {run_dir}")


def load_all_runs(base_dir: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    runs = {}
    for name in sorted(os.listdir(base_dir)):
        run_dir = os.path.join(base_dir, name)
        if not os.path.isdir(run_dir):
            continue
        try:
            events_path = find_event_file(run_dir)
        except FileNotFoundError:
            continue
        scalars = load_run_scalars(events_path)
        runs[name] = scalars
    return runs


def export_to_csv(
    runs: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for run_name, tags in runs.items():
        safe_run_name = run_name.replace("/", "_")
        for tag, df in tags.items():
            safe_tag = tag.replace("/", "_")
            out_path = os.path.join(output_dir, f"{safe_run_name}__{safe_tag}.csv")
            df.to_csv(out_path, index=False)


def build_xy_from_runs(
    runs: Dict[str, Dict[str, pd.DataFrame]],
    x_source: str,
    y_tag: str,
    x_tag: str = "",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    curves = {}
    for run_name, tags in runs.items():
        if y_tag not in tags:
            continue
        y_df = tags[y_tag]
        if x_source == "step":
            x = y_df["step"].to_numpy()
        elif x_source == "wall_time":
            x = y_df["wall_time"].to_numpy()
        elif x_source == "scalar":
            if x_tag == "":
                continue
            if x_tag not in tags:
                continue
            x_df = tags[x_tag]
            merged = pd.merge(
                x_df[["step", "value"]],
                y_df[["step", "value"]],
                on="step",
                suffixes=("_x", "_y"),
            )
            x = merged["value_x"].to_numpy()
            y = merged["value_y"].to_numpy()
            curves[run_name] = (x, y)
            continue
        else:
            continue
        y = y_df["value"].to_numpy()
        curves[run_name] = (x, y)
    return curves


def plot_curves(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    x_label: str,
    y_label: str,
    title: str,
    output_path: str,
) -> None:
    plt.figure(figsize=(6, 4))
    for run_name, (x, y) in curves.items():
        plt.plot(x, y, label=run_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()


def aggregate_values(df: pd.DataFrame, method: str) -> float:
    if df.empty:
        return float("nan")
    if method == "last":
        return float(df["value"].iloc[-1])
    return float(df["value"].mean())


def aggregate_array(values: np.ndarray, method: str) -> float:
    if values.size == 0:
        return float("nan")
    if method == "last":
        return float(values[-1])
    return float(values.mean())


def simplify_run_name(run_name: str) -> str:
    base = os.path.basename(run_name)
    parts = base.split("_")
    if len(parts) >= 4 and parts[0] == "run" and parts[-1].startswith("seed") and parts[-2].isdigit():
        return "_".join(parts[1:-2])
    if base.startswith("run_"):
        return base[len("run_"):]
    return base


def plot_dual_curves(
    curves_main: Dict[str, Tuple[np.ndarray, np.ndarray]],
    curves_eval: Dict[str, Tuple[np.ndarray, np.ndarray]],
    x_label_main: str,
    y_label_main: str,
    title_main: str,
    x_label_eval: str,
    y_label_eval: str,
    title_eval: str,
    output_path: str,
) -> None:
    if output_path:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax_main = axes[0]
        for run_name, (x, y) in curves_main.items():
            ax_main.plot(x, y, label=run_name)
        ax_main.set_xlabel(x_label_main)
        ax_main.set_ylabel(y_label_main)
        ax_main.set_title(title_main)
        if curves_main:
            ax_main.legend()
        ax_eval = axes[1]
        for run_name, (x, y) in curves_eval.items():
            ax_eval.plot(x, y, label=run_name)
        ax_eval.set_xlabel(x_label_eval)
        ax_eval.set_ylabel(y_label_eval)
        ax_eval.set_title(title_eval)
        if curves_eval:
            ax_eval.legend()
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return
    fig_main = plt.figure(figsize=(6, 4))
    ax_main = fig_main.add_subplot(111)
    for run_name, (x, y) in curves_main.items():
        ax_main.plot(x, y, label=run_name)
    ax_main.set_xlabel(x_label_main)
    ax_main.set_ylabel(y_label_main)
    ax_main.set_title(title_main)
    if curves_main:
        ax_main.legend()
    fig_eval = plt.figure(figsize=(6, 4))
    ax_eval = fig_eval.add_subplot(111)
    for run_name, (x, y) in curves_eval.items():
        ax_eval.plot(x, y, label=run_name)
    ax_eval.set_xlabel(x_label_eval)
    ax_eval.set_ylabel(y_label_eval)
    ax_eval.set_title(title_eval)
    if curves_eval:
        ax_eval.legend()
    fig_main.tight_layout()
    fig_eval.tight_layout()
    plt.show()
    plt.close(fig_main)
    plt.close(fig_eval)


def build_tri_metrics(
    runs: Dict[str, Dict[str, pd.DataFrame]],
    x_tag: str,
    y_tag: str,
    c_tag: str,
    agg: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    xs: List[float] = []
    ys: List[float] = []
    cs: List[float] = []
    names: List[str] = []
    for run_name, tags in runs.items():
        if x_tag not in tags or y_tag not in tags or c_tag not in tags:
            continue
        x_val = aggregate_values(tags[x_tag], agg)
        y_val = aggregate_values(tags[y_tag], agg)
        c_val = aggregate_values(tags[c_tag], agg)
        if np.isnan(x_val) or np.isnan(y_val) or np.isnan(c_val):
            continue
        xs.append(x_val)
        ys.append(y_val)
        cs.append(c_val)
        names.append(simplify_run_name(run_name))
    return np.array(xs), np.array(ys), np.array(cs), names


def plot_scatter_xyc(
    xs: np.ndarray,
    ys: np.ndarray,
    cs: np.ndarray,
    names: List[str],
    x_label: str,
    y_label: str,
    c_label: str,
    title: str,
    output_path: str,
) -> None:
    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(xs, ys, c=cs, cmap="viridis", edgecolors="black")
    for x, y, name in zip(xs, ys, names):
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=6)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    cbar = plt.colorbar(scatter)
    cbar.set_label(c_label)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()


def plot_scatter_with_eval(
    xs: np.ndarray,
    ys: np.ndarray,
    cs: np.ndarray,
    names: List[str],
    curves_eval: Dict[str, Tuple[np.ndarray, np.ndarray]],
    x_label: str,
    y_label: str,
    c_label: str,
    title: str,
    x_label_eval: str,
    y_label_eval: str,
    title_eval: str,
    output_path: str,
) -> None:
    if output_path:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax_scatter = axes[0]
        scatter = ax_scatter.scatter(xs, ys, c=cs, cmap="viridis", edgecolors="black")
        for x, y, name in zip(xs, ys, names):
            ax_scatter.annotate(name, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=6)
        ax_scatter.set_xlabel(x_label)
        ax_scatter.set_ylabel(y_label)
        ax_scatter.set_title(title)
        cbar = fig.colorbar(scatter, ax=ax_scatter)
        cbar.set_label(c_label)
        ax_eval = axes[1]
        for run_name, (x_e, y_e) in curves_eval.items():
            ax_eval.plot(x_e, y_e, label=run_name)
        ax_eval.set_xlabel(x_label_eval)
        ax_eval.set_ylabel(y_label_eval)
        ax_eval.set_title(title_eval)
        if curves_eval:
            ax_eval.legend()
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return
    fig_scatter = plt.figure(figsize=(6, 4))
    ax_scatter = fig_scatter.add_subplot(111)
    scatter = ax_scatter.scatter(xs, ys, c=cs, cmap="viridis", edgecolors="black")
    for x, y, name in zip(xs, ys, names):
        ax_scatter.annotate(name, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=6)
    ax_scatter.set_xlabel(x_label)
    ax_scatter.set_ylabel(y_label)
    ax_scatter.set_title(title)
    cbar = fig_scatter.colorbar(scatter, ax=ax_scatter)
    cbar.set_label(c_label)
    fig_eval = plt.figure(figsize=(6, 4))
    ax_eval = fig_eval.add_subplot(111)
    for run_name, (x_e, y_e) in curves_eval.items():
        ax_eval.plot(x_e, y_e, label=run_name)
    ax_eval.set_xlabel(x_label_eval)
    ax_eval.set_ylabel(y_label_eval)
    ax_eval.set_title(title_eval)
    if curves_eval:
        ax_eval.legend()
    fig_scatter.tight_layout()
    fig_eval.tight_layout()
    plt.show()
    plt.close(fig_scatter)
    plt.close(fig_eval)


def build_eff_return_metrics(
    runs: Dict[str, Dict[str, pd.DataFrame]],
    ess_tag: str,
    nearzero_tag: str,
    y_tag: str,
    kl_tag: str,
    agg: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    xs: List[float] = []
    ys: List[float] = []
    cs: List[float] = []
    names: List[str] = []
    for run_name, tags in runs.items():
        if ess_tag not in tags or nearzero_tag not in tags or y_tag not in tags or kl_tag not in tags:
            continue
        ess_df = tags[ess_tag]
        nz_df = tags[nearzero_tag]
        merged = pd.merge(
            ess_df[["step", "value"]],
            nz_df[["step", "value"]],
            on="step",
            suffixes=("_ess", "_nz"),
        )
        eff_values = merged["value_ess"].to_numpy() * (1.0 - merged["value_nz"].to_numpy())
        x_val = aggregate_array(eff_values, agg)
        y_val = aggregate_values(tags[y_tag], agg)
        c_val = aggregate_values(tags[kl_tag], agg)
        if np.isnan(x_val) or np.isnan(y_val) or np.isnan(c_val):
            continue
        xs.append(x_val)
        ys.append(y_val)
        cs.append(c_val)
        names.append(simplify_run_name(run_name))
    return np.array(xs), np.array(ys), np.array(cs), names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="tensorboard_all",
    )
    parser.add_argument(
        "--output_csv_dir",
        type=str,
        default="tensorboard_export",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["csv", "plot", "scatter3", "scatter_eff"],
        default="csv",
    )
    parser.add_argument(
        "--y_tag",
        type=str,
        default="",
    )
    parser.add_argument(
        "--x_source",
        type=str,
        choices=["step", "wall_time", "scalar"],
        default="step",
    )
    parser.add_argument(
        "--x_tag",
        type=str,
        default="",
    )
    parser.add_argument(
        "--x_label",
        type=str,
        default="step",
    )
    parser.add_argument(
        "--y_label",
        type=str,
        default="value",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="TensorBoard curves",
    )
    parser.add_argument(
        "--plot_output",
        type=str,
        default="",
    )
    parser.add_argument(
        "--tri_x_tag",
        type=str,
        default="",
    )
    parser.add_argument(
        "--tri_y_tag",
        type=str,
        default="",
    )
    parser.add_argument(
        "--tri_c_tag",
        type=str,
        default="",
    )
    parser.add_argument(
        "--tri_c_label",
        type=str,
        default="performance",
    )
    parser.add_argument(
        "--tri_agg",
        type=str,
        choices=["mean", "last"],
        default="mean",
    )
    parser.add_argument(
        "--eff_ess_tag",
        type=str,
        default="ESS/ESS_Eff_Norm_Old",
    )
    parser.add_argument(
        "--eff_nearzero_tag",
        type=str,
        default="Contribution/NearZero_U_Frac_Old",
    )
    parser.add_argument(
        "--eff_y_tag",
        type=str,
        default="Eval/Average_Return",
    )
    parser.add_argument(
        "--eff_kl_tag",
        type=str,
        default="Metrics/KL_Divergence",
    )
    parser.add_argument(
        "--eff_agg",
        type=str,
        choices=["mean", "last"],
        default="mean",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    runs = load_all_runs(base_dir)
    if args.mode == "csv":
        export_to_csv(runs, args.output_csv_dir)
        return
    if args.mode == "plot":
        if args.y_tag == "":
            raise ValueError("y_tag must be specified when mode=plot")
        curves = build_xy_from_runs(
            runs=runs,
            x_source=args.x_source,
            y_tag=args.y_tag,
            x_tag=args.x_tag,
        )
        eval_curves = build_xy_from_runs(
            runs=runs,
            x_source="step",
            y_tag="Eval/Average_Return",
            x_tag="",
        )
        plot_dual_curves(
            curves_main=curves,
            curves_eval=eval_curves,
            x_label_main=args.x_label,
            y_label_main=args.y_label,
            title_main=args.title,
            x_label_eval="step",
            y_label_eval="Eval/Average_Return",
            title_eval="Eval/Average_Return vs step",
            output_path=args.plot_output,
        )
        return
    if args.mode == "scatter3":
        if args.tri_x_tag == "" or args.tri_y_tag == "" or args.tri_c_tag == "":
            raise ValueError("tri_x_tag, tri_y_tag and tri_c_tag must be specified when mode=scatter3")
        xs, ys, cs, names = build_tri_metrics(
            runs=runs,
            x_tag=args.tri_x_tag,
            y_tag=args.tri_y_tag,
            c_tag=args.tri_c_tag,
            agg=args.tri_agg,
        )
        eval_curves = build_xy_from_runs(
            runs=runs,
            x_source="step",
            y_tag="Eval/Average_Return",
            x_tag="",
        )
        plot_scatter_with_eval(
            xs=xs,
            ys=ys,
            cs=cs,
            names=names,
            curves_eval=eval_curves,
            x_label=args.x_label,
            y_label=args.y_label,
            c_label=args.tri_c_label,
            title=args.title,
            x_label_eval="step",
            y_label_eval="Eval/Average_Return",
            title_eval="Eval/Average_Return vs step",
            output_path=args.plot_output,
        )
        return
    if args.mode == "scatter_eff":
        xs, ys, cs, names = build_eff_return_metrics(
            runs=runs,
            ess_tag=args.eff_ess_tag,
            nearzero_tag=args.eff_nearzero_tag,
            y_tag=args.eff_y_tag,
            kl_tag=args.eff_kl_tag,
            agg=args.eff_agg,
        )
        eval_curves = build_xy_from_runs(
            runs=runs,
            x_source="step",
            y_tag=args.eff_y_tag,
            x_tag="",
        )
        plot_scatter_with_eval(
            xs=xs,
            ys=ys,
            cs=cs,
            names=names,
            curves_eval=eval_curves,
            x_label="ESS_Eff_Norm_Old * (1 - NearZero_U_Frac_Old)",
            y_label=args.eff_y_tag,
            c_label=args.eff_kl_tag,
            title="Effective old-data utilization vs performance",
            x_label_eval="step",
            y_label_eval=args.eff_y_tag,
            title_eval=f"{args.eff_y_tag} vs step",
            output_path=args.plot_output,
        )
        return


if __name__ == "__main__":
    main()
