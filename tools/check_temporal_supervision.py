#!/usr/bin/env python3
"""Check whether current dataset setup supports t -> t+k temporal supervision.

This script focuses on the question:
Can we train with (S_task_t, action_chunk_t:t+k) -> S_task_t+k supervision?

Checks:
1) Config consistency:
   - data_mix exists in mixture registry
   - model action horizon matches dataset action_indices horizon
2) Temporal observability:
   - whether observation_indices currently include t+k
3) Optional runtime dataset statistics:
   - valid (non-tail) ratio for t+k across trajectories
   - timestamp monotonicity sanity checks
"""

from __future__ import annotations

import argparse
import difflib
import os
import random
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path("/Users/bazinga/code/my-starvla")
MIXTURE_PY = PROJECT_ROOT / "starVLA/dataloader/gr00t_lerobot/mixtures.py"
DATA_CONFIG_PY = PROJECT_ROOT / "starVLA/dataloader/gr00t_lerobot/data_config.py"


@dataclass
class CheckResult:
    ok: bool
    message: str


@dataclass
class RobotTemporalSpec:
    robot_type: str
    class_name: str
    observation_indices: list[int]
    action_indices: list[int]


def _nested_get(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _literal_int_list(node: ast.AST) -> list[int] | None:
    if isinstance(node, ast.List):
        vals: list[int] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                vals.append(int(elt.value))
            else:
                return None
        return vals
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "list":
        if len(node.args) == 1:
            inner = node.args[0]
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "range":
                ints: list[int] = []
                for arg in inner.args:
                    if not (isinstance(arg, ast.Constant) and isinstance(arg.value, int)):
                        return None
                    ints.append(int(arg.value))
                if len(ints) == 1:
                    return list(range(ints[0]))
                if len(ints) == 2:
                    return list(range(ints[0], ints[1]))
                if len(ints) == 3:
                    return list(range(ints[0], ints[1], ints[2]))
    return None


def parse_dataset_mixtures(path: Path) -> dict[str, list[tuple[str, float, str]]]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DATASET_NAMED_MIXTURES":
                    data = ast.literal_eval(node.value)
                    if not isinstance(data, dict):
                        raise ValueError("DATASET_NAMED_MIXTURES is not a dict.")
                    return data
    raise ValueError("DATASET_NAMED_MIXTURES not found.")


def parse_robot_temporal_specs(path: Path) -> dict[str, RobotTemporalSpec]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    class_specs: dict[str, dict[str, list[int]]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            obs: list[int] | None = None
            act: list[int] | None = None
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for t in stmt.targets:
                        if isinstance(t, ast.Name) and t.id == "observation_indices":
                            parsed = _literal_int_list(stmt.value)
                            if parsed is not None:
                                obs = parsed
                        if isinstance(t, ast.Name) and t.id == "action_indices":
                            parsed = _literal_int_list(stmt.value)
                            if parsed is not None:
                                act = parsed
            if obs is not None and act is not None:
                class_specs[node.name] = {
                    "observation_indices": obs,
                    "action_indices": act,
                }

    robot_map: dict[str, RobotTemporalSpec] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ROBOT_TYPE_CONFIG_MAP":
                    if not isinstance(node.value, ast.Dict):
                        raise ValueError("ROBOT_TYPE_CONFIG_MAP is not a dict AST.")
                    for k_node, v_node in zip(node.value.keys, node.value.values):
                        if not (isinstance(k_node, ast.Constant) and isinstance(k_node.value, str)):
                            continue
                        robot_type = str(k_node.value)
                        class_name = None
                        if isinstance(v_node, ast.Call) and isinstance(v_node.func, ast.Name):
                            class_name = v_node.func.id
                        if class_name is None or class_name not in class_specs:
                            continue
                        spec = class_specs[class_name]
                        robot_map[robot_type] = RobotTemporalSpec(
                            robot_type=robot_type,
                            class_name=class_name,
                            observation_indices=list(spec["observation_indices"]),
                            action_indices=list(spec["action_indices"]),
                        )
                    return robot_map
    raise ValueError("ROBOT_TYPE_CONFIG_MAP not found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-yaml",
        type=str,
        default="/Users/bazinga/code/my-starvla/starVLA/config/training/starvla_train_libero_mapanything_llava3d.yaml",
        help="Training yaml to inspect.",
    )
    parser.add_argument("--data-mix", type=str, default=None, help="Override datasets.vla_data.data_mix.")
    parser.add_argument("--data-root-dir", type=str, default=None, help="Override datasets.vla_data.data_root_dir.")
    parser.add_argument(
        "--future-action-window-size",
        type=int,
        default=None,
        help="Override framework.action_model.future_action_window_size.",
    )
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="Also open dataset metadata and compute valid-ratio for t+k.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=8,
        help="Max trajectories per dataset for timestamp monotonicity check in runtime mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for trajectory sampling in runtime mode.",
    )
    return parser.parse_args()


def check_data_mix_exists(data_mix: str, mixtures: dict[str, Any]) -> CheckResult:
    if data_mix in mixtures:
        return CheckResult(True, f"data_mix `{data_mix}` found in DATASET_NAMED_MIXTURES.")

    candidates = difflib.get_close_matches(data_mix, list(mixtures.keys()), n=5, cutoff=0.2)
    if len(candidates) == 0:
        return CheckResult(False, f"data_mix `{data_mix}` not found; no close key candidates.")
    return CheckResult(False, f"data_mix `{data_mix}` not found; close candidates: {candidates}.")


def summarize_robot_temporal_contract(
    robot_type: str,
    expected_chunk: int | None,
    robot_specs: dict[str, RobotTemporalSpec],
) -> list[CheckResult]:
    results: list[CheckResult] = []
    if robot_type not in robot_specs:
        results.append(CheckResult(False, f"robot_type `{robot_type}` not found in ROBOT_TYPE_CONFIG_MAP."))
        return results

    spec = robot_specs[robot_type]
    obs_idx = list(spec.observation_indices)
    act_idx = list(spec.action_indices)
    if len(act_idx) == 0:
        results.append(CheckResult(False, f"robot_type `{robot_type}` action_indices is empty."))
        return results

    k = max(act_idx)
    chunk = len(act_idx)
    min_a = min(act_idx)

    results.append(
        CheckResult(
            True,
            f"robot_type `{robot_type}`: observation_indices={obs_idx}, action_indices=[min={min_a}, max={k}, len={chunk}]",
        )
    )
    if expected_chunk is not None:
        if expected_chunk == chunk:
            results.append(CheckResult(True, f"action chunk length matches model: dataset={chunk}, model={expected_chunk}."))
        else:
            results.append(
                CheckResult(
                    False,
                    f"action chunk mismatch: dataset={chunk}, model={expected_chunk}.",
                )
            )

    if 0 in obs_idx:
        results.append(CheckResult(True, "observation_indices include t(=0)."))
    else:
        results.append(CheckResult(False, "observation_indices do not include t(=0)."))

    if k in obs_idx:
        results.append(CheckResult(True, f"observation_indices include t+k (k={k})."))
    else:
        results.append(
            CheckResult(
                False,
                f"observation_indices do NOT include t+k (k={k}); current setup cannot directly supervise S_task_t+k from dataloader output.",
            )
        )
    return results


def runtime_dataset_checks(
    data_root_dir: Path,
    data_name: str,
    robot_type: str,
    robot_specs: dict[str, RobotTemporalSpec],
    max_trajectories: int,
    seed: int,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    if robot_type not in robot_specs:
        return [CheckResult(False, f"runtime check skipped: robot_type `{robot_type}` missing in parsed spec.")]
    act_idx = list(robot_specs[robot_type].action_indices)
    if len(act_idx) == 0:
        return [CheckResult(False, f"runtime check skipped for `{data_name}`: empty action_indices.")]
    k = max(act_idx)

    dataset_path = data_root_dir / data_name
    if not dataset_path.exists():
        return [CheckResult(False, f"dataset path missing: {dataset_path}")]

    try:
        import sys

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset  # type: ignore
    except Exception as exc:
        return [CheckResult(False, f"runtime import failed (skip runtime checks): {exc}")]

    ds = make_LeRobotSingleDataset(
        data_root_dir=data_root_dir,
        data_name=data_name,
        robot_type=robot_type,
        delete_pause_frame=False,
        data_cfg={"video_backend": "torchvision_av"},
    )

    traj_lengths = list(ds.trajectory_lengths)
    total_steps = int(sum(traj_lengths))
    valid_steps = int(sum(max(0, int(L) - int(k)) for L in traj_lengths))
    valid_ratio = (valid_steps / total_steps) if total_steps > 0 else 0.0
    results.append(
        CheckResult(
            True,
            f"`{data_name}` steps={total_steps}, valid_for_t+k={valid_steps}, valid_ratio={valid_ratio:.4f}, k={k}",
        )
    )

    if total_steps == 0:
        results.append(CheckResult(False, f"`{data_name}` has zero steps."))
        return results

    rng = random.Random(seed)
    sampled_ids = list(ds.trajectory_ids)
    rng.shuffle(sampled_ids)
    sampled_ids = sampled_ids[: max(1, min(max_trajectories, len(sampled_ids)))]

    mono_ok = 0
    mono_fail = 0
    ts_missing = 0
    for tid in sampled_ids:
        try:
            if getattr(ds, "_lerobot_version", "v2.0") == "v3.0":
                traj = ds.get_trajectory_data_lerobot_v3(int(tid))
            else:
                traj = ds.get_trajectory_data(int(tid))
            if "timestamp" not in traj.columns:
                ts_missing += 1
                continue
            ts = traj["timestamp"].to_numpy()
            monotonic = True
            if len(ts) > 1:
                prev = ts[0]
                for cur in ts[1:]:
                    if cur < prev:
                        monotonic = False
                        break
                    prev = cur
            if monotonic:
                mono_ok += 1
            else:
                mono_fail += 1
        except Exception as exc:
            mono_fail += 1
            results.append(CheckResult(False, f"`{data_name}` trajectory {tid} timestamp check failed: {exc}"))

    results.append(
        CheckResult(
            mono_fail == 0 and ts_missing == 0,
            f"`{data_name}` sampled trajectories={len(sampled_ids)}, timestamp_monotonic_ok={mono_ok}, monotonic_fail={mono_fail}, missing_timestamp={ts_missing}",
        )
    )
    return results


def check_current_getitem_temporal_behavior() -> CheckResult:
    datasets_py = Path("/Users/bazinga/code/my-starvla/starVLA/dataloader/gr00t_lerobot/datasets.py")
    if not datasets_py.exists():
        return CheckResult(False, f"Source file missing: {datasets_py}")
    text = datasets_py.read_text(encoding="utf-8")
    class_marker = "class LeRobotMixtureDataset"
    class_pos = text.find(class_marker)
    if class_pos < 0:
        return CheckResult(False, "Cannot locate `LeRobotMixtureDataset` class in datasets.py.")

    mixture_text = text[class_pos:]
    getitem_marker = "def __getitem__(self, index: int) -> dict:"
    if getitem_marker not in mixture_text:
        return CheckResult(False, "Cannot locate `LeRobotMixtureDataset.__getitem__` implementation.")

    legacy_single_frame_marker = "image = data[video_key][0]"
    if legacy_single_frame_marker in mixture_text:
        return CheckResult(
            False,
            "Detected hard-coded single-frame selection in LeRobotMixtureDataset (`data[video_key][0]`).",
        )

    required_markers = ['"image_t"', '"image_tk"', '"valid_tk"']
    missing = [m for m in required_markers if m not in mixture_text]
    if missing:
        return CheckResult(
            False,
            f"LeRobotMixtureDataset.__getitem__ missing temporal output keys: {missing}",
        )

    return CheckResult(
        True,
        "LeRobotMixtureDataset.__getitem__ exposes temporal keys (image_t/image_tk/valid_tk) without hard-coded single-frame selection.",
    )


def print_result(prefix: str, result: CheckResult) -> None:
    flag = "PASS" if result.ok else "FAIL"
    print(f"[{flag}] {prefix}{result.message}")


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config_yaml)
    if not cfg_path.exists():
        print(f"[FAIL] config yaml not found: {cfg_path}")
        return 2

    cfg = load_yaml(cfg_path)
    data_mix = args.data_mix or _nested_get(cfg, ["datasets", "vla_data", "data_mix"], "")
    data_mix = str(data_mix).strip()
    data_root_dir = args.data_root_dir or _nested_get(cfg, ["datasets", "vla_data", "data_root_dir"], "")
    data_root_dir = str(data_root_dir).strip()
    faw = args.future_action_window_size
    if faw is None:
        faw = _nested_get(cfg, ["framework", "action_model", "future_action_window_size"], None)
    expected_chunk = (int(faw) + 1) if faw is not None else None

    print("=== Temporal Supervision Check ===")
    print(f"config_yaml={cfg_path}")
    print(f"data_mix={data_mix!r}")
    print(f"data_root_dir={data_root_dir!r}")
    print(f"expected_action_chunk_len(model)={expected_chunk}")
    print()

    critical_fail = False

    try:
        mixtures = parse_dataset_mixtures(MIXTURE_PY)
    except Exception as exc:
        print(f"[FAIL] cannot parse dataset mixtures: {exc}")
        return 2
    try:
        robot_specs = parse_robot_temporal_specs(DATA_CONFIG_PY)
    except Exception as exc:
        print(f"[FAIL] cannot parse robot temporal specs: {exc}")
        return 2

    mix_result = check_data_mix_exists(data_mix, mixtures)
    print_result("", mix_result)
    if not mix_result.ok:
        critical_fail = True
        print("\nStop: cannot continue dataset-level checks because data_mix key is invalid.")
        return 1

    mix_items = mixtures[data_mix]
    print(f"\nMixture entries ({len(mix_items)}):")
    for idx, (data_name, weight, robot_type) in enumerate(mix_items):
        print(f"  - #{idx}: data_name={data_name}, weight={weight}, robot_type={robot_type}")

    print("\n--- Contract Checks (config-level) ---")
    for idx, (data_name, weight, robot_type) in enumerate(mix_items):
        print(f"\n[{idx}] data_name={data_name}, robot_type={robot_type}")
        for res in summarize_robot_temporal_contract(
            robot_type=robot_type,
            expected_chunk=expected_chunk,
            robot_specs=robot_specs,
        ):
            print_result("  ", res)
            if not res.ok and "observation_indices include t+k" in res.message:
                critical_fail = True

    getitem_res = check_current_getitem_temporal_behavior()
    print("\n--- Pipeline Behavior Check ---")
    print_result("", getitem_res)
    if not getitem_res.ok:
        critical_fail = True

    if args.check_runtime:
        print("\n--- Runtime Dataset Checks ---")
        if len(data_root_dir) == 0:
            print("[FAIL] runtime check requires data_root_dir (from yaml or --data-root-dir).")
            return 2
        root = Path(data_root_dir)
        if not root.exists():
            print(f"[FAIL] data_root_dir does not exist: {root}")
            return 2

        for idx, (data_name, _weight, robot_type) in enumerate(mix_items):
            print(f"\n[{idx}] runtime data_name={data_name}, robot_type={robot_type}")
            for res in runtime_dataset_checks(
                data_root_dir=root,
                data_name=data_name,
                robot_type=robot_type,
                robot_specs=robot_specs,
                max_trajectories=args.max_trajectories,
                seed=args.seed,
            ):
                print_result("  ", res)

    print("\n=== Summary ===")
    if critical_fail:
        print("[FAIL] Current setup does NOT satisfy strict t -> t+k supervision requirements.")
        print("       Main reasons are usually: missing t+k in observation_indices and __getitem__ dropping temporal frames.")
        return 1
    print("[PASS] Current setup is compatible with strict t -> t+k supervision requirements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
