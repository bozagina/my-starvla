import argparse
from types import SimpleNamespace

import torch

from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
)


def build_config(tp_size: int):
    action_cfg = SimpleNamespace(
        action_model_type="DiT-B",
        hidden_size=1024,
        tp_size=tp_size,
        add_pos_embed=True,
        max_seq_len=32,
        action_dim=7,
        state_dim=7,
        future_action_window_size=3,
        action_horizon=4,
        past_action_window_size=0,
        repeated_diffusion_steps=2,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=16,
        num_inference_timesteps=2,
        num_target_vision_tokens=4,
        diffusion_model_cfg={
            "cross_attention_dim": 32,
            "dropout": 0.0,
            "final_dropout": False,
            "interleave_self_attention": False,
            "norm_type": "ada_norm",
            "num_layers": 2,
            "output_dim": 16,
            "positional_embeddings": None,
        },
    )
    mapanything_cfg = SimpleNamespace(vl_hidden_dim=128, num_vl_layers=2)
    framework_cfg = SimpleNamespace(
        action_model=action_cfg, mapanything_llava3d=mapanything_cfg
    )
    global_cfg = SimpleNamespace(framework=framework_cfg)
    return global_cfg


def run_single():
    torch.manual_seed(0)
    cfg = build_config(tp_size=1)
    model = LayerwiseFlowmatchingActionHead(global_config=cfg)
    B = 2
    T = cfg.framework.action_model.future_action_window_size + 1
    S = 8
    D = cfg.framework.mapanything_llava3d.vl_hidden_dim
    num_layers = len(model.model.transformer_blocks)
    vl_embs_list = [torch.randn(B, S, D) for _ in range(num_layers)]
    actions = torch.randn(B, T, cfg.framework.action_model.action_dim)
    state = torch.randn(B, 1, cfg.framework.action_model.state_dim)
    loss = model(vl_embs_list, actions, state)
    print("single_loss", float(loss))
    pred = model.predict_action(vl_embs_list, state)
    print("single_pred_shape", tuple(pred.shape))


def run_compare(tp_size: int):
    torch.manual_seed(0)
    cfg_ref = build_config(tp_size=1)
    cfg_tp = build_config(tp_size=tp_size)
    model_ref = LayerwiseFlowmatchingActionHead(global_config=cfg_ref)
    model_tp = LayerwiseFlowmatchingActionHead(global_config=cfg_tp)
    B = 2
    T = cfg_ref.framework.action_model.future_action_window_size + 1
    S = 8
    D = cfg_ref.framework.mapanything_llava3d.vl_hidden_dim
    num_layers = len(model_ref.model.transformer_blocks)
    vl_embs_list = [torch.randn(B, S, D) for _ in range(num_layers)]
    actions = torch.randn(B, T, cfg_ref.framework.action_model.action_dim)
    state = torch.randn(B, 1, cfg_ref.framework.action_model.state_dim)
    loss_ref = model_ref(vl_embs_list, actions, state)
    loss_tp = model_tp(vl_embs_list, actions, state)
    diff = (loss_tp - loss_ref).abs()
    print("compare_loss_ref", float(loss_ref))
    print("compare_loss_tp", float(loss_tp))
    print("compare_abs_diff", float(diff))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "compare"],
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=2,
    )
    args = parser.parse_args()
    if args.mode == "single":
        run_single()
    else:
        run_compare(args.tp_size)


if __name__ == "__main__":
    main()
