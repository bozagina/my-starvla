# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import socket
import argparse
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework
import torch, os


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    ckpt_path = os.path.abspath(os.path.expanduser(args.ckpt_path))
    logging.info("Loading policy checkpoint from: %s", ckpt_path)
    vla = baseframework.from_pretrained( # TODO should auto detect framework from model path
        ckpt_path,
    )

    # Runtime ablation toggles for Path-A inference (no checkpoint mutation).
    if hasattr(vla, "enable_causal_feedback_inference"):
        if args.disable_path_a_inference and args.enable_path_a_inference:
            raise ValueError(
                "Conflicting flags: both --disable_path_a_inference and "
                "--enable_path_a_inference are set."
            )
        if args.disable_path_a_inference:
            vla.enable_causal_feedback_inference = False
            try:
                vla.config.framework.action_model.enable_causal_feedback_inference = False
            except Exception:
                pass
            logging.info("[ablation] Path-A inference is DISABLED by runtime flag.")
        elif args.enable_path_a_inference:
            vla.enable_causal_feedback_inference = True
            try:
                vla.config.framework.action_model.enable_causal_feedback_inference = True
            except Exception:
                pass
            logging.info("[ablation] Path-A inference is ENABLED by runtime flag.")
        else:
            logging.info(
                "[ablation] Path-A inference follows checkpoint/config default: %s",
                bool(getattr(vla, "enable_causal_feedback_inference", True)),
            )

    if hasattr(vla, "causal_feedback_scale") and args.path_a_feedback_scale is not None:
        try:
            vla.causal_feedback_scale = float(args.path_a_feedback_scale)
            try:
                vla.config.framework.action_model.causal_feedback_scale = float(args.path_a_feedback_scale)
            except Exception:
                pass
            logging.info("[ablation] Override Path-A feedback scale to %.6f", float(vla.causal_feedback_scale))
        except Exception as e:
            logging.warning("Failed to override path_a_feedback_scale: %s", e)

    if args.use_bf16: # False
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "simpler_env"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout" , type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument(
        "--disable_path_a_inference",
        action="store_true",
        help="Disable Path-A causal feedback tokens at inference time for strict A/B eval.",
    )
    parser.add_argument(
        "--enable_path_a_inference",
        action="store_true",
        help="Force-enable Path-A causal feedback tokens at inference time.",
    )
    parser.add_argument(
        "--path_a_feedback_scale",
        type=float,
        default=None,
        help="Optional runtime override for Path-A feedback scale.",
    )
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
