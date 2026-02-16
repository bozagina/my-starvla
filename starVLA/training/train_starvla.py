# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].


"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).  
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.  
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).  
"""

# Standard Library
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import re

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
from starVLA.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)


def build_accelerator(cfg) -> Accelerator:
    """Build accelerator with explicit gradient accumulation from config."""
    grad_accum_steps = int(getattr(cfg.trainer, "gradient_accumulation_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError(f"Invalid gradient_accumulation_steps={grad_accum_steps}, must be >= 1")

    deepspeed_plugin = None
    try:
        deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=grad_accum_steps)
    except TypeError:
        # Backward compatibility for older accelerate versions.
        deepspeed_plugin = DeepSpeedPlugin()
        logger.warning(
            "DeepSpeedPlugin does not support `gradient_accumulation_steps` ctor arg in this environment; "
            "falling back to plugin defaults."
        )

    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=grad_accum_steps,
    )
    accelerator.print(accelerator.state)
    accelerator.print(
        f"[accum] cfg.gradient_accumulation_steps={grad_accum_steps}, "
        f"accelerator.gradient_accumulation_steps={accelerator.gradient_accumulation_steps}"
    )
    return accelerator


def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # save config (both yaml and json for easy inspection)
        try:
            OmegaConf.save(cfg, output_dir / "config.yaml")
            with open(output_dir / "config.yaml", "r") as f_yaml, open(
                output_dir / "config.json", "w"
            ) as f_json:
                yaml_cfg = yaml.safe_load(f_yaml)
                json.dump(yaml_cfg, f_json, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save config to {output_dir}: {e}")

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.mapanything_llava3d.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from starVLA.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self.best_metric = None
        self.steps_since_improvement = 0
        trackers = getattr(self.config, "trackers", None)
        backend = None
        if trackers is not None:
            try:
                if "swanlab" in trackers:
                    backend = "swanlab"
                elif "wandb" in trackers:
                    backend = "wandb"
            except TypeError:
                backend = None
        if backend is None:
            backend = "wandb"
        self.logger_backend = backend
    
    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        cfg_grad_accum = int(getattr(self.config.trainer, "gradient_accumulation_steps", 1))
        actual_grad_accum = int(getattr(self.accelerator, "gradient_accumulation_steps", 1))
        if cfg_grad_accum != actual_grad_accum:
            raise RuntimeError(
                "Gradient accumulation mismatch detected: "
                f"cfg={cfg_grad_accum}, accelerator={actual_grad_accum}. "
                "Refuse to start training to avoid silent behavior drift."
            )

        # load pretrained weights
        self._init_checkpointing() # TODO merge with load pretrained weights

        # 根据  resume 调整 lr_scheduler
        self._adjust_lr_scheduler_for_resume()

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )
        self._patch_deepspeed_no_sync_if_needed()

        base_model = self.accelerator.unwrap_model(self.model)
        self._register_grad_hooks(base_model)

        self._init_wandb()

    def _patch_deepspeed_no_sync_if_needed(self):
        """
        Accelerate's `accumulate()` uses `no_sync` on non-sync micro steps.
        DeepSpeed ZeRO stage-2/3 with gradient partitioning disallows `no_sync`
        and raises:
          "no_sync context manager is incompatible with gradient partitioning ..."
        Replace instance-level `no_sync` with a no-op context manager to keep
        accumulation flow working.
        """
        dist_type = str(getattr(self.accelerator.state, "distributed_type", ""))
        if "DEEPSPEED" not in dist_type.upper():
            return
        engine = self.model
        if getattr(engine, "_starvla_no_sync_patched", False):
            return
        zero_partition_fn = getattr(engine, "zero_optimization_partition_gradients", None)
        if not callable(zero_partition_fn):
            return
        try:
            zero_partition = bool(zero_partition_fn())
        except Exception:
            zero_partition = False
        if not zero_partition:
            return

        @contextlib.contextmanager
        def _no_sync_passthrough():
            yield

        if hasattr(engine, "no_sync"):
            engine.no_sync = _no_sync_passthrough
            engine._starvla_no_sync_patched = True
            logger.warning(
                "Patched DeepSpeedEngine.no_sync -> nullcontext for ZeRO gradient partitioning "
                "compatibility (accumulate still works; gradients may sync each micro-step)."
            )


    def _adjust_lr_scheduler_for_resume(self):
        """根据已完成的步数调整学习率调度器状态"""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            
            # 方法1: 直接模拟已完成的步数（适用于大多数调度器）
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            
            # 或者方法2: 对于某些调度器，可以直接设置最后步数
            # if hasattr(self.lr_scheduler, '_step_count'):
            #     self.lr_scheduler._step_count = self.completed_steps
            
            logger.info(f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}")

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if not self.accelerator.is_main_process:
            return
        if getattr(self, "logger_backend", None) == "swanlab":
            import swanlab
            cfg_dict = OmegaConf.to_container(self.config, resolve=True)
            swanlab.init(
                project=self.config.wandb_project,
                workspace=self.config.wandb_entity,
                experiment_name=self.config.run_id,
                config=cfg_dict,
            )
        else:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 获取预训练检查点和是否恢复训练的标志
        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        # TODO retinking resume and load from pretrained_checkpoint
        if is_resume:
            # 恢复训练状态
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}")
                return None
            else:
                logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
                self.completed_steps = 0

        # 加载预训练权重
        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            try:
                self.completed_steps = int(re.search(r"steps_(\d+)_pytorch_model\.pt", pretrained_checkpoint).group(1))
            except AttributeError:
                logger.warning(f"Could not parse steps from pretrained checkpoint: {pretrained_checkpoint}")
                self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0
    

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """save current training state"""

        if self.accelerator.is_main_process:

            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
            # ✅ Save accessed configuration only
            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                # self.config.save_accessed_config(
                #     output_dir / "config.json", 
                #     use_original_values=False
                # )
                self.config.save_accessed_config(
                    output_dir / "config.yaml", 
                    use_original_values=False 
                )
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                # add learning rate 
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0] # see lr group in yaml.trainer.learning_rate

                geom_vision_only_steps = getattr(self.config.trainer, "geom_vision_only_steps", 0)
                lang_freeze_steps = getattr(self.config.trainer, "lang_freeze_steps", 0)
                phase = 0
                if geom_vision_only_steps and self.completed_steps < geom_vision_only_steps:
                    phase = 1
                elif lang_freeze_steps and self.completed_steps < lang_freeze_steps:
                    phase = 2
                metrics["debug/training_phase"] = phase

                # add epoch info
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

                # 1) 本地 JSONL 日志（方便离线 debug）
                try:
                    metrics_path = os.path.join(self.config.output_dir, "metrics.jsonl")
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(metrics) + "\n")
                except Exception as e:
                    logger.warning(f"Failed to write local metrics.jsonl: {e}")

                # 2) 远端日志（SwanLab / WandB）
                backend = getattr(self, "logger_backend", None)
                if backend == "swanlab":
                    try:
                        import swanlab
                        swanlab.log(metrics, step=self.completed_steps)
                    except Exception as e:
                        logger.warning(f"Failed to log metrics to SwanLab: {e}")
                elif backend == "wandb":
                    try:
                        wandb.log(metrics, step=self.completed_steps)
                    except Exception as e:
                        logger.warning(f"Failed to log metrics to WandB: {e}")

                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)
        # self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()
        self.optimizer.zero_grad()

        # create progress bar
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # update progress
            sync_step = bool(self.accelerator.sync_gradients)
            if sync_step:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data - t_start_data:.3f}",
                            "model_times": f"{t_end_model - t_start_model:.3f}",
                        }
                    )

            # Only run step-level side effects once per synchronized optimizer update.
            if not sync_step:
                continue

            # evaluate model
            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            # record metrics
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            stop_training = False
            if not dist.is_initialized() or dist.get_rank() == 0:
                stop_training = self._check_early_stopping(step_metrics)

            if dist.is_initialized():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                stop_flag = torch.tensor(int(stop_training), device=device)
                dist.broadcast(stop_flag, src=0)
                stop_training = bool(stop_flag.item())

            if stop_training:
                logger.info("Stopping training loop due to early stopping")
                break

            # save checkpoint
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        examples = self._get_next_batch()
        score = 0.0
        num_samples = len(examples)
        actions = [example["action"] for example in examples]  # label
        # Predict actions using the model
        output_dict = self.model.predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]  # B, T, D
            actions = np.array(actions)  # convert actions to numpy.ndarray
            # B, Chunk, dim = actions.shape
            num_pots = np.prod(actions.shape)
            # Compute the metric score
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_score = score / num_pots
            step_metrics["mse_score"] = average_score

        del examples
        dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps (cfg) = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(
                f"  Gradient accumulation steps (accelerator) = {self.accelerator.gradient_accumulation_steps}"
            )
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _register_grad_hooks(self, base_model=None):
        """register backward hooks on key modules to track output gradient norms (compatible with ZeRO)."""
        if base_model is None:
            base_model = self.model
        modules = []
        action_model = getattr(base_model, "action_model", None)
        if action_model is not None:
            dit = getattr(action_model, "model", None)
            if dit is not None:
                modules.append(dit)
            action_decoder = getattr(action_model, "action_decoder", None)
            if action_decoder is not None:
                modules.append(action_decoder)
        vlm_interface = getattr(base_model, "mapanythingllava3d_vlm_interface", None)
        vlm_model = getattr(vlm_interface, "model", None) if vlm_interface is not None else None
        if vlm_model is not None:
            for name in [
                "geometric_projector",
                "fusion_projector",
                "vision_projector",
            ]:
                module = getattr(vlm_model, name, None)
                if module is not None:
                    modules.append(module)

        handles = []

        def make_hook():
            def hook(module, grad_input, grad_output):
                if not grad_output:
                    return
                out_grad = grad_output[0]
                if out_grad is None:
                    return
                g = out_grad.detach().float()
                if g.numel() == 0:
                    return
                sq = g * g
                module._last_grad_l2 = sq.sum().sqrt().item()
                module._last_grad_rms = sq.mean().sqrt().item()

            return hook

        for m in modules:
            try:
                h = m.register_full_backward_hook(make_hook())
                handles.append(h)
            except Exception:
                continue
        self._grad_hook_handles = handles

    def _check_early_stopping(self, metrics: dict) -> bool:
        es_cfg = getattr(self.config.trainer, "early_stopping", None)
        if not es_cfg or not getattr(es_cfg, "enabled", False):
            return False

        metric_name = getattr(es_cfg, "metric", "action_dit_loss")
        mode = getattr(es_cfg, "mode", "min")
        patience = getattr(es_cfg, "patience", 10000)
        min_delta = getattr(es_cfg, "min_delta", 0.0)

        if metric_name not in metrics:
            return False

        value = float(metrics[metric_name])

        if self.best_metric is None:
            self.best_metric = value
            self.steps_since_improvement = 0
            logger.info(f"Early stopping initialized on metric {metric_name} with value {value:.6f}")
            return False

        improved = (value < self.best_metric - min_delta) if mode == "min" else (value > self.best_metric + min_delta)

        if improved:
            self.best_metric = value
            self.steps_since_improvement = 0
            logger.info(f"Early stopping metric {metric_name} improved to {value:.6f}")
            return False

        self.steps_since_improvement += 1

        if self.steps_since_improvement >= patience:
            logger.info(
                f"Early stopping triggered on metric {metric_name}: "
                f"no improvement for {self.steps_since_improvement} steps"
            )
            return True

        return False
    
    def _collect_debug_norm_metrics(self, metrics: dict):
        """Collect parameter / gradient / weight stats for VLM, vision, geom and action model."""
        if not self.accelerator.is_main_process:
            return

        try:
            rank = dist.get_rank() if dist.is_initialized() else -1
        except Exception:
            rank = -1
        metrics["debug/grad_collect_called"] = 1
        metrics["debug/grad_collect_rank"] = int(rank)

        def module_norms(module):
            if module is None:
                return None, None
            param_sq = 0.0
            grad_sq = 0.0
            grad_count = 0
            for p in module.parameters():
                if not p.requires_grad:
                    continue
                if p.data is not None:
                    w = p.data.float()
                    param_sq += torch.sum(w * w).item()
                if p.grad is not None:
                    g = p.grad.detach().float()
                    grad_sq += torch.sum(g * g).item()
                    grad_count += 1
            if param_sq == 0.0 and grad_sq == 0.0:
                return None, None
            param_norm = param_sq**0.5 if param_sq > 0.0 else None
            hook_grad_l2 = getattr(module, "_last_grad_l2", None)
            if hook_grad_l2 is not None:
                grad_norm = float(hook_grad_l2)
            else:
                grad_norm = grad_sq**0.5 if grad_count > 0 else None
            return param_norm, grad_norm

        def module_grad_debug(name, module):
            info = {"n_params": 0, "n_grad": 0, "n_grad_nonzero": 0}
            if module is None:
                metrics[f"debug/grad_info/{name}_n_params"] = 0
                metrics[f"debug/grad_info/{name}_n_grad"] = 0
                metrics[f"debug/grad_info/{name}_n_grad_nonzero"] = 0
                return
            for p in module.parameters():
                if not p.requires_grad:
                    continue
                info["n_params"] += 1
                if p.grad is not None:
                    info["n_grad"] += 1
                    with torch.no_grad():
                        v = p.grad.detach().float()
                        if v.abs().sum().item() > 0.0:
                            info["n_grad_nonzero"] += 1
            metrics[f"debug/grad_info/{name}_n_params"] = int(info["n_params"])
            metrics[f"debug/grad_info/{name}_n_grad"] = int(info["n_grad"])
            metrics[f"debug/grad_info/{name}_n_grad_nonzero"] = int(info["n_grad_nonzero"])

        def weight_grad_stats(module):
            if module is None:
                return None, None, None, None
            param_sq = 0.0
            param_count = 0
            grad_sq = 0.0
            grad_count = 0
            for p in module.parameters():
                if not p.requires_grad:
                    continue
                if p.data is not None:
                    w = p.data.float()
                    param_sq += torch.sum(w * w).item()
                    param_count += w.numel()
                if p.grad is not None:
                    g = p.grad.detach().float()
                    grad_sq += torch.sum(g * g).item()
                    grad_count += g.numel()
            if param_count == 0 and grad_count == 0:
                return None, None, None, None
            w_l2 = param_sq**0.5 if param_sq > 0.0 else None
            w_rms = (param_sq / param_count)**0.5 if param_sq > 0.0 and param_count > 0 else None
            hook_grad_l2 = getattr(module, "_last_grad_l2", None)
            hook_grad_rms = getattr(module, "_last_grad_rms", None)
            if hook_grad_l2 is not None:
                g_l2 = float(hook_grad_l2)
            else:
                g_l2 = grad_sq**0.5 if grad_count > 0 else None
            if hook_grad_rms is not None:
                g_rms = float(hook_grad_rms)
            else:
                g_rms = (grad_sq / grad_count)**0.5 if grad_count > 0 else None
            return w_l2, w_rms, g_l2, g_rms

        try:
            base_model = self.accelerator.unwrap_model(self.model)
            action_model = getattr(base_model, "action_model", None)
            dit = getattr(action_model, "model", None) if action_model is not None else None
            action_decoder = getattr(action_model, "action_decoder", None) if action_model is not None else None

            vlm_interface = getattr(base_model, "mapanythingllava3d_vlm_interface", None)
            vlm_model = getattr(vlm_interface, "model", None) if vlm_interface is not None else None
            vlm_language = getattr(vlm_model, "language_model", None) if vlm_model is not None else vlm_model

            module_grad_debug("dit", dit)
            dit_param, dit_grad = module_norms(dit)
            if dit_param is not None:
                metrics["debug/param_norm/dit"] = dit_param
            if dit_grad is not None:
                metrics["debug/grad_norm/dit"] = dit_grad

            module_grad_debug("action_decoder", action_decoder)
            dec_param, dec_grad = module_norms(action_decoder)
            if dec_param is not None:
                metrics["debug/param_norm/action_decoder"] = dec_param
            if dec_grad is not None:
                metrics["debug/grad_norm/action_decoder"] = dec_grad

            module_grad_debug("vlm_language", vlm_language)
            vlm_param, vlm_grad = module_norms(vlm_language)
            if vlm_param is not None:
                metrics["debug/param_norm/vlm_language"] = vlm_param
            if vlm_grad is not None:
                metrics["debug/grad_norm/vlm_language"] = vlm_grad

            if vlm_model is not None:
                geom_modules = [
                    ("geometric_model", getattr(vlm_model, "geometric_model", None)),
                    ("geometric_projector", getattr(vlm_model, "geometric_projector", None)),
                    ("fusion_projector", getattr(vlm_model, "fusion_projector", None)),
                ]
                for name, module in geom_modules:
                    module_grad_debug(name, module)
                    w_l2, w_rms, g_l2, g_rms = weight_grad_stats(module)
                    metrics[f"debug/geom_grad/{name}_l2"] = float(g_l2) if g_l2 is not None else 0.0
                    metrics[f"debug/geom_grad/{name}_rms"] = float(g_rms) if g_rms is not None else 0.0
                    metrics[f"debug/core_weight/{name}_w_rms"] = float(w_rms) if w_rms is not None else 0.0
                    if w_l2 is not None:
                        init_attr = "_param_l2_init"
                        last_attr = "_param_l2_last"
                        init_val = getattr(module, init_attr, None)
                        if init_val is None:
                            setattr(module, init_attr, w_l2)
                            setattr(module, last_attr, w_l2)
                            delta_from_start = 0.0
                            delta_from_last = 0.0
                        else:
                            last_val = getattr(module, last_attr, init_val)
                            delta_from_start = abs(w_l2 - init_val)
                            delta_from_last = abs(w_l2 - last_val)
                            setattr(module, last_attr, w_l2)
                        metrics[f"debug/param_delta/{name}_l2_from_start"] = float(delta_from_start)
                        metrics[f"debug/param_delta/{name}_l2_from_last"] = float(delta_from_last)

                vision_modules = [
                    ("vision_tower", getattr(vlm_model, "vision_tower", None)),
                    ("vision_projector", getattr(vlm_model, "vision_projector", None)),
                ]
                for name, module in vision_modules:
                    module_grad_debug(name, module)
                    w_l2, w_rms, g_l2, g_rms = weight_grad_stats(module)
                    metrics[f"debug/vlm_vision_grad/{name}_l2"] = float(g_l2) if g_l2 is not None else 0.0
                    metrics[f"debug/vlm_vision_grad/{name}_rms"] = float(g_rms) if g_rms is not None else 0.0
                    metrics[f"debug/core_weight/{name}_w_rms"] = float(w_rms) if w_rms is not None else 0.0
                    if w_l2 is not None:
                        init_attr = "_param_l2_init"
                        last_attr = "_param_l2_last"
                        init_val = getattr(module, init_attr, None)
                        if init_val is None:
                            setattr(module, init_attr, w_l2)
                            setattr(module, last_attr, w_l2)
                            delta_from_start = 0.0
                            delta_from_last = 0.0
                        else:
                            last_val = getattr(module, last_attr, init_val)
                            delta_from_start = abs(w_l2 - init_val)
                            delta_from_last = abs(w_l2 - last_val)
                            setattr(module, last_attr, w_l2)
                        metrics[f"debug/param_delta/{name}_l2_from_start"] = float(delta_from_start)
                        metrics[f"debug/param_delta/{name}_l2_from_last"] = float(delta_from_last)

                module_grad_debug("language_model", vlm_language)
                w_l2, w_rms, g_l2, g_rms = weight_grad_stats(vlm_language)
                metrics["debug/vlm_vision_grad/language_model_l2"] = float(g_l2) if g_l2 is not None else 0.0
                metrics["debug/vlm_vision_grad/language_model_rms"] = float(g_rms) if g_rms is not None else 0.0
                metrics["debug/core_weight/language_model_w_rms"] = float(w_rms) if w_rms is not None else 0.0
                if w_l2 is not None:
                    init_attr = "_param_l2_init"
                    last_attr = "_param_l2_last"
                    init_val = getattr(vlm_language, init_attr, None)
                    if init_val is None:
                        setattr(vlm_language, init_attr, w_l2)
                        setattr(vlm_language, last_attr, w_l2)
                        delta_from_start = 0.0
                        delta_from_last = 0.0
                    else:
                        last_val = getattr(vlm_language, last_attr, init_val)
                        delta_from_start = abs(w_l2 - init_val)
                        delta_from_last = abs(w_l2 - last_val)
                        setattr(vlm_language, last_attr, w_l2)
                    metrics["debug/param_delta/language_model_l2_from_start"] = float(delta_from_start)
                    metrics["debug/param_delta/language_model_l2_from_last"] = float(delta_from_last)
        except Exception as e:
            logger.warning(f"Failed to collect debug norm metrics: {e}")
            try:
                metrics["debug/grad_collect_error"] = str(e)
            except Exception:
                pass

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            # VLA task forward propagation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            debug_metrics = output_dict.get("debug_metrics", None)
            step_metrics = {"action_dit_loss": float(action_loss.item())}
            if isinstance(debug_metrics, dict):
                step_metrics.update(debug_metrics)

            # Guard against loss explosion before backward.
            if not torch.isfinite(total_loss).all():
                logger.warning(
                    "Detected non-finite loss at completed_steps=%d, skip optimizer update.",
                    self.completed_steps,
                )
                step_metrics["debug/nonfinite_loss"] = 1.0
                self.optimizer.zero_grad()
                return step_metrics

            # VLA backward propagation
            self.accelerator.backward(total_loss)

            geom_vision_only_steps = getattr(self.config.trainer, "geom_vision_only_steps", 0)
            lang_freeze_steps = getattr(self.config.trainer, "lang_freeze_steps", 0)
            if geom_vision_only_steps and self.completed_steps < geom_vision_only_steps:
                try:
                    base_model = self.accelerator.unwrap_model(self.model)
                    action_model = getattr(base_model, "action_model", None)
                    if action_model is not None:
                        for p in action_model.parameters():
                            if p.grad is not None:
                                p.grad.zero_()
                    vlm_interface = getattr(base_model, "mapanythingllava3d_vlm_interface", None)
                    vlm_model = getattr(vlm_interface, "model", None) if vlm_interface is not None else None
                    language_model = getattr(vlm_model, "language_model", None) if vlm_model is not None else None
                    if language_model is not None:
                        for p in language_model.parameters():
                            if p.grad is not None:
                                p.grad.zero_()
                except Exception as e:
                    logger.warning(f"Failed to zero grads during geom_vision_only warmup: {e}")
            elif lang_freeze_steps and self.completed_steps < lang_freeze_steps:
                try:
                    base_model = self.accelerator.unwrap_model(self.model)
                    vlm_interface = getattr(base_model, "mapanythingllava3d_vlm_interface", None)
                    vlm_model = getattr(vlm_interface, "model", None) if vlm_interface is not None else None
                    language_model = getattr(vlm_model, "language_model", None) if vlm_model is not None else None
                    if language_model is not None:
                        for p in language_model.parameters():
                            if p.grad is not None:
                                p.grad.zero_()
                except Exception as e:
                    logger.warning(f"Failed to zero language model grads during warmup: {e}")

            # gradient clipping
            clipped_grad_norm = None
            if self.config.trainer.gradient_clipping is not None:
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.trainer.gradient_clipping,
                    )
                    if isinstance(grad_norm, torch.Tensor):
                        clipped_grad_norm = float(grad_norm.detach().float().cpu().item())
                    elif grad_norm is not None:
                        clipped_grad_norm = float(grad_norm)
                    if clipped_grad_norm is not None:
                        step_metrics["debug/clip_grad_norm"] = clipped_grad_norm

            # collect grad/weight stats before optimizer step (only on synchronized steps)
            if self.accelerator.sync_gradients:
                self._collect_debug_norm_metrics(step_metrics)

            # optimizer step
            if self.accelerator.sync_gradients:
                if clipped_grad_norm is not None and not math.isfinite(clipped_grad_norm):
                    logger.warning(
                        "Detected non-finite clipped grad norm at completed_steps=%d, skip optimizer update.",
                        self.completed_steps,
                    )
                    step_metrics["debug/nonfinite_grad_norm"] = 1.0
                    self.optimizer.zero_grad()
                    return step_metrics

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

        return step_metrics

    def _finalize_training(self):
        """training end processing"""
        # save final model
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")


        # close W&B
        if self.accelerator.is_main_process:
            backend = getattr(self, "logger_backend", None)
            if backend == "swanlab":
                try:
                    import swanlab
                    swanlab.finish()
                except Exception:
                    pass
            elif backend == "wandb":
                wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    #  Wrap config to enable access tracking
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")
    accelerator = build_accelerator(cfg)

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)
    # prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # if cfg.is_debug:
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
