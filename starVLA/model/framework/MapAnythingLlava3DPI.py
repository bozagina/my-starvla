from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    get_action_model,
    LayerwiseFlowmatchingActionHead,
)
from starVLA.training.trainer_utils.trainer_tools import resize_images
from deployment.model_server.tools.image_tools import to_pil_preserve


logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("MapAnythingLlava3DPI")
class MapAnythingLlava3D_PI(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.mapanythingllava3d_vlm_interface = get_vlm_model(config=self.config)

        vlm_core = getattr(self.mapanythingllava3d_vlm_interface, "model", None)
        if vlm_core is not None and hasattr(vlm_core, "enable_geom_feature_hook"):
            try:
                vlm_core.enable_geom_feature_hook(max_steps=1000)
                
            except Exception:
                pass

        llm = self.mapanythingllava3d_vlm_interface.model.language_model.model
        num_vl_layers = getattr(llm.config, "num_hidden_layers", 32)
        llm_hidden_size = getattr(llm.config, "hidden_size", self.mapanythingllava3d_vlm_interface.model.hidden_size)
        self.config.framework.mapanything_llava3d.vl_hidden_dim = llm_hidden_size
        self.config.framework.mapanything_llava3d.num_vl_layers = num_vl_layers
        self.vl_layer_selection = getattr(self.config.framework.action_model, "vl_layer_selection", "last")
        self.normalize_vl_hidden = bool(
            getattr(self.config.framework.mapanything_llava3d, "normalize_vl_hidden", False)
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.vlm_use_cache = False
        self._configure_memory_optimizations()

    @staticmethod
    def _set_module_use_cache(module, use_cache: bool, module_name: str):
        if module is None:
            return
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            setattr(cfg, "use_cache", bool(use_cache))
            logger.info(f"[memory_opt] set {module_name}.config.use_cache={bool(use_cache)}")
        generation_cfg = getattr(module, "generation_config", None)
        if generation_cfg is not None and hasattr(generation_cfg, "use_cache"):
            setattr(generation_cfg, "use_cache", bool(use_cache))

    @staticmethod
    def _enable_gradient_checkpointing(module, module_name: str):
        if module is None:
            return False
        for fn_name in ("gradient_checkpointing_enable", "enable_gradient_checkpointing"):
            fn = getattr(module, fn_name, None)
            if callable(fn):
                try:
                    fn()
                    logger.info(f"[memory_opt] enabled gradient checkpointing on {module_name} via `{fn_name}`")
                    return True
                except TypeError:
                    try:
                        fn(gradient_checkpointing_kwargs={"use_reentrant": False})
                        logger.info(f"[memory_opt] enabled gradient checkpointing on {module_name} via `{fn_name}` with kwargs")
                        return True
                    except Exception as e:
                        logger.warning(f"[memory_opt] failed enabling gradient checkpointing on {module_name}: {e}")
                except Exception as e:
                    logger.warning(f"[memory_opt] failed enabling gradient checkpointing on {module_name}: {e}")
        return False

    def _configure_memory_optimizations(self):
        fw_cfg = getattr(self.config, "framework", None)
        ma_cfg = getattr(fw_cfg, "mapanything_llava3d", None) if fw_cfg is not None else None
        if ma_cfg is not None and hasattr(ma_cfg, "use_cache"):
            self.vlm_use_cache = bool(getattr(ma_cfg, "use_cache"))
        else:
            self.vlm_use_cache = False

        vlm_interface = self.mapanythingllava3d_vlm_interface
        vlm_core = getattr(vlm_interface, "model", None)
        language_wrapper = getattr(vlm_core, "language_model", None) if vlm_core is not None else None
        llm_core = getattr(language_wrapper, "model", None) if language_wrapper is not None else None
        action_dit = getattr(self.action_model, "model", None)

        self._set_module_use_cache(vlm_core, self.vlm_use_cache, "vlm_core")
        self._set_module_use_cache(language_wrapper, self.vlm_use_cache, "language_wrapper")
        self._set_module_use_cache(llm_core, self.vlm_use_cache, "llm_core")

        enable_gc = bool(getattr(getattr(self.config, "trainer", None), "enable_gradient_checkpointing", False))
        if not enable_gc:
            logger.info("[memory_opt] gradient checkpointing disabled by config")
            return

        gc_enabled = False
        gc_enabled = self._enable_gradient_checkpointing(vlm_core, "vlm_core") or gc_enabled
        gc_enabled = self._enable_gradient_checkpointing(language_wrapper, "language_wrapper") or gc_enabled
        gc_enabled = self._enable_gradient_checkpointing(llm_core, "llm_core") or gc_enabled
        gc_enabled = self._enable_gradient_checkpointing(action_dit, "action_dit") or gc_enabled
        if not gc_enabled:
            logger.warning("[memory_opt] no module accepted gradient checkpointing enable call")

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        if not hasattr(self, "_debug_logged_instructions"):
            try:
                logger.info(f"[debug_instructions] batch_size={len(instructions)} example0={instructions[0]}")
            except Exception:
                pass
            self._debug_logged_instructions = 1
        elif self._debug_logged_instructions < 5:
            try:
                logger.info(f"[debug_instructions] batch_size={len(instructions)} example0={instructions[0]}")
            except Exception:
                pass
            self._debug_logged_instructions += 1

        vlm_inputs = self.mapanythingllava3d_vlm_interface.build_mapanythingllava3d_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_outputs = self.mapanythingllava3d_vlm_interface(
                **vlm_inputs,
                use_cache=self.vlm_use_cache,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = vlm_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            total_layers = len(all_hidden) - 1
            if expected_layers > total_layers:
                raise ValueError(f"expected_layers={expected_layers} greater than available vlm layers={total_layers}")
            if self.vl_layer_selection == "first":
                indices = range(1, 1 + expected_layers)
            else:
                indices = range(len(all_hidden) - expected_layers, len(all_hidden))
            vl_embs_list = [all_hidden[i] for i in indices]
            if self.normalize_vl_hidden:
                vl_embs_list = [F.layer_norm(h, h.shape[-1:]) for h in vl_embs_list]
            base_hidden = vl_embs_list[-1]
            debug_metrics = {}
            try:
                try:
                    h = base_hidden.detach().float()
                    debug_metrics["debug/vl_norm/base_hidden_mean"] = float(h.mean().item())
                    debug_metrics["debug/vl_norm/base_hidden_std"] = float(h.std().item())
                    debug_metrics["debug/vl_norm/base_hidden_rms"] = float((h * h).mean().sqrt().item())
                except Exception:
                    debug_metrics = debug_metrics
                vlm_core = getattr(self.mapanythingllava3d_vlm_interface, "model", None)
                geom_stats = getattr(vlm_core, "geom_feature_stats", None)
                if isinstance(geom_stats, list) and geom_stats:
                    by_tag = {}
                    for record in reversed(geom_stats):
                        tag = str(record.get("tag", ""))
                        if tag and tag not in by_tag:
                            by_tag[tag] = record
                        if len(by_tag) >= 3:
                            break
                    for tag, rec in by_tag.items():
                        prefix = f"debug/geom/{tag}"
                        debug_metrics[f"{prefix}_mean"] = float(rec.get("mean", 0.0))
                        debug_metrics[f"{prefix}_std"] = float(rec.get("std", 0.0))
                        debug_metrics[f"{prefix}_min"] = float(rec.get("min", 0.0))
                        debug_metrics[f"{prefix}_max"] = float(rec.get("max", 0.0))
                first_indices = range(1, 1 + expected_layers)
                last_indices = range(len(all_hidden) - expected_layers, len(all_hidden))
                for rel_idx, layer_idx in enumerate(first_indices):
                    h = all_hidden[layer_idx].detach().float()
                    norm = h.view(-1).norm().item()
                    debug_metrics[f"debug/vlm_first/layer_{rel_idx}_norm"] = norm
                for rel_idx, layer_idx in enumerate(last_indices):
                    h = all_hidden[layer_idx].detach().float()
                    norm = h.view(-1).norm().item()
                    debug_metrics[f"debug/vlm_last/layer_{rel_idx}_norm"] = norm
                try:
                    input_ids = vlm_inputs.get("input_ids", None)
                    image_token_index = vlm_inputs.get("image_token_index", None)
                    if input_ids is not None and image_token_index is not None and input_ids.ndim == 2:
                        img_mask = input_ids == image_token_index
                        per_sample = img_mask.sum(dim=1)
                        debug_metrics["debug/input/num_image_tokens_total"] = float(per_sample.sum().item())
                        debug_metrics["debug/input/num_image_tokens_per_sample_mean"] = float(per_sample.float().mean().item())
                        debug_metrics["debug/input/num_image_tokens_per_sample_min"] = float(per_sample.min().item())
                        debug_metrics["debug/input/num_image_tokens_per_sample_max"] = float(per_sample.max().item())
                except Exception:
                    debug_metrics = debug_metrics
            except Exception:
                debug_metrics = debug_metrics

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size + 1) :, :]

            repeated_diffusion_steps = 4
            if self.config is not None:
                try:
                    repeated_diffusion_steps = int(
                        getattr(self.config.framework.action_model, "repeated_diffusion_steps", repeated_diffusion_steps)
                    )
                except Exception:
                    repeated_diffusion_steps = repeated_diffusion_steps
                try:
                    if hasattr(self.config, "trainer") and hasattr(self.config.trainer, "repeated_diffusion_steps"):
                        repeated_diffusion_steps = int(self.config.trainer.repeated_diffusion_steps)
                except Exception:
                    repeated_diffusion_steps = repeated_diffusion_steps
            repeated_diffusion_steps = max(1, repeated_diffusion_steps)
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]

            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=base_hidden.device, dtype=base_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(vl_embs_list_repeated, actions_target_repeated, state_repeated)
            try:
                layer_means = getattr(self.action_model, "_last_dit_layer_means", None)
                layer_vars = getattr(self.action_model, "_last_dit_layer_vars", None)
                if layer_means is not None:
                    for idx, m in enumerate(layer_means):
                        debug_metrics[f"debug/dit_layer/{idx}_mean"] = float(m)
                if layer_vars is not None:
                    for idx, v in enumerate(layer_vars):
                        debug_metrics[f"debug/dit_layer/{idx}_var"] = float(v)
            except Exception:
                debug_metrics = debug_metrics

        return {"action_loss": action_loss, "debug_metrics": debug_metrics}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        vlm_inputs = self.mapanythingllava3d_vlm_interface.build_mapanythingllava3d_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_outputs = self.mapanythingllava3d_vlm_interface(
                **vlm_inputs,
                use_cache=self.vlm_use_cache,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = vlm_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            total_layers = len(all_hidden) - 1
            if expected_layers > total_layers:
                raise ValueError(f"expected_layers={expected_layers} greater than available vlm layers={total_layers}")
            if self.vl_layer_selection == "first":
                indices = range(1, 1 + expected_layers)
            else:
                indices = range(len(all_hidden) - expected_layers, len(all_hidden))
            vl_embs_list = [all_hidden[i] for i in indices]
            if self.normalize_vl_hidden:
                vl_embs_list = [F.layer_norm(h, h.shape[-1:]) for h in vl_embs_list]
            base_hidden = vl_embs_list[-1]

        state_tensor = torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype) if state is not None else None

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(vl_embs_list, state_tensor)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./starVLA/config/training/starvla_cotrain_oxe_mapanything_llava3d.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.mapanything_llava3d.base_vlm = "/2025233147/zzq/SpatialVLA_llava3d/model_zoo/mapanythingllava3d_base_v3"

    

    model = MapAnythingLlava3D_PI(cfg)
    # ckpt="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1011_qwenpi/checkpoints/need_steps_10000_pytorch_model.pt"
    # model = Qwen_PI.from_pretrained(ckpt)
    print(model)


    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake instruction for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
