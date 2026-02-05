import os
import torch
import torch.nn as nn
from typing import Optional, List, Dict
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoImageProcessor, AutoTokenizer
from starVLA.mapanything_llava3d.model.configuration_mapanything_llava3d import MapAnythingLlava3DConfig
from starVLA.mapanything_llava3d.model.modeling_mapanything_llava3d_vlm import MapAnythingLlava3DForConditionalGeneration
from starVLA.mapanything_llava3d.model.processing_mapanything_llava3d import MapAnythingLlava3DProcessor
from accelerate.logging import get_logger


logger = get_logger(__name__)


_DEFAULT_VISION_MODEL = "/2025233147/zzq/mapAnythingLlava3dPi0.5/model_zoo/siglip-so400m-patch14-224"
_DEFAULT_LANGUAGE_MODEL = "/2025233147/zzq/mapAnythingLlava3dPi0.5/model_zoo/llava3d"
_DEFAULT_MAPANYTHING_MODEL = "/2025233147/zzq/mapAnythingLlava3dPi0.5/model_zoo/mapanything"


def _is_rank0():
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


class _MapAnythingLlava3D_Interface(nn.Module):
    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        if not os.path.exists(_DEFAULT_VISION_MODEL):
            raise FileNotFoundError(_DEFAULT_VISION_MODEL)
        if not os.path.exists(_DEFAULT_LANGUAGE_MODEL):
            raise FileNotFoundError(_DEFAULT_LANGUAGE_MODEL)
        if not os.path.exists(_DEFAULT_MAPANYTHING_MODEL):
            raise FileNotFoundError(_DEFAULT_MAPANYTHING_MODEL)
        use_geom = True
        prefix_image_dropout_prob = 0.0
        prefix_lang_dropout_prob = 0.0
        base_vlm_path = None
        vision_model_name_or_path = _DEFAULT_VISION_MODEL
        language_model_name_or_path = _DEFAULT_LANGUAGE_MODEL
        try:
            fw_cfg = getattr(config, "framework", None)
            ma_cfg = getattr(fw_cfg, "mapanything_llava3d", None) if fw_cfg is not None else None
            if ma_cfg is not None:
                base_vlm_path = getattr(ma_cfg, "base_vlm", None)
                prefix_image_dropout_prob = float(getattr(ma_cfg, "prefix_image_dropout_prob", 0.0))
                prefix_lang_dropout_prob = float(getattr(ma_cfg, "prefix_lang_dropout_prob", 0.0))
                if hasattr(ma_cfg, "use_geometric_branch"):
                    use_geom = bool(getattr(ma_cfg, "use_geometric_branch"))
                elif hasattr(ma_cfg, "use_geom"):
                    use_geom = bool(getattr(ma_cfg, "use_geom"))
        except Exception:
            prefix_image_dropout_prob = 0.0
            prefix_lang_dropout_prob = 0.0
            use_geom = True
        print(f"prefix_image_dropout_prob: {prefix_image_dropout_prob}")
        print(f"prefix_lang_dropout_prob: {prefix_lang_dropout_prob}")

        model = None
        mapanything_cfg = None
        if base_vlm_path is not None:
            assert isinstance(base_vlm_path, str), f"framework.mapanything_llava3d.base_vlm must be str, got {type(base_vlm_path)}"
            assert os.path.isdir(base_vlm_path), f"base_vlm path does not exist or is not a directory: {base_vlm_path}"
            try:
                model = MapAnythingLlava3DForConditionalGeneration.from_pretrained(base_vlm_path)
                mapanything_cfg = model.config
                mapanything_cfg.prefix_image_dropout_prob = prefix_image_dropout_prob
                mapanything_cfg.prefix_lang_dropout_prob = prefix_lang_dropout_prob
                setattr(mapanything_cfg, "use_geometric_branch", use_geom)
                print(f"Loaded MapAnythingLlava3D VLM from merged checkpoint: {base_vlm_path}")
            except Exception as e:
                raise AssertionError(f"Failed to load merged VLM checkpoint from {base_vlm_path}: {e}")

        if model is None:
            vision_model_name_or_path = _DEFAULT_VISION_MODEL
            language_model_name_or_path = _DEFAULT_LANGUAGE_MODEL
            mapanything_model_name_or_path = _DEFAULT_MAPANYTHING_MODEL
            mapanything_cfg = MapAnythingLlava3DConfig(
                vision_model_name_or_path=vision_model_name_or_path,
                language_model_name_or_path=language_model_name_or_path,
                mapanything_model_name_or_path=mapanything_model_name_or_path,
                use_spatial_token=False,
                action_expert_config=None,
                use_geom=use_geom,
                prefix_image_dropout_prob=prefix_image_dropout_prob,
                prefix_lang_dropout_prob=prefix_lang_dropout_prob,
            )
            model = MapAnythingLlava3DForConditionalGeneration(mapanything_cfg)
        image_processor = AutoImageProcessor.from_pretrained(vision_model_name_or_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(language_model_name_or_path, trust_remote_code=True)
        statistics = None
        intrinsic_config = None
        action_config = None
        processor = MapAnythingLlava3DProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            statistics=statistics,
            intrinsic_config=intrinsic_config,
            action_config=action_config,
        )
        self.model = model
        self.processor = processor
        self.config = mapanything_cfg
        self.runtime_config = config
        self.model.config.hidden_size = self.model.config.hidden_size
    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**kwargs)
        return outputs

    def generate(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generation_output = self.model.generate(**kwargs)
        return generation_output

    def build_mapanythingllava3d_inputs(self, images: List[List], instructions: List[str], unnorm_key: Optional[str] = None) -> Dict[str, torch.Tensor]:
        assert len(images) == len(instructions)
        cot_prompt = None
        datasets_cfg = getattr(self.runtime_config, "datasets", None) if self.runtime_config is not None else None
        vla_cfg = getattr(datasets_cfg, "vla_data", None) if datasets_cfg is not None else None
        if vla_cfg is not None and hasattr(vla_cfg, "CoT_prompt"):
            cot_prompt = getattr(vla_cfg, "CoT_prompt")
        processed_instructions = []
        for instruction in instructions:
            if cot_prompt:
                processed_instructions.append(cot_prompt.replace("{instruction}", instruction))
            else:
                processed_instructions.append(instruction)
        if _is_rank0():
            print(f"[mapanything_llava3d] processed_instructions: {processed_instructions}")
        model_inputs = self.processor(
            text=processed_instructions,
            images=images,
            unnorm_key=unnorm_key,
            return_tensors="pt",
        )
        batch = dict(model_inputs.data)
        if "intrinsic" not in batch:
            b = batch["input_ids"].shape[0]
            pixel_values = batch.get("pixel_values")
            if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 5:
                v = pixel_values.shape[1]
                intrinsic = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(b, v, 1, 1)
            else:
                intrinsic = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(b, 1, 1)
            batch["intrinsic"] = intrinsic
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.model.device)
        return batch


if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    interface = _MapAnythingLlava3D_Interface()
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    images = [[image, image]]
    instructions = ["Describe the scene."]
    inputs = interface.build_mapanythingllava3d_inputs(images=images, instructions=instructions)
    with torch.no_grad():
        outputs = interface.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            intrinsic=inputs.get("intrinsic"),
            labels=None,
            return_dict=True,
        )
    print("logits shape:", outputs.logits.shape)
