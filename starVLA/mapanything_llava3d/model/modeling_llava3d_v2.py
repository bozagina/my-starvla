import json
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Optional, Tuple, Union, List
from transformers import PreTrainedModel, AutoConfig
from transformers.modeling_utils import load_state_dict
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from LLaVA_3D.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from LLaVA_3D.llava.model.language_model.llava_mistral import LlavaMistralForCausalLM


@contextmanager
def _suspend_llava_vision_tower():
    builder = None
    llava_arch = None
    origs = {}
    try:
        import LLaVA_3D.llava.model.multimodal_encoder.builder as builder  # type: ignore
    except Exception:
        builder = None
    try:
        import LLaVA_3D.llava.model.llava_arch as llava_arch  # type: ignore
    except Exception:
        llava_arch = None

    def _noop_build(*args, **kwargs):
        return None

    for mod in (builder, llava_arch):
        if mod is not None and hasattr(mod, "build_vision_tower"):
            origs[mod] = mod.build_vision_tower
            mod.build_vision_tower = _noop_build

    try:
        yield
    finally:
        for mod, orig in origs.items():
            mod.build_vision_tower = orig


def _resolve_index_file(pretrained_path: str) -> Optional[Path]:
    candidates = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    base = Path(pretrained_path)
    for name in candidates:
        p = base / name
        if p.is_file():
            return p
    return None


def _resolve_weight_files(pretrained_path: str) -> List[Path]:
    base = Path(pretrained_path)
    index_file = _resolve_index_file(pretrained_path)
    if index_file is not None:
        data = json.loads(index_file.read_text())
        shard_files = sorted(set(data.get("weight_map", {}).values()))
        return [base / f for f in shard_files]
    candidates = [
        "model.safetensors",
        "pytorch_model.bin",
    ]
    for name in candidates:
        p = base / name
        if p.is_file():
            return [p]
    raise FileNotFoundError(
        f"No model weights found under {pretrained_path} (missing index or weight files)."
    )


def _filter_llava_state_dict(state_dict: dict) -> dict:
    drop_substrings = (".mm_projector.", ".vision_tower.", ".video_tower.")
    filtered = {}
    for k, v in state_dict.items():
        if any(s in k for s in drop_substrings):
            continue
        filtered[k] = v
    return filtered


def _load_llava_base_weights(model: nn.Module, pretrained_path: str) -> None:
    weight_files = _resolve_weight_files(pretrained_path)
    for wf in weight_files:
        shard = load_state_dict(str(wf))
        shard = _filter_llava_state_dict(shard)
        model.load_state_dict(shard, strict=False)
        del shard


class LLaVA3DForCausalLMV2(PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        model_type = getattr(config, "llava3d_model_type", "llama")
        pretrained_path = getattr(config, "llava3d_pretrained_path", None)
        if model_type == "llama":
            if pretrained_path is not None:
                llava_cfg = AutoConfig.from_pretrained(pretrained_path, trust_remote_code=True)
                setattr(llava_cfg, "mm_video_tower", None)
                # 彻底不构建 LLaVA 的 vision tower，避免 build_vision_tower 里 os.path.exists(None) 以及联网下 clip
                if hasattr(llava_cfg, "mm_vision_tower"):
                    setattr(llava_cfg, "mm_vision_tower", "")
                if hasattr(llava_cfg, "vision_tower"):
                    setattr(llava_cfg, "vision_tower", "")
                with _suspend_llava_vision_tower():
                    self.model = LlavaLlamaForCausalLM(llava_cfg)
                _load_llava_base_weights(self.model, pretrained_path)
            else:
                setattr(config, "mm_video_tower", None)
                with _suspend_llava_vision_tower():
                    self.model = LlavaLlamaForCausalLM(config)
        elif model_type == "mistral":
            if pretrained_path is not None:
                llava_cfg = AutoConfig.from_pretrained(pretrained_path, trust_remote_code=True)
                setattr(llava_cfg, "mm_video_tower", None)
                # 彻底不构建 LLaVA 的 vision tower，避免 build_vision_tower 里 os.path.exists(None) 以及联网下 clip
                if hasattr(llava_cfg, "mm_vision_tower"):
                    setattr(llava_cfg, "mm_vision_tower", "")
                if hasattr(llava_cfg, "vision_tower"):
                    setattr(llava_cfg, "vision_tower", "")
                with _suspend_llava_vision_tower():
                    self.model = LlavaMistralForCausalLM(llava_cfg)
                _load_llava_base_weights(self.model, pretrained_path)
            else:
                setattr(config, "mm_video_tower", None)
                with _suspend_llava_vision_tower():
                    self.model = LlavaMistralForCausalLM(config)
        else:
            raise ValueError(f"Unsupported LLaVA-3D model type: {model_type}")
        self.vocab_size = getattr(self.model.config, "vocab_size", getattr(config, "vocab_size", None))

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        if hasattr(self.model, "get_output_embeddings"):
            return self.model.get_output_embeddings()
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        return None

    def set_output_embeddings(self, new_embeddings):
        if hasattr(self.model, "set_output_embeddings"):
            self.model.set_output_embeddings(new_embeddings)
        elif hasattr(self.model, "lm_head"):
            self.model.lm_head = new_embeddings
        else:
            raise AttributeError("Underlying LLaVA-3D model does not support setting output embeddings.")

    def get_decoder(self):
        if hasattr(self.model, "get_decoder"):
            return self.model.get_decoder()
        return self.model

    def set_decoder(self, value):
        self.model = value

    def tie_weights(self):
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Tuple]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        if return_dict is False:
            if labels is not None:
                loss = outputs[0]
                logits = outputs[1]
                if num_logits_to_keep and num_logits_to_keep > 0:
                    logits = logits[:, -num_logits_to_keep:, :]
                return (loss, logits) + tuple(outputs[2:])
            else:
                logits = outputs[0]
                if num_logits_to_keep and num_logits_to_keep > 0:
                    logits = logits[:, -num_logits_to_keep:, :]
                return (logits,) + tuple(outputs[1:])
        logits = outputs.logits
        if num_logits_to_keep and num_logits_to_keep > 0:
            logits = logits[:, -num_logits_to_keep:, :]
        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def encode_images(self, pixel_values):
        return pixel_values

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        if hasattr(self.model, "prepare_inputs_for_generation"):
            out = self.model.prepare_inputs_for_generation(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids,
                use_cache=use_cache,
                num_logits_to_keep=num_logits_to_keep,
                **kwargs,
            )
            # Drop unknown keys to avoid size mismatch errors
            out.pop("cache_position", None)
            return out
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "use_cache": use_cache,
        }
        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep
        return model_inputs
