# Temporal Supervision Refactor Notes (Step 2 + Step 3)

## Scope
This note documents the implemented refactor for:
1. Step 2: data-side temporal contract (`t -> t+k` sample fields)
2. Step 3: framework-side passthrough of next-step task tokens (`task_tokens_next`) into Action Head

The goal is to support SA-GR style dynamics supervision while keeping backward compatibility with existing training code paths.

## Files Changed
- `/Users/bazinga/code/my-starvla/starVLA/dataloader/gr00t_lerobot/data_config.py`
- `/Users/bazinga/code/my-starvla/starVLA/dataloader/gr00t_lerobot/datasets.py`
- `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py`
- `/Users/bazinga/code/my-starvla/tools/check_temporal_supervision.py`

## Step 2: Data Contract Refactor

### 2.1 Observation/Action index alignment
In `Libero4in1DataConfig`:
- `observation_indices` changed from `[0]` to `[0, 7]`
- `action_indices` kept as `[0..7]` (length 8)

This enforces each sample to contain both current frame `t` and supervision frame `t+k` where `k=7`.

### 2.2 Mixture dataset output fields
In `LeRobotMixtureDataset.__getitem__`, temporal fields are explicitly produced:
- `image` (legacy key, equals `image_t`)
- `image_t`
- `image_tk`
- `valid_tk` (float in `[0,1]`)
- `temporal_delta_k` (int)

If `include_state` is enabled, it also outputs:
- `state` (legacy, equals `state_t`)
- `state_t`
- `state_tk`

### 2.3 Validity semantics
`valid_tk=1.0` only when:
- `observation_indices` contains exact `k`, and
- `step + k` is inside trajectory length.

Otherwise `valid_tk=0.0`, and dataset still returns a clamped fallback frame to keep tensor shapes stable.

## Step 3: Framework passthrough (`task_tokens_next`)

In `/Users/bazinga/code/my-starvla/starVLA/model/framework/MapAnythingLlava3DPI.py`, training `forward` now does:

1. Read temporal inputs with compatibility fallback:
   - `image_t` if exists, else legacy `image`
   - `image_tk` if exists, else fallback to `image_t`

2. Run VLM on `image_t` (normal training branch):
   - produce `vl_embs_list`, `task_tokens`, attention mask

3. Run VLM on `image_tk` (target branch):
   - wrapped with `torch.no_grad()` to reduce memory + avoid extra gradient path
   - extract `task_tokens_next`

4. Build repeated tensors for diffusion repeats:
   - `task_tokens_repeated`
   - `task_tokens_next_repeated`
   - `valid_tk_repeated`

5. Boundary-safe fallback for invalid temporal pairs:
   - where `valid_tk==0`, replace `task_tokens_next` with detached `task_tokens` to avoid shape breaks.

6. Pass into action head:

```python
action_loss = self.action_model(
    vl_embs_list_repeated,
    actions_target_repeated,
    state_repeated,
    encoder_attention_mask=attention_mask_repeated,
    task_tokens=task_tokens_repeated,
    task_tokens_next=task_tokens_next_repeated,
)
```

## Tensor Flow (Per Batch)

1. Dataloader emits:
   - `image_t`: list[PIL], camera-ordered
   - `image_tk`: list[PIL], camera-ordered
   - `action`: `(H, action_dim_total)`
   - `valid_tk`: scalar float

2. Framework (`MapAnythingLlava3DPI.forward`):
   - VLM(`image_t`) -> `task_tokens`: `[B, N_task, D]`
   - VLM(`image_tk`) -> `task_tokens_next`: `[B, N_task, D]` (if available)
   - repeat by `repeated_diffusion_steps` -> `[B*R, ...]`

3. Action head (`LayerwiseFM_ActionHeader.forward`):
   - world predictor uses `(task_tokens, action/state)` to predict next latent
   - drift head uses residual `(task_tokens_next - pred_next)` for correction + losses

## Readability / Compatibility Decisions

- Kept legacy keys (`image`, `state`) so old call sites keep running.
- Added explicit temporal keys (`image_t`, `image_tk`, `state_t`, `state_tk`) to avoid hidden assumptions.
- Added debug metrics:
  - `debug/temporal/has_image_tk`
  - `debug/temporal/valid_tk_ratio`
  - `debug/temporal/task_tokens_next_available`
- Updated temporal-check tool to inspect `LeRobotMixtureDataset.__getitem__` specifically, avoiding false negatives from single-dataset code paths.

## Debug Checklist

### Static contract check
```bash
python /Users/bazinga/code/my-starvla/tools/check_temporal_supervision.py \
  --config-yaml /Users/bazinga/code/my-starvla/starVLA/config/training/starvla_train_libero_mapanything_llava3d_ab_b_concat_cross.yaml \
  --data-mix libero_all
```

### Runtime spot-check (with dataset path)
```bash
python /Users/bazinga/code/my-starvla/tools/check_temporal_supervision.py \
  --config-yaml /Users/bazinga/code/my-starvla/starVLA/config/training/starvla_train_libero_mapanything_llava3d_ab_b_concat_cross.yaml \
  --data-mix libero_all \
  --check-runtime \
  --data-root-dir <YOUR_LEROBOT_ROOT>
```

### Expected signs in training logs
- `debug/temporal/has_image_tk` should be `1.0`
- `debug/temporal/task_tokens_next_available` should mostly be `1.0`
- `debug/temporal/valid_tk_ratio` should stay high (dataset-dependent; usually >0.9)

## Current limitation
`valid_tk=0` samples are currently handled by fallback substitution rather than explicit masked loss weighting inside the Action Head. This keeps code path stable but may introduce slight supervision bias on invalid tail steps. If needed, the next optimization is to add explicit valid-mask weighting in world-loss terms.
