set -o pipefail

# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3
# 修改后：
export NCCL_SOCKET_IFNAME=eth0        # 或者干脆注释掉，让 NCCL 自动选
export NCCL_IB_DISABLE=1              # 禁用 InfiniBand
export NCCL_P2P_DISABLE=0             # 保持 GPU 之间 P2P（同机）
# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=MapAnythingLlava3DPI
# freeze_module_list='mapanythingllava3d_vlm_interface.model.vision_tower,mapanythingllava3d_vlm_interface.model.geometric_model'
freeze_module_list=''
base_vlm=/2025233147/zzq/SpatialVLA_llava3d/model_zoo/mapanythingllava3d_base_v3

config_yaml=/2025233147/zzq/SpatialVLA_llava3d/starVLA/starVLA/config/training/starvla_train_libero_mapanything_llava3d.yaml
libero_data_root=/2025233147/zzq/SpatialVLA_llava3d/playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./results/Checkpoints
seed=42
per_device_bs=2
grad_accum_steps=4
timestamp=$(date +"%Y%m%d_%H%M%S")
run_id=1229_libero4in1_MapAnythingLlava3DPI_s${seed}_${timestamp}
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
raw_log_file="${output_dir}/train.raw.log"
log_file="${output_dir}/train.log"
# mv this script to the output dir
cp $0 ${output_dir}/

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1


freeze_args=()
if [ -n "${freeze_module_list}" ]; then
  freeze_args=(--trainer.freeze_modules "${freeze_module_list}")
fi

stdbuf -oL -eL accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero3.yaml \
  --gradient_accumulation_steps ${grad_accum_steps} \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.mapanything_llava3d.base_vlm ${base_vlm} \
  --framework.mapanything_llava3d.normalize_vl_hidden true \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${per_device_bs} \
  --datasets.vla_data.video_backend torchvision_av \
  "${freeze_args[@]}" \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.gradient_accumulation_steps ${grad_accum_steps} \
  --seed ${seed} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starvla_mapanything_llava3d \
  2>&1 | tee -a "${raw_log_file}"

train_exit=${PIPESTATUS[0]}
tr '\r' '\n' < "${raw_log_file}" > "${log_file}"
echo "Saved raw log to: ${raw_log_file}"
echo "Saved normalized log to: ${log_file}"
exit ${train_exit}



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.mapanything_llava3d.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
