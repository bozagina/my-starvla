#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MetaWorld PPO Interactive Configuration Builder
# =============================================================================
# 
# 交互式实验配置工具，支持：
# 1. 通过菜单选择任务、Clip模式、AdvStruct和AuxAdv参数
# 2. 支持快速预设（baseline, part3_only, part4_only等）
# 3. 保存和加载配置文件
# 4. 批量生成多个配置组合
# 5. 一键启动并行实验
#
# Usage:
#   ./run_metaworld_ppo_interactive.sh
# =============================================================================

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_SCRIPT="/cpfs01/qianfy_workspace/openvla_oft_rl/rl/ds_metaworld_ppo_mlp_with_param_more_stats_try_new_v5_dev.py"
CLIP_CONFIG_PATH="/cpfs01/qianfy_workspace/openvla_oft_rl/rl/config/clip.yml"

# 配置文件存储目录
CONFIG_DIR="/cpfs01/qianfy_workspace/openvla_oft_rl/scripts/interactive_configs"
mkdir -p "${CONFIG_DIR}"

# 上一次运行的配置JSON文件
LAST_RUN_CONFIG_FILE="${CONFIG_DIR}/last_run_config.json"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# 全局配置变量
declare -a SELECTED_TASKS=()
declare -a SELECTED_CLIP_MODES=()
declare -a CONFIG_COMBINATIONS=()

# 自动组合生成器的参数组
declare -a AUTO_ADV_MODES=()
declare -a AUTO_ADV_TEMPS=()
declare -a AUTO_LATE_FRACS=()
declare -a AUTO_ALPHAS=()
declare -a AUTO_LR_MULTS=()
declare -a AUTO_W_MINS=()
declare -a AUTO_W_MAXS=()
declare -a AUTO_AUX_ENABLES=()
declare -a AUTO_AUX_MODES=()         # 新增：Route A (residual) / Joint
declare -a AUTO_AUX_TARGET_MODES=()  # 新增：Route C (pg_grad)
declare -a AUTO_AUX_PAIR_MODES=()    # 新增：Route B (vector)
declare -a AUTO_AUX_COEFS=()

# 默认参数
DEFAULT_SEED=42
DEFAULT_NUM_TRAINER_GPUS=1
DEFAULT_NUM_ROLLOUT_WORKERS=16
DEFAULT_NUM_EVAL_WORKERS=32
DEFAULT_TRAIN_BATCH_SIZE=512
DEFAULT_TRAIN_ITERS=100000
DEFAULT_MAX_PARALLEL_JOBS=8
DEFAULT_GPU_CONFIGS="0,1"
DEFAULT_STARTUP_DELAY=90  # 每个任务启动间隔（秒）
DEFAULT_MONITOR_TIMEOUT=300  # 监控超时时间（秒，5分钟）
DEFAULT_MAX_RETRIES=3  # 最大重试次数
DEFAULT_ENABLE_SWANLAB=1  # 是否启用SwanLab (1=启用, 0=仅TensorBoard)

# 当前配置状态
CURRENT_SEED="${DEFAULT_SEED}"
CURRENT_TRAIN_ITERS="${DEFAULT_TRAIN_ITERS}"
CURRENT_NUM_TRAINER_GPUS="${DEFAULT_NUM_TRAINER_GPUS}"
CURRENT_NUM_ROLLOUT_WORKERS="${DEFAULT_NUM_ROLLOUT_WORKERS}"
CURRENT_NUM_EVAL_WORKERS="${DEFAULT_NUM_EVAL_WORKERS}"
CURRENT_TRAIN_BATCH_SIZE="${DEFAULT_TRAIN_BATCH_SIZE}"
CURRENT_MAX_PARALLEL_JOBS="${DEFAULT_MAX_PARALLEL_JOBS}"
CURRENT_GPU_CONFIGS="${DEFAULT_GPU_CONFIGS}"
CURRENT_STARTUP_DELAY=90  # 默认启动间隔90秒
CURRENT_STARTUP_DELAY="${DEFAULT_STARTUP_DELAY}"
CURRENT_MONITOR_TIMEOUT="${DEFAULT_MONITOR_TIMEOUT}"
CURRENT_MAX_RETRIES="${DEFAULT_MAX_RETRIES}"
CURRENT_ENABLE_SWANLAB="${DEFAULT_ENABLE_SWANLAB}"

# 可用的任务列表
ALL_TASKS=(
  "sweep-into-v3"
  "drawer-open-v3"
  "door-open-v3"
  "button-press-topdown-v3"
  "handle-press-v3"
  "push-v3"
  "peg-insert-side-v3"
  "pick-place-v3"
  "plate-slide-v3"
  "coffee-button-v3"
)

# 可用的Clip模式
ALL_CLIP_MODES=(
  "clip"
  "no_clip"
  "soft_clip_alpha-0"
  "soft_clip_alpha-0-5"
  "soft_clip_alpha-1"
  "soft_clip_alpha-2"
  "sapo_soft_clip"
  "log_gauss_clip"
)

# =============================================================================
# 辅助函数
# =============================================================================

print_header() {
  echo -e "${CYAN}${BOLD}"
  echo "═══════════════════════════════════════════════════════════════════"
  echo "$1"
  echo "═══════════════════════════════════════════════════════════════════"
  echo -e "${NC}"
}

print_section() {
  echo -e "${BLUE}${BOLD}━━━ $1 ━━━${NC}"
}

print_info() {
  echo -e "${GREEN}ℹ ${NC}$1"
}

print_warning() {
  echo -e "${YELLOW}⚠ ${NC}$1"
}

print_error() {
  echo -e "${RED}✗ ${NC}$1"
}

print_success() {
  echo -e "${GREEN}✓ ${NC}$1"
}

ask_yes_no() {
  local prompt="$1"
  local response
  while true; do
    echo -ne "${YELLOW}${prompt} (y/n): ${NC}"
    read -r response
    case "$response" in
      [Yy]* ) return 0;;
      [Nn]* ) return 1;;
      * ) echo "请输入 y 或 n";;
    esac
  done
}

press_any_key() {
  echo -e "${CYAN}按任意键继续...${NC}"
  read -n 1 -s
}

# =============================================================================
# 主菜单
# =============================================================================

show_main_menu() {
  clear
  print_header "MetaWorld PPO 交互式配置工具"
  echo
  print_section "当前配置摘要"
  echo -e "  任务数量: ${MAGENTA}${#SELECTED_TASKS[@]}${NC} 个"
  echo -e "  Clip模式: ${MAGENTA}${#SELECTED_CLIP_MODES[@]}${NC} 个"
  echo -e "  配置组合: ${MAGENTA}${#CONFIG_COMBINATIONS[@]}${NC} 个"
  echo -e "  总实验数: ${BOLD}${GREEN}$((${#SELECTED_TASKS[@]} * ${#SELECTED_CLIP_MODES[@]} * ${#CONFIG_COMBINATIONS[@]}))${NC} 个"
  echo
  print_section "菜单选项"
  echo "  1) 选择任务 (Tasks)"
  echo "  2) 选择Clip模式"
  echo "  3) 配置实验组合 (AdvStruct & AuxAdv)"
  echo "  4) 使用快速预设"
  echo "  5) 训练参数设置"
  echo "  6) GPU与并行设置"
  echo
  echo "  7) 查看完整配置"
  echo "  8) 保存配置到文件"
  echo "  9) 从文件加载配置"
  echo "  ${CYAN}10) 自动组合生成器 (推荐)${NC}"
  echo "  ${GREEN}11) 加载上一次运行的配置${NC}"
  echo
  echo "  ${GREEN}${BOLD}R) 运行实验${NC}"
  echo "  ${RED}Q) 退出${NC}"
  echo
  echo -ne "${YELLOW}请选择 [1-11/R/Q]: ${NC}"
}

# =============================================================================
# 1. 选择任务
# =============================================================================

select_tasks() {
  clear
  print_header "选择任务 (Tasks)"
  echo
  print_info "可选任务列表："
  echo
  for i in "${!ALL_TASKS[@]}"; do
    local task="${ALL_TASKS[$i]}"
    local selected=""
    for selected_task in "${SELECTED_TASKS[@]}"; do
      if [[ "$selected_task" == "$task" ]]; then
        selected="[${GREEN}✓${NC}]"
        break
      fi
    done
    echo "  $((i+1))) ${selected} ${task}"
  done
  echo
  echo "  A) 全选"
  echo "  C) 清空选择"
  echo "  B) 返回主菜单"
  echo
  echo -ne "${YELLOW}选择任务编号 (多个用空格分隔, 或输入 A/C/B): ${NC}"
  read -r input
  
  case "$input" in
    [Aa])
      SELECTED_TASKS=("${ALL_TASKS[@]}")
      print_success "已选择所有任务 (${#SELECTED_TASKS[@]} 个)"
      sleep 1
      ;;
    [Cc])
      SELECTED_TASKS=()
      print_success "已清空任务选择"
      sleep 1
      ;;
    [Bb])
      return
      ;;
    *)
      SELECTED_TASKS=()
      for num in $input; do
        if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "${#ALL_TASKS[@]}" ]; then
          SELECTED_TASKS+=("${ALL_TASKS[$((num-1))]}")
        fi
      done
      print_success "已选择 ${#SELECTED_TASKS[@]} 个任务"
      sleep 1
      ;;
  esac
}

# =============================================================================
# 2. 选择Clip模式
# =============================================================================

select_clip_modes() {
  clear
  print_header "选择Clip模式"
  echo
  print_info "可选Clip模式："
  echo
  for i in "${!ALL_CLIP_MODES[@]}"; do
    local mode="${ALL_CLIP_MODES[$i]}"
    local selected=""
    for selected_mode in "${SELECTED_CLIP_MODES[@]}"; do
      if [[ "$selected_mode" == "$mode" ]]; then
        selected="[${GREEN}✓${NC}]"
        break
      fi
    done
    echo "  $((i+1))) ${selected} ${mode}"
  done
  echo
  echo "  A) 全选"
  echo "  C) 清空选择"
  echo "  B) 返回主菜单"
  echo
  echo -ne "${YELLOW}选择Clip模式编号 (多个用空格分隔, 或输入 A/C/B): ${NC}"
  read -r input
  
  case "$input" in
    [Aa])
      SELECTED_CLIP_MODES=("${ALL_CLIP_MODES[@]}")
      print_success "已选择所有Clip模式 (${#SELECTED_CLIP_MODES[@]} 个)"
      sleep 1
      ;;
    [Cc])
      SELECTED_CLIP_MODES=()
      print_success "已清空Clip模式选择"
      sleep 1
      ;;
    [Bb])
      return
      ;;
    *)
      SELECTED_CLIP_MODES=()
      for num in $input; do
        if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "${#ALL_CLIP_MODES[@]}" ]; then
          SELECTED_CLIP_MODES+=("${ALL_CLIP_MODES[$((num-1))]}")
        fi
      done
      print_success "已选择 ${#SELECTED_CLIP_MODES[@]} 个Clip模式"
      sleep 1
      ;;
  esac
}

# =============================================================================
# 3. 配置实验组合
# =============================================================================

configure_experiments() {
  clear
  print_header "配置实验组合"
  echo
  print_section "当前配置列表"
  if [ ${#CONFIG_COMBINATIONS[@]} -eq 0 ]; then
    echo "  (空)"
  else
    for i in "${!CONFIG_COMBINATIONS[@]}"; do
      echo "  $((i+1))) ${CONFIG_COMBINATIONS[$i]}"
    done
  fi
  echo
  print_section "操作选项"
  echo "  1) 添加新配置"
  echo "  2) 删除配置"
  echo "  3) 清空所有配置"
  echo "  B) 返回主菜单"
  echo
  echo -ne "${YELLOW}请选择 [1-3/B]: ${NC}"
  read -r choice
  
  case "$choice" in
    1)
      add_new_config
      ;;
    2)
      delete_config
      ;;
    3)
      if ask_yes_no "确认清空所有配置?"; then
        CONFIG_COMBINATIONS=()
        print_success "已清空所有配置"
        sleep 1
      fi
      ;;
    [Bb])
      return
      ;;
  esac
}

add_new_config() {
  local current_group_name=""
  
  while true; do
    clear
    print_header "添加新配置"
    echo
    
    # 输入组名 (如果尚未设置)
    if [[ -z "$current_group_name" ]]; then
      echo -ne "${YELLOW}组名 (用于文件夹分类, 例如: baseline, ablation, 留空则为 'default'): ${NC}"
      read -r input_group
      current_group_name="${input_group:-default}"
    else
      print_info "当前组名: ${MAGENTA}${current_group_name}${NC}"
    fi
    
    # 输入配置标签
    echo -ne "${YELLOW}配置名称 (例如: my_exp_1): ${NC}"
    read -r label
    if [[ -z "$label" ]]; then
      print_error "配置名称不能为空"
      sleep 1
      continue
    fi
    
    echo
    print_section "AdvStruct 配置"
    
    # AdvStruct Mode
    echo "选择 AdvStruct 模式:"
    echo "  1) none (禁用)"
    echo "  2) state_softmax (状态条件权重)"
    echo "  3) state_action_softmax (状态+动作条件权重)"
    echo -ne "${YELLOW}选择 [1-3]: ${NC}"
    read -r adv_mode_choice
    case "$adv_mode_choice" in
      1) adv_mode="none";;
      2) adv_mode="state_softmax";;
      3) adv_mode="state_action_softmax";;
      *) adv_mode="none";;
    esac
    
    # 如果启用了AdvStruct，询问其他参数
    if [[ "$adv_mode" != "none" ]]; then
      echo -ne "${YELLOW}Temperature (默认 0.5): ${NC}"
      read -r adv_temp
      adv_temp="${adv_temp:-0.5}"
      
      echo -ne "${YELLOW}Late Start Frac (0-1, 默认 0.0): ${NC}"
      read -r late_frac
      late_frac="${late_frac:-0.0}"
      
      echo -ne "${YELLOW}Alpha (收缩系数, 默认 1.0): ${NC}"
      read -r alpha
      alpha="${alpha:-1.0}"
      
      echo -ne "${YELLOW}LR Multiplier (学习率倍数, 默认 1.0): ${NC}"
      read -r lr_mult
      lr_mult="${lr_mult:-1.0}"
      
      echo -ne "${YELLOW}W Min (权重下限, -1禁用, 默认 -1.0): ${NC}"
      read -r w_min
      w_min="${w_min:--1.0}"
      
      echo -ne "${YELLOW}W Max (权重上限, -1禁用, 默认 -1.0): ${NC}"
      read -r w_max
      w_max="${w_max:--1.0}"
    else
      adv_temp="1.0"
      late_frac="0.0"
      alpha="1.0"
      lr_mult="1.0"
      w_min="-1.0"
      w_max="-1.0"
    fi
    
    echo
    print_section "AuxAdv 配置"
    
    if ask_yes_no "启用 AuxAdv?"; then
      aux_enable="1"
      
      # 新增：Route A/B/C 配置
      echo "AuxAdv 模式 (Route A):"
      echo "  1) residual (推荐: 残差拟合)"
      echo "  2) joint (旧版: 联合拟合)"
      echo -ne "${YELLOW}选择 [1-2, 默认1]: ${NC}"
      read -r aux_mode_choice
      case "$aux_mode_choice" in
        1) aux_mode="residual";;
        2) aux_mode="joint";;
        *) aux_mode="residual";;
      esac
      
      echo "AuxAdv 目标 (Route C):"
      echo "  1) advantage (默认: 拟合Advantage)"
      echo "  2) pg_grad (拟合PG梯度贡献)"
      echo -ne "${YELLOW}选择 [1-2, 默认1]: ${NC}"
      read -r aux_target_choice
      case "$aux_target_choice" in
        1) aux_target_mode="advantage";;
        2) aux_target_mode="pg_grad";;
        *) aux_target_mode="advantage";;
      esac

      echo "AuxAdv 交互项模式 (Route B):"
      echo "  1) scalar (默认: 标量门控)"
      echo "  2) vector (低秩向量门控)"
      echo -ne "${YELLOW}选择 [1-2, 默认1]: ${NC}"
      read -r aux_pair_choice
      case "$aux_pair_choice" in
        1) aux_pair_mode="scalar";;
        2) aux_pair_mode="vector";;
        *) aux_pair_mode="scalar";;
      esac
      
      echo -ne "${YELLOW}AuxAdv Coef (默认 0.1): ${NC}"
      read -r aux_coef
      aux_coef="${aux_coef:-0.1}"
    else
      aux_enable="0"
      aux_mode="residual"
      aux_target_mode="advantage"
      aux_pair_mode="scalar"
      aux_coef="0.0"
    fi
    
    # 组装配置字符串
    # 格式: group_name|label|adv_mode|adv_temp|late_frac|alpha|lr_mult|w_min|w_max|aux_enable|aux_coef|aux_mode|aux_target_mode|aux_pair_mode
    config_str="${current_group_name}|${label}|${adv_mode}|${adv_temp}|${late_frac}|${alpha}|${lr_mult}|${w_min}|${w_max}|${aux_enable}|${aux_coef}|${aux_mode}|${aux_target_mode}|${aux_pair_mode}"
    CONFIG_COMBINATIONS+=("${config_str}")
    
    echo
    print_success "配置已添加: [${current_group_name}] ${label}"
    echo "  - 组名: ${current_group_name}"
    echo "  - AdvStruct: ${adv_mode} (temp=${adv_temp}, late=${late_frac}, alpha=${alpha}, lr=${lr_mult}, w_min=${w_min}, w_max=${w_max})"
    echo "  - AuxAdv: enable=${aux_enable} (mode=${aux_mode}, target=${aux_target_mode}, pair=${aux_pair_mode}, coef=${aux_coef})"
    echo
    
    # 询问下一步操作
    echo "接下来做什么?"
    echo "  1) 继续在当前组 [${current_group_name}] 添加配置"
    echo "  2) 切换到新组"
    echo "  3) 完成并返回"
    echo -ne "${YELLOW}选择 [1-3, 默认1]: ${NC}"
    read -r next_action
    
    case "$next_action" in
      2)
        current_group_name=""
        ;;
      3)
        return
        ;;
      *)
        # 默认继续，不重置 current_group_name
        ;;
    esac
  done
}

delete_config() {
  if [ ${#CONFIG_COMBINATIONS[@]} -eq 0 ]; then
    print_warning "没有可删除的配置"
    sleep 1
    return
  fi
  
  clear
  print_header "删除配置"
  echo
  for i in "${!CONFIG_COMBINATIONS[@]}"; do
    echo "  $((i+1))) ${CONFIG_COMBINATIONS[$i]}"
  done
  echo
  echo -ne "${YELLOW}输入要删除的配置编号 (或 B 返回): ${NC}"
  read -r num
  
  if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "${#CONFIG_COMBINATIONS[@]}" ]; then
    unset 'CONFIG_COMBINATIONS[$((num-1))]'
    CONFIG_COMBINATIONS=("${CONFIG_COMBINATIONS[@]}")  # 重新索引
    print_success "配置已删除"
    sleep 1
  fi
}

# =============================================================================
# 4. 快速预设
# =============================================================================

use_preset() {
  clear
  print_header "快速预设"
  echo
  print_info "选择预设配置："
  echo
  echo "  1) Baseline (无 AdvStruct, 无 AuxAdv)"
  echo "  2) Part3 Only (AdvStruct 启用)"
  echo "  3) Part4 Only (AuxAdv 启用)"
  echo "  4) Part3 + Part4 (两者都启用)"
  echo "  5) Comprehensive (15个完整组合)"
  echo "  6) New Features (4个新特性测试)"
  echo "  B) 返回主菜单"
  echo
  echo -ne "${YELLOW}请选择 [1-6/B]: ${NC}"
  read -r choice
  
  case "$choice" in
    1)
      CONFIG_COMBINATIONS=("baseline|baseline|none|1.0|0.0|1.0|1.0|-1.0|-1.0|0|0.0|residual|advantage|scalar")
      print_success "已加载 Baseline 预设"
      ;;
    2)
      CONFIG_COMBINATIONS=("part3|part3_only|state_softmax|0.5|0.0|1.0|1.0|-1.0|-1.0|0|0.0|residual|advantage|scalar")
      print_success "已加载 Part3 Only 预设"
      ;;
    3)
      CONFIG_COMBINATIONS=("part4|part4_only|none|1.0|0.0|1.0|1.0|-1.0|-1.0|1|0.1|residual|advantage|scalar")
      print_success "已加载 Part4 Only 预设"
      ;;
    4)
      CONFIG_COMBINATIONS=("combined|part3_part4|state_softmax|0.5|0.0|1.0|1.0|-1.0|-1.0|1|0.1|residual|advantage|scalar")
      print_success "已加载 Part3 + Part4 预设"
      ;;
    5)
      CONFIG_COMBINATIONS=(
        "baseline|baseline|none|1.0|0.0|1.0|1.0|-1.0|-1.0|0|0.0|residual|advantage|scalar"
        "part3_basic|part3_only|state_softmax|0.5|0.0|1.0|1.0|-1.0|-1.0|0|0.0|residual|advantage|scalar"
        "part4_basic|part4_only|none|1.0|0.0|1.0|1.0|-1.0|-1.0|1|0.1|residual|advantage|scalar"
        "combined|part3_part4|state_softmax|0.5|0.0|1.0|1.0|-1.0|-1.0|1|0.1|residual|advantage|scalar"
        "part3_late|part3_late_start|state_softmax|0.5|0.2|1.0|1.0|-1.0|-1.0|0|0.0|residual|advantage|scalar"
        "part3_alpha|part3_alpha|state_softmax|0.5|0.0|0.2|1.0|-1.0|-1.0|0|0.0|residual|advantage|scalar"
        "part3_lr|part3_lr|state_softmax|0.5|0.0|1.0|0.1|-1.0|-1.0|0|0.0|residual|advantage|scalar"
        "part3_wmax|part3_wmax|state_softmax|0.5|0.0|1.0|1.0|-1.0|2.0|0|0.0|residual|advantage|scalar"
        "part3_alpha_wmax|part3_alpha_wmax|state_softmax|0.5|0.0|0.2|1.0|-1.0|2.0|0|0.0|residual|advantage|scalar"
        "part3_late_alpha|part3_late_alpha|state_softmax|0.5|0.2|0.2|1.0|-1.0|-1.0|0|0.0|residual|advantage|scalar"
        "part3_alpha_lr|part3_alpha_lr|state_softmax|0.5|0.0|0.2|0.1|-1.0|-1.0|0|0.0|residual|advantage|scalar"
        "part3_conservative|part3_conservative|state_softmax|0.5|0.2|0.2|0.1|0.5|2.0|0|0.0|residual|advantage|scalar"
        "combined_late|part3_part4_late|state_softmax|0.5|0.2|1.0|1.0|-1.0|-1.0|1|0.1|residual|advantage|scalar"
        "combined_alpha|part3_part4_alpha|state_softmax|0.5|0.0|0.2|1.0|-1.0|2.0|1|0.1|residual|advantage|scalar"
        "combined_conservative|part3_part4_conservative|state_softmax|0.5|0.2|0.2|0.1|0.5|2.0|1|0.1|residual|advantage|scalar"
      )
      print_success "已加载 Comprehensive 预设 (15个组合)"
      ;;
    6)
      CONFIG_COMBINATIONS=(
        "new_features|late_start|state_softmax|0.5|0.2|1.0|1.0|-1.0|-1.0|0|0.0|residual|advantage|scalar"
        "new_features|alpha_shrink|state_softmax|0.5|0.0|0.2|1.0|-1.0|-1.0|0|0.0|residual|advantage|scalar"
        "new_features|separate_lr|state_softmax|0.5|0.0|1.0|0.1|-1.0|-1.0|0|0.0|residual|advantage|scalar"
        "new_features|hard_clamp|state_softmax|0.5|0.0|1.0|1.0|0.5|2.0|0|0.0|residual|advantage|scalar"
      )
      print_success "已加载 New Features 预设 (4个新特性)"
      ;;
    7)
      # 新增预设：Part 5 (Routes A/B/C)
      CONFIG_COMBINATIONS=(
        "route_a|residual_pair|none|1.0|0.0|1.0|1.0|-1.0|-1.0|1|0.1|residual|advantage|scalar"
        "route_b|vector_pair|none|1.0|0.0|1.0|1.0|-1.0|-1.0|1|0.1|residual|advantage|vector"
        "route_c|pg_target|none|1.0|0.0|1.0|1.0|-1.0|-1.0|1|0.1|residual|pg_grad|scalar"
        "route_abc|all_combined|none|1.0|0.0|1.0|1.0|-1.0|-1.0|1|0.1|residual|pg_grad|vector"
      )
      print_success "已加载 Part 5 Routes 预设 (A/B/C/Combined)"
      ;;
    [Bb])
      return
      ;;
  esac
  sleep 1
}

# =============================================================================
# 5. 训练参数设置
# =============================================================================

configure_training_params() {
  clear
  print_header "训练参数设置"
  echo
  print_section "当前设置"
  echo "  Seed: ${CURRENT_SEED}"
  echo "  Train Iters: ${CURRENT_TRAIN_ITERS}"
  echo "  Num Trainer GPUs: ${CURRENT_NUM_TRAINER_GPUS}"
  echo "  Num Rollout Workers: ${CURRENT_NUM_ROLLOUT_WORKERS}"
  echo "  Num Eval Workers: ${CURRENT_NUM_EVAL_WORKERS}"
  echo "  Train Batch Size: ${CURRENT_TRAIN_BATCH_SIZE}"
  echo
  
  if ask_yes_no "是否修改训练参数?"; then
    echo
    echo -ne "${YELLOW}Seed (当前: ${CURRENT_SEED}): ${NC}"
    read -r new_seed
    CURRENT_SEED="${new_seed:-$CURRENT_SEED}"
    
    echo -ne "${YELLOW}Train Iters (当前: ${CURRENT_TRAIN_ITERS}): ${NC}"
    read -r new_iters
    CURRENT_TRAIN_ITERS="${new_iters:-$CURRENT_TRAIN_ITERS}"
    
    echo -ne "${YELLOW}Num Trainer GPUs (当前: ${CURRENT_NUM_TRAINER_GPUS}): ${NC}"
    read -r new_trainer_gpus
    CURRENT_NUM_TRAINER_GPUS="${new_trainer_gpus:-$CURRENT_NUM_TRAINER_GPUS}"
    
    echo -ne "${YELLOW}Num Rollout Workers (当前: ${CURRENT_NUM_ROLLOUT_WORKERS}): ${NC}"
    read -r new_rollout
    CURRENT_NUM_ROLLOUT_WORKERS="${new_rollout:-$CURRENT_NUM_ROLLOUT_WORKERS}"
    
    echo -ne "${YELLOW}Num Eval Workers (当前: ${CURRENT_NUM_EVAL_WORKERS}): ${NC}"
    read -r new_eval
    CURRENT_NUM_EVAL_WORKERS="${new_eval:-$CURRENT_NUM_EVAL_WORKERS}"
    
    echo -ne "${YELLOW}Train Batch Size (当前: ${CURRENT_TRAIN_BATCH_SIZE}): ${NC}"
    read -r new_batch
    CURRENT_TRAIN_BATCH_SIZE="${new_batch:-$CURRENT_TRAIN_BATCH_SIZE}"
    
    print_success "训练参数已更新"
    sleep 1
  fi
}

# =============================================================================
# 6. GPU与并行设置
# =============================================================================

configure_gpu_parallel() {
  clear
  print_header "GPU与并行设置"
  echo
  print_section "当前设置"
  echo "  Max Parallel Jobs: ${CURRENT_MAX_PARALLEL_JOBS}"
  echo "  GPU Configs: ${CURRENT_GPU_CONFIGS}"
  echo "  Startup Delay: ${CURRENT_STARTUP_DELAY}s"
  echo "  Monitor Timeout: ${CURRENT_MONITOR_TIMEOUT}s (检测Ray启动超时)"
  echo "  Max Retries: ${CURRENT_MAX_RETRIES} (任务失败最大重试次数)"
  if [[ "${CURRENT_ENABLE_SWANLAB}" == "1" ]]; then
    echo "  Enable SwanLab: ${GREEN}是${NC} (同时使用SwanLab和TensorBoard)"
  else
    echo "  Enable SwanLab: ${YELLOW}否${NC} (仅使用TensorBoard)"
  fi
  echo
  
  if ask_yes_no "是否修改GPU与并行设置?"; then
    echo
    echo -ne "${YELLOW}Max Parallel Jobs (当前: ${CURRENT_MAX_PARALLEL_JOBS}): ${NC}"
    read -r new_parallel
    CURRENT_MAX_PARALLEL_JOBS="${new_parallel:-$CURRENT_MAX_PARALLEL_JOBS}"
    
    echo -ne "${YELLOW}GPU Configs (逗号分隔, 如 '0,1', 当前: ${CURRENT_GPU_CONFIGS}): ${NC}"
    read -r new_gpus
    CURRENT_GPU_CONFIGS="${new_gpus:-$CURRENT_GPU_CONFIGS}"
    
    echo -ne "${YELLOW}Startup Delay (秒, 避免端口冲突, 当前: ${CURRENT_STARTUP_DELAY}): ${NC}"
    read -r new_delay
    CURRENT_STARTUP_DELAY="${new_delay:-$CURRENT_STARTUP_DELAY}"
    
    echo -ne "${YELLOW}Monitor Timeout (秒, 检测Ray启动超时, 当前: ${CURRENT_MONITOR_TIMEOUT}): ${NC}"
    read -r new_timeout
    CURRENT_MONITOR_TIMEOUT="${new_timeout:-$CURRENT_MONITOR_TIMEOUT}"
    
    echo -ne "${YELLOW}Max Retries (任务失败最大重试次数, 当前: ${CURRENT_MAX_RETRIES}): ${NC}"
    read -r new_retries
    CURRENT_MAX_RETRIES="${new_retries:-$CURRENT_MAX_RETRIES}"
    
    echo
    echo "日志记录选项:"
    echo "  1) 启用SwanLab (同时使用SwanLab和TensorBoard)"
    echo "  2) 仅使用TensorBoard (禁用SwanLab)"
    echo -ne "${YELLOW}选择 [1-2, 当前: $([[ "${CURRENT_ENABLE_SWANLAB}" == "1" ]] && echo "1" || echo "2")]: ${NC}"
    read -r swanlab_choice
    case "$swanlab_choice" in
      1) CURRENT_ENABLE_SWANLAB="1";;
      2) CURRENT_ENABLE_SWANLAB="0";;
      *) ;;  # 保持当前值
    esac
    
    print_success "GPU与并行设置已更新"
    sleep 1
  fi
}

# =============================================================================
# 10. 自动组合生成器
# =============================================================================

auto_combination_generator() {
  clear
  print_header "自动组合生成器"
  echo
  print_info "通过定义参数组自动生成所有可能的组合（笛卡尔积）"
  echo
  print_section "子菜单"
  echo "  1) 定义参数组"
  echo "  2) 查看当前参数组"
  echo "  3) 生成组合 (笛卡尔积)"
  echo "  4) 清空参数组"
  echo "  B) 返回主菜单"
  echo
  echo -ne "${YELLOW}请选择 [1-4/B]: ${NC}"
  read -r choice
  
  case "$choice" in
    1) define_parameter_groups;;
    2) view_parameter_groups;;
    3) generate_cartesian_product;;
    4)
      if ask_yes_no "确认清空所有参数组?"; then
        AUTO_ADV_MODES=()
        AUTO_ADV_TEMPS=()
        AUTO_LATE_FRACS=()
        AUTO_ALPHAS=()
        AUTO_LR_MULTS=()
        AUTO_W_MINS=()
        AUTO_W_MAXS=()
        AUTO_AUX_ENABLES=()
        AUTO_AUX_COEFS=()
        print_success "已清空所有参数组"
        sleep 1
      fi
      auto_combination_generator
      ;;
    [Bb]) return;;
    *) auto_combination_generator;;
  esac
}

define_parameter_groups() {
  clear
  print_header "定义参数组"
  echo
  print_info "为每个参数定义多个候选值，空格分隔"
  print_warning "例如: 0.5 1.0 2.0"
  echo
  
  # AdvStruct Mode
  print_section "AdvStruct Mode"
  echo "可选值: none, state_softmax, state_action_softmax"
  echo -ne "${YELLOW}当前: [${AUTO_ADV_MODES[@]:-未设置}]${NC}\n"
  echo -ne "${YELLOW}输入 (回车跳过): ${NC}"
  read -r input
  if [[ -n "$input" ]]; then
    read -ra AUTO_ADV_MODES <<< "$input"
    print_success "已设置 ${#AUTO_ADV_MODES[@]} 个 AdvStruct Mode"
  fi
  echo
  
  # Temperature
  print_section "Temperature"
  echo "推荐范围: 0.1 - 2.0"
  echo -ne "${YELLOW}当前: [${AUTO_ADV_TEMPS[@]:-未设置}]${NC}\n"
  echo -ne "${YELLOW}输入 (回车跳过): ${NC}"
  read -r input
  if [[ -n "$input" ]]; then
    read -ra AUTO_ADV_TEMPS <<< "$input"
    print_success "已设置 ${#AUTO_ADV_TEMPS[@]} 个 Temperature"
  fi
  echo
  
  # Late Start Frac
  print_section "Late Start Frac"
  echo "推荐范围: 0.0 - 0.3"
  echo -ne "${YELLOW}当前: [${AUTO_LATE_FRACS[@]:-未设置}]${NC}\n"
  echo -ne "${YELLOW}输入 (回车跳过): ${NC}"
  read -r input
  if [[ -n "$input" ]]; then
    read -ra AUTO_LATE_FRACS <<< "$input"
    print_success "已设置 ${#AUTO_LATE_FRACS[@]} 个 Late Start Frac"
  fi
  echo
  
  # Alpha
  print_section "Alpha (收缩系数)"
  echo "推荐范围: 0.1 - 1.0"
  echo -ne "${YELLOW}当前: [${AUTO_ALPHAS[@]:-未设置}]${NC}\n"
  echo -ne "${YELLOW}输入 (回车跳过): ${NC}"
  read -r input
  if [[ -n "$input" ]]; then
    read -ra AUTO_ALPHAS <<< "$input"
    print_success "已设置 ${#AUTO_ALPHAS[@]} 个 Alpha"
  fi
  echo
  
  # LR Multiplier
  print_section "LR Multiplier"
  echo "推荐范围: 0.01 - 1.0"
  echo -ne "${YELLOW}当前: [${AUTO_LR_MULTS[@]:-未设置}]${NC}\n"
  echo -ne "${YELLOW}输入 (回车跳过): ${NC}"
  read -r input
  if [[ -n "$input" ]]; then
    read -ra AUTO_LR_MULTS <<< "$input"
    print_success "已设置 ${#AUTO_LR_MULTS[@]} 个 LR Multiplier"
  fi
  echo
  
  # W Min
  print_section "W Min (权重下限)"
  echo "推荐: -1.0 (禁用) 或 0.1-0.8"
  echo -ne "${YELLOW}当前: [${AUTO_W_MINS[@]:-未设置}]${NC}\n"
  echo -ne "${YELLOW}输入 (回车跳过): ${NC}"
  read -r input
  if [[ -n "$input" ]]; then
    read -ra AUTO_W_MINS <<< "$input"
    print_success "已设置 ${#AUTO_W_MINS[@]} 个 W Min"
  fi
  echo
  
  # W Max
  print_section "W Max (权重上限)"
  echo "推荐: -1.0 (禁用) 或 1.5-3.0"
  echo -ne "${YELLOW}当前: [${AUTO_W_MAXS[@]:-未设置}]${NC}\n"
  echo -ne "${YELLOW}输入 (回车跳过): ${NC}"
  read -r input
  if [[ -n "$input" ]]; then
    read -ra AUTO_W_MAXS <<< "$input"
    print_success "已设置 ${#AUTO_W_MAXS[@]} 个 W Max"
  fi
  echo
  
  # AuxAdv Enable
  print_section "AuxAdv Enable"
  echo "可选值: 0 (禁用) 或 1 (启用)"
  echo -ne "${YELLOW}当前: [${AUTO_AUX_ENABLES[@]:-未设置}]${NC}\n"
  echo -ne "${YELLOW}输入 (回车跳过): ${NC}"
  read -r input
  if [[ -n "$input" ]]; then
    read -ra AUTO_AUX_ENABLES <<< "$input"
    print_success "已设置 ${#AUTO_AUX_ENABLES[@]} 个 AuxAdv Enable"
  fi
  echo
  
  # AuxAdv Coef
  print_section "AuxAdv Coef"
  echo "推荐范围: 0.05 - 0.3"
  echo -ne "${YELLOW}当前: [${AUTO_AUX_COEFS[@]:-未设置}]${NC}\n"
  echo -ne "${YELLOW}输入 (回车跳过): ${NC}"
  read -r input
  if [[ -n "$input" ]]; then
    read -ra AUTO_AUX_COEFS <<< "$input"
    print_success "已设置 ${#AUTO_AUX_COEFS[@]} 个 AuxAdv Coef"
  fi
  
  echo
  print_success "参数组定义完成！"
  press_any_key
  auto_combination_generator
}

view_parameter_groups() {
  clear
  print_header "当前参数组"
  echo
  
  local total_combinations=1
  
  echo "AdvStruct Mode (${#AUTO_ADV_MODES[@]} 个):"
  if [ ${#AUTO_ADV_MODES[@]} -eq 0 ]; then
    echo "  ${RED}(未设置, 默认: state_softmax)${NC}"
  else
    for val in "${AUTO_ADV_MODES[@]}"; do echo "  - $val"; done
    total_combinations=$((total_combinations * ${#AUTO_ADV_MODES[@]}))
  fi
  echo
  
  echo "Temperature (${#AUTO_ADV_TEMPS[@]} 个):"
  if [ ${#AUTO_ADV_TEMPS[@]} -eq 0 ]; then
    echo "  ${RED}(未设置, 默认: 0.5)${NC}"
  else
    for val in "${AUTO_ADV_TEMPS[@]}"; do echo "  - $val"; done
    total_combinations=$((total_combinations * ${#AUTO_ADV_TEMPS[@]}))
  fi
  echo
  
  echo "Late Start Frac (${#AUTO_LATE_FRACS[@]} 个):"
  if [ ${#AUTO_LATE_FRACS[@]} -eq 0 ]; then
    echo "  ${RED}(未设置, 默认: 0.0)${NC}"
  else
    for val in "${AUTO_LATE_FRACS[@]}"; do echo "  - $val"; done
    total_combinations=$((total_combinations * ${#AUTO_LATE_FRACS[@]}))
  fi
  echo
  
  echo "Alpha (${#AUTO_ALPHAS[@]} 个):"
  if [ ${#AUTO_ALPHAS[@]} -eq 0 ]; then
    echo "  ${RED}(未设置, 默认: 1.0)${NC}"
  else
    for val in "${AUTO_ALPHAS[@]}"; do echo "  - $val"; done
    total_combinations=$((total_combinations * ${#AUTO_ALPHAS[@]}))
  fi
  echo
  
  echo "LR Multiplier (${#AUTO_LR_MULTS[@]} 个):"
  if [ ${#AUTO_LR_MULTS[@]} -eq 0 ]; then
    echo "  ${RED}(未设置, 默认: 1.0)${NC}"
  else
    for val in "${AUTO_LR_MULTS[@]}"; do echo "  - $val"; done
    total_combinations=$((total_combinations * ${#AUTO_LR_MULTS[@]}))
  fi
  echo
  
  echo "W Min (${#AUTO_W_MINS[@]} 个):"
  if [ ${#AUTO_W_MINS[@]} -eq 0 ]; then
    echo "  ${RED}(未设置, 默认: -1.0)${NC}"
  else
    for val in "${AUTO_W_MINS[@]}"; do echo "  - $val"; done
    total_combinations=$((total_combinations * ${#AUTO_W_MINS[@]}))
  fi
  echo
  
  echo "W Max (${#AUTO_W_MAXS[@]} 个):"
  if [ ${#AUTO_W_MAXS[@]} -eq 0 ]; then
    echo "  ${RED}(未设置, 默认: -1.0)${NC}"
  else
    for val in "${AUTO_W_MAXS[@]}"; do echo "  - $val"; done
    total_combinations=$((total_combinations * ${#AUTO_W_MAXS[@]}))
  fi
  echo
  
  echo "AuxAdv Enable (${#AUTO_AUX_ENABLES[@]} 个):"
  if [ ${#AUTO_AUX_ENABLES[@]} -eq 0 ]; then
    echo "  ${RED}(未设置, 默认: 0)${NC}"
  else
    for val in "${AUTO_AUX_ENABLES[@]}"; do echo "  - $val"; done
    total_combinations=$((total_combinations * ${#AUTO_AUX_ENABLES[@]}))
  fi
  echo
  
  echo "AuxAdv Coef (${#AUTO_AUX_COEFS[@]} 个):"
  if [ ${#AUTO_AUX_COEFS[@]} -eq 0 ]; then
    echo "  ${RED}(未设置, 默认: 0.0)${NC}"
  else
    for val in "${AUTO_AUX_COEFS[@]}"; do echo "  - $val"; done
    total_combinations=$((total_combinations * ${#AUTO_AUX_COEFS[@]}))
  fi
  echo
  
  print_section "预计生成组合数"
  echo -e "  ${BOLD}${GREEN}${total_combinations}${NC} 个配置组合"
  echo
  
  press_any_key
  auto_combination_generator
}

generate_cartesian_product() {
  clear
  print_header "生成笛卡尔积组合"
  echo
  
  # 设置默认值（如果未设置）
  local adv_modes=("${AUTO_ADV_MODES[@]}")
  [[ ${#adv_modes[@]} -eq 0 ]] && adv_modes=("state_softmax")
  
  local adv_temps=("${AUTO_ADV_TEMPS[@]}")
  [[ ${#adv_temps[@]} -eq 0 ]] && adv_temps=("0.5")
  
  local late_fracs=("${AUTO_LATE_FRACS[@]}")
  [[ ${#late_fracs[@]} -eq 0 ]] && late_fracs=("0.0")
  
  local alphas=("${AUTO_ALPHAS[@]}")
  [[ ${#alphas[@]} -eq 0 ]] && alphas=("1.0")
  
  local lr_mults=("${AUTO_LR_MULTS[@]}")
  [[ ${#lr_mults[@]} -eq 0 ]] && lr_mults=("1.0")
  
  local w_mins=("${AUTO_W_MINS[@]}")
  [[ ${#w_mins[@]} -eq 0 ]] && w_mins=("-1.0")
  
  local w_maxs=("${AUTO_W_MAXS[@]}")
  [[ ${#w_maxs[@]} -eq 0 ]] && w_maxs=("-1.0")
  
  local aux_enables=("${AUTO_AUX_ENABLES[@]}")
  [[ ${#aux_enables[@]} -eq 0 ]] && aux_enables=("0")
  
  local aux_coefs=("${AUTO_AUX_COEFS[@]}")
  [[ ${#aux_coefs[@]} -eq 0 ]] && aux_coefs=("0.0")
  
  # 计算总组合数
  local total=$((${#adv_modes[@]} * ${#adv_temps[@]} * ${#late_fracs[@]} * ${#alphas[@]} * \
                 ${#lr_mults[@]} * ${#w_mins[@]} * ${#w_maxs[@]} * ${#aux_enables[@]} * ${#aux_coefs[@]}))
  
  print_warning "即将生成 ${total} 个配置组合"
  echo
  
  # 询问分组策略
  print_section "选择分组策略"
  echo "  1) 不分组 (所有配置在同一目录)"
  echo "  2) 按主变化参数分组 (推荐)"
  echo "  3) 自定义统一组名"
  echo "  4) 智能分组 (每${MAGENTA}10${NC}个配置一组)"
  echo
  echo -ne "${YELLOW}请选择 [1-4, 默认2]: ${NC}"
  read -r grouping_strategy
  grouping_strategy="${grouping_strategy:-2}"
  
  local group_prefix="auto_gen"
  
  case "$grouping_strategy" in
    1)
      group_prefix="all"
      print_info "所有配置将放在 'all' 组中"
      ;;
    3)
      echo -ne "${YELLOW}输入统一组名: ${NC}"
      read -r custom_group
      group_prefix="${custom_group:-auto_gen}"
      print_info "所有配置将放在 '${group_prefix}' 组中"
      ;;
    4)
      print_info "将按每10个配置自动分组"
      ;;
    2|*)
      print_info "将按主要变化参数自动分组"
      ;;
  esac
  echo
  
  if ! ask_yes_no "是否覆盖现有的配置组合列表 (选项3中的配置)?"; then
    print_info "已取消"
    sleep 1
    auto_combination_generator
    return
  fi
  
  # 清空现有配置
  CONFIG_COMBINATIONS=()
  
  # 生成笛卡尔积
  local combo_idx=1
  for adv_mode in "${adv_modes[@]}"; do
    for adv_temp in "${adv_temps[@]}"; do
      for late_frac in "${late_fracs[@]}"; do
        for alpha in "${alphas[@]}"; do
          for lr_mult in "${lr_mults[@]}"; do
            for w_min in "${w_mins[@]}"; do
              for w_max in "${w_maxs[@]}"; do
                for aux_enable in "${aux_enables[@]}"; do
                  for aux_coef in "${aux_coefs[@]}"; do
                    # 生成组名（根据策略）
                    local group_name
                    case "$grouping_strategy" in
                      1|3)
                        # 统一组名
                        group_name="${group_prefix}"
                        ;;
                      4)
                        # 智能分组：每10个一组
                        local group_num=$(( (combo_idx - 1) / 10 + 1 ))
                        group_name="group_${group_num}"
                        ;;
                      2|*)
                        # 按主变化参数分组
                        group_name="auto"
                        # 找出与默认值不同的参数
                        [[ "$adv_mode" != "state_softmax" ]] && group_name="${group_name}_${adv_mode}"
                        [[ "$adv_temp" != "0.5" ]] && group_name="${group_name}_t${adv_temp}"
                        [[ "$late_frac" != "0.0" ]] && group_name="${group_name}_late"
                        [[ "$alpha" != "1.0" ]] && group_name="${group_name}_alpha"
                        [[ "$lr_mult" != "1.0" ]] && group_name="${group_name}_lr"
                        [[ "$w_min" != "-1.0" ]] && group_name="${group_name}_wmin"
                        [[ "$w_max" != "-1.0" ]] && group_name="${group_name}_wmax"
                        [[ "$aux_enable" == "1" ]] && group_name="${group_name}_aux"
                        # 如果所有都是默认值，使用baseline
                        [[ "$group_name" == "auto" ]] && group_name="baseline"
                        ;;
                    esac
                    
                    # 生成标签
                    local label="cfg_${combo_idx}"
                    
                    # 添加描述性后缀
                    [[ "$adv_mode" != "state_softmax" ]] && label="${label}_${adv_mode}"
                    [[ "$adv_temp" != "0.5" ]] && label="${label}_t${adv_temp}"
                    [[ "$late_frac" != "0.0" ]] && label="${label}_late${late_frac}"
                    [[ "$alpha" != "1.0" ]] && label="${label}_a${alpha}"
                    [[ "$lr_mult" != "1.0" ]] && label="${label}_lr${lr_mult}"
                    [[ "$w_min" != "-1.0" ]] && label="${label}_wmin${w_min}"
                    [[ "$w_max" != "-1.0" ]] && label="${label}_wmax${w_max}"
                    [[ "$aux_enable" == "1" ]] && label="${label}_aux${aux_coef}"
                    
                    # 组装配置字符串
                    # 默认使用 residual, advantage, scalar 作为自动生成的 Aux 参数（暂不支持在 auto gen 中 sweep 这些，除非继续加循环）
                    # 也可以在这里加上对 AUTO_AUX_MODES 等的循环支持
                    # 暂时先用默认值，避免层级过深
                    local config_str="${group_name}|${label}|${adv_mode}|${adv_temp}|${late_frac}|${alpha}|${lr_mult}|${w_min}|${w_max}|${aux_enable}|${aux_coef}|residual|advantage|scalar"
                    CONFIG_COMBINATIONS+=("${config_str}")
                    
                    combo_idx=$((combo_idx + 1))
                  done
                done
              done
            done
          done
        done
      done
    done
  done
  
  echo
  print_success "已生成 ${#CONFIG_COMBINATIONS[@]} 个配置组合！"
  echo
  print_info "提示: 您可以在主菜单选项7中查看完整配置"
  echo
  press_any_key
}

# =============================================================================
# 7. 查看完整配置
# =============================================================================

view_full_config() {
  clear
  print_header "完整配置摘要"
  echo
  
  print_section "任务列表 (${#SELECTED_TASKS[@]} 个)"
  if [ ${#SELECTED_TASKS[@]} -eq 0 ]; then
    echo "  ${RED}(未选择)${NC}"
  else
    for task in "${SELECTED_TASKS[@]}"; do
      echo "  - ${task}"
    done
  fi
  echo
  
  print_section "Clip模式 (${#SELECTED_CLIP_MODES[@]} 个)"
  if [ ${#SELECTED_CLIP_MODES[@]} -eq 0 ]; then
    echo "  ${RED}(未选择)${NC}"
  else
    for mode in "${SELECTED_CLIP_MODES[@]}"; do
      echo "  - ${mode}"
    done
  fi
  echo
  
  print_section "实验配置组合 (${#CONFIG_COMBINATIONS[@]} 个)"
  if [ ${#CONFIG_COMBINATIONS[@]} -eq 0 ]; then
    echo "  ${RED}(未配置)${NC}"
  else
    for i in "${!CONFIG_COMBINATIONS[@]}"; do
      IFS='|' read -r group_name label adv_mode adv_temp late_frac alpha lr_mult w_min w_max aux_enable aux_coef aux_mode aux_target_mode aux_pair_mode <<< "${CONFIG_COMBINATIONS[$i]}"
      echo "  $((i+1))) ${CYAN}[${group_name}]${NC} ${BOLD}${label}${NC}"
      echo "     - AdvStruct: ${adv_mode} (temp=${adv_temp}, late=${late_frac}, alpha=${alpha}, lr=${lr_mult}, w_min=${w_min}, w_max=${w_max})"
      echo "     - AuxAdv: enable=${aux_enable} (mode=${aux_mode:-residual}, target=${aux_target_mode:-advantage}, pair=${aux_pair_mode:-scalar}, coef=${aux_coef})"
    done
  fi
  echo
  
  print_section "训练参数"
  echo "  Seed: ${CURRENT_SEED}"
  echo "  Train Iters: ${CURRENT_TRAIN_ITERS}"
  echo "  Num Trainer GPUs: ${CURRENT_NUM_TRAINER_GPUS}"
  echo "  Num Rollout Workers: ${CURRENT_NUM_ROLLOUT_WORKERS}"
  echo "  Num Eval Workers: ${CURRENT_NUM_EVAL_WORKERS}"
  echo "  Train Batch Size: ${CURRENT_TRAIN_BATCH_SIZE}"
  echo
  
  print_section "GPU与并行设置"
  echo "  Max Parallel Jobs: ${CURRENT_MAX_PARALLEL_JOBS}"
  echo "  GPU Configs: ${CURRENT_GPU_CONFIGS}"
  echo "  Startup Delay: ${CURRENT_STARTUP_DELAY}s"
  echo "  Monitor Timeout: ${CURRENT_MONITOR_TIMEOUT}s"
  echo "  Max Retries: ${CURRENT_MAX_RETRIES}"
  if [[ "${CURRENT_ENABLE_SWANLAB}" == "1" ]]; then
    echo "  Enable SwanLab: ${GREEN}是${NC} (同时使用SwanLab和TensorBoard)"
  else
    echo "  Enable SwanLab: ${YELLOW}否${NC} (仅使用TensorBoard)"
  fi
  echo
  
  print_section "总实验数"
  local total_experiments=$((${#SELECTED_TASKS[@]} * ${#SELECTED_CLIP_MODES[@]} * ${#CONFIG_COMBINATIONS[@]}))
  echo -e "  ${BOLD}${GREEN}${total_experiments}${NC} 个实验"
  echo
  
  press_any_key
}

# =============================================================================
# 8. 保存配置
# =============================================================================

save_config() {
  clear
  print_header "保存配置到文件"
  echo
  echo -ne "${YELLOW}配置文件名 (不含扩展名): ${NC}"
  read -r config_name
  
  if [[ -z "$config_name" ]]; then
    print_error "文件名不能为空"
    sleep 1
    return
  fi
  
  local config_file="${CONFIG_DIR}/${config_name}.conf"
  
  # 保存配置
  {
    echo "# MetaWorld PPO Interactive Config"
    echo "# Generated: $(date)"
    echo
    echo "# Tasks"
    for task in "${SELECTED_TASKS[@]}"; do
      echo "TASK=${task}"
    done
    echo
    echo "# Clip Modes"
    for mode in "${SELECTED_CLIP_MODES[@]}"; do
      echo "CLIP_MODE=${mode}"
    done
    echo
    echo "# Configurations"
    for config in "${CONFIG_COMBINATIONS[@]}"; do
      echo "CONFIG=${config}"
    done
    echo
    echo "# Training Parameters"
    echo "SEED=${CURRENT_SEED}"
    echo "TRAIN_ITERS=${CURRENT_TRAIN_ITERS}"
    echo "NUM_TRAINER_GPUS=${CURRENT_NUM_TRAINER_GPUS}"
    echo "NUM_ROLLOUT_WORKERS=${CURRENT_NUM_ROLLOUT_WORKERS}"
    echo "NUM_EVAL_WORKERS=${CURRENT_NUM_EVAL_WORKERS}"
    echo "TRAIN_BATCH_SIZE=${CURRENT_TRAIN_BATCH_SIZE}"
    echo
    echo "# GPU & Parallel"
    echo "MAX_PARALLEL_JOBS=${CURRENT_MAX_PARALLEL_JOBS}"
    echo "GPU_CONFIGS=${CURRENT_GPU_CONFIGS}"
    echo "STARTUP_DELAY=${CURRENT_STARTUP_DELAY}"
    echo "MONITOR_TIMEOUT=${CURRENT_MONITOR_TIMEOUT}"
    echo "MAX_RETRIES=${CURRENT_MAX_RETRIES}"
    echo "ENABLE_SWANLAB=${CURRENT_ENABLE_SWANLAB}"
  } > "${config_file}"
  
  print_success "配置已保存到: ${config_file}"
  sleep 2
}

# =============================================================================
# 9. 加载配置
# =============================================================================

load_config() {
  clear
  print_header "从文件加载配置"
  echo
  
  local config_files=("${CONFIG_DIR}"/*.conf)
  if [ ! -e "${config_files[0]}" ]; then
    print_warning "没有找到保存的配置文件"
    sleep 2
    return
  fi
  
  print_info "可用的配置文件:"
  echo
  local idx=1
  for file in "${config_files[@]}"; do
    echo "  ${idx}) $(basename "$file")"
    idx=$((idx+1))
  done
  echo
  echo -ne "${YELLOW}选择配置文件编号 (或 B 返回): ${NC}"
  read -r choice
  
  if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#config_files[@]}" ]; then
    local selected_file="${config_files[$((choice-1))]}"
    
    # 重置当前配置
    SELECTED_TASKS=()
    SELECTED_CLIP_MODES=()
    CONFIG_COMBINATIONS=()
    
    # 加载配置
    while IFS= read -r line; do
      # 跳过注释和空行
      [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
      
      if [[ "$line" =~ ^TASK=(.*)$ ]]; then
        SELECTED_TASKS+=("${BASH_REMATCH[1]}")
      elif [[ "$line" =~ ^CLIP_MODE=(.*)$ ]]; then
        SELECTED_CLIP_MODES+=("${BASH_REMATCH[1]}")
      elif [[ "$line" =~ ^CONFIG=(.*)$ ]]; then
        CONFIG_COMBINATIONS+=("${BASH_REMATCH[1]}")
      elif [[ "$line" =~ ^SEED=(.*)$ ]]; then
        CURRENT_SEED="${BASH_REMATCH[1]}"
      elif [[ "$line" =~ ^TRAIN_ITERS=(.*)$ ]]; then
        CURRENT_TRAIN_ITERS="${BASH_REMATCH[1]}"
      elif [[ "$line" =~ ^NUM_TRAINER_GPUS=(.*)$ ]]; then
        CURRENT_NUM_TRAINER_GPUS="${BASH_REMATCH[1]}"
      elif [[ "$line" =~ ^NUM_ROLLOUT_WORKERS=(.*)$ ]]; then
        CURRENT_NUM_ROLLOUT_WORKERS="${BASH_REMATCH[1]}"
      elif [[ "$line" =~ ^NUM_EVAL_WORKERS=(.*)$ ]]; then
        CURRENT_NUM_EVAL_WORKERS="${BASH_REMATCH[1]}"
      elif [[ "$line" =~ ^TRAIN_BATCH_SIZE=(.*)$ ]]; then
        CURRENT_TRAIN_BATCH_SIZE="${BASH_REMATCH[1]}"
      elif [[ "$line" =~ ^MAX_PARALLEL_JOBS=(.*)$ ]]; then
        CURRENT_MAX_PARALLEL_JOBS="${BASH_REMATCH[1]}"
      elif [[ "$line" =~ ^GPU_CONFIGS=(.*)$ ]]; then
        CURRENT_GPU_CONFIGS="${BASH_REMATCH[1]}"
      elif [[ "$line" =~ ^STARTUP_DELAY=(.*)$ ]]; then
        CURRENT_STARTUP_DELAY="${BASH_REMATCH[1]}"
      elif [[ "$line" =~ ^MONITOR_TIMEOUT=(.*)$ ]]; then
        CURRENT_MONITOR_TIMEOUT="${BASH_REMATCH[1]}"
      elif [[ "$line" =~ ^MAX_RETRIES=(.*)$ ]]; then
        CURRENT_MAX_RETRIES="${BASH_REMATCH[1]}"
      elif [[ "$line" =~ ^ENABLE_SWANLAB=(.*)$ ]]; then
        CURRENT_ENABLE_SWANLAB="${BASH_REMATCH[1]}"
      fi
    done < "${selected_file}"
    
    print_success "配置已加载: $(basename "$selected_file")"
    echo "  - 任务: ${#SELECTED_TASKS[@]} 个"
    echo "  - Clip模式: ${#SELECTED_CLIP_MODES[@]} 个"
    echo "  - 配置组合: ${#CONFIG_COMBINATIONS[@]} 个"
    sleep 2
  fi
}

# =============================================================================
# 10.5. 保存上一次运行的配置
# =============================================================================

save_last_run_config() {
  # 保存当前配置到JSON文件（使用Python生成JSON，更可靠）
  local json_file="${LAST_RUN_CONFIG_FILE}"
  local temp_dir="${CONFIG_DIR}/.temp"
  mkdir -p "${temp_dir}"
  
  if command -v python3 &> /dev/null || command -v python &> /dev/null; then
    local python_cmd="python3"
    command -v python3 &> /dev/null || python_cmd="python"
    
    # 将数组数据写入临时文件（每行一个元素）
    local tasks_file="${temp_dir}/tasks_$$.tmp"
    local clip_modes_file="${temp_dir}/clip_modes_$$.tmp"
    local config_combinations_file="${temp_dir}/config_combinations_$$.tmp"
    
    printf '%s\n' "${SELECTED_TASKS[@]}" > "${tasks_file}"
    printf '%s\n' "${SELECTED_CLIP_MODES[@]}" > "${clip_modes_file}"
    printf '%s\n' "${CONFIG_COMBINATIONS[@]}" > "${config_combinations_file}"
    
    # 使用Python生成JSON
    ${python_cmd} << PYTHON_EOF > "${json_file}"
import json
import sys
from datetime import datetime

# 从临时文件读取数组数据
tasks_file = "${tasks_file}"
clip_modes_file = "${clip_modes_file}"
config_combinations_file = "${config_combinations_file}"

def read_lines(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.rstrip('\n') for line in f if line.rstrip('\n')]
    except FileNotFoundError:
        return []

tasks = read_lines(tasks_file)
clip_modes = read_lines(clip_modes_file)
config_combinations = read_lines(config_combinations_file)

# 构建配置字典
config = {
    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "tasks": tasks,
    "clip_modes": clip_modes,
    "config_combinations": config_combinations,
    "seed": ${CURRENT_SEED},
    "train_iters": ${CURRENT_TRAIN_ITERS},
    "num_trainer_gpus": ${CURRENT_NUM_TRAINER_GPUS},
    "num_rollout_workers": ${CURRENT_NUM_ROLLOUT_WORKERS},
    "num_eval_workers": ${CURRENT_NUM_EVAL_WORKERS},
    "train_batch_size": ${CURRENT_TRAIN_BATCH_SIZE},
    "max_parallel_jobs": ${CURRENT_MAX_PARALLEL_JOBS},
    "gpu_configs": "${CURRENT_GPU_CONFIGS}",
    "startup_delay": ${CURRENT_STARTUP_DELAY},
    "monitor_timeout": ${CURRENT_MONITOR_TIMEOUT},
    "max_retries": ${CURRENT_MAX_RETRIES},
    "enable_swanlab": ${CURRENT_ENABLE_SWANLAB}
}

# 写入JSON文件
try:
    with open("${json_file}", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
except Exception as e:
    print(f"Error writing config file: {e}", file=sys.stderr)
    sys.exit(1)

# 清理临时文件
import os
for f in [tasks_file, clip_modes_file, config_combinations_file]:
    try:
        os.remove(f)
    except:
        pass
PYTHON_EOF

    # 清理临时文件（如果Python脚本失败）
    rm -f "${tasks_file}" "${clip_modes_file}" "${config_combinations_file}" 2>/dev/null
    
    if [[ $? -ne 0 ]]; then
      print_warning "保存上一次运行配置失败（但实验将继续运行）"
    fi
  else
    # 如果没有Python，使用简单的文本格式保存（备用方案）
    print_warning "未找到Python，使用备用格式保存配置"
    {
      echo "# Last Run Config"
      echo "# Generated: $(date '+%Y-%m-%d %H:%M:%S')"
      echo
      for task in "${SELECTED_TASKS[@]}"; do
        echo "TASK=${task}"
      done
      echo
      for mode in "${SELECTED_CLIP_MODES[@]}"; do
        echo "CLIP_MODE=${mode}"
      done
      echo
      for config in "${CONFIG_COMBINATIONS[@]}"; do
        echo "CONFIG=${config}"
      done
      echo
      echo "SEED=${CURRENT_SEED}"
      echo "TRAIN_ITERS=${CURRENT_TRAIN_ITERS}"
      echo "NUM_TRAINER_GPUS=${CURRENT_NUM_TRAINER_GPUS}"
      echo "NUM_ROLLOUT_WORKERS=${CURRENT_NUM_ROLLOUT_WORKERS}"
      echo "NUM_EVAL_WORKERS=${CURRENT_NUM_EVAL_WORKERS}"
      echo "TRAIN_BATCH_SIZE=${CURRENT_TRAIN_BATCH_SIZE}"
      echo "MAX_PARALLEL_JOBS=${CURRENT_MAX_PARALLEL_JOBS}"
      echo "GPU_CONFIGS=${CURRENT_GPU_CONFIGS}"
      echo "STARTUP_DELAY=${CURRENT_STARTUP_DELAY}"
      echo "MONITOR_TIMEOUT=${CURRENT_MONITOR_TIMEOUT}"
      echo "MAX_RETRIES=${CURRENT_MAX_RETRIES}"
      echo "ENABLE_SWANLAB=${CURRENT_ENABLE_SWANLAB}"
    } > "${json_file}.txt"
  fi
}

# =============================================================================
# 11. 加载上一次运行的配置
# =============================================================================

load_last_run_config() {
  clear
  print_header "加载上一次运行的配置"
  echo
  
  if [[ ! -f "${LAST_RUN_CONFIG_FILE}" ]]; then
    print_warning "没有找到上一次运行的配置文件"
    echo "文件路径: ${LAST_RUN_CONFIG_FILE}"
    echo
    print_info "提示: 只有运行过实验后才会保存上一次配置"
    sleep 2
    return
  fi
  
  # 使用Python解析JSON（更可靠）
  if command -v python3 &> /dev/null || command -v python &> /dev/null; then
    local python_cmd="python3"
    command -v python3 &> /dev/null || python_cmd="python"
    
    # 使用Python解析并显示配置信息
    local config_info=$(${python_cmd} << PYTHON_EOF
import json
import sys
import os

config_file = "${LAST_RUN_CONFIG_FILE}"
if not config_file or not os.path.exists(config_file):
    print("ERROR: Config file not found")
    sys.exit(1)

try:
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"TIMESTAMP:{config.get('timestamp', 'Unknown')}")
    print(f"TASKS:{','.join(config.get('tasks', []))}")
    print(f"CLIP_MODES:{','.join(config.get('clip_modes', []))}")
    print(f"CONFIG_COMBINATIONS:{len(config.get('config_combinations', []))}")
    print(f"SEED:{config.get('seed', 42)}")
    print(f"TRAIN_ITERS:{config.get('train_iters', 100000)}")
    print(f"NUM_TRAINER_GPUS:{config.get('num_trainer_gpus', 1)}")
    print(f"NUM_ROLLOUT_WORKERS:{config.get('num_rollout_workers', 16)}")
    print(f"NUM_EVAL_WORKERS:{config.get('num_eval_workers', 32)}")
    print(f"TRAIN_BATCH_SIZE:{config.get('train_batch_size', 512)}")
    print(f"MAX_PARALLEL_JOBS:{config.get('max_parallel_jobs', 8)}")
    print(f"GPU_CONFIGS:{config.get('gpu_configs', '0,1')}")
    print(f"STARTUP_DELAY:{config.get('startup_delay', 90)}")
    print(f"MONITOR_TIMEOUT:{config.get('monitor_timeout', 300)}")
    print(f"MAX_RETRIES:{config.get('max_retries', 3)}")
    print(f"ENABLE_SWANLAB:{config.get('enable_swanlab', 1)}")
    
    # 输出配置组合（每行一个）
    for combo in config.get('config_combinations', []):
        print(f"COMBO:{combo}")
        
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
PYTHON_EOF
    )
    
    if [[ $? -ne 0 ]] || [[ "${config_info}" == ERROR:* ]]; then
      print_error "解析配置文件失败: ${config_info#ERROR: }"
      sleep 2
      return
    fi
    
    # 显示配置信息
    print_info "上一次运行配置信息:"
    echo
    local timestamp=$(echo "${config_info}" | grep "^TIMESTAMP:" | cut -d: -f2-)
    echo "  运行时间: ${timestamp}"
    echo
    
    local task_count=$(echo "${config_info}" | grep "^TASKS:" | cut -d: -f2- | tr ',' '\n' | wc -l)
    echo "  任务数量: ${task_count} 个"
    local clip_mode_count=$(echo "${config_info}" | grep "^CLIP_MODES:" | cut -d: -f2- | tr ',' '\n' | wc -l)
    echo "  Clip模式: ${clip_mode_count} 个"
    local combo_count=$(echo "${config_info}" | grep "^CONFIG_COMBINATIONS:" | cut -d: -f2-)
    echo "  配置组合: ${combo_count} 个"
    echo
    
    if ! ask_yes_no "确认加载上一次运行的配置?"; then
      print_info "已取消"
      sleep 1
      return
    fi
    
    # 重置当前配置
    SELECTED_TASKS=()
    SELECTED_CLIP_MODES=()
    CONFIG_COMBINATIONS=()
    
    # 加载任务
    local tasks_line=$(echo "${config_info}" | grep "^TASKS:" | cut -d: -f2-)
    if [[ -n "$tasks_line" ]]; then
      IFS=',' read -ra TASK_ARRAY <<< "$tasks_line"
      SELECTED_TASKS=("${TASK_ARRAY[@]}")
    fi
    
    # 加载Clip模式
    local clip_modes_line=$(echo "${config_info}" | grep "^CLIP_MODES:" | cut -d: -f2-)
    if [[ -n "$clip_modes_line" ]]; then
      IFS=',' read -ra CLIP_ARRAY <<< "$clip_modes_line"
      SELECTED_CLIP_MODES=("${CLIP_ARRAY[@]}")
    fi
    
    # 加载配置组合
    while IFS= read -r line; do
      if [[ "$line" == COMBO:* ]]; then
        CONFIG_COMBINATIONS+=("${line#COMBO:}")
      fi
    done <<< "${config_info}"
    
    # 加载其他参数
    CURRENT_SEED=$(echo "${config_info}" | grep "^SEED:" | cut -d: -f2-)
    CURRENT_TRAIN_ITERS=$(echo "${config_info}" | grep "^TRAIN_ITERS:" | cut -d: -f2-)
    CURRENT_NUM_TRAINER_GPUS=$(echo "${config_info}" | grep "^NUM_TRAINER_GPUS:" | cut -d: -f2-)
    CURRENT_NUM_ROLLOUT_WORKERS=$(echo "${config_info}" | grep "^NUM_ROLLOUT_WORKERS:" | cut -d: -f2-)
    CURRENT_NUM_EVAL_WORKERS=$(echo "${config_info}" | grep "^NUM_EVAL_WORKERS:" | cut -d: -f2-)
    CURRENT_TRAIN_BATCH_SIZE=$(echo "${config_info}" | grep "^TRAIN_BATCH_SIZE:" | cut -d: -f2-)
    CURRENT_MAX_PARALLEL_JOBS=$(echo "${config_info}" | grep "^MAX_PARALLEL_JOBS:" | cut -d: -f2-)
    CURRENT_GPU_CONFIGS=$(echo "${config_info}" | grep "^GPU_CONFIGS:" | cut -d: -f2-)
    CURRENT_STARTUP_DELAY=$(echo "${config_info}" | grep "^STARTUP_DELAY:" | cut -d: -f2-)
    CURRENT_MONITOR_TIMEOUT=$(echo "${config_info}" | grep "^MONITOR_TIMEOUT:" | cut -d: -f2-)
    CURRENT_MAX_RETRIES=$(echo "${config_info}" | grep "^MAX_RETRIES:" | cut -d: -f2-)
    CURRENT_ENABLE_SWANLAB=$(echo "${config_info}" | grep "^ENABLE_SWANLAB:" | cut -d: -f2-)
    
    print_success "上一次运行配置已加载"
    echo "  - 任务: ${#SELECTED_TASKS[@]} 个"
    echo "  - Clip模式: ${#SELECTED_CLIP_MODES[@]} 个"
    echo "  - 配置组合: ${#CONFIG_COMBINATIONS[@]} 个"
    sleep 2
    
  else
    print_error "未找到Python解释器，无法解析JSON配置文件"
    print_info "请安装Python3或Python"
    sleep 2
    return
  fi
}

# =============================================================================
# R. 运行实验
# =============================================================================

run_experiments() {
  # 验证配置
  if [ ${#SELECTED_TASKS[@]} -eq 0 ]; then
    print_error "错误: 未选择任务"
    sleep 2
    return
  fi
  
  if [ ${#SELECTED_CLIP_MODES[@]} -eq 0 ]; then
    print_error "错误: 未选择Clip模式"
    sleep 2
    return
  fi
  
  if [ ${#CONFIG_COMBINATIONS[@]} -eq 0 ]; then
    print_error "错误: 未配置实验组合"
    sleep 2
    return
  fi
  
  clear
  print_header "运行实验"
  echo
  
  local total_experiments=$((${#SELECTED_TASKS[@]} * ${#SELECTED_CLIP_MODES[@]} * ${#CONFIG_COMBINATIONS[@]}))
  
  print_warning "即将启动 ${total_experiments} 个实验"
  echo
  echo "  - 任务数: ${#SELECTED_TASKS[@]}"
  echo "  - Clip模式: ${#SELECTED_CLIP_MODES[@]}"
  echo "  - 配置组合: ${#CONFIG_COMBINATIONS[@]}"
  echo "  - 最大并行数: ${CURRENT_MAX_PARALLEL_JOBS}"
  echo
  
  if ! ask_yes_no "确认启动?"; then
    print_info "已取消"
    sleep 1
    return
  fi
  
  # 保存当前配置为上一次运行配置
  save_last_run_config
  
  # 创建日志目录
  local timestamp=$(date +%Y%m%d_%H%M%S)
  local log_dir="logs/interactive_run_${timestamp}"
  mkdir -p "${log_dir}"
  
  print_success "日志目录: ${log_dir}"
  echo
  
  # 生成临时脚本
  local temp_script="${log_dir}/run_script.sh"
  generate_run_script "${temp_script}" "${log_dir}"
  
  print_info "正在启动实验..."
  bash "${temp_script}"
  
  echo
  print_success "所有实验已完成!"
  echo "日志保存在: ${log_dir}"
  echo
  
  # 显示分组摘要
  print_section "日志文件分组摘要"
  local -A group_counts
  for config in "${CONFIG_COMBINATIONS[@]}"; do
    IFS='|' read -r group_name _ <<< "${config}"
    group_counts["${group_name}"]=$((${group_counts["${group_name}"]:-0} + 1))
  done
  
  for group in "${!group_counts[@]}"; do
    local count=$((${group_counts[$group]} * ${#SELECTED_TASKS[@]} * ${#SELECTED_CLIP_MODES[@]}))
    echo "  ${CYAN}${group}/${NC}: ${count} 个日志文件"
  done
  echo
  
  press_any_key
}

generate_run_script() {
  local script_file="$1"
  local log_dir="$2"
  
  cat > "${script_file}" << 'SCRIPT_HEADER'
#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_SCRIPT="/cpfs01/qianfy_workspace/openvla_oft_rl/rl/ds_metaworld_ppo_mlp_with_param_more_stats_try_new_v5_dev.py"
CLIP_CONFIG_PATH="/cpfs01/qianfy_workspace/openvla_oft_rl/rl/config/clip.yml"

# 存储所有后台任务的PID
declare -a BACKGROUND_PIDS=()
declare -a MONITOR_PIDS=()

# 清理函数：kill所有后台任务
cleanup_jobs() {
  echo ""
  echo "捕获到退出信号 (Ctrl+C)，正在终止所有后台任务..."
  
  # 终止监控进程
  if [ ${#MONITOR_PIDS[@]} -gt 0 ]; then
    echo "终止监控进程: ${MONITOR_PIDS[@]}"
    for pid in "${MONITOR_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        echo "  - 终止监控进程 PID: $pid"
        kill -TERM "$pid" 2>/dev/null || true
      fi
    done
  fi
  
  # 终止训练进程
  if [ ${#BACKGROUND_PIDS[@]} -gt 0 ]; then
    echo "终止训练进程: ${BACKGROUND_PIDS[@]}"
    for pid in "${BACKGROUND_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        echo "  - 终止训练进程 PID: $pid"
        kill -TERM "$pid" 2>/dev/null || true
      fi
    done
  fi
  
  # 等待进程终止
  sleep 2
  
  # 强制kill仍在运行的进程
  for pid in "${MONITOR_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "  - 强制终止监控进程 PID: $pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  
  for pid in "${BACKGROUND_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "  - 强制终止训练进程 PID: $pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  
  echo "所有后台任务已终止"
  exit 1
}

# 设置信号捕获
trap cleanup_jobs INT TERM

# 计算当前运行的训练任务数（基于 PID 数组）
count_running_jobs() {
  local running_count=0
  for pid in "${BACKGROUND_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      running_count=$((running_count + 1))
    fi
  done
  echo "$running_count"
}

wait_for_slot() {
  local max_jobs=$1
  local running_count
  
  # 统计 BACKGROUND_PIDS 数组中仍在运行的训练进程数
  # 只统计训练进程，不统计监控进程（因为监控进程数量 = 训练进程数量）
  running_count=$(count_running_jobs)
  
  while [[ $running_count -ge ${max_jobs} ]]; do
    sleep 5
    running_count=$(count_running_jobs)
  done
}

# 监控函数：检测任务是否卡在Ray启动阶段
monitor_task() {
  local log_file="$1"
  local task_pid="$2"
  local timeout_seconds="$3"
  local max_retries="$4"
  local job_info="$5"
  local retry_count=0
  
  while true; do
    # 检查进程是否还在运行
    if ! kill -0 "$task_pid" 2>/dev/null; then
      # 进程已结束，监控任务完成
      break
    fi
    
    # 检查日志文件是否存在
    if [[ ! -f "$log_file" ]]; then
      sleep 10
      continue
    fi
    
    # 检查日志中是否包含Ray启动信息
    if grep -q "Started a local Ray instance" "$log_file" 2>/dev/null; then
      # 获取日志文件的最后修改时间
      local last_modified=$(stat -c %Y "$log_file" 2>/dev/null || echo "0")
      local current_time=$(date +%s)
      local time_since_update=$((current_time - last_modified))
      
      # 如果超过超时时间没有更新
      if [[ $time_since_update -gt $timeout_seconds ]]; then
        echo "[MONITOR] $(date '+%Y-%m-%d %H:%M:%S') - 检测到任务卡住: ${job_info}"
        echo "[MONITOR] 日志文件超过 ${timeout_seconds} 秒未更新，最后更新于 $((time_since_update / 60)) 分钟前"
        
        # 检查是否还有重试次数
        if [[ $retry_count -lt $max_retries ]]; then
          retry_count=$((retry_count + 1))
          echo "[MONITOR] 尝试重启任务 (第 ${retry_count}/${max_retries} 次重试)"
          
          # 终止原进程及其子进程
          if kill -0 "$task_pid" 2>/dev/null; then
            echo "[MONITOR] 终止进程 PID: $task_pid"
            kill -TERM "$task_pid" 2>/dev/null || true
            sleep 2
            # 强制kill如果还在运行
            if kill -0 "$task_pid" 2>/dev/null; then
              kill -9 "$task_pid" 2>/dev/null || true
            fi
          fi
          
          # 清理Ray进程（如果存在）
          pkill -f "ray.*${TASK_NAME}" 2>/dev/null || true
          sleep 2
          
          # 备份旧日志
          if [[ -f "$log_file" ]]; then
            mv "$log_file" "${log_file}.failed_retry_${retry_count}_$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
          fi
          
          # 重新启动任务
          echo "[MONITOR] 重新启动任务..."
          (
            export CUDA_VISIBLE_DEVICES
            export ENABLE_SWANLAB
            {
              echo "[PROCESS_INFO] PID=$$ | Task=${TASK_NAME} | ClipMode=${CLIP_MODE} | Seed=${SEED} | AdvStructMode=${ADV_STRUCT_MODE} | EnableSwanLab=${ENABLE_SWANLAB} | Retry=${retry_count} | StartTime=$(date '+%Y-%m-%d %H:%M:%S')"
              ${PYTHON_BIN} "${PYTHON_SCRIPT}" \
                --task_name "${TASK_NAME}" \
                --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
                --num-trainer-gpus "${NUM_TRAINER_GPUS}" \
                --num-rollout-workers "${NUM_ROLLOUT_WORKERS}" \
                --num-eval-workers "${NUM_EVAL_WORKERS}" \
                --train-batch-size "${TRAIN_BATCH_SIZE}" \
                --seed "${SEED}" \
                --clip-mode "${CLIP_MODE}" \
                --clip-config "${CLIP_CONFIG}" \
                --train-iters "${TRAIN_ITERS}" \
                --adv-struct-mode "${ADV_STRUCT_MODE}" \
                --adv-weight-temp "${ADV_WEIGHT_TEMP}" \
                --adv-weight-reg "${ADV_WEIGHT_REG}" \
                --adv-w-min "${ADV_W_MIN}" \
                --adv-w-max "${ADV_W_MAX}" \
                --adv-struct-late-start-frac "${ADV_STRUCT_LATE_START_FRAC}" \
                --adv-struct-alpha "${ADV_STRUCT_ALPHA}" \
                --adv-struct-lr-mult "${ADV_STRUCT_LR_MULT}" \
                --aux-adv-enable "${AUX_ADV_ENABLE}" \
                --aux-adv-mode "${AUX_ADV_MODE}" \
                --aux-adv-target-mode "${AUX_ADV_TARGET_MODE}" \
                --aux-adv-pair-mode "${AUX_ADV_PAIR_MODE}" \
                --aux-adv-coef "${AUX_ADV_COEF}" \
                2>&1
            } > "${LOG_FILE}"
          ) &
          
          task_pid=$!
          echo "[MONITOR] 新进程 PID: $task_pid"
          
          # 等待新日志文件创建并重置时间戳检查
          sleep 15
        else
          echo "[MONITOR] 已达到最大重试次数 (${max_retries})，停止重试"
          echo "[MONITOR] 终止进程 PID: $task_pid"
          kill -TERM "$task_pid" 2>/dev/null || true
          sleep 2
          kill -9 "$task_pid" 2>/dev/null || true
          break
        fi
      fi
    fi
    
    # 每30秒检查一次
    sleep 30
  done
  
  echo "[MONITOR] 监控任务结束: ${job_info}"
}

run_training() {
  export CUDA_VISIBLE_DEVICES
  export ENABLE_SWANLAB
  
  {
    echo "[PROCESS_INFO] PID=$$ | Task=${TASK_NAME} | ClipMode=${CLIP_MODE} | Seed=${SEED} | AdvStructMode=${ADV_STRUCT_MODE} | EnableSwanLab=${ENABLE_SWANLAB} | StartTime=$(date '+%Y-%m-%d %H:%M:%S')"
    ${PYTHON_BIN} "${PYTHON_SCRIPT}" \
      --task_name "${TASK_NAME}" \
      --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
      --num-trainer-gpus "${NUM_TRAINER_GPUS}" \
      --num-rollout-workers "${NUM_ROLLOUT_WORKERS}" \
      --num-eval-workers "${NUM_EVAL_WORKERS}" \
      --train-batch-size "${TRAIN_BATCH_SIZE}" \
      --seed "${SEED}" \
      --clip-mode "${CLIP_MODE}" \
      --clip-config "${CLIP_CONFIG}" \
      --train-iters "${TRAIN_ITERS}" \
      --adv-struct-mode "${ADV_STRUCT_MODE}" \
      --adv-weight-temp "${ADV_WEIGHT_TEMP}" \
      --adv-weight-reg "${ADV_WEIGHT_REG}" \
      --adv-w-min "${ADV_W_MIN}" \
      --adv-w-max "${ADV_W_MAX}" \
      --adv-struct-late-start-frac "${ADV_STRUCT_LATE_START_FRAC}" \
      --adv-struct-alpha "${ADV_STRUCT_ALPHA}" \
      --adv-struct-lr-mult "${ADV_STRUCT_LR_MULT}" \
      --aux-adv-enable "${AUX_ADV_ENABLE}" \
      --aux-adv-mode "${AUX_ADV_MODE}" \
      --aux-adv-target-mode "${AUX_ADV_TARGET_MODE}" \
      --aux-adv-pair-mode "${AUX_ADV_PAIR_MODE}" \
      --aux-adv-coef "${AUX_ADV_COEF}" \
      2>&1
  } > "${LOG_FILE}"
}

SCRIPT_HEADER
  
  # 添加配置变量
  cat >> "${script_file}" << EOF

# Configuration
LOG_DIR="${log_dir}"
MAX_PARALLEL_JOBS=${CURRENT_MAX_PARALLEL_JOBS}
GPU_CONFIGS="${CURRENT_GPU_CONFIGS}"
STARTUP_DELAY=${CURRENT_STARTUP_DELAY}
MONITOR_TIMEOUT=${CURRENT_MONITOR_TIMEOUT}
MAX_RETRIES=${CURRENT_MAX_RETRIES}
ENABLE_SWANLAB=${CURRENT_ENABLE_SWANLAB}

# Training parameters
SEED=${CURRENT_SEED}
TRAIN_ITERS=${CURRENT_TRAIN_ITERS}
NUM_TRAINER_GPUS=${CURRENT_NUM_TRAINER_GPUS}
NUM_ROLLOUT_WORKERS=${CURRENT_NUM_ROLLOUT_WORKERS}
NUM_EVAL_WORKERS=${CURRENT_NUM_EVAL_WORKERS}
TRAIN_BATCH_SIZE=${CURRENT_TRAIN_BATCH_SIZE}
CLIP_CONFIG="${CLIP_CONFIG_PATH}"

# Tasks
declare -a TASKS=(
EOF
  
  for task in "${SELECTED_TASKS[@]}"; do
    echo "  \"${task}\"" >> "${script_file}"
  done
  
  cat >> "${script_file}" << EOF
)

# Clip modes
declare -a CLIP_MODES=(
EOF
  
  for mode in "${SELECTED_CLIP_MODES[@]}"; do
    echo "  \"${mode}\"" >> "${script_file}"
  done
  
  cat >> "${script_file}" << EOF
)

# Configurations
declare -a CONFIGS=(
EOF
  
  for config in "${CONFIG_COMBINATIONS[@]}"; do
    echo "  \"${config}\"" >> "${script_file}"
  done
  
  cat >> "${script_file}" << 'EOF'
)

# Run experiments
job_idx=0
total_jobs=$((${#TASKS[@]} * ${#CLIP_MODES[@]} * ${#CONFIGS[@]}))

echo "===== Starting ${total_jobs} experiments ====="
echo "Max parallel jobs: ${MAX_PARALLEL_JOBS}"
echo "Startup delay: ${STARTUP_DELAY}s"
echo "Monitor timeout: ${MONITOR_TIMEOUT}s"
echo "Max retries: ${MAX_RETRIES}"
if [[ "${ENABLE_SWANLAB}" == "1" ]]; then
  echo "Logging: SwanLab + TensorBoard"
else
  echo "Logging: TensorBoard only (SwanLab disabled)"
fi
echo ""

# 存储监控进程PID
declare -a MONITOR_PIDS=()

for task in "${TASKS[@]}"; do
  for clip_mode in "${CLIP_MODES[@]}"; do
    for config in "${CONFIGS[@]}"; do
      wait_for_slot "${MAX_PARALLEL_JOBS}"
      
      IFS='|' read -r group_name label adv_mode adv_temp late_frac alpha lr_mult w_min w_max aux_enable aux_coef aux_mode aux_target_mode aux_pair_mode <<< "${config}"
      
      # 创建组目录
      mkdir -p "${LOG_DIR}/${group_name}"
      
      log_file="${LOG_DIR}/${group_name}/job_${job_idx}_${task}_${clip_mode}_${label}.log"
      job_info="[${group_name}] ${task} | ${clip_mode} | ${label}"
      
      echo "[Job $((job_idx+1))/${total_jobs}] Starting: ${job_info}"
      
      (
        TASK_NAME="${task}"
        CLIP_MODE="${clip_mode}"
        CUDA_VISIBLE_DEVICES="${GPU_CONFIGS}"
        ENABLE_SWANLAB="${ENABLE_SWANLAB}"
        ADV_STRUCT_MODE="${adv_mode}"
        ADV_WEIGHT_TEMP="${adv_temp}"
        ADV_WEIGHT_REG="0.0"
        ADV_W_MIN="${w_min}"
        ADV_W_MAX="${w_max}"
        ADV_STRUCT_LATE_START_FRAC="${late_frac}"
        ADV_STRUCT_ALPHA="${alpha}"
        ADV_STRUCT_LR_MULT="${lr_mult}"
        AUX_ADV_ENABLE="${aux_enable}"
        AUX_ADV_COEF="${aux_coef}"
        AUX_ADV_MODE="${aux_mode:-residual}"
        AUX_ADV_TARGET_MODE="${aux_target_mode:-advantage}"
        AUX_ADV_PAIR_MODE="${aux_pair_mode:-scalar}"
        LOG_FILE="${log_file}"
        SEED="${SEED}"
        NUM_TRAINER_GPUS="${NUM_TRAINER_GPUS}"
        NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS}"
        NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS}"
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}"
        CLIP_CONFIG="${CLIP_CONFIG}"
        TRAIN_ITERS="${TRAIN_ITERS}"
        PYTHON_BIN="${PYTHON_BIN}"
        PYTHON_SCRIPT="${PYTHON_SCRIPT}"
        
        run_training
      ) &
      
      # 记录训练进程PID
      training_pid=$!
      BACKGROUND_PIDS+=("${training_pid}")
      echo "[Job $((job_idx+1))/${total_jobs}] Training process started with PID: ${training_pid}"
      
      # 启动监控进程
      (
        TASK_NAME="${task}"
        LOG_FILE="${log_file}"
        MONITOR_TIMEOUT="${MONITOR_TIMEOUT}"
        MAX_RETRIES="${MAX_RETRIES}"
        PYTHON_BIN="${PYTHON_BIN}"
        PYTHON_SCRIPT="${PYTHON_SCRIPT}"
        CUDA_VISIBLE_DEVICES="${GPU_CONFIGS}"
        ENABLE_SWANLAB="${ENABLE_SWANLAB}"
        ADV_STRUCT_MODE="${adv_mode}"
        ADV_WEIGHT_TEMP="${adv_temp}"
        ADV_WEIGHT_REG="0.0"
        ADV_W_MIN="${w_min}"
        ADV_W_MAX="${w_max}"
        ADV_STRUCT_LATE_START_FRAC="${late_frac}"
        ADV_STRUCT_ALPHA="${alpha}"
        ADV_STRUCT_LR_MULT="${lr_mult}"
        AUX_ADV_ENABLE="${aux_enable}"
        AUX_ADV_COEF="${aux_coef}"
        AUX_ADV_MODE="${aux_mode:-residual}"
        AUX_ADV_TARGET_MODE="${aux_target_mode:-advantage}"
        AUX_ADV_PAIR_MODE="${aux_pair_mode:-scalar}"
        SEED="${SEED}"
        NUM_TRAINER_GPUS="${NUM_TRAINER_GPUS}"
        NUM_ROLLOUT_WORKERS="${NUM_ROLLOUT_WORKERS}"
        NUM_EVAL_WORKERS="${NUM_EVAL_WORKERS}"
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}"
        CLIP_CONFIG="${CLIP_CONFIG}"
        TRAIN_ITERS="${TRAIN_ITERS}"
        CLIP_MODE="${clip_mode}"
        
        monitor_task "${log_file}" "${training_pid}" "${MONITOR_TIMEOUT}" "${MAX_RETRIES}" "${job_info}"
      ) &
      
      monitor_pid=$!
      MONITOR_PIDS+=("${monitor_pid}")
      echo "[Job $((job_idx+1))/${total_jobs}] Monitor process started with PID: ${monitor_pid}"
      
      job_idx=$((job_idx + 1))
      
      # 启动延迟：避免端口冲突和资源竞争
      if [[ $job_idx -lt ${total_jobs} ]]; then
        echo "Waiting ${STARTUP_DELAY}s before starting next job..."
        echo "Current running jobs: $(count_running_jobs)"
        sleep ${STARTUP_DELAY}
        echo ""
      fi
    done
  done
done

echo ""
echo "All jobs have been launched. Waiting for completion..."
echo "Press Ctrl+C to terminate all jobs."
echo ""

wait
echo "All experiments completed!"
EOF
  
  chmod +x "${script_file}"
}

# =============================================================================
# 主循环
# =============================================================================

main() {
  while true; do
    show_main_menu
    read -r choice
    
    case "$choice" in
      1) select_tasks;;
      2) select_clip_modes;;
      3) configure_experiments;;
      4) use_preset;;
      5) configure_training_params;;
      6) configure_gpu_parallel;;
      7) view_full_config;;
      8) save_config;;
      9) load_config;;
      10) auto_combination_generator;;
      11) load_last_run_config;;
      [Rr]) run_experiments;;
      [Qq]) 
        echo
        print_info "再见!"
        exit 0
        ;;
      *) 
        print_warning "无效选择"
        sleep 1
        ;;
    esac
  done
}

# 启动
main

