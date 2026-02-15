import dataclasses
import datetime as dt
import json
import logging
import math
import os
import pathlib
from pathlib import Path
import requests
import time

import imageio
import numpy as np
import tqdm
import tyro
try:
    from libero.libero import benchmark, get_libero_path
except ImportError:
    from libero.libero import benchmark
    get_libero_path = None

try:
    from libero.libero.envs import OffScreenRenderEnv
except ImportError:
    from libero.libero.envs.offscreen_env import OffScreenRenderEnv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224,224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    post_process_action: bool = True

    job_name: str = "test"

    # Debug / input validation
    use_state: bool = True
    expected_state_dim: int = 8
    auto_pad_state_to_expected_dim: bool = False
    log_payload_every_n_steps: int = 20
    repeat_infer_debug_times: int = 1


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"
    
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        policy_ckpt_path=args.pretrained_path, # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )


    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(
                f"\nTask id: {task_id}, episode: {episode_idx}, description: {task_description}"
            )

            # Reset environment
            client_model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            full_actions = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0
            
            # full_actions = np.load("./debug/action.npy")
            
            while t < max_steps + args.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )

                # Save preprocessed image for replay video
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )
                state = np.asarray(state, dtype=np.float32).reshape(-1)
                if args.use_state and args.expected_state_dim > 0 and state.shape[0] != args.expected_state_dim:
                    if args.auto_pad_state_to_expected_dim and state.shape[0] < args.expected_state_dim:
                        pad_n = args.expected_state_dim - state.shape[0]
                        state = np.concatenate([state, np.zeros((pad_n,), dtype=np.float32)], axis=0)
                        logging.warning(
                            f"[state] auto pad state dim from {state.shape[0] - pad_n} to {state.shape[0]}"
                        )
                    else:
                        raise ValueError(
                            f"State dim mismatch at task_id={task_id}, episode={episode_idx}, step={step}: "
                            f"got {state.shape[0]}, expected {args.expected_state_dim}"
                        )

                observation = { # 
                    "observation.primary": np.expand_dims(
                        img, axis=0
                    ),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(
                        wrist_img, axis=0
                    ),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # align key with model API --> 这里给了两个图像 --> check training
                example_dict = {
                    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "lang": observation["instruction"][0],
                }
                if args.use_state:
                    example_dict["state"] = observation["observation.state"].astype(np.float32)

                log_every = max(1, int(args.log_payload_every_n_steps))
                if step % log_every == 0:
                    logging.info(
                        "[payload] task_id=%s episode=%s step=%s lang='%s' img0=%s img1=%s state_shape=%s",
                        task_id,
                        episode_idx,
                        step,
                        example_dict["lang"],
                        tuple(example_dict["image"][0].shape),
                        tuple(example_dict["image"][1].shape),
                        None if "state" not in example_dict else tuple(example_dict["state"].shape),
                    )

                logging.info(
                    f"Sending to policy server | task_id={task_id}, episode={episode_idx}, lang='{example_dict['lang']}'"
                )

                
                start_time = time.time()
                
                repeat_times = max(1, int(args.repeat_infer_debug_times))
                response = client_model.step(example=example_dict, step=step)
                if repeat_times > 1:
                    cached_raw_actions = getattr(client_model, "raw_actions", None)
                    response_list = [response]
                    for _ in range(repeat_times - 1):
                        response_list.append(client_model.step(example=example_dict, step=step))
                    repeated_actions = []
                    for rep in response_list:
                        rep_raw = rep["raw_action"]
                        rep_action = np.concatenate(
                            [
                                np.asarray(rep_raw.get("world_vector"), dtype=np.float32).reshape(-1),
                                np.asarray(rep_raw.get("rotation_delta"), dtype=np.float32).reshape(-1),
                                np.asarray(rep_raw.get("open_gripper"), dtype=np.float32).reshape(-1),
                            ],
                            axis=0,
                        )
                        repeated_actions.append(rep_action)
                    repeated_actions = np.stack(repeated_actions, axis=0)
                    logging.info(
                        "[repeat_debug] task_id=%s episode=%s step=%s repeats=%s action_std=%s",
                        task_id,
                        episode_idx,
                        step,
                        repeat_times,
                        np.array2string(repeated_actions.std(axis=0), precision=6),
                    )
                    if cached_raw_actions is not None:
                        client_model.raw_actions = cached_raw_actions
                
                end_time = time.time()
                # print(f"time: {end_time - start_time}")
                
                # # 
                raw_action = response["raw_action"]
                
                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(f"Unexpected action sizes: "
                                    f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                                    f"Falling back to LIBERO_DUMMY_ACTION.")
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)
                
                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            
            full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)
            
            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

        # Log final results
        logging.info(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logging.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )

    logging.info(
        f"Total success rate: {float(total_successes) / float(total_episodes)}"
    )
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    if get_libero_path is not None:
        base_bddl_path = pathlib.Path(get_libero_path("bddl_files"))
    else:
        base = os.environ.get("LIBERO_CONFIG_PATH") or os.environ.get("LIBERO_HOME")
        if base is None:
            raise RuntimeError("LIBERO_CONFIG_PATH or LIBERO_HOME must be set when get_libero_path is unavailable")
        base_bddl_path = pathlib.Path(base) / "libero" / "bddl_files"
    task_bddl_file = base_bddl_path / task.problem_folder / task.bddl_file
    debug_lines = [
        f"[DEBUG] task_description = {task_description}",
        f"[DEBUG] task.problem_folder = {task.problem_folder}",
        f"[DEBUG] task.bddl_file = {task.bddl_file}",
        f"[DEBUG] task_bddl_file = {task_bddl_file}",
    ]
    for line in debug_lines:
        logging.info(line)
        print(line, flush=True)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True

if __name__ == "__main__":
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
