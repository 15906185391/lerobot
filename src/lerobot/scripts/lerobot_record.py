# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Records a dataset via teleoperation.  This is a pure data-collection
tool — no policy inference.  For deploying trained policies, use
``lerobot-rollout`` instead.

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Example:

```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --display_data=true
```

To stream the data to Foxglove instead of Rerun, add ``--display_mode=foxglove`` (then connect the
Foxglove app to ``ws://127.0.0.1:8765``; override the port with ``--display_port=<port>``).

Example recording with bimanual so100:
```shell
lerobot-record \\
  --robot.type=bi_so_follower \\
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \\
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \\
  --robot.id=bimanual_follower \\
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    front: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
  }' \\
  --teleop.type=bi_so_leader \\
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \\
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \\
  --teleop.id=bimanual_leader \\
  --display_data=true \\
  --dataset.repo_id=${HF_USER}/bimanual-so-handover-cube \\
  --dataset.num_episodes=25 \\
  --dataset.single_task="Grab and handover the red cube to the other arm" \\
  --dataset.streaming_encoding=true \\
  --dataset.encoder_threads=2
```

Example recording with custom video encoding parameters:
```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --dataset.rgb_encoder.vcodec=h264 \\
    --dataset.rgb_encoder.preset=fast \\
    --dataset.rgb_encoder.extra_options={"tune": "film", "profile:v": "high", "bf": 2} \\
    --display_data=true
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.lerobot_camera_ros2 import ROS2CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.orbbec import OrbbecCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
    wheeled_arm,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_openarm_mini,
    bi_rebot_102_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
    wheeled_arm_pico,
)
from lerobot.teleoperators.keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.keyboard_input import apply_recording_control, init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.ros2_action_publisher import ROS2ActionPublisher
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Teleoperator to control the robot (required)
    teleop: TeleoperatorConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Visualization backend used when display_data is True: "rerun" or "foxglove".
    display_mode: str = "rerun"
    # For "rerun": IP of a remote server to send to. For "foxglove": interface to bind the WebSocket
    # server to (127.0.0.1 for local only, 0.0.0.0 for all interfaces).
    display_ip: str | None = None
    # For "rerun": port of the remote server. For "foxglove": port to bind the WebSocket server to.
    display_port: int | None = None
    # Whether to display compressed (JPEG) images instead of raw frames
    display_compressed_images: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False
    # If True, run teleop without saving until the operator advances with keyboard/PICO controls.
    # Useful when the operator cannot reach the keyboard while staging the robot.
    wait_for_episode_start: bool = False
    # Publish the final sent action as sensor_msgs/msg/JointState for jitter inspection in ROS 2.
    publish_action_ros2: bool = False
    action_ros2_topic: str = "/lerobot/action"
    action_ros2_node_name: str = "lerobot_action_publisher"

    def __post_init__(self):
        if self.teleop is None:
            raise ValueError(
                "A teleoperator is required for recording. "
                "Use --teleop.type=... to specify one. "
                "For policy-based deployment, use lerobot-rollout instead."
            )


def _next_dataset_frame_index(dataset: LeRobotDataset | None) -> int | None:
    if dataset is None:
        return None

    frame_index = dataset.num_frames
    writer = getattr(dataset, "writer", None)
    episode_buffer = getattr(writer, "episode_buffer", None)
    if episode_buffer is not None:
        frame_index += episode_buffer["size"]
    return frame_index


def _poll_teleop_recording_control(teleop: Teleoperator | list[Teleoperator] | None, events: dict) -> None:
    if isinstance(teleop, list):
        teleops = teleop
    elif teleop is None:
        teleops = []
    else:
        teleops = [teleop]

    for teleop_item in teleops:
        get_recording_control = getattr(teleop_item, "get_recording_control", None)
        if not callable(get_recording_control):
            continue
        control = get_recording_control()
        if control is None:
            continue
        apply_recording_control(control, events)
        return


def _teleop_emergency_stop_requested(teleop: Teleoperator | list[Teleoperator] | None) -> bool:
    if isinstance(teleop, list):
        teleops = teleop
    elif teleop is None:
        teleops = []
    else:
        teleops = [teleop]

    for teleop_item in teleops:
        emergency_stop_requested = getattr(teleop_item, "emergency_stop_requested", None)
        if callable(emergency_stop_requested) and emergency_stop_requested():
            return True
    return False


def _refresh_teleop_visualization_from_feedback(
    teleop: Teleoperator | list[Teleoperator] | None,
    feedback: dict,
    *,
    display_data: bool = False,
    display_mode: str = "rerun",
    frame_index: int | None = None,
    timestamp: float | None = None,
    collision_status: str = "sync",
    force: bool = False,
) -> None:
    if not isinstance(teleop, Teleoperator):
        return

    refresh_visualization = getattr(teleop, "refresh_visualization_from_feedback", None)
    if callable(refresh_visualization):
        refresh_visualization(feedback, collision_status=collision_status)
    else:
        send_feedback = getattr(teleop, "send_feedback", None)
        if callable(send_feedback):
            send_feedback(feedback)

    if display_data and display_mode == "rerun":
        log_robot_visualization = getattr(teleop, "log_rerun_robot_visualization", None)
        if callable(log_robot_visualization):
            log_robot_visualization(frame_index=frame_index, timestamp=timestamp, force=force)


def _sync_teleop_feedback_from_robot(
    robot: Robot,
    teleop: Teleoperator | list[Teleoperator] | None,
    *,
    display_data: bool = False,
    display_mode: str = "rerun",
    frame_index: int | None = None,
    timestamp: float | None = None,
    collision_status: str = "sync",
) -> None:
    if not isinstance(teleop, Teleoperator):
        return
    if robot.name != "wheeled_arm":
        return
    require_feedback = bool(getattr(getattr(robot, "config", None), "require_fresh_feedback", False))
    if require_feedback:
        require_valid_feedback = getattr(robot, "require_valid_feedback", None)
        if callable(require_valid_feedback):
            require_valid_feedback("initial teleop feedback sync")
    elif not getattr(robot, "has_valid_feedback", True):
        return

    feedback = robot.get_observation()
    _refresh_teleop_visualization_from_feedback(
        teleop,
        feedback,
        display_data=display_data,
        display_mode=display_mode,
        frame_index=frame_index,
        timestamp=timestamp,
        collision_status=collision_status,
        force=True,
    )
    reset_baseline = getattr(teleop, "reset_baseline", None)
    if callable(reset_baseline):
        reset_baseline()


def _make_reset_visualization_callback(
    teleop: Teleoperator | list[Teleoperator] | None,
    *,
    display_data: bool,
    display_mode: str,
    dataset: LeRobotDataset,
    fps: int,
):
    if not isinstance(teleop, Teleoperator):
        return None
    if not (display_data or getattr(teleop, "_visualizer", None) is not None):
        return None

    last_update_t = 0.0
    update_period_s = 0.1

    def callback(feedback: dict) -> None:
        nonlocal last_update_t
        now = time.perf_counter()
        if now - last_update_t < update_period_s:
            return
        last_update_t = now
        frame_index = _next_dataset_frame_index(dataset)
        _refresh_teleop_visualization_from_feedback(
            teleop,
            feedback,
            display_data=display_data,
            display_mode=display_mode,
            frame_index=frame_index,
            timestamp=frame_index / fps if frame_index is not None else None,
            collision_status="reset",
            force=True,
        )

    return callback


def _action_values_for_dataset(
    robot: Robot, requested_action: RobotAction, sent_action: RobotAction
) -> RobotAction:
    if robot.name != "wheeled_arm":
        return requested_action

    action_features = getattr(robot, "action_features", {})
    recorded_action = {
        key: value for key, value in requested_action.items() if key in action_features
    }
    recorded_action.update({key: value for key, value in sent_action.items() if key in action_features})
    return recorded_action


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     [ Teleoperator ]
     |
     |  [teleop.get_action] -> raw_action
     |          |
     |          V
     | [teleop_action_processor]
     |          |
     '---> processed_teleop_action
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    control_time_s: int | float | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_mode: str = "rerun",
    display_compressed_images: bool = False,
    display_episode_index: int | None = None,
    action_publisher: ROS2ActionPublisher | None = None,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    control_interval = 1 / fps

    no_action_count = 0
    timestamp = 0
    local_frame_index = 0
    start_episode_t = time.perf_counter()
    while control_time_s is None or timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from teleop
        if isinstance(teleop, Teleoperator):
            if (
                robot.name == "wheeled_arm"
                and getattr(robot, "has_valid_feedback", True)
                and getattr(getattr(teleop, "config", None), "use_continuous_robot_feedback", True)
            ):
                teleop.send_feedback(obs)
            act = teleop.get_action()
            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)
            _poll_teleop_recording_control(teleop, events)

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        elif isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            _poll_teleop_recording_control(teleop, events)
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning(
                    "No teleoperator provided, skipping action generation. "
                    "This is likely to happen when resetting the environment without a teleop device. "
                    "The robot won't be at its rest position at the start of the next episode."
                )
            continue

        if events["exit_early"]:
            break

        display_frame_index = _next_dataset_frame_index(dataset)

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        _sent_action = robot.send_action(robot_action_to_send)
        if action_publisher is not None:
            action_publisher.publish(_sent_action, robot.action_features)
        dataset_action_values = _action_values_for_dataset(robot, action_values, _sent_action)

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, dataset_action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            elapsed_s = time.perf_counter() - start_episode_t
            metadata = {
                "elapsed_s": elapsed_s,
                "is_recording": dataset is not None,
                "loop_frame_index": local_frame_index,
            }
            if display_episode_index is not None:
                metadata["episode_index"] = display_episode_index
            elif dataset is not None:
                metadata["episode_index"] = dataset.num_episodes

            log_visualization_data(
                display_mode,
                observation=obs_processed,
                action=action_values,
                compress_images=display_compressed_images,
                metadata=metadata,
                frame_index=display_frame_index,
                timestamp=display_frame_index / fps if display_frame_index is not None else None,
            )
            if display_mode == "rerun":
                log_robot_visualization = getattr(teleop, "log_rerun_robot_visualization", None)
                if callable(log_robot_visualization):
                    log_robot_visualization(
                        frame_index=display_frame_index,
                        timestamp=display_frame_index / fps if display_frame_index is not None else None,
                    )

        dt_s = time.perf_counter() - start_loop_t

        sleep_time_s: float = control_interval - dt_s
        if sleep_time_s < 0:
            logging.warning(
                f"Record loop is running slower ({1 / dt_s:.1f} Hz) than the target FPS ({fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
            )

        precise_sleep(max(sleep_time_s, 0.0))

        timestamp = time.perf_counter() - start_episode_t
        local_frame_index += 1


@parser.wrap()
def record(
    cfg: RecordConfig,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_visualization(
            cfg.display_mode, session_name="recording", ip=cfg.display_ip, port=cfg.display_port
        )
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Fall back to identity pipelines when the caller doesn't supply processors.
    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None
    action_publisher = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            # Reject eval_ prefix — for policy evaluation use lerobot-rollout
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for policy evaluation. "
                    "lerobot-record is for data collection only. Use lerobot-rollout for policy deployment."
                )
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                rgb_encoder=cfg.dataset.rgb_encoder,
                depth_encoder=cfg.dataset.depth_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        # Connect the teleoperator before the robot so the robot isn't left idle (and possibly
        # tripping a firmware watchdog) during teleop init. Matches lerobot_teleoperate.py.
        if teleop is not None:
            teleop.connect()
        robot.connect()
        frame_index = _next_dataset_frame_index(dataset)
        _sync_teleop_feedback_from_robot(
            robot,
            teleop,
            display_data=cfg.display_data,
            display_mode=cfg.display_mode,
            frame_index=frame_index,
            timestamp=frame_index / cfg.dataset.fps if frame_index is not None else None,
            collision_status="sync",
        )

        listener, events = init_keyboard_listener()

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 # --dataset.rgb_encoder.vcodec=auto. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

        if cfg.publish_action_ros2:
            action_publisher = ROS2ActionPublisher(
                topic=cfg.action_ros2_topic,
                node_name=cfg.action_ros2_node_name,
            )
            logging.info("Publishing sent actions to ROS 2 topic '%s'.", cfg.action_ros2_topic)

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                if cfg.wait_for_episode_start:
                    log_say(
                        f"Waiting for episode {dataset.num_episodes} start command",
                        cfg.play_sounds,
                    )
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=None,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_mode=cfg.display_mode,
                        display_compressed_images=display_compressed_images,
                        display_episode_index=dataset.num_episodes,
                        action_publisher=action_publisher,
                    )
                    if events["stop_recording"]:
                        break
                    events["exit_early"] = False
                    events["rerecord_episode"] = False

                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_mode=cfg.display_mode,
                    display_compressed_images=display_compressed_images,
                    display_episode_index=dataset.num_episodes,
                    action_publisher=action_publisher,
                )

                # Execute a few seconds without recording to give time to manually reset the environment
                # Skip reset for the last episode to be recorded
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    reset_to_rest_pose = getattr(robot, "reset_to_rest_pose", None)
                    if callable(reset_to_rest_pose):
                        reset_completed = reset_to_rest_pose(
                            stop_requested=lambda: _teleop_emergency_stop_requested(teleop),
                            progress_callback=_make_reset_visualization_callback(
                                teleop,
                                display_data=cfg.display_data,
                                display_mode=cfg.display_mode,
                                dataset=dataset,
                                fps=cfg.dataset.fps,
                            ),
                        )
                        if not reset_completed:
                            events["stop_recording"] = True
                            events["exit_early"] = True
                            log_say("Emergency stop requested during robot reset", cfg.play_sounds)
                            break
                        frame_index = _next_dataset_frame_index(dataset)
                        _sync_teleop_feedback_from_robot(
                            robot,
                            teleop,
                            display_data=cfg.display_data,
                            display_mode=cfg.display_mode,
                            frame_index=frame_index,
                            timestamp=frame_index / cfg.dataset.fps
                            if frame_index is not None
                            else None,
                            collision_status="reset",
                        )

                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_mode=cfg.display_mode,
                        display_compressed_images=display_compressed_images,
                        display_episode_index=dataset.num_episodes,
                        action_publisher=action_publisher,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if action_publisher is not None:
            action_publisher.close()

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if listener is not None:
            listener.stop()

        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
