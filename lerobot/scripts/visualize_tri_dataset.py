#!/usr/bin/env python

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
""" Visualize data of **all** frames of any episode of a TRI LBM dataset.

Note: The last frame of the episode doesnt always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossly compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Examples:

- Visualize data stored on a local machine:
```
local$ python lerobot/scripts/visualize_tri_dataset.py \
    --path ~/efs/data/tasks/PutSpatulaInUtensilCrock/riverway/sim/bc/rollout/2024-06-21T17-23-09+00-00/example/
    --episode-index 0
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ python lerobot/scripts/visualize_tri_dataset.py \
    --path ~/efs/data/tasks/PutSpatulaInUtensilCrock/riverway/sim/bc/rollout/2024-06-21T17-23-09+00-00/example/
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episode_0.rrd .
local$ rerun lerobot_pusht_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
(You need to forward the websocket port to the distant machine, with
`ssh -L 9087:localhost:9087 username@remote-host`)
```
distant$ python lerobot/scripts/visualize_tri_dataset.py \
    --path ~/efs/data/tasks/PutSpatulaInUtensilCrock/riverway/sim/bc/rollout/2024-06-21T17-23-09+00-00/example/
    --episode-index 0 \
    --mode distant \
    --ws-port 9087

local$ rerun ws://localhost:9087
```

The lbm_eval task status tracker, with paths to the demonstrations on efs, is
here (requires TRI bits):
https://docs.google.com/spreadsheets/d/1c9k5CewVeXoKvM3ikbEZiTNPmOA_jKTVFshyW95jdkk/edit

"""

import argparse
import gc
import logging
import time
from tqdm import tqdm
from pathlib import Path

import numpy as np
import rerun as rr

def visualize_dataset(
    path: Path,
    episode_index: int,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    if save:
        assert (
            output_dir is not None
        ), "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"episode_{episode_index}", spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    gc.collect()

    if mode == "distant":
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

    logging.info("Logging to Rerun")

    logging.info("Loading actions.")
    # summary = np.load(path / "diffusion_spartan" / f"episode_{episode_index}" / "processed" / "summary.npz")
    actions = np.load(path / "diffusion_spartan" / f"episode_{episode_index}" / "processed" / "actions.npz")

    dt = 0.1
    t = 0
    for [i, action] in enumerate(actions['actions']):
        rr.set_time_sequence("frame_index", i)
        rr.set_time_seconds("timestamp", t)
        t += dt
        for dim_idx, val in enumerate(action):
            rr.log(f"action/{dim_idx}", rr.Scalar(val))
                
    observations = np.load(path / "diffusion_spartan" / f"episode_{episode_index}" / "processed" / "observations.npz")

    for key, value in observations.items():
        if 'robot_' in key:
            for [i, robot_value] in enumerate(value):
                rr.set_time_sequence("frame_index", i)
                for dim_idx, val in enumerate(robot_value):
                    rr.log(f"{key}/{dim_idx}", rr.Scalar(val))
        elif '_depth' in key:
            pass  # skip depth images
        else: # handle cameras
            for [i, image] in enumerate(tqdm(value, desc=key)):
                rr.set_time_sequence("frame_index", i)
                # Note: adding .compress() did not seem to speed things up, so i've simply downsampled here.
                # TODO(russt): make this a command-line argument
                rr.log(key, rr.Image(image[::2, ::2]))
    
    if mode == "local" and save:
        # save .rrd locally
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        print("Data sent; press Ctrl-C to terminate the websocket connection.")
        # stop the process from exiting since it is serving the websocket connection
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to a TRI log pkl file named e.g. episode_0.pkl.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode to visualize.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun ws://localhost:PORT` on the local machine."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9087,
        help="Web socket port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )

    args = parser.parse_args()
    visualize_dataset(**vars(args))


if __name__ == "__main__":
    main()
