import numpy as np
import mujoco
import mujoco_viewer
import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--envdir",
    type=str,
    default="envs/stir",
)
parser.add_argument(
    "--env",
    type=str,
    default="stir-v2",
)
args = parser.parse_args()


mesh_path = os.path.join(
    args.envdir,
    "mesh",
)

if os.path.exists(mesh_path):
    mesh_files = os.listdir(mesh_path)
    assets = dict()
    for file in mesh_files:
        with open(os.path.join(mesh_path, file), "rb") as f:
            assets[file] = f.read()

    model = mujoco.MjModel.from_xml_path(
        os.path.join(args.envdir, "xmls", f"{args.env}.xml"), assets
    )

else:
    model = mujoco.MjModel.from_xml_path(
        os.path.join(args.envdir, "xmls", f"{args.env}.xml")
    )


data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
ctrl_t = time.time()
t = time.time()
for i in range(10000000000):
    if time.time() - ctrl_t > 0.05:
        # data.ctrl[0] = 0.04
        # data.ctrl[1] = 0.04
        # data.ctrl[2] = 0.054 - 0.008
        data.ctrl[0] = 0.01
        data.ctrl[2] = 0.02
        print(data.qpos[0] * 1000, end=", ")
        # print(data.qpos[1], end=", ")
        print((data.qpos[2]) * 1000)
        # data.ctrl[0] = 0.7 * np.sin(i * 0.05)
        # data.ctrl[1] = 0.7 * np.cos(i * 0.05)
        # data.ctrl[4] = 20.7 * np.cos(i * 0.05)
        # data.qpos[:] = np.zeros(data.qpos.size)
        viewer.render()
        ctrl_t = time.time()
    if viewer.is_alive:
        if time.time() - t > 0.0025:
            mujoco.mj_step(model, data)
            # viewer.render()
            t = time.time()
    else:
        print("break")
        break

# close
viewer.close()
