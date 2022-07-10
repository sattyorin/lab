import argparse

"""
TODO(sara): write a naming rules
"""


def get_xml() -> str:
    return '<?xml version="1.0"?>\n'


def get_mujoco(model_name: str, worldbody: str, actuator: str) -> str:
    mujoco: str = f'<mujoco model="{model_name}">\n'
    end_mujoco: str = "</mujoco>\n"
    return mujoco + worldbody + actuator + end_mujoco


def get_worldbody(light: str, objects: str, floor: str, body: str) -> str:
    worldbody: str = "<worldbody>\n"
    end_worldbody: str = "</worldbody>\n"
    return worldbody + light + objects + floor + body + end_worldbody


def get_light() -> str:
    light1: str = '<light name="light0" directional="true" \
        diffuse=".3 .3 .3" pos="-1 -1 1" dir="1 1 -1" />\n'
    light2: str = '<light name="light1" directional="true" \
        diffuse=".3 .3 .3" pos="1 -1 1" dir="-1 1 -1" />\n'
    light3: str = '<light name="light2" directional="true" \
        diffuse=".3 .3 .3" pos="0 1 1" dir="0 -1 -1" />\n'
    return light1 + light2 + light3


def get_object(
    object_id: int,
    position_x: float,
    position_y: float,
    position_z: float,
    size: float,
) -> str:
    body = f'<body name="object{object_id}" \
        pos="{position_x} {position_y} {position_z}">\n'
    freejoint = "<freejoint />\n"
    sphere = f'<geom name="sphere{object_id}" type="sphere" size="{size}" />\n'
    end_body = "</body>\n"
    return body + freejoint + sphere + end_body


def get_floor(x: float = 5.0, y: float = 5.0) -> str:
    return f'<geom name="floor" pos="0 0 0" \
        size="{x} {y} .1" type="plane" condim="3" />\n'


def get_palm(modules: str, position_z: float) -> str:
    body: str = f'<body name="palm" pos="0 0 {position_z}">\n'
    end_body: str = "</body>\n"
    return body + modules + end_body


def get_module(
    module_id: int,
    position_x: float,
    position_y: float,
    size_x: float = 0.05,
    size_y: float = 0.05,
    size_z: float = 0.4,
) -> str:
    body: str = (
        f'<body name="module{module_id}" pos="{position_x} {position_y} 0">\n'
    )
    geom: str = f'<geom name="box{module_id}" type="box" \
        size="{size_x} {size_y} {size_z}" />\n'
    joint: str = f'<joint name="joint{module_id}" \
        type="slide" pos="0 0 0" axis="0 0 1" range="-0.01 0" damping="10000" />\n'
    end_body: str = "</body>\n"
    return body + geom + joint + end_body


def get_actuator(motors: str) -> str:
    actuator: str = "<actuator>\n"
    end_actuator: str = "</actuator>\n"
    return actuator + motors + end_actuator


def get_motor(motor_id: int) -> str:
    return f'<motor name="motor{motor_id}" gear="20000" joint="joint{motor_id}" />\n'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        default="envs/linear_actuator_array/xmls/linear_actuator_array-v0.xml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module: str = get_module(0, 0.0, 0.0)
    body: str = get_palm(module, 4.0)
    objects: str = get_object(0, 0.0, 0.0, 6.5, 0.1)
    worldbody: str = get_worldbody(get_light(), objects, get_floor(), body)
    actuator: str = get_actuator(get_motor(0))
    mujoco: str = get_mujoco(args.path, worldbody, actuator)
    with open(args.path, mode="w") as f:
        f.write(get_xml() + mujoco)


if __name__ == "__main__":
    main()
