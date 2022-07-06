import argparse


def get_xml() -> str:
    return '<?xml version="1.0"?>\n'


def get_mujoco(model_name: str, worldbody: str, actuator: str) -> str:
    return (
        f'<mujoco model="{model_name}">\n'
        + worldbody
        + actuator
        + "</mujoco>\n"
    )


def get_worldbody(light: str, floor: str, body: str) -> str:
    return "<worldbody>\n" + light + floor + body + "</worldbody>\n"


def get_light() -> str:
    return (
        '<light directional="true" diffuse=".3 .3 .3" pos="-1 -1 1" dir="1 1 -1" />\n'
        + f'<light directional="true" diffuse=".3 .3 .3" pos="1 -1 1" dir="-1 1 -1" />\n'
        + f'<light directional="true" diffuse=".3 .3 .3" pos="0 1 1" dir="0 -1 -1" />\n'
    )


def get_floor(x: float = 5.0, y: float = 5.0) -> str:
    return f'<geom name="floor" pos="0 0 0" size="{x} {y} .1" type="plane" condim="3" />\n'


def get_body(name: str, modules: str, position_z: float) -> str:
    return (
        f'<body name="{name}" pos="0 0 {position_z}">\n' + modules + "</body>\n"
    )


def get_module(
    module_id: int,
    position_x: float,
    position_y: float,
    size_x: float = 0.05,
    size_y: float = 0.05,
    size_z: float = 0.4,
) -> str:
    return (
        f'<body name="module{module_id}" pos="{position_x} {position_y} 0">\n'
        + f'<geom name="box{module_id}" type="box" size="{size_x} {size_y} {size_z}" />\n'
        + f'<joint name="joint{module_id}" type="slide" pos="0 0 0" axis="0 0 1" range="-0.01 0" damping="10000" />\n'
        + "</body>\n"
    )


def get_actuator(motors: str) -> str:
    return "<actuator>\n" + motors + "</actuator>\n"


def get_motor(motor_id: int) -> str:
    return (
        f'<motor name="a{motor_id}" gear="20000" joint="joint{motor_id}" />\n'
    )


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
    body: str = get_body("palm", module, 4.0)
    worldbody: str = get_worldbody(get_light(), get_floor(), body)
    actuator: str = get_actuator(get_motor(0))
    with open(args.path, mode="w") as f:
        f.write(get_xml() + get_mujoco(args.path, worldbody, actuator))


if __name__ == "__main__":
    main()
