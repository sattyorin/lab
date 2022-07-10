import mujoco_py

if __name__ == "__main__":
    xml_path = "envs/linear_actuator_array/xmls/linear_actuator_array-v0.xml"

    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)

    for _ in range(1000):
        sim.step()
        viewer.render()
