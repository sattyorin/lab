import mujoco
import numpy as np
import glfw

# https://github.com/rohanpsingh/mujoco-python-viewer

if __name__ == "__main__":
    XML_PATH = "xmls/stir-v0.xml"

    glfw.init()
    width, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
    # glfw.window_hint(glfw.VISIBLE, 0)
    window = glfw.create_window(width=width, height=height,
                                title='Invisible window', monitor=None,
                                share=None)
    framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(
        window)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    assets = dict()
    with open('mesh/bowl4.stl', 'rb') as f:
        assets['bowl.stl'] = f.read()

    model = mujoco.MjModel.from_xml_path(XML_PATH, assets)
    data = mujoco.MjData(model)
    context = mujoco.MjrContext(
        model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    option = mujoco.MjvOption()
    camera = mujoco.MjvCamera()
    camera.lookat = np.array([0.0, 0.0, 0.2])
    camera.distance = 1.0
    camera.azimuth = -10.0
    camera.elevation = -40.0
    perturb = mujoco.MjvPerturb()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    viewport = mujoco.MjrRect(
        0, 0, framebuffer_width, framebuffer_height)

    mujoco.mjv_updateScene(model, data, option, perturb,
                           camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    # mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)

    for i in range(500):
        if glfw.window_should_close(window):
            break

        mujoco.mj_step(model, data)
        mujoco.mjv_updateScene(model, data, option, perturb,
                               camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)

        camera.lookat[2] += 0.0005
        camera.azimuth += 0.5

        glfw.poll_events()

        # data.xfrc_applied = np.zeros_like(data.xfrc_applied)
        # mujoco.mjv_applyPerturbPose(model, data, perturb, 0)
        # mujoco.mjv_applyPerturbForce(model, data, perturb)

    context.free()

    if window:
        if glfw.get_current_context() == window:
            glfw.make_context_current(None)
        glfw.destroy_window(window)
        glfw.terminate()

    print("done")
