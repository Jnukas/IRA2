import swift


if __name__ == "__main__":
    r = CR16_cleanup()
    env = swift.Swift(); env.launch(realtime=True)
    r.add_to_env(env)

    # Check meshes line up at home
    r.verify_mesh_alignment(env)

    # Bring up sliders
    r.teach(env, block=True)