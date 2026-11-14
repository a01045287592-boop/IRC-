from roboid import Raccoon

PORT = "/dev/ttyACM1"

robot = Raccoon(port_name=PORT)

try:
    # Joint encoder
    q = robot.encoder(-1)

    # ★ 보정 전 Wrist XYZ
    xyz = robot.xyz()

    print()
    print("Encoder [deg]")
    print(
        f"J1={q[0]:.4f}, "
        f"J2={q[1]:.4f}, "
        f"J3={q[2]:.4f}, "
        f"J4={q[3]:.4f}"
    )

    print()
    print("★ 보정 전 Wrist XYZ [cm]")
    print(
        f"[{xyz[0]:.4f}, "
        f"{xyz[1]:.4f}, "
        f"{xyz[2]:.4f}]"
    )

finally:
    robot.dispose()