import math
import numpy as np
from roboid import Raccoon


# ============================================================
# 설정
# ============================================================

PORT = "/dev/ttyACM1"

# Raccoon 링크 길이 [cm]
L1 = 8.25
L2 = 10.0
L3 = 10.0


# ============================================================
# Forward Kinematics
# ============================================================

def forward_kinematics(j1, j2, j3, j4, RX, RY, RZ):

    th1 = math.radians(j1)
    th2 = math.radians(j2)
    th3 = math.radians(j3)
    th4 = math.radians(j4)

    c1 = math.cos(th1)
    s1 = math.sin(th1)

    c4 = math.cos(th4)
    s4 = math.sin(th4)

    c23 = math.cos(th2 + th3)
    s23 = math.sin(th2 + th3)

    M1 = c4 * RX - s4 * RY + L3
    M2 = s4 * RX + c4 * RY

    M = (
        s23 * M1
        + c23 * M2
        + math.sin(th2) * L2
    )

    x = -c1 * M + s1 * RZ
    y = -s1 * M - c1 * RZ

    z = (
        c23 * M1
        - s23 * M2
        + math.cos(th2) * L2
        + L1
    )

    return np.array([
        -y,
        x,
        z
    ])


# ============================================================
# Raccoon 연결
# ============================================================

robot = Raccoon(port_name=PORT)


try:

    # ========================================================
    # Encoder 읽기
    # ========================================================

    q = robot.encoder(-1)

    j1 = float(q[0])
    j2 = float(q[1])
    j3 = float(q[2])
    j4 = float(q[3])


    # ========================================================
    # 보정 전 좌표
    # Raccoon 기본 TCP
    # ========================================================

    xyz_original = forward_kinematics(
        j1,
        j2,
        j3,
        j4,
        RX=8.0,
        RY=-0.6,
        RZ=0.0
    )


    # ========================================================
    # ★ 10cm TCP 보정 좌표
    # Calibration에 사용할 좌표
    # ========================================================

    xyz_tcp10 = forward_kinematics(
        j1,
        j2,
        j3,
        j4,
        RX=10.0,
        RY=-0.6,
        RZ=0.0
    )


    # ========================================================
    # 출력
    # ========================================================

    print()
    print("=" * 60)

    print("Encoder [deg]")
    print(
        f"J1={j1:.4f}, "
        f"J2={j2:.4f}, "
        f"J3={j3:.4f}, "
        f"J4={j4:.4f}"
    )


    print()
    print("보정 전 XYZ [cm]")
    print(
        f"[{xyz_original[0]:.4f}, "
        f"{xyz_original[1]:.4f}, "
        f"{xyz_original[2]:.4f}]"
    )


    print()
    print("★ 10cm TCP 보정 XYZ [cm]")
    print(
        f"[{xyz_tcp10[0]:.4f}, "
        f"{xyz_tcp10[1]:.4f}, "
        f"{xyz_tcp10[2]:.4f}]"
    )


    print()
    print("★ Calibration용 Robot XYZ [cm]")
    print(
        f"[{xyz_tcp10[0]:.4f}, "
        f"{xyz_tcp10[1]:.4f}, "
        f"{xyz_tcp10[2]:.4f}]"
    )

    print("=" * 60)


finally:

    robot.dispose()