from loco_manipulation_gym.envs.go2_arx.go2_arx_config import Go2ArxRoughCfg, Go2ArxRoughCfgPPO


class TennisRoughCfg(Go2ArxRoughCfg):
    class env(Go2ArxRoughCfg.env):
        # actor obs: 75 base + 3 ekf ball pos + 2 ekf ball vel (x,y) + 1 visibility flag
        num_observations = 81
        # critic obs: compact privileged state for asymmetric training
        num_privileged_obs = 88

    class tennis_ball:
        radius = 0.033
        mass = 0.058
        restitution = 0.65
        friction = 0.4
        linear_damping = 0.01
        angular_damping = 0.01

        spawn_x = [2.0, 3.0]
        spawn_y = [-0.35, 0.35]
        spawn_z = [0.9, 1.4]

        vel_x = [-8.0, -5.0]
        vel_y = [-0.6, 0.6]
        vel_z = [0.8, 2.6]

        strike_plane_x = 0.5
        min_intercept_time = 0.05
        max_intercept_time = 2.5

    class perception:
        camera_fov_deg = 60.0
        motion_blur_omega_thresh = 1.5
        base_pos_noise = 0.005
        base_vel_noise = 0.03
        distance_noise_scale = 0.015  # sigma_pos grows with d^2
        motion_blur_gain = 0.3
        out_of_fov_noise_boost = 8.0

        ekf_process_pos = 1e-3
        ekf_process_vel = 5e-3
        ekf_meas_pos = 3e-2
        ekf_meas_vel = 6e-2

    class rewards(Go2ArxRoughCfg.rewards):
        class scales(Go2ArxRoughCfg.rewards.scales):
            # active perception: penalize EKF prediction error to GT ball position
            ekf_error = -1.0


class TennisRoughCfgPPO(Go2ArxRoughCfgPPO):
    class runner(Go2ArxRoughCfgPPO.runner):
        experiment_name = "tennis_go2_arx_phase2"
