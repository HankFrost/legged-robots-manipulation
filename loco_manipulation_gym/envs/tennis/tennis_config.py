from loco_manipulation_gym.envs.go2_arx.go2_arx_config import Go2ArxRoughCfg, Go2ArxRoughCfgPPO


class TennisRoughCfg(Go2ArxRoughCfg):
    class env(Go2ArxRoughCfg.env):
        # 75 (go2_arx base) + 3 (ball pos) + 3 (ball vel)
        num_observations = 81
        num_privileged_obs = num_observations + 187

    class tennis_ball:
        # tennis ball parameters
        radius = 0.033
        mass = 0.058
        restitution = 0.65
        friction = 0.4
        linear_damping = 0.01
        angular_damping = 0.01

        # spawn ranges (in each env local frame before adding env origin)
        spawn_x = [2.0, 3.0]
        spawn_y = [-0.35, 0.35]
        spawn_z = [0.9, 1.4]

        # initial velocity ranges (towards robot, so vx < 0)
        vel_x = [-8.0, -5.0]
        vel_y = [-0.6, 0.6]
        vel_z = [0.8, 2.6]

        # intercept-plane for temporal grounding
        strike_plane_x = 0.5
        min_intercept_time = 0.05
        max_intercept_time = 2.5


class TennisRoughCfgPPO(Go2ArxRoughCfgPPO):
    class runner(Go2ArxRoughCfgPPO.runner):
        experiment_name = "tennis_go2_arx"
