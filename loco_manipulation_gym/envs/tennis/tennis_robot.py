import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import (
    get_axis_params,
    quat_rotate_inverse,
    to_torch,
    torch_rand_float,
)

from loco_manipulation_gym.envs.go2_arx.go2_arx_robot import Go2ArxRobot


class BallEKF:
    """Batch EKF for 3D constant-velocity ball tracking in robot body frame."""

    def __init__(self, num_envs: int, dt: float, device: str, cfg):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device

        self.state = torch.zeros(num_envs, 6, device=device)
        self.P = torch.eye(6, device=device).unsqueeze(0).repeat(num_envs, 1, 1) * 0.5

        self.F = torch.eye(6, device=device)
        self.F[0:3, 3:6] = torch.eye(3, device=device) * dt
        self.H = torch.eye(6, device=device)

        q_pos = cfg.ekf_process_pos
        q_vel = cfg.ekf_process_vel
        self.Q = torch.diag(torch.tensor([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel], device=device))

        r_pos = cfg.ekf_meas_pos
        r_vel = cfg.ekf_meas_vel
        self.R_base = torch.diag(torch.tensor([r_pos, r_pos, r_pos, r_vel, r_vel, r_vel], device=device))

        self.eye6 = torch.eye(6, device=device)

    def reset(self, env_ids, init_state=None):
        if len(env_ids) == 0:
            return
        self.P[env_ids] = self.eye6 * 0.5
        if init_state is None:
            self.state[env_ids] = 0.0
        else:
            self.state[env_ids] = init_state

    def predict(self):
        self.state = (self.F @ self.state.unsqueeze(-1)).squeeze(-1)
        self.P = self.F.unsqueeze(0) @ self.P @ self.F.t().unsqueeze(0) + self.Q.unsqueeze(0)

    def update(self, visible_mask, measurement, meas_var_scale):
        if torch.sum(visible_mask) == 0:
            return

        ids = torch.where(visible_mask)[0]
        x = self.state[ids]
        P = self.P[ids]
        z = measurement[ids]

        R = self.R_base.unsqueeze(0).repeat(len(ids), 1, 1)
        R = R * meas_var_scale[ids].view(-1, 1, 1)

        y = z - (self.H @ x.unsqueeze(-1)).squeeze(-1)
        S = self.H.unsqueeze(0) @ P @ self.H.t().unsqueeze(0) + R
        K = P @ torch.linalg.inv(S)

        x_post = x + (K @ y.unsqueeze(-1)).squeeze(-1)
        P_post = (self.eye6.unsqueeze(0) - K @ self.H.unsqueeze(0)) @ P

        self.state[ids] = x_post
        self.P[ids] = P_post


class TennisRobot(Go2ArxRobot):
    """Phase-2 tennis environment: noisy perception + EKF + asymmetric observations."""

    def _create_envs(self):
        super()._create_envs()

        ball_cfg = self.cfg.tennis_ball
        ball_asset_options = gymapi.AssetOptions()
        ball_asset_options.fix_base_link = False
        ball_asset_options.disable_gravity = False
        ball_asset_options.override_com = True
        ball_asset_options.override_inertia = True

        self.ball_asset = self.gym.create_sphere(self.sim, ball_cfg.radius, ball_asset_options)

        shape_props = self.gym.get_asset_rigid_shape_properties(self.ball_asset)
        for p in shape_props:
            p.friction = ball_cfg.friction
            p.rolling_friction = ball_cfg.friction
            p.torsion_friction = ball_cfg.friction
            p.restitution = ball_cfg.restitution
        self.gym.set_asset_rigid_shape_properties(self.ball_asset, shape_props)

        body_props = self.gym.get_asset_rigid_body_properties(self.ball_asset)
        for p in body_props:
            p.mass = ball_cfg.mass
            p.linear_damping = ball_cfg.linear_damping
            p.angular_damping = ball_cfg.angular_damping
        self.gym.set_asset_rigid_body_properties(self.ball_asset, body_props)

        self.ball_actor_handles = []
        self.robot_actor_indices = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.ball_actor_indices = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        ball_pose = gymapi.Transform()
        ball_pose.p = gymapi.Vec3(3.0, 0.0, 1.0)

        for env_id, env_handle in enumerate(self.envs):
            self.robot_actor_indices[env_id] = self.gym.get_actor_index(
                env_handle, self.actor_handles[env_id], gymapi.DOMAIN_SIM
            )

            ball_actor = self.gym.create_actor(
                env_handle,
                self.ball_asset,
                ball_pose,
                f"tennis_ball_{env_id}",
                0,
                0,
                0,
            )
            self.ball_actor_handles.append(ball_actor)
            self.ball_actor_indices[env_id] = self.gym.get_actor_index(
                env_handle, ball_actor, gymapi.DOMAIN_SIM
            )

    def _init_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = self.all_root_states[self.robot_actor_indices.long()]
        self.ball_root_states = self.all_root_states[self.ball_actor_indices.long()]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))

        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.init_ee_goal_variale()

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.ball_pos = self.ball_root_states[:, 0:3]
        self.ball_vel = self.ball_root_states[:, 7:10]
        self.time_to_intercept = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.visibility_flag = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.perception_sigma = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)

        self.ekf = BallEKF(self.num_envs, self.dt, self.device, self.cfg.perception)

        self.ee_body_handle = -1
        for ee_name in ["gripper", "arm_link6", "link_6", "link6"]:
            handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], ee_name)
            if handle != -1:
                self.ee_body_handle = handle
                break

    def _get_noise_scale_vec(self, cfg):
        base_noise = super()._get_noise_scale_vec(cfg)
        if base_noise.shape[0] == cfg.env.num_observations:
            return base_noise
        pad = torch.zeros(cfg.env.num_observations - base_noise.shape[0], device=self.device)
        return torch.cat((base_noise, pad), dim=0)

    def _reset_root_states(self, env_ids):
        if len(env_ids) == 0:
            return

        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)

        actor_ids = self.robot_actor_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.all_root_states),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )

    def _push_robots(self):
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.all_root_states))

    def _reset_ball(self, env_ids):
        if len(env_ids) == 0:
            return

        ball_cfg = self.cfg.tennis_ball
        n = len(env_ids)

        p0_x = torch_rand_float(ball_cfg.spawn_x[0], ball_cfg.spawn_x[1], (n, 1), device=self.device)
        p0_y = torch_rand_float(ball_cfg.spawn_y[0], ball_cfg.spawn_y[1], (n, 1), device=self.device)
        p0_z = torch_rand_float(ball_cfg.spawn_z[0], ball_cfg.spawn_z[1], (n, 1), device=self.device)
        p0 = torch.cat((p0_x, p0_y, p0_z), dim=1)
        p0[:, :2] += self.env_origins[env_ids, :2]

        vx = torch_rand_float(ball_cfg.vel_x[0], ball_cfg.vel_x[1], (n, 1), device=self.device)
        vy = torch_rand_float(ball_cfg.vel_y[0], ball_cfg.vel_y[1], (n, 1), device=self.device)
        vz = torch_rand_float(ball_cfg.vel_z[0], ball_cfg.vel_z[1], (n, 1), device=self.device)
        v0 = torch.cat((vx, vy, vz), dim=1)

        self.ball_root_states[env_ids, 0:3] = p0
        self.ball_root_states[env_ids, 3:7] = 0.0
        self.ball_root_states[env_ids, 6] = 1.0
        self.ball_root_states[env_ids, 7:10] = v0
        self.ball_root_states[env_ids, 10:13] = 0.0

        strike_plane_world_x = self.env_origins[env_ids, 0] + ball_cfg.strike_plane_x
        safe_vx = torch.where(torch.abs(v0[:, 0]) < 1e-4, torch.full_like(v0[:, 0], -1e-4), v0[:, 0])
        t_hit = (strike_plane_world_x - p0[:, 0]) / safe_vx
        t_hit = torch.clamp(t_hit, min=ball_cfg.min_intercept_time, max=ball_cfg.max_intercept_time)
        self.time_to_intercept[env_ids] = t_hit

        actor_ids = self.ball_actor_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.all_root_states),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )

    def _get_ball_state_body_frame(self):
        rel_pos_world = self.ball_pos - self.root_states[:, 0:3]
        rel_vel_world = self.ball_vel - self.root_states[:, 7:10]

        ball_pos_body = quat_rotate_inverse(self.base_quat, rel_pos_world)
        ball_vel_body = quat_rotate_inverse(self.base_quat, rel_vel_world)
        return ball_pos_body, ball_vel_body

    def _compute_visibility(self, ball_pos_body):
        fov_half = 0.5 * self.cfg.perception.camera_fov_deg * torch.pi / 180.0
        xy_norm = torch.norm(ball_pos_body[:, :2], dim=-1).clamp(min=1e-6)
        cos_angle = ball_pos_body[:, 0] / xy_norm
        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        in_front = ball_pos_body[:, 0] > 0.0
        visible = torch.logical_and(in_front, angle <= fov_half)
        return visible

    def _apply_perception_noise(self, ball_pos_body, ball_vel_body, visible):
        perception_cfg = self.cfg.perception

        d = torch.norm(ball_pos_body, dim=-1)
        omega_norm = torch.norm(self.base_ang_vel, dim=-1)

        sigma_pos = perception_cfg.base_pos_noise + perception_cfg.distance_noise_scale * (d ** 2)
        blur_boost = torch.where(
            omega_norm > perception_cfg.motion_blur_omega_thresh,
            1.0 + perception_cfg.motion_blur_gain * (omega_norm - perception_cfg.motion_blur_omega_thresh),
            torch.ones_like(omega_norm),
        )
        sigma_pos = sigma_pos * blur_boost

        sigma_vel = perception_cfg.base_vel_noise * blur_boost

        oof_boost = torch.where(visible, torch.ones_like(sigma_pos), torch.full_like(sigma_pos, perception_cfg.out_of_fov_noise_boost))
        sigma_pos = sigma_pos * oof_boost
        sigma_vel = sigma_vel * oof_boost

        noisy_pos = ball_pos_body + torch.randn_like(ball_pos_body) * sigma_pos.unsqueeze(-1)
        noisy_vel = ball_vel_body + torch.randn_like(ball_vel_body) * sigma_vel.unsqueeze(-1)

        self.perception_sigma[:, 0] = sigma_pos
        return noisy_pos, noisy_vel, d, omega_norm

    def _update_ekf(self):
        ball_pos_body, ball_vel_body = self._get_ball_state_body_frame()
        visible = self._compute_visibility(ball_pos_body)

        noisy_pos, noisy_vel, distance, omega_norm = self._apply_perception_noise(ball_pos_body, ball_vel_body, visible)

        z = torch.cat((noisy_pos, noisy_vel), dim=-1)
        meas_scale = 1.0 + self.perception_sigma[:, 0]

        self.ekf.predict()
        self.ekf.update(visible, z, meas_scale)

        self.visibility_flag[:, 0] = visible.float()

        return ball_pos_body, ball_vel_body, distance, omega_norm


    def _build_robot_obs(self, add_noise=False):
        """Build 75-dim robot observation; optionally add default observation noise."""
        self.dof_err = self.dof_pos - self.default_dof_pos
        self.dof_err[:, self.wheel_indices] = 0
        self.dof_pos[:, self.wheel_indices] = 0

        robot_obs = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                self.dof_err * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self._local_gripper_pos * self.obs_scales.gripper_track,
                self.curr_ee_goal_cart * self.obs_scales.gripper_track,
                (self._local_gripper_pos - self.curr_ee_goal_cart) * self.obs_scales.gripper_track,
                self.actions,
            ),
            dim=-1,
        )

        if add_noise and self.add_noise:
            robot_obs = robot_obs + (2 * torch.rand_like(robot_obs) - 1) * self.noise_scale_vec[: robot_obs.shape[1]]
        return robot_obs

    def _get_env_friction(self):
        if hasattr(self, "friction_coeffs"):
            coeff = self.friction_coeffs.squeeze(-1).to(self.device)
            return coeff.view(-1, 1)
        return torch.full((self.num_envs, 1), self.cfg.terrain.static_friction, dtype=torch.float, device=self.device)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._reset_ball(env_ids)

        # reset EKF covariance/state each episode reset
        ball_pos_body, ball_vel_body = self._get_ball_state_body_frame()
        init_state = torch.cat((ball_pos_body, ball_vel_body), dim=-1)
        self.ekf.reset(env_ids, init_state[env_ids])
        self.visibility_flag[env_ids] = 1.0
        self.perception_sigma[env_ids] = 0.0

    def compute_observations(self):
        # Build robot observations once for actor(noisy) and critic(privileged, GT robot state).
        robot_obs_actor = self._build_robot_obs(add_noise=True)
        robot_obs_priv = self._build_robot_obs(add_noise=False)

        gt_ball_pos_body, gt_ball_vel_body, distance, omega_norm = self._update_ekf()
        ekf_state = self.ekf.state

        # Actor obs (81): 75 base + EKF pos(3) + EKF vel_xy(2) + visibility flag(1)
        actor_ball = torch.cat((ekf_state[:, 0:3], ekf_state[:, 3:5], self.visibility_flag), dim=-1)
        self.obs_buf = torch.cat((robot_obs_actor, actor_ball), dim=-1)

        # Critic privileged obs (88): GT robot states + GT ball states + env/temporal factors.
        friction = self._get_env_friction()
        self.privileged_obs_buf = torch.cat(
            (
                robot_obs_priv,  # 75
                gt_ball_pos_body,  # +3 => 78
                gt_ball_vel_body,  # +3 => 81
                friction,  # +1 => 82
                self.time_to_intercept.unsqueeze(-1),  # +1 => 83
                self.visibility_flag,  # +1 => 84
                distance.unsqueeze(-1),  # +1 => 85
                omega_norm.unsqueeze(-1),  # +1 => 86
                torch.norm(ekf_state[:, 0:3] - gt_ball_pos_body, dim=-1, keepdim=True),  # +1 => 87
                self.perception_sigma,  # +1 => 88
            ),
            dim=-1,
        )

        if self.obs_buf.shape[1] != self.cfg.env.num_observations:
            raise RuntimeError(f"Actor obs dim mismatch: {self.obs_buf.shape[1]} != {self.cfg.env.num_observations}")
        if self.privileged_obs_buf.shape[1] != self.cfg.env.num_privileged_obs:
            raise RuntimeError(
                f"Privileged obs dim mismatch: {self.privileged_obs_buf.shape[1]} != {self.cfg.env.num_privileged_obs}"
            )

    def _reward_ekf_error(self):
        gt_ball_pos_body, _ = self._get_ball_state_body_frame()
        ekf_error = torch.norm(self.ekf.state[:, 0:3] - gt_ball_pos_body, dim=-1)
        return ekf_error
