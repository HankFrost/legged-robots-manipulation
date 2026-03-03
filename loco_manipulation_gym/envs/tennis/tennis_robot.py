import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import (
    get_axis_params,
    quat_rotate_inverse,
    to_torch,
    torch_rand_float,
)

from loco_manipulation_gym.envs.go2_arx.go2_arx_robot import Go2ArxRobot


class TennisRobot(Go2ArxRobot):
    """Phase-1 tennis environment skeleton built on top of Go2ArxRobot."""

    def _create_envs(self):
        # create robot envs first (from parent)
        super()._create_envs()

        ball_cfg = self.cfg.tennis_ball

        # ball asset
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

            # NOTE: collision_group=0 to allow broad interaction with scene actors/ground.
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
        """Override to handle two root actors per env (robot + ball)."""
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
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.init_ee_goal_variale()

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(
            2,
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        ) + str_rng[0]

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

        # Explicitly define EE handle for future strike reward logic.
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
        """Reset robot root states only (avoid touching ball states)."""
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
        """Reset ball with sampled initial state and compute per-env intercept time T."""
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

        # T from x-dimension crossing of strike plane: P0_x + Vx * T = x_plane
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

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._reset_ball(env_ids)

    def compute_observations(self):
        super().compute_observations()

        # Append projectile state to keep obs dimension aligned to 81.
        ball_pos_local = self.ball_pos - self.root_states[:, 0:3]
        self.obs_buf = torch.cat((self.obs_buf, ball_pos_local, self.ball_vel), dim=-1)
