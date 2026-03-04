# 四足机械臂网球/羽毛球技能复现改造路线图

> 目标：基于 `loco_manipulation_gym` 复现“Go2 + ARX6 拦截并击打高速抛体（网球/羽毛球）”任务。

## 0. 先做差距分析（1~2 天）

1. 盘点现有资产：`Go2ArxRobot` 的观测、动作、奖励、重置流程。
2. 确认球体是否已有资产；若没有，增加 `sphere` 资产与发射器逻辑。
3. 梳理 `rsl_rl` 接口：当前是否仅支持单一 obs，确定改造点（rollout storage / actor-critic / runner）。
4. 明确训练指标：
   - 拦截成功率（interception success）
   - 击球后目标方向误差
   - 能耗/电流约束违例率

## 1. 阶段一：环境与任务骨架（优先级 P0）

### 1.1 新建任务与配置

1. 新建 `TennisRobot`（继承 `Go2ArxRobot`），并创建独立 config：
   - `env`: 并行环境数、episode 长度
   - `ball`: 初始位置/速度采样范围、重力、是否空气阻力
   - `intercept`: 目标拦截时间 `T` 与时间窗口 `Δt`
2. 保留“可回退”开关：支持 `tennis`（无空气阻力）和 `shuttlecock`（后续扩展气动模型）。

### 1.2 抛体动力学与重置

1. 实现标准抛体：
   \[
   P_t=P_0+V_0t+\frac{1}{2}gt^2
   \]
2. 每回合 reset 随机化：
   - 球初始位姿与速度
   - 机器人初始 base yaw 与 arm 初始姿态
3. 终止条件：
   - 球落地/越界
   - 机器人不安全姿态
   - 超时

### 1.3 任务状态机（最关键）

建议至少 4 个阶段：
1. `BALLISTIC_TRACKING`：飞行追踪 + 预测更新
2. `PRE_STRIKE`：进入 `|t-T|<Δt_pre` 的准备区间
3. `STRIKE_WINDOW`：`|t-T|<Δt`，激活击球主奖励
4. `FOLLOW_THROUGH`：击球后随挥与稳定恢复

并在 buffer 中持久记录：`phase_id`, `t_to_intercept`, `predicted_intercept_point`。

## 2. 阶段二：观测与噪声建模（P0）

### 2.1 非对称 Actor-Critic 观测拆分

1. **Actor obs（部署可得）**：
   - 本体感知（关节位置/速度、base 状态、上一时刻动作）
   - EKF 输出球状态（位置/速度）
   - 所有球相关量都走噪声模型
2. **Critic privileged obs（训练专用）**：
   - 球真值轨迹（GT）
   - 机器人 GT 状态
   - 环境参数（质量/摩擦/增益）
   - `T` 和 `t_to_intercept`

### 2.2 感知噪声模型 σ

实现：
\[
\sigma = f(d_{robot-ball}, \|\omega_{base}\|, \mathbb{1}_{inFOV})
\]

建议形态：
- 距离越远噪声越大
- base 角速度越大噪声越大（模拟运动模糊）
- 若不在 FOV：观测退化为“丢失观测+外推”

### 2.3 EKF 管线

1. 维护 `predict/update` 两阶段。
2. 在观测丢失时仅 predict。
3. 向 Actor 只暴露 EKF 状态，不暴露 GT。
4. 记录 EKF 误差用于奖励与日志。

## 3. 阶段三：奖励系统重构（P0）

## 3.1 时间触发主奖励（Strike Reward）

仅在 `|t-T|<Δt` 激活，避免稀释信用分配：
1. `r_pos`：拍心与球距离
2. `r_rot`：拍面法线与目标回球方向夹角
3. `r_vel`：拍面速度/动量是否达到阈值

可写成：
\[
r_s = w_{pos}r_{pos}+w_{rot}r_{rot}+w_{vel}r_{vel}
\]
并乘时间门控 `g_t = 1(|t-T|<Δt)`。

### 3.2 主动感知奖励

1. 奖励 EKF 误差下降：`r_ap = e_{t-1}-e_t`
2. 奖励球在 FOV 内停留比例
3. 避免作弊：加入 base 抖动惩罚

### 3.3 安全与能耗约束

两条路线二选一（建议先易后难）：
1. PPO + 拉格朗日乘子（约束项如总功率/总电流）
2. N-P3O 形式（需要更大改动）

建议先落地约束指标：
- 机械臂总电流 `< 8A`
- 关节峰值扭矩
- 单回合约束违例次数

## 4. 阶段四：算法侧改造（P0）

## 4.1 Actor-Critic 网络

1. Actor MLP: `[512, 256, 128]`
2. Critic 可更宽（如 `[512, 512, 256]`）提高 value 拟合能力
3. 确保 actor/critic 输入维度独立可配

### 4.2 rsl_rl 改造点

1. Rollout storage 同时存储：
   - `obs`
   - `privileged_obs`
2. `act()` 用 `obs`，`evaluate()` 用 `privileged_obs`
3. Runner/Env 接口返回二元观测
4. 日志分离：policy loss / value loss / constraint loss

### 4.3 训练稳定性

1. GAE 与 value clip 参数单独调优
2. Strike window 稀疏奖励下提高 entropy warmup
3. 课程学习：
   - 先固定球速、固定方向
   - 再扩展速度与角度分布

## 5. 阶段五：系统辨识与域随机化（P1）

1. 训练期随机化：
   - 连杆质量
   - 电机增益/延迟
   - 地面摩擦
2. 观测延迟随机化：相机和控制链路延迟
3. 对抗性随机化：偶发观测丢失与突变光照（抽象为噪声尖峰）

## 6. 阶段六：验证闭环（P0）

### 6.1 指标看板（建议最先搭）

- `success@intercept`
- `mean strike error (cm)`
- `return direction error (deg)`
- `FOV ratio`
- `EKF RMSE`
- `energy/current violation rate`

### 6.2 Ablation（论文复现关键）

至少做 5 组：
1. 对称 AC vs 非对称 AC
2. 无时间门控 vs 有时间门控
3. 无主动感知奖励 vs 有主动感知奖励
4. 无约束 vs 约束优化
5. 无域随机化 vs 有域随机化

## 7. 推荐实施顺序（可直接开工）

1. **第 1 周**：完成任务类、抛体、状态机、基础奖励（无 EKF）
2. **第 2 周**：接入 EKF 与噪声模型，完成 actor obs 闭环
3. **第 3 周**：改造 rsl_rl 非对称 AC + 时间触发奖励完善
4. **第 4 周**：加约束优化 + 域随机化 + ablation 脚本

## 8. 最小可行里程碑（MVP）定义

满足以下条件即可认定“第一版复现成功”：
1. 在固定球速场景中，拦截成功率 > 70%
2. Actor 全程不访问 GT 球状态
3. 能耗/电流约束违例率低于预设阈值
4. 至少完成“对称 vs 非对称 AC”的 ablation

## 9. 建议你下一步立刻做的 3 件事

1. 在 repo 中创建 `tennis` 任务骨架与 config（先跑通 reset/step）。
2. 定义观测字典结构（`obs` 与 `privileged_obs`）并把维度写入配置。
3. 实现时间门控的 `strike reward`，先用 GT 球状态验证奖励是否工作，再切到 EKF 输出。
