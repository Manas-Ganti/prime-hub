# Not a fit: continuous-control / pixel-policy projects

One file for the three projects whose verdict is **Not a fit**. Each section gives the reason.

## self-play-highwayenv (`git@github.com:Manas-Ganti/self-play-highwayenv.git`, public)
- **What it actually is:** PPO vs critic-free GRPO on highway-env racetracks. Agents are trained
  solo with IDM traffic and evaluated head-to-head. It is not an ELO self-play league as the
  checklist assumed.
- **Action/obs:** continuous steering/throttle at the control rate; kinematic observations.
- **Results:** `results/` is empty. The latest commits log an IDM reference (49/50 laps) and an SB3
  pilot. No trained-agent numbers exist.
- **Why not a fit:** decisions happen at the control rate, so a text wrapper would mean an LLM
  emitting steering values tens of times per second. A coarse "lane change / overtake / hold"
  meta-action interface over highway-env's discrete meta-actions is *possible*, and a multi-agent
  `vf.Env` could pit two LLMs against each other. But that would be a different project, it would
  not test the research question (GRPO vs PPO under distribution shift), and highway-env LLM
  driving envs already exist. Argued both ways: the multi-agent `Env` angle is the only
  interesting one, and it doesn't justify the cost right now.

## maze-navigation-drone (`git@github.com:Manas-Ganti/maze-navigation-drone.git`, public)
- Isaac Lab Crazyflie, 64×64 RGB camera, `[vx, vy, vz, yaw_rate]` at 20 Hz, PPO via rsl_rl.
  Potential-based geodesic shaping, time penalty, collision penalty, goal bonus (`env/reward.py`).
- Results: geometry checks only (`results/maze_checks/summary.json`). Not trained.
- **Not a fit:** a pixel policy under continuous control, and it needs Isaac Sim (GPU,
  non-pip-installable). An LLM wrapper would be contrived.

## robotics-rl-payload-transport (`git@github.com:Manas-Ganti/robotics-rl-payload-transport.git`, public)
- Mobile robot carrying a variable payload over sloped terrain (Isaac), Nav2 baseline, OOD axes
  (density, slope, friction, payload, sensor noise).
- Results: `results/clips/` only (diagnostic clips). No metrics file.
- **Not a fit:** continuous control in a physics simulator that a Hub user can't install.
