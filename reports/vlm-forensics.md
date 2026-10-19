# VLM forensics: investigative AI-image detection (source: `VLM-RL-aicontent-detection/`)

Repo remote: `RL-Based-AI-content-detector.git`. It appears publicly as
`Manas-Ganti/Visual-reasoning-rlvr` (renamed). Last commit `6b82e70`.
Proposed Hub name: **`vlm-image-forensics`**.

## 1. Task

One episode is one 1024×1024 image, label AI or REAL. The agent first sees a heavily
downsampled overview (`OVERVIEW_LONG_EDGE=56` on synth1024) with a 4×4 grid. Each turn it emits
a labelled text block: RECONCILIATION, BELIEF_UPDATE `P(fake)=x`, OBSERVATION, REASONING,
HYPOTHESIS, ACTION (parser: `env/trajectory.py:178-218`).
- `INSPECT <n>`: reveals cell n (1–16) at full resolution and costs one unit of inspect budget.
- `VERDICT <AI|REAL> confidence=<c>`: terminal.

Termination: verdict, or `max_turns = max_inspects + 3` (`env/environment.py:88`). The default
budget is 4–6 inspects.

## 2. Reward

`compute_episode_reward`, `env/reward.py:217-244`, with weights from `RewardConfig` (`reward.py:57-95`):

```
R = 1.0·(±1 verdict correct; 0 if unanswered)
  + 0.40·(1 - 2(p_final - y)²)          belief_brier, proper rule on final P(fake)
  + 0.30·verdict_consistency            ∈ [0,1], truth-independent
  + 0.0·belief_coherence                (disabled, run 2 mistrained on it)
  + 0.0·prediction_tracking             (disabled, run 1 farmed "CONFIRMED" 91% of the time)
  - 0.05·n_inspects
  - 0.50·conf·(1.5 if wrong REAL else 1.0)   on wrong verdicts only
  - 1.0 if unanswered
```

**The README formula is stale.** It still lists coherence 0.30 and tracking 0.10 and omits
belief_brier. The Hub README must use the code's weights. Everything is terminal. Breakdown
terms are logged.

## 3. Verifiability

Programmatic, from the image-level label plus parsed numeric fields. No judge. The "intervention
test" and the GradCAM evidence slice are eval-only (`eval/harness.py`), not part of the reward.
The substrate gates (ceiling AUC 0.930, floor 0.591, geometry AUC 0.500) show the task needs
inspection to reach high accuracy (`results/substrate_synth1024.md`).

## 4. Reward-hacking audit

**Exploit found: a policy that never looks scores ~+0.50 (CRITICAL).**
Constant completion `BELIEF_UPDATE: P(fake)=0.52 / ACTION: VERDICT AI confidence=0`:
- verdict_consistency gives +0.30 on every answered episode. The final belief agrees with the
  verdict, and nothing checks the truth (`reward.py:139-161`).
- belief_brier near p = 0.5 gives about +0.20 on either label (`reward.py:176-194`).
- `confidence=0` is accepted and clamped to [0,1] (`trajectory.py:209-211`), which zeroes the
  confident-wrong penalty (`reward.py:197-210`).
- Zero inspects means zero action cost.

Expected reward on balanced labels: **+0.4997** (computed with the source code). For comparison,
the GRPO cycle-1 mean reward was +0.183 (README; `results/grpo_run1.md`). The constant policy
beats every trained policy on reward while having **50% accuracy**. Test:
`reports/exploit_probes/test_vlm_noinspect_exploit.py` (passes, i.e. the exploit is live).
This may also help explain why accuracy never moved: about 0.5 of reward is available with no
discrimination at all.

Fix options (any of them breaks parity with the ARC runs):
1. Gate verdict_consistency on correctness, or remove it.
2. Floor the confidence used in the penalty (e.g. `max(conf, |p-0.5|·2)`), or derive the
   penalty from P(fake) instead of a separately stated confidence. Two calibration channels
   (P(fake) and confidence) let the agent zero one of them.
3. Require at least one INSPECT before a verdict counts, or score zero-inspect verdicts as unanswered.

Other probes:
- Always-CONFIRMED: already neutralised (weight 0).
- Prose / length gaming: the reward reads only parsed fields. Not exploitable.
- Last-ACTION-line parsing plus a fallback that scans the whole text for `VERDICT` (`trajectory.py:198-202`):
  the model can't get two actions. Fine.

## 5. Data

`synth1024`: DIV2K photographs (center-cropped) paired with SDXL generations captioned from
those crops, 1570 images (`data/build_paired_synthetic.py`). Built on ARC; **not on Hugging Face**.
- DIV2K license, checked 2026-10-18 on the official page (data.vision.ee.ethz.ch/cvl/DIV2K): "made
  available for academic research purpose only. All the images are collected from the Internet, and
  the copyright belongs to the original owners", with a removal-on-request clause. No redistribution
  grant and no license for the individual images. **Rehosting DIV2K crops on HF/the Hub is not
  allowed**: the Hub is a public registry used for training, including commercially, which ETH cannot
  authorise for images it does not own. (An existing HF mirror, eugenesiow/Div2k, is marked
  `license: other` and repeats the same terms; it is not a permission.) SDXL outputs are fine under
  the OpenRAIL++ license, but captions/generations derived from DIV2K crops should be hosted only
  without the source photos.
- Ways forward: (a) host only the SDXL half + per-image DIV2K filenames/crop boxes and have `load()`
  download the official DIV2K zip from ETH (synth1024 uses all 800 `DIV2K_train_HR` images, the
  multi-GB train zip, per `docs/arc_runbook.md`; the user accepts the academic-only terms; the env
  inherits that restriction and must say so); (b) rebuild the real half from openly licensed photos
  (e.g. Pexels/Unsplash-licensed or CC0/CC-BY sets with per-image licences), regenerate the SDXL
  pairs, and re-run the substrate gates; (c) keep the env private/eval-only.
- Generating in `load()` is not realistic (it needs SDXL plus a VLM captioner on a GPU).
- So the package needs a hosted HF dataset with a clear license. This is a **blocker** until resolved.
- Earlier substrates (faces, GenImage) failed the gates. GenImage has a geometry confound
  (`results/geometry_confound.md`) and should not be used.

## 6. Evals and ARC results (separate harness)

| measurement | setup | headline | file |
|---|---|---|---|
| substrate gates | Qwen2.5-VL-32B, n=156 (78/class) | ceiling AUC 0.930, floor 0.591, budget-6 random cells 0.820 AUC / 0.763 acc | `results/substrate_synth1024.md`, README |
| GRPO cycle 1 | 600 steps, budget 4 | discrimination gap -3.4% → +15.8%; held-out acc 0.545 vs 0.513 base (McNemar p=0.23); mean reward -0.032 → +0.183 | `results/grpo_run1.md` |
| GRPO cycle 3 | 140 steps, budget 6; eval clean, budget 2 | acc base 0.404 / SFT 0.417 / GRPO 0.385 (n.s.) | `results/grpo_run3.md` |
| trace analysis | 988 rollouts | the verdict matches the first-inspection belief 80.1% of the time | `results/reasoning_trace_analysis.md` |

Honest headline: no accuracy gain over the base model in three cycles.

## 7. Hub fit verdict: **Native**, but not publishable yet

It is LLM-native (VLM, multi-turn, tool-like INSPECT), and v1 has image utilities
(`verifiers/v1/utils/image.py`). Three blockers: (1) the no-look exploit; (2) data hosting and
license; (3) it needs a vision-capable API model for evals, which is fine but should be stated.
Port shape: Taskset + Toolset with an `inspect(cell)` tool that returns an image; verdict as the
final reply or a tool. The shared predict-then-verify idea with dxenv is conceptual only. The two
codebases share no reward code (dxenv scores predicted lab buckets against revealed values; this
env reads self-reported reconciliations). A shared module would be artificial; vendor each.
