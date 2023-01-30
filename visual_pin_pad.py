"""Continuous action space version of the visual pin pad environment.

Adapted from the Director paper (https://danijar.com/project/director/).
"""

import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
from hydra.utils import instantiate

import gym
import imageio
import numpy as np
from gym.spaces import Box
from gym.utils.seeding import _int_list_from_bigint, create_seed, hash_seed
from omegaconf import DictConfig, OmegaConf

from cfg import init_hydra, init_config, to_normalized_str

TimeStep = tuple[np.ndarray, float, bool, dict[str, Any]]


@dataclass
class PinPad:
    """Data structure for information about a single pin pad."""

    lower_left: np.ndarray
    upper_right: np.ndarray
    color: np.ndarray


class VisualPinPadEnv(gym.Env):
    """Implementation of the visual pin pad environment."""

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        # Creeate a config from the inputted parameters.
        self.env_cfg = OmegaConf.create(kwargs)
        # Frame stack is important in this environment because
        # otherwise the agent doesn't get velocity information.
        assert self.env_cfg.frame_stack > 1
        # Actions are 2D acceleration vectors in the x and y direction.
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        if self.env_cfg.modality == "pixels":
            assert not self.env_cfg.wind.is_observed
            obs_shape = (3, self.env_cfg.img_size, self.env_cfg.img_size)
            self.observation_space = Box(
                low=np.zeros(obs_shape, dtype=np.uint8),
                high=np.full(obs_shape, 255, dtype=np.uint8),
                shape=obs_shape,
                dtype=np.uint8,
            )
        elif self.env_cfg.modality == "state":
            # Observation: [x position, y position, x velocity, y velocity,
            #               history of last len(pad_goal_sequence) touched pads]
            # each element in the history is a one-hot vector of length num_pads + 1
            low = [-1.0, -1.0, -1.0 * self.env_cfg.max_vel, -1.0 * self.env_cfg.max_vel]
            high = [1.0, 1.0, 1.0 * self.env_cfg.max_vel, 1.0 * self.env_cfg.max_vel]
            if self.env_cfg.wind.is_observed:
                low += [-self.env_cfg.wind.max_abs] * 2
                high += [self.env_cfg.wind.max_abs] * 2
            for _1 in range(len(self.env_cfg.pad_goal_sequence)):
                for _2 in range(self.env_cfg.num_pads + 1):
                    low.append(0.0)
                    high.append(1.0)
            self.observation_space = Box(
                low=np.array(low, dtype=np.float32),
                high=np.array(high, dtype=np.float32),
                dtype=np.float32,
            )
        self.seed(self.env_cfg.seed)
        self._current_wind = np.zeros(2, dtype=np.float32)
        self._pin_pads = self._get_pin_pad_info()
        # Check that self.env_cfg.pad_goal_sequence is well-formed. A pad cannot
        # be touched twice in a row, so that should never be in the goal.
        for i in range(len(self.env_cfg.pad_goal_sequence) - 1):
            assert (
                self.env_cfg.pad_goal_sequence[i]
                != self.env_cfg.pad_goal_sequence[i + 1]
            )
            assert 0 <= self.env_cfg.pad_goal_sequence[i] < self.env_cfg.num_pads
        assert 0 <= self.env_cfg.pad_goal_sequence[-1] < self.env_cfg.num_pads

    def get_dataset(self) -> dict[str, np.ndarray]:
        """Implement data generation for offline mode.

        To generate this dataset, first run: PYTHONPATH=src python
        src/envs/debug/visual_pin_pad_env.py.
        """
        with self.get_dataset_path().open("rb") as f:
            return pickle.load(f)

    def get_dataset_path(self) -> Path:
        offline_dir = Path(self.env_cfg.dataset.base_dir)
        pad_goal_sequence_str = "-".join(map(str, self.env_cfg.pad_goal_sequence))
        fprefix = "__".join(
            to_normalized_str(x)
            for x in [
                "pin_pad",
                self.env_cfg.modality,
                pad_goal_sequence_str,
                self.env_cfg.dt,
                self.env_cfg.max_vel,
                self.env_cfg.dataset.p_reach_target,
                self.env_cfg.dataset.seed,
                self.env_cfg.dataset.num_transitions,
                self.env_cfg.state_obs_pos_vel_noise,
                self.env_cfg.wind.delta,
                self.env_cfg.wind.max_abs,
                self.env_cfg.wind.is_observed,
            ]
        )
        return offline_dir / f"{fprefix}.pkl"

    def get_normalized_score(self, ep_reward: float) -> float:
        """Dummy reward normalization for evaluation in offline mode.

        Reverses the multiplication by 100 done by the caller.
        """
        return ep_reward / 100.0

    def reset(self) -> np.ndarray:
        # The internal state has agent position (with (0, 0) being the center of the
        # image), agent velocity, and the ordered history of touched pads.
        self._reset_agent_position()
        self._reset_wind()
        self._agent_vel = np.zeros(2, dtype=np.float32)
        self._touched_pads: list[int] = []
        self._current_step = 0
        obs = self._make_obs()
        assert self.observation_space.contains(obs)
        return obs

    def seed(self, seed: Optional[int] = None) -> list[int]:
        seed = create_seed(seed)
        seed_list = _int_list_from_bigint(hash_seed(seed))
        self.rng = np.random.default_rng(seed_list)
        obs_seed = self.observation_space.seed(seed)
        act_seed = self.action_space.seed(seed)
        return [seed] + obs_seed + act_seed

    def step(self, action: np.ndarray) -> TimeStep:
        action = action.clip(self.action_space.low, self.action_space.high)
        if self.env_cfg.wind.max_abs > 0.0:
            self._update_wind()
            action = (action + self._current_wind).clip(
                self.action_space.low, self.action_space.high
            )
        assert self.action_space.contains(action)
        self._current_step += 1
        # The action corresponds to acceleration. Compute new agent_pos & agent_vel.
        self._agent_pos: np.ndarray = (
            self._agent_pos
            + self._agent_vel * self.env_cfg.dt
            + 0.5 * action * (self.env_cfg.dt**2)
        )
        self._agent_vel = self._agent_vel + action * self.env_cfg.dt
        self._agent_vel = np.clip(
            self._agent_vel, -1.0 * self.env_cfg.max_vel, self.env_cfg.max_vel
        )
        for i in range(2):
            if self._agent_pos[i] < -1:
                self._agent_pos[i] = -1
                self._agent_vel[i] = 0
            elif self._agent_pos[i] > 1:
                self._agent_pos[i] = 1
                self._agent_vel[i] = 0
        maybe_touched_pad = self._maybe_get_touched_pad()
        if maybe_touched_pad is not None:
            assert 0 <= maybe_touched_pad < self.env_cfg.num_pads  # 0-indexed
            self._touched_pads.append(maybe_touched_pad)
        # Sparse reward only given if the sequence matches exactly.
        recent_pads = self._touched_pads[-len(self.env_cfg.pad_goal_sequence) :]
        reward = float(recent_pads == self.env_cfg.pad_goal_sequence)
        if reward > 0.0:
            # If the agent gets reward, reset it, but don't reset self._current_step!
            self._reset_agent_position()
            self._agent_vel = np.zeros(2, dtype=np.float32)
            self._touched_pads = []
        obs = self._make_obs()
        assert self.observation_space.contains(obs)
        ep_length = self.env_cfg.fixed_episode_length * self.env_cfg.action_repeat
        done = self._current_step == ep_length
        info: dict[str, Any] = {}
        return obs, reward, done, info

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        recent_pads = self._touched_pads[-len(self.env_cfg.pad_goal_sequence) :]
        obs = self._make_image_obs(
            self.env_cfg, self._agent_pos, self._pin_pads, recent_pads
        )
        return obs.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

    def render_obs(self, obs: np.ndarray) -> np.ndarray:
        """Render the given observation (as opposed to the one currently in
        memory)."""
        # Of the arguments to self._make_image_obs(), the env_cfg and
        # pin_pads are static, so we only need to extract agent_pos and
        # recent_pads from the given obs.
        agent_pos = obs[:2]
        recent_pads: list[int] = []
        pad_start = 6 if self.env_cfg.wind.is_observed else 4
        for i in range(len(self.env_cfg.pad_goal_sequence)):
            onehot_vec = obs[
                i * (self.env_cfg.num_pads + 1)
                + pad_start : (i + 1) * (self.env_cfg.num_pads + 1)
                + pad_start
            ]
            assert len(onehot_vec) == self.env_cfg.num_pads + 1
            # Note that the passed-in observation may be malformed (e.g., if it comes
            # out of a learned decoder), so we can't even be sure that we're working
            # with a one-hot vector here. Nevertheless, we can do this argmax, because
            # we're essentially turning the observation into an image "by force".
            offset = int(np.argmax(onehot_vec))
            if offset == 0:
                break
            recent_pads.append(offset - 1)
        obs = self._make_image_obs(self.env_cfg, agent_pos, self._pin_pads, recent_pads)
        return obs.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

    def _make_obs(self) -> np.ndarray:
        if self.env_cfg.modality == "pixels":
            recent_pads = self._touched_pads[-len(self.env_cfg.pad_goal_sequence) :]
            obs = self._make_image_obs(
                self.env_cfg, self._agent_pos, self._pin_pads, recent_pads
            )
        elif self.env_cfg.modality == "state":
            obs = self._make_state_obs()
        return obs

    @staticmethod
    def _make_image_obs(
        env_cfg: DictConfig,
        agent_pos: np.ndarray,
        pin_pads: list[PinPad],
        recent_pads: list[int],
    ) -> np.ndarray:
        # Show arena with pads and agent on a white background.
        arena = np.full(
            (env_cfg.arena_size, env_cfg.arena_size, 3),
            fill_value=255,
            dtype=np.uint8,
        )
        for pad in pin_pads:
            low_x, low_y = VisualPinPadEnv.pos_to_arena_coords(
                pad.lower_left, env_cfg.arena_size
            )
            high_x, high_y = VisualPinPadEnv.pos_to_arena_coords(
                pad.upper_right, env_cfg.arena_size
            )
            color_mult = (
                1.0 if VisualPinPadEnv.pos_touching_pad(agent_pos, pad) else 0.5
            )
            arena[low_x:high_x, low_y:high_y] = color_mult * pad.color
        agent_pos_arena = np.clip(
            VisualPinPadEnv.pos_to_arena_coords(agent_pos, env_cfg.arena_size),
            env_cfg.agent_halfsize,
            env_cfg.arena_size - env_cfg.agent_halfsize,
        )
        low_x, low_y = agent_pos_arena - env_cfg.agent_halfsize
        high_x, high_y = agent_pos_arena + env_cfg.agent_halfsize
        arena[low_x:high_x, low_y:high_y] = 0.0
        # Show history area on a dark gray background.
        history_area = np.full(
            (env_cfg.history_area_size, env_cfg.arena_size, 3),
            fill_value=150,
            dtype=np.uint8,
        )
        # Show touched pads in the history area. Only show the most
        # recent len(env_cfg.pad_goal_sequence) touched pads.
        x = env_cfg.history_area_size // 2
        low_x = x - env_cfg.history_square_halfsize
        high_x = x + env_cfg.history_square_halfsize
        ys = np.linspace(0, env_cfg.arena_size, num=len(env_cfg.pad_goal_sequence) + 2)[
            1:-1
        ]
        # Note: ys will often be longer than recent_pads, but this zip still works.
        for pad_idx, y in zip(recent_pads, ys):
            pad = pin_pads[pad_idx]
            low_y = int(y) - env_cfg.history_square_halfsize
            high_y = int(y) + env_cfg.history_square_halfsize
            history_area[low_x:high_x, low_y:high_y] = pad.color
        img = np.concatenate([arena, history_area], axis=0)
        # Pad the left and right so that the final image is square.
        assert img.shape[0] > img.shape[1]
        pad_amt = (img.shape[0] - img.shape[1]) // 2
        padding = np.full(
            (img.shape[0], pad_amt, 3),
            fill_value=150,
            dtype=np.uint8,
        )
        img = np.concatenate([padding, img, padding], axis=1)
        assert img.shape[0] == img.shape[1] == env_cfg.img_size
        return img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

    def _make_state_obs(self) -> np.ndarray:
        obs = np.full(self.observation_space.shape, fill_value=0.0, dtype=np.float32)
        obs[:2] = self._agent_pos
        obs[2:4] = self._agent_vel
        if (std := self.env_cfg.state_obs_pos_vel_noise) > 0.0:
            noise = self.rng.normal(0, std, size=4)
            noise[2:4] *= self.env_cfg.max_vel
            np.clip(
                obs[0:4] + noise,
                a_min=self.observation_space.low[0:4],
                a_max=self.observation_space.high[0:4],
                out=obs[0:4],
            )
        if self.env_cfg.wind.is_observed:
            obs[4:6] = self._current_wind
            pad_start = 6
        else:
            pad_start = 4
        recent_pads = self._touched_pads[-len(self.env_cfg.pad_goal_sequence) :]
        for i in range(len(self.env_cfg.pad_goal_sequence)):
            # Compute the offset into the associated one-hot vector.
            if i < len(recent_pads):
                assert 0 <= recent_pads[i] < self.env_cfg.num_pads
                offset = recent_pads[i] + 1
            else:
                offset = 0
            # Each one-hot vector is length self.env_cfg.num_pads + 1. We shift the index
            # by `pad_start` to account for previous data in the observation vector.
            obs[i * (self.env_cfg.num_pads + 1) + pad_start + offset] = 1.0
        return obs

    def _maybe_get_touched_pad(self) -> Optional[int]:
        for i, pad in enumerate(self._pin_pads):
            if self.pos_touching_pad(self._agent_pos, pad):
                if self._touched_pads and i == self._touched_pads[-1]:
                    # A pad cannot be touched twice in a row.
                    return None
                return i
        return None

    def _reset_agent_position(self) -> None:
        """Reset agent position to a randomly sampled empty location."""
        while True:
            pos = self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            if self.env_cfg.fixed_start_position:
                pos *= 0.0  # always start at the center
            if any(self.pos_touching_pad(pos, pad) for pad in self._pin_pads):
                # Collision; resample.
                continue
            break
        self._agent_pos = pos
        assert self._maybe_get_touched_pad() is None

    def _reset_wind(self) -> None:
        if (max_abs := self.env_cfg.wind.max_abs) > 0:
            self._current_wind[:] = self.rng.uniform(low=-max_abs, high=max_abs, size=2)
        else:
            self._current_wind.fill(0)

    def _update_wind(self) -> None:
        if (delta := self.env_cfg.wind.delta) > 0:
            self._current_wind += delta * ((self.rng.integers(0, 2, size=2)) * 2 - 1)
            self._current_wind = self._current_wind.clip(
                -self.env_cfg.wind.max_abs, self.env_cfg.wind.max_abs
            )

    @staticmethod
    def pos_touching_pad(pos: np.ndarray, pad: PinPad) -> bool:
        return bool(np.all(pad.lower_left <= pos) and np.all(pos <= pad.upper_right))

    @staticmethod
    def pos_to_arena_coords(pos: np.ndarray, arena_size: int) -> np.ndarray:
        """Convert coordinates in [-1, 1] to coordinates in image."""
        return np.interp(pos, [-1.0, 1.0], [0, arena_size]).astype(np.uint8)

    def _get_pin_pad_info(self) -> list[PinPad]:
        """Get the information about the pin pads, as a list of PinPad data
        structures."""
        if self.env_cfg.num_pads == 3:
            info = [
                PinPad(
                    lower_left=np.array([-1.0, -1.0]),
                    upper_right=np.array([-0.5, -0.5]),
                    color=np.array([255, 0, 0]),
                ),
                PinPad(
                    lower_left=np.array([0.5, -0.25]),
                    upper_right=np.array([1.0, 0.25]),
                    color=np.array([0, 255, 0]),
                ),
                PinPad(
                    lower_left=np.array([-1.0, 0.5]),
                    upper_right=np.array([-0.5, 1.0]),
                    color=np.array([0, 0, 255]),
                ),
            ]
        else:
            raise Exception("Unsupported num_pads")
        assert len(info) == self.env_cfg.num_pads
        return info


def build_env(
    overrides: list[str], return_env_cfg: bool = False
) -> Union["VisualPinPadEnv", tuple["VisualPinPadEnv", DictConfig]]:
    init_hydra()
    overrides = [
        "benchmark=debug",
        "task=visual_pin_pad",
        "frame_stack=7",
        "benchmark.debug.task.fixed_episode_length=${.dataset.num_transitions}",
        "benchmark.debug.task.fixed_start_position=false",
        "warmup_steps=0",
    ] + overrides
    with init_config(overrides=overrides) as cfg:
        env = instantiate(cfg.benchmark.debug.task)
        env_cfg = cfg.benchmark.debug.task
        # Resolve config for faster access.
        OmegaConf.resolve(env_cfg)
    return (env, env_cfg) if return_env_cfg else env


def _generate_offline_dataset(overrides: list[str]) -> None:
    env, env_cfg = build_env(overrides=overrides, return_env_cfg=True)
    dataset_cfg = env_cfg.dataset
    rng = np.random.default_rng(seed=dataset_cfg.seed)
    pin_pads = env.unwrapped._get_pin_pad_info()
    if dataset_cfg.write_video:
        # Set `maxlen` to keep only the last N frames in the video.
        frames: deque[np.ndarray] = deque(maxlen=None)
    total_steps = 0
    last_touched_steps = 0
    count_touched_pads = 0
    avg_num_steps_between_pads: float = 0.0
    p_change_target_per_step = 0
    total_reward = 0.0
    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []
    done = True  # so that we call reset() on the first iteration
    print("Running data collection with the following settings:")
    print("  " + "\n  ".join(overrides))
    while True:
        if done:
            obs = env.reset()
            if dataset_cfg.write_video:
                # Save the first frame of the new episode
                frames.append(env.render())
            cur_pad_idx = rng.integers(0, len(pin_pads))
            cur_pad = pin_pads[cur_pad_idx]
            cur_destination = rng.uniform(cur_pad.lower_left, cur_pad.upper_right)
            assert env.unwrapped._touched_pads == []
            # Use copy() here to avoid needing a mypy annotation.
            last_touched_pads = env.unwrapped._touched_pads.copy()
        act = (
            (cur_destination - env.unwrapped._agent_pos)
            .astype(env.action_space.dtype)
            .clip(env.action_space.low, env.action_space.high)
        )
        observations.append(obs)
        actions.append(act)
        obs, rew, done, _ = env.step(act)
        rewards.append(rew)
        terminals.append(False)  # this env can never terminate, only time out
        timeouts.append(done)
        total_steps += 1
        if total_steps % 100_000 == 0:
            print(f"Step {total_steps}, total reward: {total_reward}")
        total_reward += rew
        if dataset_cfg.write_video:
            frames.append(env.render())
        if total_steps == dataset_cfg.num_transitions:
            break
        touched_new_pad = last_touched_pads != (
            touched_pads := env.unwrapped._touched_pads
        )
        if touched_new_pad or rng.uniform() < p_change_target_per_step:
            if touched_new_pad:
                count_touched_pads += 1
                # `avg_num_steps_between_pads` is a running average of the number of
                # steps it takes to go from one pad to another.
                r = 1 / count_touched_pads
                avg_num_steps_between_pads = (
                    1 - r
                ) * avg_num_steps_between_pads + r * (total_steps - last_touched_steps)
                # The formula below to compute `p_change_target_per_step` ensures that
                #   p_reach_target = (1 - p_change_target_per_step)**avg_num_steps_between_pads
                # i.e., that on average the probability of changing target over the
                # sub-trajectory from one pad to another is equal to `p_reach_target`.
                p_change_target_per_step = 1 - dataset_cfg.p_reach_target ** (
                    1 / avg_num_steps_between_pads
                )
                last_touched_steps = total_steps
                last_touched_pads = touched_pads.copy()

            # Note that the agent may "accidentally" hit another pad than the one
            # it was aiming for.
            if touched_new_pad and touched_pads:
                cur_pad_idx = touched_pads[-1]
            while True:  # resample a target pad different from the current one
                new_pad_idx = rng.integers(0, len(pin_pads))
                if new_pad_idx != cur_pad_idx:
                    break
            cur_pad_idx = new_pad_idx
            cur_pad = pin_pads[cur_pad_idx]
            cur_destination = rng.uniform(cur_pad.lower_left, cur_pad.upper_right)
    print(f"Got reward {total_reward} in {total_steps} steps")
    observations_np = np.array(observations)
    actions_np = np.array(actions)
    rewards_np = np.array(rewards)
    terminals_np = np.array(terminals)
    timeouts_np = np.array(timeouts)
    del observations, actions, rewards, terminals, timeouts
    assert terminals_np.sum() == 0
    assert (
        observations_np.shape[0]
        == actions_np.shape[0]
        == rewards_np.shape[0]
        == terminals_np.shape[0]
        == timeouts_np.shape[0]
        == dataset_cfg.num_transitions
    )
    dataset = {
        "observations": observations_np,
        "actions": actions_np,
        "rewards": rewards_np,
        "terminals": terminals_np,
        "timeouts": timeouts_np,
    }
    offline_dir = Path(dataset_cfg.base_dir)
    offline_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = env.get_dataset_path()
    with dataset_path.open("wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved dataset to {dataset_path}")
    if dataset_cfg.write_video:
        video_fname = dataset_path.parent / f"{dataset_path.stem}.gif"
        imageio.mimsave(video_fname, frames, fps=30)
        print(f"Saved video to {video_fname}")


def _register_as_gym_env() -> None:
    """Register the pin pad environment in gym."""
    for name, dt, max_vel in [["hard", 0.01, 1.0], ["medium", 0.03, 3.0]]:
        gym.envs.register(
            # Note that the "visual" version should use `modality=pixels`.
            id=f"pinpad-{name}-v0",
            entry_point="visual_pin_pad:build_env",
            # We use a fixed episode length of 10K steps for evaluation.
            max_episode_steps=10_000,
            kwargs=dict(
                overrides=[
                    "modality=state",
                    "seed=0",
                    "benchmark.debug.task.dataset.seed=0",
                    "benchmark.debug.task.pad_goal_sequence=[0, 1, 2]",
                    "benchmark.debug.task.dataset.num_transitions=1_000_000",
                    "benchmark.debug.task.dataset.p_reach_target=0.75",
                    "benchmark.debug.task.dataset.write_video=false",
                    f"benchmark.debug.task.dt={dt}",
                    f"benchmark.debug.task.max_vel={max_vel}",
                ]
            ),
        )


_register_as_gym_env()


if __name__ == "__main__":
    for dt, max_vel in [[0.01, 1.0], [0.03, 3.0]]:
        _generate_offline_dataset(
            [
                "modality=state",
                "seed=0",
                "benchmark.debug.task.dataset.seed=0",
                "benchmark.debug.task.pad_goal_sequence=[0, 1, 2]",
                "benchmark.debug.task.dataset.num_transitions=1_000_000",
                "benchmark.debug.task.dataset.p_reach_target=0.75",
                "benchmark.debug.task.dataset.write_video=false",
                f"benchmark.debug.task.dt={dt}",
                f"benchmark.debug.task.max_vel={max_vel}",
            ]
        )