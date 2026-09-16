"""Wrap a trained SB3 model as a BaseController for the shared eval harness."""
import numpy as np
from campus_env import BaseController, GreedyCtrl

try:
    from sb3_contrib import MaskablePPO, RecurrentPPO
    SB3_CONTRIB = True
except Exception:
    SB3_CONTRIB = False


class RLCtrl(BaseController):
    def __init__(self, env, model, name):
        super().__init__(env)
        self.model = model
        self.name = name
        self._is_masked = SB3_CONTRIB and isinstance(model, MaskablePPO)
        self._is_recurrent = SB3_CONTRIB and isinstance(model, RecurrentPPO)
        self._lstm_state = None
        self._ep_start = True

    def select_action(self, obs, info=None):
        if self._is_masked:
            masks = self.env.action_masks() if hasattr(self.env, "action_masks") else None
            a, _ = self.model.predict(obs, deterministic=True, action_masks=masks)
        elif self._is_recurrent:
            a, self._lstm_state = self.model.predict(
                obs, state=self._lstm_state,
                episode_start=np.array([self._ep_start]), deterministic=True)
            self._ep_start = False
        else:
            a, _ = self.model.predict(obs, deterministic=True)
        if isinstance(a, np.ndarray) and a.ndim == 0:
            return int(a)
        return a

    def evaluate(self, n_episodes=100, seed=0):
        rewards, served, curtail, critical, fair, runtime = [], [], [], [], [], []
        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=seed + ep)
            self._lstm_state = None; self._ep_start = True
            r_tot = s_tot = c_tot = cr_tot = 0.0
            import time
            t0 = time.perf_counter()
            done = False
            while not done:
                a = self.select_action(obs)
                obs, r, done, _, info = self.env.step(a)
                r_tot += r
                s_tot += info["served"]
                c_tot += info["curtailed"]
                cr_tot += info["breakdown"]["critical"]
            runtime.append(time.perf_counter() - t0)
            served_by = np.sum([h["served"] for h in self.env._history], axis=0)
            demand_by = np.sum([h["demand"] for h in self.env._history], axis=0)
            ratios = served_by / np.maximum(demand_by, 1e-9)
            from campus_env import jain
            rewards.append(r_tot); served.append(s_tot); curtail.append(c_tot)
            critical.append(cr_tot); fair.append(jain(ratios))
        return dict(reward=np.array(rewards), served=np.array(served),
                    curtail=np.array(curtail), critical=np.array(critical),
                    fairness=np.array(fair), runtime=np.array(runtime))


class HybridController(BaseController):
    name = "Hybrid-PPO+Greedy"

    def __init__(self, env, ppo_model, threshold=0.40):
        super().__init__(env)
        self.ppo = ppo_model
        self.greedy = GreedyCtrl(env)
        self.thr = threshold

    def select_action(self, obs, info=None):
        deficit = float(obs[5])
        if deficit >= self.thr:
            return self.greedy.select_action(obs)
        a, _ = self.ppo.predict(obs, deterministic=True)
        return int(a)
