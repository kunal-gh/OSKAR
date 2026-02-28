from typing import Any, Optional

import numpy as np

from src.core.cognitive_engine import CognitiveEngine


class RiskFusionEngine:
    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations
        self.cognitive_engine = CognitiveEngine()

    def calculate_risk(
        self,
        hate_score: float,
        misinfo_score: float,
        bot_score: float,
        trust_score: float,
        burst_score: float = 0.0,
        profile: Optional[Any] = None,
    ) -> dict:
        """
        Combines all module risk signals into a Monte Carlo risk distribution.

        v0.4 update: burst_score (LSTM Autoencoder) adds an additive
        amplifier when temporal coordination is detected.

        Weights:   Misinfo: 0.6, Hate: 0.4
        Modifiers: trust_modifier * bot_modifier * burst_amplifier

        Returns:
          {"mean_risk": float, "confidence_interval": [low, high], "route": str}
        """
        weights = np.array([0.6, 0.4])
        base_scores = np.array([misinfo_score, hate_score])

        # Trust modifier: trusted user lowers risk, bad actor amplifies it
        trust_modifier = max(0.1, 1.5 - trust_score)

        # GNN bot multiplier (OSKAR 2.0): bot_score 0.9 → 1.9x risk
        bot_modifier = 1.0 + bot_score

        # Burst amplifier (v0.4): temporal coordination spike → +up to 0.15
        burst_amplifier = 1.0 + (burst_score * 0.15)

        adjusted_scores = base_scores * trust_modifier * bot_modifier * burst_amplifier
        adjusted_scores = np.clip(adjusted_scores, 0.0, 1.0)

        # Monte Carlo Simulation
        simulated_risks = []
        for _ in range(self.num_simulations):
            noise = np.random.normal(0, 0.1, 2)
            sample_scores = np.clip(adjusted_scores + noise, 0.0, 1.0)
            sample_risk = float(np.sum(sample_scores * weights))
            simulated_risks.append(sample_risk)

        simulated_risks = np.array(simulated_risks)
        mean_risk = float(np.mean(simulated_risks))
        ci_low = float(np.percentile(simulated_risks, 2.5))
        ci_high = float(np.percentile(simulated_risks, 97.5))

        # --- Advanced Compliance-Aware Routing ---
        if profile:
            # EU-style hard block for bot-driven hate/misinfo
            # (If mean risk or bot score is extreme, and it's a regulated region)
            if (
                mean_risk > (profile.hate_threshold * 0.8) and bot_score > profile.bot_threshold
            ) or (mean_risk > 0.95):
                route = "hard_block"
            elif mean_risk > profile.hate_threshold:
                route = "human_review"
            elif mean_risk > (profile.hate_threshold * 0.7):
                route = "soft_warning"
            else:
                route = "auto_action"
        else:
            # Default baseline routing
            if mean_risk > 0.8:
                route = "human_review"
            elif mean_risk > 0.5:
                route = "soft_warning"
            else:
                route = "auto_action"

        return {
            "mean_risk": round(mean_risk, 4),
            "confidence_interval": [round(ci_low, 4), round(ci_high, 4)],
            "route": route,
        }
