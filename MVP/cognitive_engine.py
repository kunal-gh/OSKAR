import math
import numpy as np

class CognitiveEngine:
    def __init__(self, temperature=1.5):
        """
        Temperature scaling parameter T.
        In a full training loop, this is fit on a validation set using NLL loss.
        """
        self.temperature = temperature

    def apply_temperature_scaling(self, logits):
        """
        Apply temperature scaling to raw logits and return softmax probabilities.
        """
        logits_tensor = np.array(logits)
        scaled_logits = logits_tensor / self.temperature
        
        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits)) # numeric stability
        probs = exp_logits / np.sum(exp_logits)
        
        return probs.tolist()

    def compute_entropy(self, probs):
        """
        Compute Shannon entropy of a probability distribution.
        """
        return -sum(p * math.log(p) if p > 0 else 0 for p in probs)

    def entropy_router(self, entropy_val: float) -> str:
        """
        Route the item based on uncertainty (entropy).
        """
        if entropy_val > 0.8:
            return "human_review"
        elif entropy_val < 0.6:
            return "auto_action"
        else:
            return "soft_warning"
