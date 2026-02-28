import logging
import random
import threading
import time

logger = logging.getLogger("CanaryRouter")
logger.setLevel(logging.INFO)


class CanaryRouter:
    """
    Application-level traffic router that randomly directs a percentage of requests
    to a 'Canary' experimental model, while sending the rest to the 'Stable' model.
    Includes an automated rollback mechanism if the Canary model degrades.
    """

    def __init__(
        self,
        stable_model,
        canary_model,
        initial_split=0.0,
        max_latency_ms=500.0,
        max_error_rate=0.05,
        evaluation_window_seconds=10,
    ):
        self.stable_model = stable_model
        self.canary_model = canary_model

        # Traffic split percentage (0.0 to 1.0)
        self.traffic_split = initial_split

        # Safety thresholds
        self.max_latency_ms = max_latency_ms
        self.max_error_rate = max_error_rate

        # Metrics
        self.canary_requests = 0
        self.canary_errors = 0
        self.canary_latency_sum = 0.0

        # Rollback monitoring thread
        self.window = evaluation_window_seconds
        self._lock = threading.Lock()
        self._monitor_thread = threading.Thread(target=self._monitor_health, daemon=True)
        self._monitor_thread.start()

    def set_split(self, split: float):
        """Admin endpoint to manually adjust traffic split."""
        with self._lock:
            self.traffic_split = max(0.0, min(1.0, split))
            # Reset metrics on split change
            self.canary_requests = 0
            self.canary_errors = 0
            self.canary_latency_sum = 0.0
            logger.info(f"Canary traffic split updated to {self.traffic_split * 100}%")

    def _monitor_health(self):
        """Background thread that automatically rolls back if metrics fail bounds."""
        while True:
            time.sleep(self.window)
            with self._lock:
                if self.traffic_split > 0 and self.canary_requests > 5:
                    error_rate = self.canary_errors / self.canary_requests
                    avg_latency = (self.canary_latency_sum / self.canary_requests) * 1000

                    if error_rate > self.max_error_rate or avg_latency > self.max_latency_ms:
                        logger.warning(
                            f"CANARY TRIPPED: Error Route={error_rate:.2f}, Latency={avg_latency:.1f}ms. "
                            f"Rolling back traffic split to 0%."
                        )
                        self.traffic_split = 0.0
                        self.canary_requests = 0
                        self.canary_errors = 0
                        self.canary_latency_sum = 0.0

    def predict(self, text: str) -> dict:
        """Route prediction to either Stable or Canary model based on split."""
        use_canary = random.random() < self.traffic_split

        if not use_canary:
            return self.stable_model.predict(text)

        # Route to Canary and track metrics
        start_time = time.time()
        try:
            result = self.canary_model.predict(text)
            latency = time.time() - start_time

            with self._lock:
                self.canary_requests += 1
                self.canary_latency_sum += latency

            # Tag the result so the dashboard knows it was an experimental route
            result["_canary_route"] = True
            return result

        except Exception as e:
            # Canary failed, record error and fallback to stable
            latency = time.time() - start_time
            with self._lock:
                self.canary_requests += 1
                self.canary_latency_sum += latency
                self.canary_errors += 1

            logger.error(f"Canary model evaluation failed: {e}. Falling back to stable.")
            fallback_result = self.stable_model.predict(text)
            fallback_result["_canary_fallback"] = True
            return fallback_result
