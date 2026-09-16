import numpy as np


class LinUCB:
    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        alpha: float = 1.0,
    ):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha

        self.A = [
            np.eye(context_dim)
            for _ in range(n_arms)
        ]

        self.b = [
            np.zeros(context_dim)
            for _ in range(n_arms)
        ]

    def predict(self, context: np.ndarray) -> np.ndarray:
        context = np.asarray(
            context,
            dtype=float,
        )

        scores = []

        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(
                self.A[arm]
            )

            theta = A_inv @ self.b[arm]

            exploitation = (
                theta @ context
            )

            exploration = (
                self.alpha
                * np.sqrt(
                    context
                    @ A_inv
                    @ context
                )
            )

            score = (
                exploitation
                + exploration
            )

            scores.append(score)

        return np.array(scores)

    def select_arm(
        self,
        context: np.ndarray,
    ) -> int:
        scores = self.predict(context)

        return int(
            np.argmax(scores)
        )

    def update(
        self,
        arm: int,
        context: np.ndarray,
        reward: float,
    ):
        context = np.asarray(
            context,
            dtype=float,
        )

        reward = float(reward)

        self.A[arm] += np.outer(
            context,
            context,
        )

        self.b[arm] += (
            reward * context
        )

    def get_parameters(self):
        parameters = []

        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(
                self.A[arm]
            )

            theta = (
                A_inv @ self.b[arm]
            )

            parameters.append({
                "arm": arm,
                "theta": theta.copy(),
                "A": self.A[arm].copy(),
                "b": self.b[arm].copy(),
            })

        return parameters