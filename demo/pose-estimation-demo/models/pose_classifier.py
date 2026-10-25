
"""Paper-faithful coarse pose classification network."""

from __future__ import annotations

import torch
from torch import nn

from estimators.interfaces import ANN_INPUT_DIM, ANN_NUM_CLASSES


class PoseClassifier(nn.Module):
    """42 -> 100 -> 100 -> 4800 multilayer perceptron."""

    def __init__(
        self,
        input_dim: int = ANN_INPUT_DIM,
        hidden_dims: tuple[int, int] = (100, 100),
        num_classes: int = ANN_NUM_CLASSES,
    ):
        super().__init__()

        if input_dim <= 0 or num_classes <= 1:
            raise ValueError("Invalid network dimensions")

        if len(hidden_dims) != 2 or any(
            value <= 0 for value in hidden_dims
        ):
            raise ValueError("Exactly two positive hidden sizes required")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = tuple(hidden_dims)

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, "
                f"got {features.shape[-1]}"
            )
        return self.network(features)

    @torch.inference_mode()
    def predict_proba(
        self,
        features: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        was_training = self.training
        self.eval()
        try:
            logits = self(features)
            return torch.softmax(logits / temperature, dim=-1)
        finally:
            self.train(was_training)

    @torch.inference_mode()
    def predict(
        self,
        features: torch.Tensor,
        temperature: float = 1.0,
    ):
        probabilities = self.predict_proba(features, temperature)
        confidence, labels = probabilities.max(dim=-1)
        return labels, confidence


def create_paper_classifier() -> PoseClassifier:
    return PoseClassifier()


def classifier_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
