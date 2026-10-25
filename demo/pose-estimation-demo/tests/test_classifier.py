
"""Consolidated Phase 3 ANN tests.

Run:
    python -m pytest tests/test_classifier.py -v
"""

import numpy as np
import pytest
import torch

from models.pose_classifier import (
    PoseClassifier,
    create_paper_classifier,
    classifier_parameter_count,
)
from training.train_classifier import (
    make_loader,
    train_model,
    load_classifier,
    resolve_device,
)
from training.calibrate_classifier import (
    calculate_error_samples,
    calibration_statistics,
    sample_error_proposals,
)
from training.validate_models import evaluate_classifier


@pytest.fixture
def tiny_dataset():
    rng = np.random.default_rng(42)

    X = rng.random((32, 42)).astype(np.float32)

    # Ensure visibility channels are valid binary masks.
    X.reshape(-1, 14, 3)[:, :, 2] = 1.0

    y = np.array(
        [100, 200, 300, 400] * 8,
        dtype=np.int64,
    )

    return {
        "X": X,
        "y": y,
        "pose_4d": np.tile(
            np.array([
                [35.0, 0.0, 0.0, 0.0],
                [35.0, 5.0, 0.0, 0.0],
                [30.0, 0.0, 0.0, 0.0],
                [30.0, 5.0, 0.0, 0.0],
            ]),
            (8, 1),
        ),
        "trajectory_id": np.repeat(
            np.arange(4), 8
        ),
    }


def test_paper_architecture():
    model = create_paper_classifier()

    assert model.input_dim == 42
    assert model.hidden_dims == (100, 100)
    assert model.num_classes == 4800

    assert classifier_parameter_count(model) == 499200


def test_forward_pass():
    model = create_paper_classifier()

    features = torch.randn(8, 42)
    logits = model(features)

    assert logits.shape == (8, 4800)
    assert torch.isfinite(logits).all()


def test_probability_distribution():
    model = create_paper_classifier()

    features = torch.randn(4, 42)

    probabilities = model.predict_proba(features)

    assert probabilities.shape == (4, 4800)

    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones(4),
    )

    assert torch.all(probabilities >= 0)


def test_prediction_interface():
    model = create_paper_classifier()

    labels, confidence = model.predict(
        torch.randn(5, 42)
    )

    assert labels.shape == (5,)
    assert confidence.shape == (5,)

    assert torch.all(labels >= 0)
    assert torch.all(labels < 4800)

    assert torch.all(confidence >= 0)
    assert torch.all(confidence <= 1)


def test_gradient_backpropagation():
    model = create_paper_classifier()

    features = torch.randn(6, 42)
    targets = torch.tensor(
        [0, 100, 200, 300, 400, 4799]
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )

    optimizer.zero_grad()

    logits = model(features)

    loss = torch.nn.functional.cross_entropy(
        logits, targets
    )

    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    optimizer.step()

    assert torch.isfinite(loss)


def test_checkpoint_round_trip(tmp_path, tiny_dataset):
    torch.manual_seed(42)

    model = create_paper_classifier()

    train_loader = make_loader(
        tiny_dataset, batch_size=8,
        shuffle=True, seed=42,
    )

    val_loader = make_loader(
        tiny_dataset, batch_size=8,
        shuffle=False,
    )

    path = tmp_path / "classifier.pt"

    report = train_model(
        model,
        train_loader,
        val_loader,
        epochs=2,
        learning_rate=0.001,
        weight_decay=0.0,
        patience=2,
        device=torch.device("cpu"),
        checkpoint_path=path,
        seed=42,
    )

    assert path.exists()
    assert report["best_epoch"] >= 1

    restored, checkpoint = load_classifier(
        path, torch.device("cpu")
    )

    model.eval()

    inputs = torch.from_numpy(
        tiny_dataset["X"][:4]
    )

    with torch.inference_mode():
        original = model(inputs)
        reloaded = restored(inputs)

    # Compare against the saved checkpoint, not necessarily
    # the last epoch, because early checkpointing selects
    # the best validation epoch.
    saved_model = create_paper_classifier()
    saved_model.load_state_dict(checkpoint["state_dict"])
    saved_model.eval()

    with torch.inference_mode():
        saved_logits = saved_model(inputs)

    torch.testing.assert_close(
        reloaded, saved_logits
    )

    assert original.shape == reloaded.shape


def test_classifier_error_calibration():
    truth = np.array([
        [25.0, 0.0, 0.0, 0.0],
        [30.0, 5.0, 0.0, 0.1],
        [35.0, -5.0, 2.0, -0.1],
    ])

    labels = np.array([100, 200, 300])

    errors = calculate_error_samples(
        truth, labels
    )

    assert errors.shape == (3, 4)
    assert np.all(np.isfinite(errors))

    stats = calibration_statistics(errors)

    assert stats["sample_count"] == 3
    assert stats["position_error_mean_m"] >= 0


def test_empirical_error_sampling():
    errors = np.array([
        [1.0, 0.0, 0.0, 0.1],
        [-1.0, 2.0, 0.0, -0.1],
    ])

    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)

    samples_a = sample_error_proposals(
        errors, 100, rng_a
    )

    samples_b = sample_error_proposals(
        errors, 100, rng_b
    )

    assert samples_a.shape == (100, 4)

    np.testing.assert_array_equal(
        samples_a, samples_b
    )


def test_evaluation_metrics(tiny_dataset):
    model = create_paper_classifier()

    metrics = evaluate_classifier(
        model,
        tiny_dataset,
        device=torch.device("cpu"),
        batch_size=8,
    )

    assert metrics["sample_count"] == 32

    assert 0 <= metrics["exact_accuracy"] <= 1
    assert 0 <= metrics["top5_accuracy"] <= 1
    assert 0 <= metrics["top10_accuracy"] <= 1

    assert (
        metrics["top10_accuracy"]
        >= metrics["top5_accuracy"]
        >= metrics["exact_accuracy"]
    )

    assert metrics["mean_position_error_m"] >= 0
    assert metrics["mean_roll_error_deg"] >= 0


def test_invalid_input_dimension():
    model = create_paper_classifier()

    with pytest.raises(ValueError):
        model(torch.randn(2, 28))


def test_device_resolution():
    device = resolve_device("cpu")

    assert device.type == "cpu"
