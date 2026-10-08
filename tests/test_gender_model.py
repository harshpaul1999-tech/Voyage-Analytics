"""Regression test for the persisted educational historic-label model."""

from src.inference.predict_gender import predict_gender


def test_gender_label_model_returns_a_known_training_label():
    prediction = predict_gender("Roy", 21)
    assert prediction in {"male", "female"}
