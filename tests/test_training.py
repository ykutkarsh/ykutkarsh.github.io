from aab.learning.training_pipeline import evaluate_xor_model, train_xor_model


def test_xor_model_learns_measurable_task():
    model, _ = train_xor_model()
    accuracy, loss = evaluate_xor_model(model)

    assert accuracy == 1.0
    assert loss < 0.1
