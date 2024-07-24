import numpy as np
import pytest
from code_loader.contract.datasetclasses import ConfusionMatrixValue  # type: ignore
from code_loader.helpers.detection.metrics.object_detection_confusion_matrix_metric import od_confusion_matrix_metric  # type: ignore


@pytest.fixture
def setup_params():
    label_id_to_name = {1: "car", 2: "person"}
    threshold = 0.5
    background_label = 0
    return label_id_to_name, threshold, background_label


def test_no_predictions_no_ground_truth(setup_params):
    label_id_to_name, threshold, background_label = setup_params
    pred_bboxes = [np.array([], dtype=np.float32).reshape(0, 6)]
    gt_bboxes = np.array([], dtype=np.float32).reshape(1, 0, 5)

    result = od_confusion_matrix_metric(
        pred_bboxes,
        gt_bboxes,
        label_id_to_name,
        threshold,
        background_label,
    )

    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0].label == ""
    assert result[0][0].expected_outcome == ConfusionMatrixValue.Positive
    assert result[0][0].predicted_probability == 0.0


def test_no_predictions_with_ground_truth(setup_params):
    label_id_to_name, threshold, background_label = setup_params
    pred_bboxes = [np.array([], dtype=np.float32).reshape(0, 6)]
    gt_bboxes = np.array([[[0, 0, 1, 1, 1], [0, 0, 2, 2, 2]]], dtype=np.float32)

    result = od_confusion_matrix_metric(
        pred_bboxes,
        gt_bboxes,
        label_id_to_name,
        threshold,
        background_label,
    )

    assert len(result) == 1
    assert len(result[0]) == 2
    for elem in result[0]:
        assert elem.label in ["car", "person"]
        assert elem.expected_outcome == ConfusionMatrixValue.Positive
        assert elem.predicted_probability == 0.0


def test_predictions_no_ground_truth(setup_params):
    label_id_to_name, threshold, background_label = setup_params
    pred_bboxes = [
        np.array([[0, 0, 1, 1, 0.9, 1], [0, 0, 2, 2, 0.8, 2]], dtype=np.float32)
    ]
    gt_bboxes = np.array([], dtype=np.float32).reshape(1, 0, 5)

    result = od_confusion_matrix_metric(
        pred_bboxes,
        gt_bboxes,
        label_id_to_name,
        threshold,
        background_label,
    )

    assert len(result) == 1
    assert len(result[0]) == 2
    for elem in result[0]:
        assert elem.label in ["car", "person"]
        assert elem.expected_outcome == ConfusionMatrixValue.Negative
        assert any(np.isclose(elem.predicted_probability, prob) for prob in [0.9, 0.8])


def test_predictions_and_ground_truth(setup_params):
    label_id_to_name, threshold, background_label = setup_params
    pred_bboxes = [
        np.array(
            [[0, 0, 1, 1, 0.9, 1], [1, 1, 2, 2, 0.8, 2], [2, 2, 3, 3, 0.7, 1]],
            dtype=np.float32,
        )
    ]
    gt_bboxes = np.array(
        [[[0, 0, 1, 1, 1], [1, 1, 2, 2, 2], [3, 3, 4, 4, 1]]], dtype=np.float32
    )

    result = od_confusion_matrix_metric(
        pred_bboxes,
        gt_bboxes,
        label_id_to_name,
        threshold,
        background_label,
    )

    assert len(result) == 1
    assert len(result[0]) == 4  # 2 TP, 1 FP, 1 FN

    tp_count = fp_count = fn_count = 0
    for elem in result[0]:
        if elem.expected_outcome == ConfusionMatrixValue.Positive and not np.isclose(
            elem.predicted_probability, 0
        ):
            tp_count += 1
        elif elem.expected_outcome == ConfusionMatrixValue.Negative:
            fp_count += 1
        elif (
            elem.expected_outcome == ConfusionMatrixValue.Positive
            and elem.predicted_probability == 0
        ):
            fn_count += 1

    assert tp_count == 2
    assert fp_count == 1
    assert fn_count == 1


def test_multiple_batches(setup_params):
    label_id_to_name, threshold, background_label = setup_params
    pred_bboxes = [
        np.array([[0, 0, 1, 1, 0.9, 1]], dtype=np.float32),
        np.array([[1, 1, 2, 2, 0.8, 2]], dtype=np.float32),
    ]
    gt_bboxes = np.array([[[0, 0, 1, 1, 1]], [[1, 1, 2, 2, 2]]], dtype=np.float32)

    result = od_confusion_matrix_metric(
        pred_bboxes,
        gt_bboxes,
        label_id_to_name,
        threshold,
        background_label,
    )

    assert len(result) == 2
    assert len(result[0]) == 1
    assert len(result[1]) == 1
    assert result[0][0].label == "car"
    assert result[1][0].label == "person"


def test_mismatched_predictions(setup_params):
    label_id_to_name, threshold, background_label = setup_params
    pred_bboxes = [
        np.array([[0, 0, 1, 1, 0.9, 2]], dtype=np.float32)
    ]  # Predicted as person
    gt_bboxes = np.array([[[0, 0, 1, 1, 1]]], dtype=np.float32)  # Ground truth is car

    result = od_confusion_matrix_metric(
        pred_bboxes,
        gt_bboxes,
        label_id_to_name,
        threshold,
        background_label,
    )

    assert len(result) == 1
    assert len(result[0]) == 2  # One FP for person, one FN for car
    assert result[0][0].label == "car"
    assert result[0][0].expected_outcome == ConfusionMatrixValue.Negative
    assert result[0][1].label == "person"
    assert result[0][1].expected_outcome == ConfusionMatrixValue.Positive
