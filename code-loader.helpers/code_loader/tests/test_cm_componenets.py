import numpy as np
import pytest

import tensorflow as tf  # type: ignore
from typing import Dict, List

from code_loader.helpers.detection.metrics.od_cm_components_metric import od_detection_metrics  # type: ignore


@pytest.fixture
def setup_params():
    label_id_to_name = {1: "class1", 2: "class2"}
    threshold = 0.5
    background_label = 0
    return label_id_to_name, threshold, background_label


def convert_results_to_numpy(results):
    return {k: v.numpy() for k, v in results.items()}


def test_empty_predictions_and_gt(setup_params):
    label_id_to_name, threshold, background_label = setup_params
    pred_bboxes = [np.empty((0, 6))]
    gt_bboxes = np.empty((1, 0, 5))
    true_metrics, false_metrics = od_detection_metrics(
        pred_bboxes,
        gt_bboxes,
        label_id_to_name,
        threshold,
        background_label,
    )
    true_metrics = convert_results_to_numpy(true_metrics)
    false_metrics = convert_results_to_numpy(false_metrics)
    assert sum(true_metrics.values()) == 0
    assert sum(false_metrics.values()) == 0


def test_no_predictions_but_gt_present(setup_params):
    label_id_to_name, threshold, background_label = setup_params
    pred_bboxes = [np.empty((0, 6))]
    gt_bboxes = np.array([[[0, 0, 1, 1, 1]]])
    true_metrics, false_metrics = od_detection_metrics(
        pred_bboxes,
        gt_bboxes,
        label_id_to_name,
        threshold,
        background_label,
    )
    true_metrics = convert_results_to_numpy(true_metrics)
    false_metrics = convert_results_to_numpy(false_metrics)
    assert true_metrics["class1_TP"][0] == 0
    assert false_metrics["class1_FN"][0] == 1


def test_predictions_but_no_gt_present(setup_params):
    label_id_to_name, threshold, background_label = setup_params
    pred_bboxes = [np.array([[0, 0, 1, 1, 0.9, 1]])]
    gt_bboxes = np.empty((1, 0, 5))
    true_metrics, false_metrics = od_detection_metrics(
        pred_bboxes,
        gt_bboxes,
        label_id_to_name,
        threshold,
        background_label,
    )
    true_metrics = convert_results_to_numpy(true_metrics)
    false_metrics = convert_results_to_numpy(false_metrics)
    assert true_metrics["class1_TP"][0] == 0
    assert false_metrics["class1_FP"][0] == 1


def test_matching_predictions_and_gt(setup_params):
    label_id_to_name, threshold, background_label = setup_params
    pred_bboxes = [np.array([[0, 0, 1, 1, 0.9, 1]])]
    gt_bboxes = np.array([[[0, 0, 1, 1, 1]]])
    true_metrics, false_metrics = od_detection_metrics(
        pred_bboxes,
        gt_bboxes,
        label_id_to_name,
        threshold,
        background_label,
    )
    true_metrics = convert_results_to_numpy(true_metrics)
    false_metrics = convert_results_to_numpy(false_metrics)
    assert true_metrics["class1_TP"][0] == 1
    assert false_metrics["class1_FP"][0] == 0
    assert false_metrics["class1_FN"][0] == 0


def test_non_matching_predictions_and_gt(setup_params):
    label_id_to_name, threshold, background_label = setup_params
    pred_bboxes = [np.array([[0.5, 0.5, 1, 1, 0.9, 1], [0, 0, 0.5, 0.5, 0.9, 2]])]
    gt_bboxes = np.array([[[0.5, 0.5, 1, 1, 2]]])
    true_metrics, false_metrics = od_detection_metrics(
        pred_bboxes,
        gt_bboxes,
        label_id_to_name,
        threshold,
        background_label,
    )
    true_metrics = convert_results_to_numpy(true_metrics)
    false_metrics = convert_results_to_numpy(false_metrics)
    assert true_metrics["class1_TP"][0] == 0
    assert true_metrics["class2_TP"][0] == 0
    assert false_metrics["class1_FP"][0] == 1
    assert false_metrics["class2_FN"][0] == 1


def test_multiple_gt_and_predictions(setup_params):
    label_id_to_name, threshold, background_label = setup_params
    pred_bboxes = [
        np.array(
            [
                [0.05, 0.05, 0.25, 0.25, 0.9, 1],
                [0.5, 0.5, 1, 1, 0.9, 2],
                [0.25, 0.25, 0.27, 0.27, 0.9, 1],
            ]
        )
    ]
    gt_bboxes = np.array(
        [
            [
                [0.05, 0.05, 0.25, 0.25, 1],
                [0.5, 0.5, 1, 1, 2],
                [0, 0, 0.01, 0.01, 2],
            ]
        ]
    )
    true_metrics, false_metrics = od_detection_metrics(
        pred_bboxes,
        gt_bboxes,
        label_id_to_name,
        threshold,
        background_label,
    )
    true_metrics = convert_results_to_numpy(true_metrics)
    false_metrics = convert_results_to_numpy(false_metrics)
    assert true_metrics["class1_TP"][0] == 1
    assert true_metrics["class2_TP"][0] == 1
    assert false_metrics["class1_FP"][0] == 1
    assert false_metrics["class2_FN"][0] == 1
