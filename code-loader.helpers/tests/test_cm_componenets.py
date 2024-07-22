import unittest
import numpy as np
import tensorflow as tf  # type: ignore
from typing import Dict, List

from code_loader.helpers.code_loader.helpers.detection.metrics.od_cm_components_metric import od_detection_metrics  # type: ignore


class TestODDetectionMetrics(unittest.TestCase):

    def setUp(self):
        self.label_id_to_name = {1: "class1", 2: "class2"}
        self.threshold = 0.5
        self.background_label = 0

    def convert_results_to_numpy(self, results):
        return {k: v.numpy() for k, v in results.items()}

    def test_empty_predictions_and_gt(self):
        pred_bboxes = [np.empty((0, 6))]
        gt_bboxes = np.empty((1, 0, 5))
        true_metrics, false_metrics = od_detection_metrics(
            pred_bboxes,
            gt_bboxes,
            self.label_id_to_name,
            self.threshold,
            self.background_label,
        )
        true_metrics = self.convert_results_to_numpy(true_metrics)
        false_metrics = self.convert_results_to_numpy(false_metrics)
        self.assertEqual(sum(true_metrics.values()), 0)
        self.assertEqual(sum(false_metrics.values()), 0)

    def test_no_predictions_but_gt_present(self):
        pred_bboxes = [np.empty((0, 6))]
        gt_bboxes = np.array([[[0, 0, 1, 1, 1]]])
        true_metrics, false_metrics = od_detection_metrics(
            pred_bboxes,
            gt_bboxes,
            self.label_id_to_name,
            self.threshold,
            self.background_label,
        )
        true_metrics = self.convert_results_to_numpy(true_metrics)
        false_metrics = self.convert_results_to_numpy(false_metrics)
        self.assertEqual(true_metrics["class1_TP"][0], 0)
        self.assertEqual(false_metrics["class1_FN"][0], 1)

    def test_predictions_but_no_gt_present(self):
        pred_bboxes = [np.array([[0, 0, 1, 1, 0.9, 1]])]
        gt_bboxes = np.empty((1, 0, 5))
        true_metrics, false_metrics = od_detection_metrics(
            pred_bboxes,
            gt_bboxes,
            self.label_id_to_name,
            self.threshold,
            self.background_label,
        )
        true_metrics = self.convert_results_to_numpy(true_metrics)
        false_metrics = self.convert_results_to_numpy(false_metrics)
        self.assertEqual(true_metrics["class1_TP"][0], 0)
        self.assertEqual(false_metrics["class1_FP"][0], 1)

    def test_matching_predictions_and_gt(self):
        pred_bboxes = [np.array([[0, 0, 1, 1, 0.9, 1]])]
        gt_bboxes = np.array([[[0, 0, 1, 1, 1]]])
        true_metrics, false_metrics = od_detection_metrics(
            pred_bboxes,
            gt_bboxes,
            self.label_id_to_name,
            self.threshold,
            self.background_label,
        )
        true_metrics = self.convert_results_to_numpy(true_metrics)
        false_metrics = self.convert_results_to_numpy(false_metrics)
        self.assertEqual(true_metrics["class1_TP"][0], 1)
        self.assertEqual(false_metrics["class1_FP"][0], 0)
        self.assertEqual(false_metrics["class1_FN"][0], 0)

    def test_non_matching_predictions_and_gt(self):
        pred_bboxes = [np.array([[0.5, 0.5, 1, 1, 0.9, 1], [0, 0, 0.5, 0.5, 0.9, 2]])]
        gt_bboxes = np.array([[[0.5, 0.5, 1, 1, 2]]])
        true_metrics, false_metrics = od_detection_metrics(
            pred_bboxes,
            gt_bboxes,
            self.label_id_to_name,
            self.threshold,
            self.background_label,
        )
        true_metrics = self.convert_results_to_numpy(true_metrics)
        false_metrics = self.convert_results_to_numpy(false_metrics)
        self.assertEqual(true_metrics["class1_TP"][0], 0)
        self.assertEqual(true_metrics["class2_TP"][0], 0)
        self.assertEqual(false_metrics["class1_FP"][0], 1)
        self.assertEqual(false_metrics["class2_FN"][0], 1)

    def test_multiple_gt_and_predictions(self):
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
            self.label_id_to_name,
            self.threshold,
            self.background_label,
        )
        true_metrics = self.convert_results_to_numpy(true_metrics)
        false_metrics = self.convert_results_to_numpy(false_metrics)
        self.assertEqual(true_metrics["class1_TP"][0], 1)
        self.assertEqual(true_metrics["class2_TP"][0], 1)
        self.assertEqual(false_metrics["class1_FP"][0], 1)
        self.assertEqual(false_metrics["class2_FN"][0], 1)


if __name__ == "__main__":
    unittest.main()
