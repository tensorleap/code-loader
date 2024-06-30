import unittest
import numpy as np
from code_loader.contract.datasetclasses import (
    ConfusionMatrixValue,
)
from cv_ai_dl.tl.metrics.new_cm import od_confusion_matrix_metric


class TestODConfusionMatrixMetric(unittest.TestCase):
    def setUp(self):
        self.label_id_to_name = {1: "car", 2: "person"}
        self.threshold = 0.5
        self.background_label = 0

    def test_no_predictions_no_ground_truth(self):
        pred_bboxes = [np.array([], dtype=np.float32).reshape(0, 6)]
        gt_bboxes = np.array([], dtype=np.float32).reshape(1, 0, 5)

        result = od_confusion_matrix_metric(
            pred_bboxes,
            gt_bboxes,
            self.label_id_to_name,
            self.threshold,
            self.background_label,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0].label, "")
        self.assertEqual(result[0][0].expected_outcome, ConfusionMatrixValue.Positive)
        self.assertEqual(result[0][0].predicted_probability, 0.0)

    def test_no_predictions_with_ground_truth(self):
        pred_bboxes = [np.array([], dtype=np.float32).reshape(0, 6)]
        gt_bboxes = np.array([[[0, 0, 1, 1, 1], [0, 0, 2, 2, 2]]], dtype=np.float32)

        result = od_confusion_matrix_metric(
            pred_bboxes,
            gt_bboxes,
            self.label_id_to_name,
            self.threshold,
            self.background_label,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)
        for elem in result[0]:
            self.assertIn(elem.label, ["car", "person"])
            self.assertEqual(elem.expected_outcome, ConfusionMatrixValue.Positive)
            self.assertEqual(elem.predicted_probability, 0.0)

    def test_predictions_no_ground_truth(self):
        pred_bboxes = [
            np.array([[0, 0, 1, 1, 0.9, 1], [0, 0, 2, 2, 0.8, 2]], dtype=np.float32)
        ]
        gt_bboxes = np.array([], dtype=np.float32).reshape(1, 0, 5)

        result = od_confusion_matrix_metric(
            pred_bboxes,
            gt_bboxes,
            self.label_id_to_name,
            self.threshold,
            self.background_label,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)
        for elem in result[0]:
            self.assertIn(elem.label, ["car", "person"])
            self.assertEqual(elem.expected_outcome, ConfusionMatrixValue.Negative)
            self.assertTrue(
                any(np.isclose(elem.predicted_probability, prob) for prob in [0.9, 0.8])
            )

    def test_predictions_and_ground_truth(self):
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
            self.label_id_to_name,
            self.threshold,
            self.background_label,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 4)  # 2 TP, 1 FP, 1 FN

        tp_count = fp_count = fn_count = 0
        for elem in result[0]:
            if (
                elem.expected_outcome == ConfusionMatrixValue.Positive
                and not np.isclose(elem.predicted_probability, 0)
            ):
                tp_count += 1
            elif elem.expected_outcome == ConfusionMatrixValue.Negative:
                fp_count += 1
            elif (
                elem.expected_outcome == ConfusionMatrixValue.Positive
                and elem.predicted_probability == 0
            ):
                fn_count += 1

        self.assertEqual(tp_count, 2)
        self.assertEqual(fp_count, 1)
        self.assertEqual(fn_count, 1)

    def test_multiple_batches(self):
        pred_bboxes = [
            np.array([[0, 0, 1, 1, 0.9, 1]], dtype=np.float32),
            np.array([[1, 1, 2, 2, 0.8, 2]], dtype=np.float32),
        ]
        gt_bboxes = np.array([[[0, 0, 1, 1, 1]], [[1, 1, 2, 2, 2]]], dtype=np.float32)

        result = od_confusion_matrix_metric(
            pred_bboxes,
            gt_bboxes,
            self.label_id_to_name,
            self.threshold,
            self.background_label,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 1)
        self.assertEqual(result[0][0].label, "car")
        self.assertEqual(result[1][0].label, "person")

    def test_mismatched_predictions(self):
        pred_bboxes = [
            np.array([[0, 0, 1, 1, 0.9, 2]], dtype=np.float32)
        ]  # Predicted as person
        gt_bboxes = np.array(
            [[[0, 0, 1, 1, 1]]], dtype=np.float32
        )  # Ground truth is car

        result = od_confusion_matrix_metric(
            pred_bboxes,
            gt_bboxes,
            self.label_id_to_name,
            self.threshold,
            self.background_label,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)  # One FP for person, one FN for car
        self.assertEqual(result[0][0].label, "car")
        self.assertEqual(result[0][0].expected_outcome, ConfusionMatrixValue.Negative)
        self.assertEqual(result[0][1].label, "person")
        self.assertEqual(result[0][1].expected_outcome, ConfusionMatrixValue.Positive)


if __name__ == "__main__":
    unittest.main()
