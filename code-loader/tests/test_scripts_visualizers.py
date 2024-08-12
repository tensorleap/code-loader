import unittest

import numpy as np

from code_loader.contract.visualizer_classes import (
    LeapImage,
    LeapText,
    LeapHorizontalBar,
    LeapGraph,
    LeapImageWithBBox,
    LeapImageMask,
    LeapTextMask,
    BoundingBox
)


class TestPlotLeapVisualizer(unittest.TestCase):

    def test_plot_leap_image(self):
        data = np.random.rand(100, 100, 3)*255
        data = data.astype(np.uint8)
        visualizer = LeapImage(data=data)
        visualizer.plot_visualizer()

    def test_plot_leap_text(self):
        data = ['I', 'ate', 'a', 'banana']
        visualizer = LeapText(data=data)
        visualizer.plot_visualizer()

    def test_plot_leap_horizontal_bar(self):
        data = np.random.rand(5).astype(np.float32)
        labels = ["Class1", "Class2", "Class3", "Class4", "Class5"]
        visualizer = LeapHorizontalBar(body=data, labels=labels)
        visualizer.plot_visualizer()

    def test_plot_leap_graph(self):
        data = np.random.rand(100, 3).astype(np.float32)
        visualizer = LeapGraph(data=data)
        visualizer.plot_visualizer()

    def test_plot_leap_image_mask(self):
        image = np.random.rand(100, 100, 3).astype(np.float32)
        mask = np.random.randint(0, 2, (100, 100)).astype(np.uint8)
        labels = ["background", "object"]
        visualizer = LeapImageMask(image=image, mask=mask, labels=labels)
        visualizer.plot_visualizer()

    def test_plot_leap_image_with_boxes(self):
        data = np.random.rand(100, 100, 3).astype(np.float32)
        bounding_boxes = [BoundingBox(**{'x': 0.5, 'y': 0.5, 'width': 0.1, 'height': 0.1, 'confidence': 0.9, 'label': 'object'})]
        visualizer = LeapImageWithBBox(data=data, bounding_boxes=bounding_boxes)
        visualizer.plot_visualizer()

    def test_plot_leap_text_mask(self):
        text = ['I', 'ate', 'a', 'banana']
        mask = np.array([0, 1, 1, 0], dtype=np.uint8)
        labels = ["O", "Entity"]
        visualizer = LeapTextMask(text=text, mask=mask, labels=labels)
        visualizer.plot_visualizer()


if __name__ == '__main__':
    unittest.main()
