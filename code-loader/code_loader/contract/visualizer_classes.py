from typing import List, Any, Union

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

import matplotlib.pyplot as plt     # type: ignore

from code_loader.contract.enums import LeapDataType
from code_loader.contract.responsedataclasses import BoundingBox


class LeapValidationError(Exception):
    pass


def validate_type(actual: Any, expected: Any, prefix_message: str = '') -> None:
    if not isinstance(expected, list):
        expected = [expected]
    if actual not in expected:
        if len(expected) == 1:
            raise LeapValidationError(
                f'{prefix_message}.\n'
                f'visualizer returned unexpected type. got {actual}, instead of {expected[0]}')
        else:
            raise LeapValidationError(
                f'{prefix_message}.\n'
                f'visualizer returned unexpected type. got {actual}, allowed is one of {expected}')


@dataclass
class LeapImage:
    """
    Visualizer representing an image for Tensorleap.

    Attributes:
    data (npt.NDArray[np.float32] | npt.NDArray[np.uint8]): The image data.
    type (LeapDataType): The data type, default is LeapDataType.Image.

    Example:
        image_data = np.random.rand(100, 100, 3).astype(np.float32)
        leap_image = LeapImage(data=image_data)
    """
    data: Union[npt.NDArray[np.float32], npt.NDArray[np.uint8]]
    type: LeapDataType = LeapDataType.Image

    def __post_init__(self) -> None:
        validate_type(self.type, LeapDataType.Image)
        validate_type(type(self.data), np.ndarray)
        validate_type(self.data.dtype, [np.uint8, np.float32])
        validate_type(len(self.data.shape), 3, 'Image must be of shape 3')
        validate_type(self.data.shape[2], [1, 3], 'Image channel must be either 3(rgb) or 1(gray)')

    def plot_visualizer(self) -> None:
        """
        Display the image contained in the LeapImage object.

        Returns:
            None

        Example:
            image_data = np.random.rand(100, 100, 3).astype(np.float32)
            leap_image = LeapImage(data=image_data)
            leap_image.plot_visualizer()
        """
        image_data = self.data

        # If the image has one channel, convert it to a 3-channel image for display
        if image_data.shape[2] == 1:
            image_data = np.repeat(image_data, 3, axis=2)

        fig, ax = plt.subplots()
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        ax.imshow(image_data)

        plt.axis('off')
        plt.title('Leap Image Visualization', color='white')
        plt.show()


@dataclass
class LeapImageWithBBox:
    """
    Visualizer representing an image with bounding boxes for Tensorleap, used for object detection tasks.

    Attributes:
    data (npt.NDArray[np.float32] | npt.NDArray[np.uint8]): The image data, shaped [H, W, 3] or [H, W, 1].
    bounding_boxes (List[BoundingBox]): List of Tensorleap bounding boxes objects in relative size to image size.
    type (LeapDataType): The data type, default is LeapDataType.ImageWithBBox.

    Example:
        image_data = np.random.rand(100, 100, 3).astype(np.float32)
        bbox = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.2, confidence=0.9, label="object")
        leap_image_with_bbox = LeapImageWithBBox(data=image_data, bounding_boxes=[bbox])
    """
    data: Union[npt.NDArray[np.float32], npt.NDArray[np.uint8]]
    bounding_boxes: List[BoundingBox]
    type: LeapDataType = LeapDataType.ImageWithBBox

    def __post_init__(self) -> None:
        validate_type(self.type, LeapDataType.ImageWithBBox)
        validate_type(type(self.data), np.ndarray)
        validate_type(self.data.dtype, [np.uint8, np.float32])
        validate_type(len(self.data.shape), 3, 'Image must be of shape 3')
        validate_type(self.data.shape[2], [1, 3], 'Image channel must be either 3(rgb) or 1(gray)')

    def plot_visualizer(self) -> None:
        """
        Plot an image with overlaid bounding boxes.

        Returns:
            None

        Example:
            image_data = np.random.rand(100, 100, 3).astype(np.float32)
            bbox = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.2, confidence=0.9, label="object")
            leap_image_with_bbox = LeapImageWithBBox(data=image_data, bounding_boxes=[bbox])
            leap_image_with_bbox.plot_visualizer()
        """

        image = self.data
        bounding_boxes = self.bounding_boxes

        # Create figure and axes
        fig, ax = plt.subplots(1)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Display the image
        ax.imshow(image)
        ax.set_title('Leap Image With BBox Visualization', color='white')

        # Draw bounding boxes on the image
        for bbox in bounding_boxes:
            x, y, width, height = bbox.x, bbox.y, bbox.width, bbox.height
            confidence, label = bbox.confidence, bbox.label

            # Convert relative coordinates to absolute coordinates
            abs_x = x * image.shape[1]
            abs_y = y * image.shape[0]
            abs_width = width * image.shape[1]
            abs_height = height * image.shape[0]

            # Create a rectangle patch
            rect = plt.Rectangle(
                (abs_x - abs_width / 2, abs_y - abs_height / 2),
                abs_width, abs_height,
                linewidth=3, edgecolor='r', facecolor='none'
            )

            # Add the rectangle to the axes
            ax.add_patch(rect)

            # Display label and confidence
            ax.text(abs_x - abs_width / 2, abs_y - abs_height / 2 - 5,
                    f"{label} {confidence:.2f}", color='r', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

        # Show the image with bounding boxes
        plt.show()

@dataclass
class LeapGraph:
    """
    Visualizer representing a line chart data for Tensorleap.

    Attributes:
    data (npt.NDArray[np.float32]): The array data, shaped [M, N] where M is the number of data points and N is the number of variables.
    type (LeapDataType): The data type, default is LeapDataType.Graph.

    Example:
        graph_data = np.random.rand(100, 3).astype(np.float32)
        leap_graph = LeapGraph(data=graph_data)
    """
    data: npt.NDArray[np.float32]
    type: LeapDataType = LeapDataType.Graph

    def __post_init__(self) -> None:
        validate_type(self.type, LeapDataType.Graph)
        validate_type(type(self.data), np.ndarray)
        validate_type(self.data.dtype, np.float32)
        validate_type(len(self.data.shape), 2, 'Graph must be of shape 2')

    def plot_visualizer(self) -> None:
        """
        Display the line chart contained in the LeapGraph object.

        Returns:
            None

        Example:
            graph_data = np.random.rand(100, 3).astype(np.float32)
            leap_graph = LeapGraph(data=graph_data)
            leap_graph.plot_visualizer()
        """
        graph_data = self.data
        num_variables = graph_data.shape[1]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Set the background color to black
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        for i in range(num_variables):
            plt.plot(graph_data[:, i], label=f'Variable {i + 1}')

        ax.set_xlabel('Data Points', color='white')
        ax.set_ylabel('Values', color='white')
        ax.set_title('Leap Graph Visualization', color='white')
        ax.legend()
        ax.grid(True, color='white')

        # Change the color of the tick labels to white
        ax.tick_params(colors='white')

        plt.show()

@dataclass
class LeapText:
    """
    Visualizer representing text data for Tensorleap.

    Attributes:
    data (List[str]): The text data, consisting of a list of text tokens. If the model requires fixed-length inputs,
    it is recommended to maintain the fixed length, using empty strings ('') instead of padding tokens ('PAD') e.g., ['I', 'ate', 'a', 'banana', '', '', '', ...]
    type (LeapDataType): The data type, default is LeapDataType.Text.

    Example:
        text_data = ['I', 'ate', 'a', 'banana', '', '', '']
        leap_text = LeapText(data=text_data)  # Create LeapText object
        LeapText(leap_text)
    """
    data: List[str]
    type: LeapDataType = LeapDataType.Text

    def __post_init__(self) -> None:
        validate_type(self.type, LeapDataType.Text)
        validate_type(type(self.data), list)
        for value in self.data:
            validate_type(type(value), str)

    def plot_visualizer(self) -> None:
        """
        Display the text contained in the LeapText object.

        Returns:
            None

        Example:
            text_data = ['I', 'ate', 'a', 'banana', '', '', '']
            leap_text = LeapText(data=text_data)
            leap_text.plot_visualizer()
        """
        text_data = self.data
        # Join the text tokens into a single string, ignoring empty strings
        display_text = ' '.join([token for token in text_data if token])

        # Create a black image using Matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Hide the axes
        ax.axis('off')

        # Set the text properties
        font_size = 20
        font_color = 'white'

        # Add the text to the image
        ax.text(0.5, 0.5, display_text, color=font_color, fontsize=font_size, ha='center', va='center')
        ax.set_title('Leap Text Visualization', color='white')

        # Display the image
        plt.show()


@dataclass
class LeapHorizontalBar:
    """
    Visualizer representing horizontal bar data for Tensorleap.
    For example, this can be used to visualize the model's prediction scores in a classification problem.

    Attributes:
    body (npt.NDArray[np.float32]): The data for the bar, shaped [C], where C is the number of data points.
    labels (List[str]): Labels for the horizontal bar; e.g., when visualizing the model's classification output, labels are the class names.
    Length of `body` should match the length of `labels`, C.
    type (LeapDataType): The data type, default is LeapDataType.HorizontalBar.

    Example:
        body_data = np.random.rand(5).astype(np.float32)
        labels = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E']
        leap_horizontal_bar = LeapHorizontalBar(body=body_data, labels=labels)
    """
    body: npt.NDArray[np.float32]
    labels: List[str]
    type: LeapDataType = LeapDataType.HorizontalBar

    def __post_init__(self) -> None:
        validate_type(self.type, LeapDataType.HorizontalBar)
        validate_type(type(self.body), np.ndarray)
        validate_type(self.body.dtype, np.float32)
        validate_type(len(self.body.shape), 1, 'HorizontalBar body must be of shape 1')

        validate_type(type(self.labels), list)
        for label in self.labels:
            validate_type(type(label), str)

    def plot_visualizer(self) -> None:
        """
        Display the horizontal bar chart contained in the LeapHorizontalBar object.

        Returns:
            None

        Example:
            body_data = np.random.rand(5).astype(np.float32)
            labels = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E']
            leap_horizontal_bar = LeapHorizontalBar(body=body_data, labels=labels)
            leap_horizontal_bar.plot_visualizer()
        """
        body_data = self.body
        labels = self.labels

        fig, ax = plt.subplots()

        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Plot horizontal bar chart
        ax.barh(labels, body_data, color='green')

        # Set the color of the labels and title to white
        ax.set_xlabel('Scores', color='white')
        ax.set_title('Leap Horizontal Bar Visualization', color='white')

        # Set the color of the ticks to white
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        plt.show()

@dataclass
class LeapImageMask:
    """
    Visualizer representing an image with a mask for Tensorleap.
    This can be used for tasks such as segmentation, and other applications where it is important to highlight specific regions within an image.

    Attributes:
    mask (npt.NDArray[np.uint8]): The mask data, shaped [H, W].
    image (npt.NDArray[np.float32] | npt.NDArray[np.uint8]): The image data, shaped [H, W, 3] or shaped [H, W, 1].
    labels (List[str]): Labels associated with the mask regions; e.g., class names for segmented objects. The length of `labels` should match the number of unique values in `mask`.
    type (LeapDataType): The data type, default is LeapDataType.ImageMask.

    Example:
        image_data = np.random.rand(100, 100, 3).astype(np.float32)
        mask_data = np.random.randint(0, 2, (100, 100)).astype(np.uint8)
        labels = ["background", "object"]
        leap_image_mask = LeapImageMask(image=image_data, mask=mask_data, labels=labels)
    """
    mask: npt.NDArray[np.uint8]
    image: Union[npt.NDArray[np.float32], npt.NDArray[np.uint8]]
    labels: List[str]
    type: LeapDataType = LeapDataType.ImageMask

    def __post_init__(self) -> None:
        validate_type(self.type, LeapDataType.ImageMask)
        validate_type(type(self.mask), np.ndarray)
        validate_type(self.mask.dtype, np.uint8)
        validate_type(len(self.mask.shape), 2, 'image mask must be of shape 2')
        validate_type(type(self.image), np.ndarray)
        validate_type(self.image.dtype, [np.uint8, np.float32])
        validate_type(len(self.image.shape), 3, 'Image must be of shape 3')
        validate_type(self.image.shape[2], [1, 3], 'Image channel must be either 3(rgb) or 1(gray)')
        validate_type(type(self.labels), list)
        for label in self.labels:
            validate_type(type(label), str)

    def plot_visualizer(self) -> None:
        """
        Plots an image with overlaid masks given a LeapImageMask visualizer object.

        Returns:
            None


        Example:
            image_data = np.random.rand(100, 100, 3).astype(np.float32)
            mask_data = np.random.randint(0, 2, (100, 100)).astype(np.uint8)
            labels = ["background", "object"]
            leap_image_mask = LeapImageMask(image=image_data, mask=mask_data, labels=labels)
            leap_image_mask.plot_visualizer()
        """

        image = self.image
        mask = self.mask
        labels = self.labels

        # Create a color map for each label
        colors = plt.cm.jet(np.linspace(0, 1, len(labels)))

        # Make a copy of the image to draw on
        overlayed_image = image.copy()

        # Iterate through the unique values in the mask (excluding 0)
        for i, label in enumerate(labels):
            # Extract binary mask for the current instance
            instance_mask = (mask == (i + 1))

            # fill the instance mask with a translucent color
            overlayed_image[instance_mask] = (
                    overlayed_image[instance_mask] * (1 - 0.5) + np.array(colors[i][:3], dtype=np.uint8) * 0.5)

        # Display the result using matplotlib
        fig, ax = plt.subplots(1)
        fig.patch.set_facecolor('black')  # Set the figure background to black
        ax.set_facecolor('black')  # Set the axis background to black

        ax.imshow(overlayed_image)
        ax.set_title('Leap Image With Mask Visualization', color='white')
        plt.axis('off')  # Hide the axis
        plt.show()


@dataclass
class LeapTextMask:
    """
    Visualizer representing text data with a mask for Tensorleap.
    This can be used for tasks such as named entity recognition (NER), sentiment analysis, and other applications where it is important to highlight specific tokens or parts of the text.

    Attributes:
    mask (npt.NDArray[np.uint8]): The mask data, shaped [L].
    text (List[str]): The text data, consisting of a list of text tokens, length of L.
    labels (List[str]): Labels associated with the masked tokens; e.g., named entities or sentiment categories. The length of `labels` should match the number of unique values in `mask`.
    type (LeapDataType): The data type, default is LeapDataType.TextMask.

    Example:
        text_data = ['I', 'ate', 'a', 'banana', '', '', '']
        mask_data = np.array([0, 0, 0, 1, 0, 0, 0]).astype(np.uint8)
        labels = ["object"]
        leap_text_mask = LeapTextMask(text=text_data, mask=mask_data, labels=labels)
        leap_text_mask.plot_visualizer()
    """
    mask: npt.NDArray[np.uint8]
    text: List[str]
    labels: List[str]
    type: LeapDataType = LeapDataType.TextMask

    def __post_init__(self) -> None:
        validate_type(self.type, LeapDataType.TextMask)
        validate_type(type(self.mask), np.ndarray)
        validate_type(self.mask.dtype, np.uint8)
        validate_type(len(self.mask.shape), 1, 'text mask must be of shape 1')
        validate_type(type(self.text), list)
        for t in self.text:
            validate_type(type(t), str)
        validate_type(type(self.labels), list)
        for label in self.labels:
            validate_type(type(label), str)

    def plot_visualizer(self) -> None:
        """
        Plots text with overlaid masks given a LeapTextMask visualizer object.

        Returns:
            None

        Example:
            text_data = ['I', 'ate', 'a', 'banana', '', '', '']
            mask_data = np.array([0, 0, 0, 1, 0, 0, 0]).astype(np.uint8)
            labels = ["object"]
            leap_text_mask = LeapTextMask(text=text_data, mask=mask_data, labels=labels)
        """

        text_data = self.text
        mask_data = self.mask
        labels = self.labels

        # Create a color map for each label
        colors = plt.cm.jet(np.linspace(0, 1, len(labels)))

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Set background to black
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.set_title('Leap Text Mask Visualization', color='white')
        ax.axis('off')

        # Set initial position
        x_pos, y_pos = 0.01, 0.5  # Adjusted initial position for better visibility

        # Display the text with colors
        for token, mask_value in zip(text_data, mask_data):
            if mask_value > 0:
                color = colors[mask_value % len(colors)]
                bbox = dict(facecolor=color, edgecolor='none',
                            boxstyle='round,pad=0.3')  # Background color for masked tokens
            else:
                bbox = None

            ax.text(x_pos, y_pos, token, fontsize=12, color='white', ha='left', va='center', bbox=bbox)

            # Update the x position for the next token
            x_pos += len(token) * 0.03 + 0.02  # Adjust the spacing between tokens

        plt.show()


@dataclass
class LeapImageWithHeatmap:
    """
    Visualizer representing an image with heatmaps for Tensorleap.
    This can be used for tasks such as highlighting important regions in an image, visualizing attention maps, and other applications where it is important to overlay heatmaps on images.

    Attributes:
    image (npt.NDArray[np.float32]): The image data, shaped [H, W, C], where C is the number of channels.
    heatmaps (npt.NDArray[np.float32]): The heatmap data, shaped [N, H, W], where N is the number of heatmaps.
    labels (List[str]): Labels associated with the heatmaps; e.g., feature names or attention regions. The length of `labels` should match the number of heatmaps, N.
    type (LeapDataType): The data type, default is LeapDataType.ImageWithHeatmap.

    Example:
        image_data = np.random.rand(100, 100, 3).astype(np.float32)
        heatmaps = np.random.rand(3, 100, 100).astype(np.float32)
        labels = ["heatmap1", "heatmap2", "heatmap3"]
        leap_image_with_heatmap = LeapImageWithHeatmap(image=image_data, heatmaps=heatmaps, labels=labels)
    """
    image: npt.NDArray[np.float32]
    heatmaps: npt.NDArray[np.float32]
    labels: List[str]
    type: LeapDataType = LeapDataType.ImageWithHeatmap

    def __post_init__(self) -> None:
        validate_type(self.type, LeapDataType.ImageWithHeatmap)
        validate_type(type(self.heatmaps), np.ndarray)
        validate_type(self.heatmaps.dtype, np.float32)
        validate_type(type(self.image), np.ndarray)
        validate_type(self.image.dtype, np.float32)
        validate_type(type(self.labels), list)
        for label in self.labels:
            validate_type(type(label), str)
        if self.heatmaps.shape[0] != len(self.labels):
            raise LeapValidationError(
                'Number of heatmaps and labels must be equal')

    def plot_visualizer(self) -> None:
        """
        Display the image with overlaid heatmaps contained in the LeapImageWithHeatmap object.

        Returns:
            None

        Example:
            image_data = np.random.rand(100, 100, 3).astype(np.float32)
            heatmaps = np.random.rand(3, 100, 100).astype(np.float32)
            labels = ["heatmap1", "heatmap2", "heatmap3"]
            leap_image_with_heatmap = LeapImageWithHeatmap(image=image_data, heatmaps=heatmaps, labels=labels)
            leap_image_with_heatmap.plot_visualizer()
        """
        image = self.image
        heatmaps = self.heatmaps
        labels = self.labels

        # Plot the base image
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('black')  # Set the figure background to black
        ax.set_facecolor('black')  # Set the axis background to black
        ax.imshow(image, cmap='gray')

        # Overlay each heatmap with some transparency
        for i in range(len(labels)):
            heatmap = heatmaps[i]
            ax.imshow(heatmap, cmap='jet', alpha=0.5)  # Adjust alpha for transparency
            ax.set_title(f'Heatmap: {labels[i]}', color='white')

            # Display a colorbar for the heatmap
            cbar = plt.colorbar(ax.imshow(heatmap, cmap='jet', alpha=0.5))
            cbar.set_label(labels[i], color='white')
            cbar.ax.yaxis.set_tick_params(color='white')  # Set color for the colorbar ticks
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')  # Set color for the colorbar labels

        plt.axis('off')
        plt.title('Leap Image With Heatmaps Visualization', color='white')
        plt.show()


map_leap_data_type_to_visualizer_class = {
    LeapDataType.Image.value: LeapImage,
    LeapDataType.Graph.value: LeapGraph,
    LeapDataType.Text.value: LeapText,
    LeapDataType.HorizontalBar.value: LeapHorizontalBar,
    LeapDataType.ImageMask.value: LeapImageMask,
    LeapDataType.TextMask.value: LeapTextMask,
    LeapDataType.ImageWithBBox.value: LeapImageWithBBox,
    LeapDataType.ImageWithHeatmap.value: LeapImageWithHeatmap
}
