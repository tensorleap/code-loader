import sys

from code_loader.contract.datasetclasses import LeapData  # type: ignore
from code_loader.helpers.visualizer.plot_functions import plot_switch


def visualize(leap_data: LeapData) -> None:
    vis_function = plot_switch.get(leap_data.type)
    if vis_function is None:
        print(f"Error: leap data type is not supported, leap data type: {leap_data.type}")
        sys.exit(1)

    vis_function(leap_data)
