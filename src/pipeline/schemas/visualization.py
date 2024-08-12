from dataclasses import dataclass


@dataclass
class Padding:
    """
    Padding for plot.

    Attributes
    ----
    tpad: float, the top padding of the plot.
    lpad: float, the left padding of the plot.
    bpad: float, the bottom padding of the plot.
    """

    tpad: float = 2.5
    lpad: float = 0.1
    bpad: float = 0.12


@dataclass
class Labels:
    """
    Labels for plot.

    Attributes
    ---
    title: str, the title of the plot.
    xlabel: str, the x-axis label of the plot.
    ylabel: str, the y-axis label of the plot.
    """

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""


@dataclass
class Line:
    """
    Line for plot.

    Attributes
    ---
    left_bottom: tuple[float, float], the left bottom point of the line.
    right_top: tuple[float, float], the right top point of the line.
    color: str, the color of the line.
    linestyle: str, the linestyle of the line.
    label: str, the label of the line.
    """

    left_bottom: tuple[float, float]
    right_top: tuple[float, float]
    color: str = "blue"
    linestyle: str = "--"
    label: str = ""
