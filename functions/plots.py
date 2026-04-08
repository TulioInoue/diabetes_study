from typing import TypedDict, Literal
from matplotlib.axes import Axes

class TextConfig(TypedDict):
    text: str
    size: float
    font: str = "arial"

def plot_bar(
    ax: Axes,
    x_values: list[str],
    y_values: list[float],
    colors: list[str],
    legends: list[str],
    title: TextConfig,
    x_label: TextConfig,
    y_label: TextConfig,
    tooltip_align: Literal["center", "edge"] = "center",
    formater: Literal["{:.0f}", "{:,.2f}", "{:.0%}", "{:,.2%}"] = "{:,.0f}",
) -> None:
    
    """
        Plots a customized bar chart on the provided Matplotlib Axes.

        Args:
            ax (Axes): The Matplotlib axes object to plot on.
            x_values (list[str]): The labels for the x-axis.
            y_values (list[float]): The heights of the bars.
            colors (list[str]): List of colors for the bars.
            legends (list[str]): Labels for the legend.
            title (TextConfig): Dictionary containing 'text', 'size', and 'font'.
            x_label (TextConfig): Configuration for the x-axis label.
            y_label (TextConfig): Configuration for the y-axis label.
            tooltip_align (str): Alignment of the bar labels.
            formater (str): Format string for the bar labels.
    """

    bar_plot = ax.bar(
        x = x_values,
        height = y_values,
        align = "center",
        color = colors,
        label = legends,
        edgecolor = "lightgray",
    )

    ax.bar_label(
        container = bar_plot,
        fmt = formater,
        label_type = tooltip_align,
    )

    ax.set_title(
        label = title.get("text", "title"),
        fontsize = title.get("size", 12),
        fontname = title.get("font", "arial"),
        fontweight = "bold",
    )

    ax.set_xlabel(
        xlabel = x_label.get("text", "x axis"),
        fontsize = x_label.get("size", 12),
        fontname = x_label.get("font", "arial"),
        fontweight = "bold",
    )

    ax.set_ylabel(
        ylabel = y_label.get("text", "y axis"),
        fontsize = y_label.get("size", 12),
        fontname = y_label.get("font", "arial"),
        fontweight = "bold"
    )

    ax.spines[["right", "left", "top"]].set_visible(False)
    ax.set_yticks(ticks = [])
