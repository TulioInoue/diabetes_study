from typing import TypedDict, Literal
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from seaborn import heatmap
import polars as pl
from numpy import triu, ones_like, linspace
from scipy.stats import gaussian_kde

# Setting cmap color:
my_cmap = LinearSegmentedColormap.from_list("my_custom_red_green_cmap", ["red", "pink", "lightgray", "lime", "green"])

class TextConfig(TypedDict):
    text: str
    size: float
    font: str = "arial"
    pady: float = 0

class PieConfig(TypedDict):
    inner_radius: float
    outer_radius: float
    tooltip_fmt: str

class TooltipConfig(TypedDict):
    show: bool
    rotation: Literal["vertical", "horizontal"] = "horizontal"
    verticalalignment: Literal["center", "top", "bottom"] = "center"
    horizontalalignment: Literal["center", "left", "right"] = "center"

class KDEConfig(TypedDict):
    show: bool
    color: str

def plot_pie(
    ax: Axes,
    values: list[float],
    labels: list[str],
    colors: list[str],
    title: TextConfig,
    style: PieConfig = {},
) -> None:
    
    """
        Plots a customized pie chart on the provided Matplotlib Axes.

        Args:
            ax (Axes): The Matplotlib axes object to plot on.
            values (list[float]): values for each category.
            labels (list[str]): Labels related to the data.
            colors (list[str]): List of colors for the pie slices.
            title (TextConfig): Dictionary containing 'text', 'size', and 'font'.
            style (str): Dictionary containing 'inner_radius', 'outer_radius', and 'tooltip_fmt'.
    """
    
    ax.pie(
        x = values,
        autopct = style.get("tooltip_fmt", "%.2f%%"),
        labels = labels,
        colors = colors,
        radius = style.get("outer_radius", 1),
    )

    ax.pie(
        x = [1],
        colors = ["white"],
        radius = style.get("inner_radius", 0.25)
    )

    ax.set_title(
        label = title.get("text", "title"),
        fontsize = title.get("size", 12),
        fontname = title.get("font", "arial"),
        fontweight = "bold",
        pad = title.get("pady", 0),
    )


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
    formater: Literal["{:,.0f}", "{:,.2f}", "{:,.0%}", "{:,.2%}"] = "{:,.0f}",
    width: float = 0.5,
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
        width = width
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
        pad = title.get("pady", 0),
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


def plot_heatmap(
    ax: Axes,
    corr_dataset: pl.DataFrame,
    title: TextConfig,
    annot: bool = True,
    formater: Literal[",.0f", ",.2f", ",.0%", ",.2%"] = ",.2f",
    tooltip_size: float = 10,
    cmap: LinearSegmentedColormap = my_cmap,
) -> None:
    
    """
        Plots a customized heatmap chart on the provided Matplotlib Axes.

        Args:
            ax (Axes): The Matplotlib axes object to plot on.
            corr_dataset (pl.DataFrame): Correlation dataframe calculated by `.corr()` method.
            title (TextConfig): Dictionary containing 'text', 'size', and 'font'.
            annot (bool): If heatmap chart has or not tooltips.
            formater (str): Format string for the tooltips.
            tooltip_size (float): Size of tooltip.
            cmap (LinearSegmentedColormap): cmap created by you (optional).
    """

    heatmap(
        data = corr_dataset,
        ax = ax,
        xticklabels = corr_dataset.columns,
        yticklabels = corr_dataset.columns,
        annot = annot,
        fmt = formater,
        cmap = cmap,
        annot_kws = {"size": tooltip_size},
        linewidths = .5,
        mask = triu(
            ones_like(
                a = corr_dataset,
                dtype = bool
            )
        ),
    )

    ax.set_title(
        label = title.get("text", "title"),
        fontsize = title.get("size", 12),
        fontname = title.get("font", "arial"),
        fontweight = "bold",
        pad = title.get("pady", 0),
    )

    ax.set_xticklabels(ax.get_xticklabels(), fontweight = 'bold', rotation = 50, ha = "right")
    ax.set_yticklabels(ax.get_xticklabels(), fontweight = 'bold', rotation = 0)

def plot_histogram(
    ax: Axes,
    values: list[float],
    title: TextConfig,
    color: str,
    kde: KDEConfig = {
        "show": True,
        "color": "black"
    },
    tooltip: TooltipConfig = {
        "show": True,
        "rotation": "horizontal",
        "verticalalignment": "center",
        "horizontalalignment": "center",
    },
    xy_delta: tuple[float, float] = (0, 0),
    bins: list[float] | Literal["auto", "sturges", "sqrt", "doane", "scott", "stone", "rice", "fd"] = "auto",
) -> None:
    
    """
        Plots a customized histogram chart on the provided Matplotlib Axes.

        Args:
            ax (Axes): The Matplotlib axes object to plot on.
            values (list[float]): The labels for the x-axis.
            bins (list | Literal): bins of histogram.
            tooltip (TooltipConfig): Dictionary containing show, rotation, verticalalignment, horizontalalignment of the tooltip.
            title (TextConfig): Dictionary containing 'text', 'size', and 'font'.
            xy_delta (tuple[float, float]): Text coordinates (x, y). 
    """
    
    totals, bins, _ = ax.hist(
        x = values,
        bins = bins,
        color = color
    )

    if tooltip.get("show", True):

        gap = 0 if tooltip.get("rotation", "horizontal") == "horizontal" else (bins[1] - bins[0])/2

        for total, bin in zip(totals, bins):
            ax.annotate(
                text = f"{total:,.0f}",
                xy = (bin + gap + xy_delta[0], total + xy_delta[1]),
                verticalalignment = tooltip.get("verticalalignment", "center"),
                horizontalalignment = tooltip.get("horizontalalignment", "center"),
                rotation = 0 if tooltip.get("rotation", "horizontal") == "horizontal" else 90
            )

    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.set_yticks(ticks = [])

    if kde.get("show", True):
        ax2 = ax.twinx()

        kde_ = gaussian_kde(values)
        dist_space = linspace(min(values), max(values), 100)
        ax2.plot(
            dist_space, kde_(dist_space), color = kde.get("color", "black")
        )

        ax2.spines[["top", "left", "right"]].set_visible(False)
        ax2.set_yticks(ticks = [])

    ax.set_title(
        label = title.get("text", "title"),
        fontsize = title.get("size", 12),
        fontname = title.get("font", "arial"),
        fontweight = "bold",
        pad = title.get("pady", 0),
    )

def plot_violin(
    ax: Axes,
    data: list[float],
    colors: list[str] | None,
    title: TextConfig
) -> None: 
    
    parts = ax.violinplot(
        dataset = data,
        showmeans = True,
        showextrema = True,
        side = "both",
    )

    parts["cmaxes"].set_color("gray")
    parts["cmins"].set_color("gray")
    parts["cbars"].set_color("gray")

    if colors != None:
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        label = title.get("text", "title"),
        fontsize = title.get("size", 12),
        fontname = title.get("font", "arial"),
        fontweight = "bold",
        pad = title.get("pady", 0),
    )

