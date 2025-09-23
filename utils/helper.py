#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/22 22:09
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   helper.py
# @Desc     :

from numpy import meshgrid, linspace, c_
from pandas import DataFrame, concat
from plotly.express import scatter, scatter_3d, line
from plotly.graph_objects import Contour
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.callbacks import Callback
from time import perf_counter
from typing import override


class Timer(object):
    """ timing code blocks using a context manager """

    def __init__(self, description: str = None, precision: int = 5):
        """ Initialise the Timer class
        :param description: the description of a timer
        :param precision: the number of decimal places to round the elapsed time
        """
        self._description: str = description
        self._precision: int = precision
        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self):
        """ Start the timer """
        self._start = perf_counter()
        print("-" * 50)
        print(f"{self._description} has started.")
        print("-" * 50)
        return self

    def __exit__(self, *args):
        """ Stop the timer and calculate the elapsed time """
        self._end = perf_counter()
        self._elapsed = self._end - self._start

    def __repr__(self):
        """ Return a string representation of the timer """
        if self._elapsed != 0.0:
            # print("-" * 50)
            return f"{self._description} took {self._elapsed:.{self._precision}f} seconds."
        return f"{self._description} has NOT started."


def scatter_visualiser(data: DataFrame, categories: DataFrame = None, dims: int = 1):
    """ Visualise the data using scatter plots.
    :param data: the DataFrame containing the data
    :param categories: the DataFrame containing the categories for colouring and symbolising the data points
    :param dims: number of dimensions to reduce to if data has more than 3 dimensions (2 or 3)
    :return: a scatter plot with different colours and symbols for each category
    """
    if categories is not None:
        df = concat([data, categories], axis=1)
        category_name = categories.columns[0]
    else:
        df = data
        category_name = None
        print(category_name)

    fig = None

    match dims:
        case 1:
            dimensions = data.shape[1]
            if dimensions == 2:
                fig = scatter(
                    df,
                    x=data.columns[0],
                    y=data.columns[1],
                    color=category_name,
                    symbol=category_name,
                    hover_data=[data.columns[0], data.columns[1], category_name]
                ).update_layout(coloraxis_showscale=False)
            else:
                fig = scatter_3d(
                    df,
                    x=data.columns[0],
                    y=data.columns[1],
                    z=data.columns[2],
                    color=category_name,
                    symbol=category_name,
                    hover_data=[data.columns[0], data.columns[1], data.columns[2], category_name]
                ).update_layout(coloraxis_showscale=False)
        case 2:
            pca = PCA(n_components=2)
            components = pca.fit_transform(data)
            df = DataFrame(components, columns=["PAC-X", "PAC-Y"])
            fig = scatter(
                df,
                x="PAC-X",
                y="PAC-Y",
                color=category_name,
                symbol=category_name,
                hover_data=["PAC-X", "PAC-Y"] + ([category_name] if category_name else [])
            ).update_layout(coloraxis_showscale=False)
        case 3:
            pca = PCA(n_components=3)
            components = pca.fit_transform(data)
            df = DataFrame(components, columns=["PAC-X", "PAC-Y", "PAC-Z"])
            fig = scatter_3d(
                df,
                x="PAC-X",
                y="PAC-Y",
                z="PAC-Z",
                color=category_name,
                symbol=category_name,
                hover_data=["PAC-X", "PAC-Y", "PAC-Z"] + ([category_name] if category_name else [])
            ).update_layout(coloraxis_showscale=False)
        case _:
            raise ValueError("dims must be 1, 2, or 3")

    return fig


def decision_boundary_adder(fig, model, X: DataFrame, pad_ratio: float = 0.05):
    """ Add decision boundary to a scatter plot.
    :param fig: the scatter plot figure
    :param model: the trained model
    :param X: the DataFrame containing the features used for training
    :param pad_ratio: the ratio to pad the decision boundary
    :return: the scatter plot figure with the decision boundary added
    """
    x_range = X.iloc[:, 0].max() - X.iloc[:, 0].min()
    y_range = X.iloc[:, 1].max() - X.iloc[:, 1].min()

    x_min = X.iloc[:, 0].min() - pad_ratio * x_range
    x_max = X.iloc[:, 0].max() + pad_ratio * x_range
    y_min = X.iloc[:, 1].min() - pad_ratio * y_range
    y_max = X.iloc[:, 1].max() + pad_ratio * y_range

    x = linspace(x_min, x_max, 100)
    y = linspace(y_min, y_max, 100)
    xx, yy = meshgrid(x, y)

    Z = model.predict(c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig.add_trace(Contour(
        x=x,
        y=y,
        z=Z,
        showscale=False,
        opacity=0.3,
        colorscale=["rgba(0,0,255,0.3)", "rgba(255,0,0,0.3)"],
        contours=dict(showlines=False),
        name="Decision Boundary"
    ))
    return fig


class TFKerasLogger(Callback):
    """ Custom Keras Callback to log training metrics and update Streamlit placeholders.
    :param num_placeholders: a dictionary of Streamlit placeholders for metrics
    :return: None
    """

    def __init__(self, num_placeholders: dict = None):
        super().__init__()
        # The key name must match the callback logs
        self._history = {k: [] for k in [
            "loss", "accuracy", "precision", "recall", "auc",
            "val_loss", "val_accuracy", "val_precision", "val_recall", "val_auc"
        ]}
        self._placeholders = num_placeholders

    @override
    def on_epoch_end(self, epoch, logs=None):
        """ At the end of each epoch, log the metrics and update the placeholders.
        :param epoch: the current epoch number
        :param logs: the logs dictionary containing the metrics
        :return: None
        """
        logs = logs or {}
        # Save the training history per epoch
        for key in self._history.keys():
            self._history[key].append(logs.get(key, None))
        # Update the placeholders with the latest metrics
        if self._placeholders:
            for key, placeholder in self._placeholders.items():
                if key in logs and placeholder is not None:
                    placeholder.metric(
                        label=f"Epoch {epoch + 1}: {key.replace("val_", "Valid ").capitalize()}",
                        value=f"{logs[key]:.4f}"
                    )

    def get_history(self):
        """ Get the training history."""
        return self._history


def binary_roc_plotter(y_true, y_pred):
    """ Plot the ROC curve for a binary classification problem.
    :param y_true: true labels (0 or 1)
    :param y_pred: predicted probabilities for the positive class
    :return: fig (ROC curve), area_under_curve (AUC value)
    """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_pred)
    area_under_the_roc_curve = auc(false_positive_rate, true_positive_rate)  # [0, 1]
    fig = line(
        x=false_positive_rate,
        y=true_positive_rate,
        text=[f"AUC = {area_under_the_roc_curve:.3f}"] * len(false_positive_rate),
        labels={"x": "False Positive Rate", "y": "True Positive Rate"}
    )
    return fig, area_under_the_roc_curve
