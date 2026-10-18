"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices = np.asarray(matrices, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    return np.sum(matrices @ vectors, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    matrix = np.asarray(matrix)
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    matrix = np.asarray(matrix)
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    matrix = np.asarray(matrix)
    return [np.unique(column).tolist() for column in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    generator = np.random.default_rng(seed)
    matrix = generator.normal(mean, std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def plot_histograms(statistics: MatrixStatistics) -> None:
    import matplotlib.pyplot as plt

    matrix = statistics.matrix
    rows, columns = matrix.shape

    figure, axes = plt.subplots(1, rows, figsize=(3 * rows, 3), squeeze=False)
    for i in range(rows):
        axes[0][i].hist(matrix[i, :])
        axes[0][i].set_title(f"Строка {i}")
    figure.tight_layout()

    figure, axes = plt.subplots(1, columns, figsize=(3 * columns, 3), squeeze=False)
    for j in range(columns):
        axes[0][j].hist(matrix[:, j])
        axes[0][j].set_title(f"Столбец {j}")
    figure.tight_layout()

    plt.show()


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    row_index, column_index = np.indices((rows, columns))
    return np.where((row_index + column_index) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    color_type = np.result_type(np.asarray(shape_color), np.asarray(background_color))
    image = np.full((image_height, image_width, 3), background_color, dtype=color_type)
    width = min(width, image_width)
    height = min(height, image_height)
    top = (image_height - height) // 2
    left = (image_width - width) // 2
    image[top:top + height, left:left + width] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    color_type = np.result_type(np.asarray(shape_color), np.asarray(background_color))
    image = np.full((image_height, image_width, 3), background_color, dtype=color_type)
    if semi_axis_x <= 0 or semi_axis_y <= 0:
        return image
    y, x = np.indices((image_height, image_width))
    x0 = (image_width - 1) / 2
    y0 = (image_height - 1) / 2
    inside = (x - x0) ** 2 / semi_axis_x ** 2 + (y - y0) ** 2 / semi_axis_y ** 2 <= 1
    image[inside] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values, dtype=float)
    middle = values[1:-1]
    left = values[:-2]
    right = values[2:]
    local_maxima_indices = np.where((middle > left) & (middle > right))[0] + 1
    local_minima_indices = np.where((middle < left) & (middle < right))[0] + 1
    moving_average = np.convolve(values, np.ones(window) / window, mode="valid")
    return TimeSeriesStatistics(
        mean=float(values.mean()),
        variance=float(values.var()),
        std=float(values.std()),
        local_maxima_indices=local_maxima_indices,
        local_minima_indices=local_minima_indices,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels, dtype=int)
    if class_count is None:
        class_count = labels.max() + 1
    result = np.zeros((len(labels), class_count), dtype=int)
    result[np.arange(len(labels)), labels] = 1
    return result
