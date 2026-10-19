import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)
import matplotlib.pyplot as plt

def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    m = np.asarray(matrices)
    v = np.asarray(vectors)
    return np.sum(np.matmul(m, v), axis=0)

def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return np.where(matrix > threshold, 1, 0)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [list(np.unique(row)) for row in matrix]



def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [list(np.unique(row)) for row in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(mean, std, (rows, columns))
    row_mean = np.mean(matrix, axis=1)
    column_mean = np.mean(matrix, axis=0)
    row_var = np.var(matrix, axis=1)
    column_var = np.var(matrix, axis=0)
    return MatrixStatistics(
        matrix = matrix,
        row_means = row_mean,
        column_means = column_mean,
        row_variances = row_var,
        column_variances = column_var
    )



def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    matrix = np.full((rows, columns), first)
    matrix[::2, 1::2] = second
    matrix[1::2, ::2] = second
    return matrix


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.full((image_height, image_width, 3), background_color)
    y0 = image_height // 2
    x0 = image_width // 2
    y_start = y0 - height // 2
    x_start = x0 - width // 2
    y_end = y_start + height
    x_end = x_start + width
    image[y_start:y_end, x_start:x_end] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.full((image_height, image_width, 3), background_color)
    y0, x0 = image_height // 2, image_width // 2
    y, x = np.ogrid[:image_height, :image_width]
    mask = ((x - x0) ** 2) / (semi_axis_x ** 2) + ((y - y0) ** 2) / (semi_axis_y ** 2) <= 1
    image[mask] = shape_color
    return image

def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    arr = np.asarray(values)
    mean = np.mean(arr)
    var = np.var(arr)
    std = np.std(arr)
    center = arr[1:-1]
    left = arr[:-2]
    right = arr[2:]
    max_mask = (center > left) & (center > right)
    min_mask = (center < right) & (center < left)
    local_max = np.where(max_mask)[0] + 1
    local_min = np.where(min_mask)[0] + 1
    kernel = np.divide(np.ones(window), window)
    moving_avg = np.convolve(arr, kernel, mode="valid")
    return TimeSeriesStatistics(
        mean = mean,
        variance = var,
        std = std,
        local_maxima_indices = local_max,
        local_minima_indices = local_min,
        moving_average = moving_avg
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    arr_labels = np.asarray(labels)
    if class_count is None:
        class_count = np.max(arr_labels) + 1
    num_labels = len(arr_labels)
    result = np.zeros((num_labels, class_count))
    result[np.arange(num_labels), arr_labels] = 1
    return result


def plot_statistic_histograms(stats: MatrixStatistics) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].hist(stats.row_means, bins=20)
    axes[0].set_title("Распределение средних значений строк")
    axes[0].set_xlabel("Среднее значение")
    axes[0].set_ylabel("Частота")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[1].hist(stats.column_means, bins=20)
    axes[1].set_title("Распределение средних значений столбцов")
    axes[1].set_xlabel("Среднее значение")
    axes[1].set_ylabel("Частота")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

def test_plot_statistic_histograms():
    data = RandomMatrixInput(
        rows=5,
        columns=5,
        mean=0,
        std=1,
        seed=42
    )

    stats = matrix_statistics(data)

    plot_statistic_histograms(stats)