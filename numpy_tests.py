import numpy as np

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixVectorBatchInput,
    OneHotInput, RandomMatrixInput, RectangleInput, TimeSeriesInput,
)
from numpy_tasks import (
    analyze_time_series,
    binarize,
    chess,
    draw_ellipse,
    draw_rectangle,
    matrix_statistics,
    one_hot,
    sum_prod,
    unique_columns,
    unique_rows,
)

# Задача 1. Сумма произведений матриц на векторы
def test_sum_prod():
    result = sum_prod(MatrixVectorBatchInput(matrices=[np.eye(2), 2 * np.eye(2)], vectors=[[[1], [2]], [[3], [4]]]))
    assert result.shape == (2, 1)
    assert result.tolist() == [[7], [10]]
    assert sum_prod(MatrixVectorBatchInput(matrices=[[[5]]], vectors=[[[3]]])).tolist() == [[15]]


# Задача 2. Бинаризация матрицы
def test_binarize():
    matrix = np.array([[1, 5], [3, 7]])
    assert binarize(BinarizeInput(matrix=matrix, threshold=3)).tolist() == [[0, 1], [0, 1]]
    assert matrix.tolist() == [[1, 5], [3, 7]]
    assert binarize(BinarizeInput(matrix=matrix, threshold=100)).tolist() == [[0, 0], [0, 0]]


# Задача 3. Уникальные элементы строк и столбцов
def test_unique_rows_and_columns():
    matrix = [[1, 1, 2], [3, 4, 3]]
    assert unique_rows(MatrixInput(matrix=matrix)) == [[1, 2], [3, 4]]
    assert unique_columns(MatrixInput(matrix=matrix)) == [[1, 3], [1, 4], [2, 3]]
    assert unique_rows(MatrixInput(matrix=[[7]])) == [[7]]
    assert unique_columns(MatrixInput(matrix=[[7]])) == [[7]]


# Задача 4. Статистики случайной матрицы
def test_matrix_statistics():
    stats = matrix_statistics(RandomMatrixInput(rows=3, columns=4, mean=5, std=2, seed=1))
    assert stats.matrix.shape == (3, 4)
    assert np.allclose(stats.row_means, stats.matrix.mean(axis=1))
    assert np.allclose(stats.column_means, stats.matrix.mean(axis=0))
    assert np.allclose(stats.row_variances, stats.matrix.var(axis=1))
    assert np.allclose(stats.column_variances, stats.matrix.var(axis=0))
    again = matrix_statistics(RandomMatrixInput(rows=3, columns=4, mean=5, std=2, seed=1))
    assert np.array_equal(stats.matrix, again.matrix)


# Задача 5. Шахматная матрица
def test_chess():
    assert chess(ChessInput(rows=3, columns=4, first=0, second=1)).tolist() == [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    assert chess(ChessInput(rows=1, columns=1, first=5, second=7)).tolist() == [[5]]
    assert chess(ChessInput(rows=1, columns=3, first=5, second=7)).tolist() == [[5, 7, 5]]


# Задача 6. Прямоугольник
def test_draw_rectangle():
    image = draw_rectangle(RectangleInput(width=2, height=2, image_height=4, image_width=4,
                                          shape_color=(1, 1, 1), background_color=(0, 0, 0)))
    assert image.shape == (4, 4, 3)
    assert image[:, :, 0].tolist() == [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    full = draw_rectangle(RectangleInput(width=3, height=2, image_height=2, image_width=3,
                                         shape_color=(1, 1, 1), background_color=(0, 0, 0)))
    assert (full == 1).all()


# Задача 6. Эллипс
def test_draw_ellipse():
    image = draw_ellipse(EllipseInput(semi_axis_x=2, semi_axis_y=1, image_height=5, image_width=5,
                                      shape_color=(1, 1, 1), background_color=(0, 0, 0)))
    assert image.shape == (5, 5, 3)
    assert image[:, :, 0].tolist() == [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    empty = draw_ellipse(EllipseInput(semi_axis_x=0, semi_axis_y=0, image_height=3, image_width=3,
                                      shape_color=(1, 1, 1), background_color=(0, 0, 0)))
    assert (empty == 0).all()


# Задача 7. Анализ временного ряда
def test_analyze_time_series():
    result = analyze_time_series(TimeSeriesInput(values=[2, 4, 1, 5, 3], window=2))
    assert result.mean == 3
    assert result.variance == 2
    assert np.isclose(result.std, 2 ** 0.5)
    assert list(result.local_maxima_indices) == [1, 3]
    assert list(result.local_minima_indices) == [2]
    assert result.moving_average.tolist() == [3, 2.5, 3, 4]

    plateau = analyze_time_series(TimeSeriesInput(values=[1, 2, 2, 1], window=4))
    assert list(plateau.local_maxima_indices) == []
    assert plateau.moving_average.tolist() == [1.5]


# Задача 8. One-hot encoding
def test_one_hot():
    assert one_hot(OneHotInput(labels=[0, 2, 3, 0])).tolist() == [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    assert one_hot(OneHotInput(labels=[0, 1], class_count=3)).tolist() == [[1, 0, 0], [0, 1, 0]]
    assert one_hot(OneHotInput(labels=[0])).tolist() == [[1]]