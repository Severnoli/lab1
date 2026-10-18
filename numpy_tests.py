import numpy as np

from numpy_tasks import (
    sum_prod,
    binarize,
    unique_rows,
    unique_columns,
    matrix_statistics,
    chess,
    draw_rectangle,
    draw_ellipse,
    analyze_time_series,
    one_hot,
)

from grader_contracts.numpy_tasks import (
    MatrixVectorBatchInput,
    BinarizeInput,
    MatrixInput,
    RandomMatrixInput,
    ChessInput,
    RectangleInput,
    EllipseInput,
    TimeSeriesInput,
    OneHotInput,
)


# 1. Сумма произведений матриц на векторы
def test_sum_prod():
    data = MatrixVectorBatchInput(
        matrices=np.array([
            [[1, 2], [3, 4]],
            [[2, 0], [1, 2]]
        ]),
        vectors=np.array([
            [[1], [2]],
            [[3], [4]]
        ])
    )

    result = sum_prod(data)

    expected = np.array([
        [11],
        [22]
    ])

    np.testing.assert_array_equal(result, expected)


# 2. Бинаризация матрицы
def test_binarize():
    matrix = np.array([
        [1, 5, 3],
        [7, 2, 5]
    ])

    data = BinarizeInput(
        matrix=matrix,
        threshold=5
    )

    result = binarize(data)

    expected = np.array([
        [0, 0, 0],
        [1, 0, 0]
    ])

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(matrix, [
        [1, 5, 3],
        [7, 2, 5]
    ])


# 3. Уникальные элементы строк
def test_unique_rows():
    data = MatrixInput(
        matrix=np.array([
            [1, 2, 2, 3],
            [4, 4, 5, 4],
            [6, 7, 6, 7]
        ])
    )

    result = unique_rows(data)

    assert result == [
        [1, 2, 3],
        [4, 5],
        [6, 7]
    ]


# 3. Уникальные элементы столбцов
def test_unique_columns():
    data = MatrixInput(
        matrix=np.array([
            [1, 2, 1],
            [3, 2, 4],
            [1, 5, 4]
        ])
    )

    result = unique_columns(data)

    assert result == [
        [1, 3],
        [2, 5],
        [1, 4]
    ]


# 4. Статистики случайной матрицы
def test_matrix_statistics():
    data = RandomMatrixInput(
        rows=3,
        columns=4,
        mean=10,
        std=2,
        seed=42
    )

    result = matrix_statistics(data)

    rng = np.random.default_rng(42)
    expected_matrix = rng.normal(10, 2, (3, 4))

    np.testing.assert_allclose(result.matrix, expected_matrix)
    np.testing.assert_allclose(
        result.row_means,
        np.mean(expected_matrix, axis=1)
    )
    np.testing.assert_allclose(
        result.column_means,
        np.mean(expected_matrix, axis=0)
    )
    np.testing.assert_allclose(
        result.row_variances,
        np.var(expected_matrix, axis=1)
    )
    np.testing.assert_allclose(
        result.column_variances,
        np.var(expected_matrix, axis=0)
    )


# 5. Шахматная матрица
def test_chess():
    data = ChessInput(
        rows=3,
        columns=4,
        first=0,
        second=1
    )

    result = chess(data)

    expected = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])

    np.testing.assert_array_equal(result, expected)


# 6. Прямоугольник
def test_draw_rectangle():
    data = RectangleInput(
        image_height=5,
        image_width=7,
        width=3,
        height=2,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0)
    )

    result = draw_rectangle(data)

    assert result.shape == (5, 7, 3)

    # Проверяем, что фон и цвет фигуры присутствуют
    assert np.all(result[0, 0] == [0, 0, 0])
    assert np.any(np.all(result == [255, 0, 0], axis=2))


# 6. Эллипс
def test_draw_ellipse():
    data = EllipseInput(
        image_height=7,
        image_width=7,
        semi_axis_x=2,
        semi_axis_y=3,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0)
    )

    result = draw_ellipse(data)

    assert result.shape == (7, 7, 3)

    # Центр должен быть цветом эллипса
    assert np.all(result[3, 3] == [255, 0, 0])

    # Угол изображения должен оставаться фоном
    assert np.all(result[0, 0] == [0, 0, 0])


# 7. Анализ временного ряда
def test_analyze_time_series():
    data = TimeSeriesInput(
        values=[1, 3, 2, 5, 4, 2, 6],
        window=3
    )

    result = analyze_time_series(data)

    values = np.array([1, 3, 2, 5, 4, 2, 6])

    np.testing.assert_allclose(result.mean, np.mean(values))
    np.testing.assert_allclose(result.variance, np.var(values))
    np.testing.assert_allclose(result.std, np.std(values))

    np.testing.assert_array_equal(
        result.local_maxima_indices,
        [1, 3]
    )

    np.testing.assert_array_equal(
        result.local_minima_indices,
        [2, 5]
    )

    np.testing.assert_allclose(
        result.moving_average,
        [2, 10 / 3, 11 / 3, 11 / 3, 4]
    )


# 8. One-hot encoding
def test_one_hot():
    data = OneHotInput(
        labels=[0, 2, 3, 0],
        class_count=None
    )

    result = one_hot(data)

    expected = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])

    np.testing.assert_array_equal(result, expected)


# 8. One-hot с явно заданным количеством классов
def test_one_hot_class_count():
    data = OneHotInput(
        labels=[0, 2],
        class_count=5
    )

    result = one_hot(data)

    expected = np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0]
    ])

    np.testing.assert_array_equal(result, expected)