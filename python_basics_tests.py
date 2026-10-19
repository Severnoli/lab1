from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from python_basics import (
    count_vowels,
    has_unique_characters,
    mse,
    prime_factorization,
    pyramid, count_one_bits,
    multiplicative_persistence,
    is_balanced_number)

# 1. Подсчёт гласных
def test_count_vowels():
    assert count_vowels(TextInput("cat dog")) == 2
    assert count_vowels(TextInput("OUIEA")) == 5
    assert count_vowels(TextInput("rhythm")) == 0

# 2. Уникальные символы
def test_has_unique_characters():
    assert has_unique_characters(TextInput("xyz")) is True
    assert has_unique_characters(TextInput("apple")) is False
    assert has_unique_characters(TextInput("q")) is True

# 3. Единичные биты
def test_count_one_bits():
    assert count_one_bits(PositiveIntegerInput(2)) == 1
    assert count_one_bits(PositiveIntegerInput(11)) == 3
    assert count_one_bits(PositiveIntegerInput(32)) == 1

# 4. Мультипликативная устойчивость
def test_multiplicative_persistence():
    assert multiplicative_persistence(PositiveIntegerInput(7)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(68)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(77)) == 4

# 5. Средняя квадратичная ошибка
def test_mse():
    assert mse(VectorPairInput([5, 6], [5, 6])) == 0
    assert mse(VectorPairInput([4, 5, 6], [5, 6, 7])) == 1
    assert mse(VectorPairInput([2, 4], [3, 6])) == 2.5

# 6. Простые множители
def test_prime_factorization():
    assert prime_factorization(PositiveIntegerInput(2)) == "(2)"
    assert prime_factorization(PositiveIntegerInput(12)) == "(2**2)(3)"
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"

# 7. Пирамида
def test_pyramid():
    assert pyramid(PositiveIntegerInput(1)) == 1
    assert pyramid(PositiveIntegerInput(14)) == 3
    assert pyramid(PositiveIntegerInput(7)) == "It is impossible"

# 8. Сбалансированное число
def test_is_balanced_number():
    assert is_balanced_number(PositiveIntegerInput(1239411)) is True
    assert is_balanced_number(PositiveIntegerInput(987654)) is False
    assert is_balanced_number(PositiveIntegerInput(242)) is True