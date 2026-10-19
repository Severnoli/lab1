from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value.lower()
    return sum(text.count(ch) for ch in "aeiou")


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    result = 0
    while number > 9:
        number2 = 1
        for char in str(number):
            number2 *= int(char)
        result += 1
        number = number2
    return result


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    return sum((predicted - expected) ** 2 for predicted, expected
                in zip(predicted, expected))/len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = dict()
    d = 2
    while d * d <= number:
        while number % d == 0:
            factors[d] = factors.get(d, 0) + 1
            number /= d
        if d == 2: d = 3
        else: d += 2
    if number > 1:
        factors[int(number)] = factors.get(number, 0) + 1
    result = ""
    for factor, count in factors.items():
        if count > 1:
            result += f"({factor}**{count})"
        else:
            result += f"({factor})"
    return result


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 1
    while total < cube_count:
        total += k * k
        if total == cube_count:
            return k
        k += 1
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    middle_n = len(str(number)) // 2
    num_second = sum(map(int, str(number)[middle_n + 1:]))
    if len(str(number)) % 2 == 0: middle_n -= 1
    num_first = sum(map(int, str(number)[:middle_n:]))
    return num_first == num_second