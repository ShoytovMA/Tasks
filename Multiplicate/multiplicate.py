def multiplicate(a: list[int]) -> list[int]:

    multiplication = 1  # Произведение ненулевых элементов массива
    zeros_count = 0  # Количество нулей во входном массиве
    single_zero_index = None  # Индекс нулевого элемента, если такой один

    for num, elem in enumerate(a):
        if elem == 0:
            zeros_count += 1
            single_zero_index = num 
            if zeros_count > 1:  # Если во входном массиве больше одного нуля,
                return [0] * len(a)  # то выходной будет состоять из одних нулей
        else:
            multiplication *= elem

    if zeros_count == 1:  # Если во входном массиве один ноль,
        result = [0] * len(a)  # то в выходном массиве на его месте будет стоять произведение всех ненулевых элементов,
        result[single_zero_index] = multiplication  # а на остальных местах нули
    else:
        result = []
        for elem in a:
            result.append(multiplication // elem)
    return result
