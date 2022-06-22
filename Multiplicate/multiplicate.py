def multiplicate(a: list[int]) -> list[int]:

    multiplication = 1
    zeros_count = 0
    single_zero_index = None

    for num, elem in enumerate(a):
        if elem == 0:
            zeros_count += 1
            single_zero_index = num
            if zeros_count > 1:
                return [0] * len(a)
        else:
            multiplication *= elem

    if zeros_count == 1:
        result = [0] * len(a)
        result[single_zero_index] = multiplication
    else:
        result = []
        for elem in a:
            result.append(multiplication // elem)
    return result
