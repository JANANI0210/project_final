for _ in range(n):
        center = (left + right) / 2
        radius = (right - left) / 2
        left = center - radius
        right = center + radius
    return right