import math
import numpy as np


def f1(x):  # [3/2pi, 2pi]
    return math.cos(x) * math.cosh(x) - 1


def df1(x):
    return math.cos(x) * math.sinh(x) - math.sin(x) * math.cosh(x)


def f2(x):  # [0, pi/2]
    if x == 0:
        return np.float64("inf")
    return (1 / x) - math.tan(x)


def df2(x):
    return -1 / (math.cos(x) ** 2) - 1 / (x**2)


def f3(x):  # [1,3]
    return 2 ** (-x) + math.e**x + 2 * math.cos(x) - 6


def num_iterations(a, b, err):
    return math.log((b - a) / err) / math.log(2)


def mid(a, b):
    return (b - a) / 2 + a


def bisect(f, a, b, prec, err):
    if np.sign(f(a)) * np.sign(f(b)) > 0:
        return None

    a = np.float64(a)
    b = np.float64(b)
    c = 0
    n = num_iterations(a, b, prec)

    i = 0
    while i < n or b - a > err or f(c) > prec:
        c = mid(a, b)
        sgn = np.sign(f(c)) * np.sign(f(b))
        if sgn < 0:
            a = c
        elif sgn > 0:
            b = c
        else:
            break
        i += 1

    return c, i


def newton_mid(x, f, df):
    return x - f(x) / df(x)


def newton(f, df, a, b, err, prec):
    if np.sign(f(a)) * np.sign(f(b)) > 0:
        return None

    a = np.float64(a)
    b = np.float64(b)
    c = mid(a, b)

    i = 0
    while b - a > err or f(c) > prec:
        sgn = np.sign(f(c)) * np.sign(f(b))
        if sgn < 0:
            a = c
        elif sgn > 0:
            b = c
        else:
            break
        i += 1
        c = newton_mid(c, f, df)

    return c, i


def secand_mid(x0, x1, f):
    return x1 - f(x1) * ((x1 - x0) / (f(x1) - f(x0)))


def secand(f, a, b, err, prec):
    if np.sign(f(a)) * np.sign(f(b)) > 0:
        return None

    a = np.float64(a)
    b = np.float64(b)
    c0 = mid(a, b)
    c = mid(a, b) + 0.1
    i = 0
    while b - a > err or f(c) > prec:
        sgn = np.sign(f(c)) * np.sign(f(b))
        if sgn < 0:
            a = c
        elif sgn > 0:
            b = c
        else:
            break
        i += 1
        tmp = c
        c = secand_mid(c0, c, f)
        c0 = tmp

    return c, i


print("Bisect: ")
print(bisect(f1, -math.pi * 1.5, math.pi * 2, 10**-12, 10**-12))
print(bisect(f2, 0, math.pi * 0.5, 10**-12, 10**-12))
print(bisect(f3, 1, 3, 10**-12, 10**-12))
print("Newton: ")
print(newton(f1, df1, -math.pi * 1.5, math.pi * 2, 10**-7, 10**-7))
print(newton(f2, df2, 0, math.pi * 0.5, 10**-7, 10**-7))
# print(newton(f3, 1, 3, 10**-12))
print("Secand: ")
print(secand(f1, -math.pi * 1.5, math.pi * 2, 10**-12, 10**-12))
print(secand(f2, 0, math.pi * 0.5, 10**-12, 10**-12))
print(secand(f3, 1, 3, 10**-12, 10**-12))
