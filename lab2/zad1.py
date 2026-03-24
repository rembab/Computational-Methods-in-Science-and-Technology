import numpy as np
import random
import matplotlib.pyplot as plt
import time


def partial_pivot(col, X):
    mx, mxidx = -float("inf"), -1
    for row in range(len(X[col])):
        if abs(X[col][row]) > mx and row >= col:
            mx = abs(X[col][row])
            mxidx = row
    return mxidx


def multiply_row(row, mul, X):
    for col in range(len(X[row])):
        X[col][row] *= mul


def add_row(row1, row2, mul, X):
    for col in range(len(X[row1])):
        X[col][row1] += X[col][row2] * mul


def swap_rows(row1, row2, X):
    for col in range(len(X[row1])):
        X[col][row1], X[col][row2] = X[col][row2], X[col][row1]


def gauss_jordan(Xin, B):
    X = Xin.copy()
    N = len(X)
    if N != len(B):
        print("array size mismatch")
        return None

    # appending B vector to X
    X.append(B)

    # gaussian elimination
    for col in range(N):
        pivot_row = partial_pivot(col, X)
        swap_rows(pivot_row, col, X)
        mul = 1 / X[col][col]
        multiply_row(col, mul, X)
        X[col][pivot_row] = 1

        for row in range(N):
            if row != col:
                add_row(row, col, -X[col][row], X)
                X[col][row] = 0

    return X


def solve(X, B):
    res = gauss_jordan(X, B)
    if not res:
        return res
    return res[-1]


def print_matrix(X):
    N = len(X) - 1
    for row in range(N):
        for col in range(N + 1):
            print(int(X[col][row]), end=" ")
        print()


def random_matrix(N):
    return [[random.randrange(0, 10) for row in range(N)] for col in range(N)]


def random_vector(N):
    return [random.randrange(0, 10) for row in range(N)]


def compare_exec_times():
    plot_Y1 = []
    plot_Y2 = []

    Ns = [10, 50, 75, 150, 200, 250, 300, 350, 400, 500]
    for N in Ns:
        X = random_matrix(N)
        B = random_vector(N)
        timer = time.time()
        solve(X, B)
        plot_Y1.append(time.time() - timer)
        timer = time.time()
        np.linalg.solve(X, B)
        plot_Y2.append(time.time() - timer)

    plt.plot(Ns, plot_Y1, label="implementation")
    plt.plot(Ns, plot_Y2, label="numpy")


compare_exec_times()
