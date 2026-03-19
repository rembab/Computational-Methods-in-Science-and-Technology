import numpy as np
import matplotlib.pyplot as plt
import time


def f32_sum(A, N):
    s = np.float32(0.0)
    for n in A:
        s += n
    return s


def f32_sum_vis():
    v = np.float32(0.53125)
    N = 10**7
    A = [v] * N
    s = np.float32(0.0)

    plot_x = []
    plot_y = []

    for i in range(N):
        s += A[i]
        if i % 25000 == 0:
            plot_x.append(i)
            plot_y.append(abs(i * v - s))

    plt.plot(plot_x, plot_y)
    plt.show()


def f32_sum_rec(A, N):

    def rec_step(A, l, r):
        if r - l > 1:
            m = (r + l) // 2
            return rec_step(A, l, m) + rec_step(A, m + 1, r)
        if l == r:
            return A[r]
        if l == r - 1:
            return A[l] + A[r]
        return np.float32(0)

    return rec_step(A, 0, N - 1)


def f32_sum_kahan(A, N):
    s = np.float32(0)
    err = np.float32(0)
    for i in range(N):
        y = A[i] - err
        tmp = s + y
        err = (tmp - s) - y
        s = tmp
    return s


def test(sumator):
    v = np.float32(0.53125)
    N = 10**7
    A = [v] * N
    s = sumator(A, N)
    e = abs(v * N - s)
    print("Error: ", e)
    print("Relative error: ", e / (v * N))


def compare_exec_times():
    v = np.float32(0.53125)
    N = 10**7
    A = [v] * N
    t1 = time.time()
    f32_sum(A, N)
    print("f32_sum time [s]: ", time.time() - t1)
    t2 = time.time()
    f32_sum_rec(A, N)
    print("f32_sum_rec time [s]: ", time.time() - t2)


def f32_riemann(S, N, rev=False):
    s = np.float32(0)
    if rev:
        r = range(N - 1, 0, -1)
    else:
        r = range(1, N, 1)
    for i in r:
        s += 1 / (i**S)

    return s


def f64_riemann(S, N, rev=False):
    s = np.float64(0)
    if rev:
        r = range(N - 1, 0, -1)
    else:
        r = range(1, N, 1)
    for i in r:
        s += 1 / (i**S)

    return s


def riemann_compare():
    S = [2, 3.6667, 5, 7.2, 10]
    N = [50, 100, 200, 500, 1000]
    print("Riemann: ")
    for s in S:
        for n in N:
            print("s = ", s, ", n = ", n)
            res1 = f32_riemann(np.float32(s), n)
            res2 = f32_riemann(np.float32(s), n, True)
            print("f32 difference: ", abs(res1 - res2))
            res1 = f64_riemann(np.float64(s), n)
            res2 = f64_riemann(np.float64(s), n, True)
            print("f64 difference: ", abs(res1 - res2))
            print()


def f32_dirichlet(S, N, rev=False):
    s = np.float32(0)
    if rev:
        r = range(N - 1, 0, -1)
    else:
        r = range(1, N, 1)
    for i in r:
        s += (-1) ** (i - 1) * (1 / (i**S))

    return s


def f64_dirichlet(S, N, rev=False):
    s = np.float64(0)
    if rev:
        r = range(N - 1, 0, -1)
    else:
        r = range(1, N, 1)
    for i in r:
        s += (-1) ** (i - 1) * (1 / (i**S))

    return s


def dirichlet_compare():
    S = [2, 3.6667, 5, 7.2, 10]
    N = [50, 100, 200, 500, 1000]
    print("Dirichlet: ")
    for s in S:
        for n in N:
            print("s = ", s, ", n = ", n)
            res1 = f32_dirichlet(np.float32(s), n)
            res2 = f32_dirichlet(np.float32(s), n, True)
            print("f32 difference: ", abs(res1 - res2))
            res1 = f64_dirichlet(np.float64(s), n)
            res2 = f64_dirichlet(np.float64(s), n, True)
            print("f64 difference: ", abs(res1 - res2))
            print()


riemann_compare()
dirichlet_compare()
