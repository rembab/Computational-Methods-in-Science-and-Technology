import numpy as np
import matplotlib.pyplot as plt
import math
import random


def show_path(points, path):
    x, y = zip(*points)
    plt.scatter(x, y)
    plt.plot(x, y, linestyle="--")
    plt.show()


def show_prog(scores):
    plt.plot(scores)
    plt.show()


def annealed_salesman(n):

    def temp_fun(perm):
        nonlocal points
        dist = 0
        for i in range(len(perm) - 1):
            x, y = perm[i], perm[i + 1]
            dist += math.dist(points[x], points[y])
        return dist

    def next_perm_smart(perm):
        i = random.randrange(0, len(perm) - 1)
        j = random.randrange(i + 1, len(perm))
        perm[i], perm[j] = perm[j], perm[i]
        return perm

    def next_perm_stupid(perm):
        return np.random.permutation(perm)

    def annealing(tempf, next_fun, n, temp, iters):
        s_temp = temp
        best = np.random.permutation(list(range(n)))
        best_val = tempf(best)
        current, current_eval = best, best_val
        scores = [best_val]
        for i in range(iters):
            temp = s_temp * (iters - i / iters)

            next = next_fun(current)
            next_eval = tempf(next)

            if next_eval < best_val or random.random() < math.exp(
                (current_eval - next_eval) / temp
            ):

                current, current_eval = next, next_eval
                if next_eval < best_val:
                    best, best_eval = next, next_eval
                    scores.append(best_eval)

        return best, best_val, scores

    points = np.random.multivariate_normal([0, 0], [[50, 0], [0, 50]], n).T
    points = list(zip(points[0], points[1]))

    best_path, best_dist, scores = annealing(temp_fun, next_perm_stupid, n, 1000, 1000)

    show_path(points, best_path)

    show_prog(scores)


annealed_salesman(100)
