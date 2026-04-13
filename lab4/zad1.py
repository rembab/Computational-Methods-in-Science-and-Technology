import numpy as np
import matplotlib.pyplot as plt
import math
import random


def show_path(points, path):
    ordered = [points[i] for i in path]
    ordered.append(ordered[0])
    x, y = zip(*ordered)
    plt.scatter(x, y)
    plt.plot(x, y, linestyle="--")
    plt.show()


def show_prog(scores):
    plt.plot(scores)
    plt.show()


def annealed_salesman(n):
    def obj_fun(perm):
        nonlocal points
        dist = 0
        for i in range(len(perm)):
            x, y = perm[i], perm[(i + 1) % len(perm)]
            dist += math.dist(points[x], points[y])
        return dist

    def next_perm_smart(perm):
        new_perm = perm.copy()
        i, j = random.sample(range(len(new_perm)), 2)
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        return new_perm

    def next_perm_stupid(perm):
        return np.random.permutation(perm)

    def annealing(objf, next_fun, n, start_temp, iters):
        best = np.random.permutation(n)
        best_val = objf(best)

        current, current_eval = best, best_val
        scores = [current_eval]

        for i in range(iters):
            temp = start_temp * ((iters - i) / iters) ** 2

            candidate = next_fun(current)
            candidate_eval = objf(candidate)

            if candidate_eval < current_eval or random.random() < math.exp(
                (current_eval - candidate_eval) / temp
            ):
                current, current_eval = candidate, candidate_eval

                if current_eval < best_val:
                    best, best_val = current.copy(), current_eval

            scores.append(current_eval)

        return best, best_val, scores

    points = np.random.multivariate_normal([0, 0], [[50, 0], [0, 50]], n).tolist()

    best_path, best_dist, scores = annealing(obj_fun, next_perm_smart, n, 10, 10000)

    show_path(points, best_path)
    show_prog(scores)


annealed_salesman(20)
