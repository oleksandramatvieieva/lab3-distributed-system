"""Математичне ядро ЛР2, частина А (варіант 11).

x = y2*y1' + y2'*y2*Y3 + y1'*Y3^2*y2 + Y3 + Y3*y1*y1'*Y3
y1 = A*b;  y2 = A1*(b1 - 2*c1);  Y3 = A2*(B2 - 2*C2);  C2[i][j] = 1/(i^2 + j)
b[i] = 11*i^2 (i парне), b[i] = 11/i (i непарне)

Модуль не містить жодного коду паралелізації: той самий compute() виконується
і послідовно, і паралельно (змінюється лише функція run_stage).
"""
import os

# BLAS має працювати в одному потоці, інакше numpy сам розпаралелить
# добутки матриць і спотворить вимірювання прискорення.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

SHARED = ["A", "A1", "A2", "B2", "b", "b1", "c1", "y1", "y2",
          "w1", "w2", "u", "v", "D", "Y3", "x"]
MATRICES = {"A", "A1", "A2", "B2", "D", "Y3", "x"}


def gen_inputs(n, seed=11):
    """Вхідні дані (випадкові матриці/вектори + b за формулою варіанта)."""
    rng = np.random.default_rng(seed)
    g = {"n": n}
    for name in ("A", "A1", "A2", "B2"):
        g[name] = rng.random((n, n))
    g["b1"] = rng.random(n)
    g["c1"] = rng.random(n)
    i = np.arange(1, n + 1, dtype=np.float64)
    g["b"] = np.where(np.arange(1, n + 1) % 2 == 0, 11.0 * i ** 2, 11.0 / i)
    for name in ("y1", "y2", "w1", "w2", "u", "v"):
        g[name] = np.zeros(n)
    for name in ("D", "Y3", "x"):
        g[name] = np.zeros((n, n))
    return g


def run_task(g, kind, r0, r1, s=None):
    """Одна підзадача над рядками [r0, r1) (середньоблокова одиниця роботи)."""
    n = g["n"]
    if kind == "y1":                       # (y1)_i = sum_j a_ij b_j
        g["y1"][r0:r1] = g["A"][r0:r1] @ g["b"]
    elif kind == "y2":                     # y2 = A1 (b1 - 2 c1)
        g["y2"][r0:r1] = g["A1"][r0:r1] @ (g["b1"] - 2.0 * g["c1"])
    elif kind == "D":                      # D = B2 - 2 C2, C2_ij = 1/(i^2+j)
        i = np.arange(r0 + 1, r1 + 1, dtype=np.float64)[:, None]
        j = np.arange(1, n + 1, dtype=np.float64)[None, :]
        g["D"][r0:r1] = g["B2"][r0:r1] - 2.0 / (i * i + j)
    elif kind == "Y3":                     # Y3 = A2 D
        g["Y3"][r0:r1] = g["A2"][r0:r1] @ g["D"]
    elif kind == "w1u":                    # w1 = Y3 y2, u = Y3 y1, часткова v
        blk = g["Y3"][r0:r1]
        g["w1"][r0:r1] = blk @ g["y2"]
        g["u"][r0:r1] = blk @ g["y1"]
        return g["y1"][r0:r1] @ blk        # внесок блоку в v = Y3' y1
    elif kind == "w2":                     # w2 = Y3 w1 = Y3^2 y2
        g["w2"][r0:r1] = g["Y3"][r0:r1] @ g["w1"]
    elif kind == "x":                      # складання результату
        s1, s3 = s
        blk = g["Y3"][r0:r1]
        g["x"][r0:r1] = (np.outer(g["y2"][r0:r1], g["y1"])   # y2 y1'
                         + s1 * blk                          # (y2'y2) Y3
                         + s3                               # y1' Y3^2 y2
                         + blk                              # Y3
                         + np.outer(g["u"][r0:r1], g["v"]))  # (Y3 y1)(y1' Y3)
    return None


def compute(g, run_stage, blocks):
    """Граф залежностей: S1 -> S2 -> S3 -> S4 -> S5.

    run_stage(tasks) виконує список незалежних підзадач і чекає на їх
    завершення (бар'єр між стадіями). blocks - список (r0, r1).
    """
    # S1: y1, y2, D незалежні (функціональна + дані)
    run_stage([(k, a, b) for k in ("y1", "y2", "D") for a, b in blocks])
    # S2: Y3 = A2 D (потрібна повна D)
    run_stage([("Y3", a, b) for a, b in blocks])
    # S3: w1, u, часткові суми v (потрібна повна Y3)
    parts = run_stage([("w1u", a, b) for a, b in blocks])
    g["v"][:] = np.sum(parts, axis=0)      # редукція
    s1 = float(g["y2"] @ g["y2"])          # y2' y2
    # S4: w2 = Y3 w1 (потрібна повна w1)
    run_stage([("w2", a, b) for a, b in blocks])
    s3 = float(g["y1"] @ g["w2"])          # y1' Y3^2 y2
    # S5: x
    run_stage([("x", a, b, (s1, s3)) for a, b in blocks])
    return g["x"]
