"""MAE4700 HW8 Problem 5: 2D Euler-Bernoulli plane frame solver.

Run `python main.py` in a directory containing the seven input .txt files.
All inputs and outputs are Python list literals, with SI-consistent units.
Global DOFs per node: [ux, uy, theta] (positive theta counterclockwise).
Local element DOFs: [u1, w1, theta1, u2, w2, theta2].
Distributed p is constant, positive along the element's local +y axis.
"""
from ast import literal_eval
from pathlib import Path
import math
import numpy as np

INPUTS = (
    'nodal coordinates.txt',
    'connectivity array.txt',
    'element stiffnesses.txt',
    'element flexural rigidities.txt',
    'distributed loads.txt',
    'external nodal forces.txt',
    'displacement BCs.txt',
)
OUTPUTS = ('nodal displacements.txt', 'reaction forces.txt', 'internal forces.txt')


def read_list(filename):
    data = literal_eval(Path(filename).read_text(encoding='utf-8').strip())
    if not isinstance(data, list):
        raise ValueError(f'{filename} must contain a Python list')
    return data


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(coords, conn, axial, bending, distributed, nodal, bcs):
    if not coords or any(not isinstance(row, (list, tuple)) or len(row) != 2
                         or not all(finite_number(x) for x in row) for row in coords):
        raise ValueError('nodal coordinates.txt: each node needs [x, y]')
    n = len(coords)
    m = len(conn)
    if not m or any(len(seq) != m for seq in (axial, bending, distributed)):
        raise ValueError('Connectivity, axial stiffness, EI and distributed loads must have equal nonzero lengths')
    for pair in conn:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(type(i) is not int or i < 0 or i >= n for i in pair):
            raise ValueError('Each connectivity entry must be [node1, node2] with valid zero-based node indices')
        if pair[0] == pair[1] or math.dist(coords[pair[0]], coords[pair[1]]) <= 0:
            raise ValueError('Elements must have nonzero length')
    if any(not finite_number(x) or x <= 0 for x in axial + bending):
        raise ValueError('Axial stiffness EA/L and flexural rigidity EI must be finite and positive')
    if any(not finite_number(x) for x in distributed):
        raise ValueError('Distributed loads must be finite scalar values (constant local transverse loads)')
    if len(nodal) != n or len(bcs) != n:
        raise ValueError('Nodal loads and BCs must each have one three-value list per node')
    for label, rows, allow_none in [('external nodal forces', nodal, False), ('displacement BCs', bcs, True)]:
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 3 or any(
                not (allow_none and v is None) and not finite_number(v) for v in row
            ):
                raise ValueError(f'{label}: each node must have three finite values (or None for free BCs)')


def element_matrices(x1, y1, x2, y2, axial_stiffness, EI, p):
    """Return local k, local equivalent load f, and T with d_local = T @ d_global."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    c, s = dx / L, dy / L
    k = np.zeros((6, 6), dtype=float)
    # Axial part: supplied axial stiffness is EA/L, not EA.
    k[np.ix_([0, 3], [0, 3])] = axial_stiffness * np.array([[1, -1], [-1, 1]], dtype=float)
    # Euler-Bernoulli transverse bending part.
    kb = EI / L**3 * np.array([
        [12, 6*L, -12, 6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12, -6*L, 12, -6*L],
        [6*L, 2*L**2, -6*L, 4*L**2],
    ], dtype=float)
    idx = [1, 2, 4, 5]
    k[np.ix_(idx, idx)] = kb
    f = p * L / 2 * np.array([0, 1, L/6, 0, 1, -L/6], dtype=float)
    T = np.array([
        [ c, s, 0,  0, 0, 0],
        [-s, c, 0,  0, 0, 0],
        [ 0, 0, 1,  0, 0, 0],
        [ 0, 0, 0,  c, s, 0],
        [ 0, 0, 0, -s, c, 0],
        [ 0, 0, 0,  0, 0, 1],
    ], dtype=float)
    return k, f, T


def solve(coords, conn, axial, bending, distributed, nodal, bcs):
    n = len(coords)
    K = np.zeros((3*n, 3*n), dtype=float)
    F = np.asarray(nodal, dtype=float).reshape(-1).copy()
    elements = []
    for pair, ka, EI, p in zip(conn, axial, bending, distributed):
        i, j = pair
        kl, fl, T = element_matrices(*coords[i], *coords[j], ka, EI, p)
        dofs = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
        K[np.ix_(dofs, dofs)] += T.T @ kl @ T
        F[dofs] += T.T @ fl
        elements.append((dofs, kl, fl, T))
    bc_flat = [v for row in bcs for v in row]
    fixed = np.array([i for i, v in enumerate(bc_flat) if v is not None], dtype=int)
    free = np.array([i for i, v in enumerate(bc_flat) if v is None], dtype=int)
    d = np.zeros(3*n, dtype=float)
    d[fixed] = [bc_flat[i] for i in fixed]
    if free.size:
        try:
            d[free] = np.linalg.solve(K[np.ix_(free, free)], F[free] - K[np.ix_(free, fixed)] @ d[fixed])
        except np.linalg.LinAlgError as exc:
            raise ValueError('Singular frame stiffness matrix: check connectivity and supports') from exc
    reactions = K @ d - F
    reactions[free] = 0.0  # Required output format: free DOF reactions are zero.
    internal = []
    for dofs, kl, fl, T in elements:
        d_local = T @ d[dofs]
        internal.append((kl @ d_local - fl).tolist())
    return d.reshape(n, 3).tolist(), reactions.reshape(n, 3).tolist(), internal


def main():
    data = [read_list(filename) for filename in INPUTS]
    validate(*data)
    results = solve(*data)
    for filename, values in zip(OUTPUTS, results):
        Path(filename).write_text(repr(values) + '\n', encoding='utf-8')
    print(f'Solved {len(data[0])} nodes, {len(data[1])} elements. Wrote three output files.')


if __name__ == '__main__':
    main()
