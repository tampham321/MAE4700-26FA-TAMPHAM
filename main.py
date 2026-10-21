"""MAE 4700 HW6 Q3: general 1D finite-element BVP solver.


"""
from ast import literal_eval
from pathlib import Path
import numpy as np


def read_literal(filename):
    return literal_eval(Path(filename).read_text().strip())


def reference_nodes(nnpe):
    return np.linspace(-1.0, 1.0, nnpe)


def shape_functions(xi, nnpe):
    nodes = reference_nodes(nnpe)
    N = np.ones(nnpe)
    for i in range(nnpe):
        for j in range(nnpe):
            if i != j:
                N[i] *= (xi - nodes[j]) / (nodes[i] - nodes[j])
    return N


def shape_derivatives(xi, nnpe):
    nodes = reference_nodes(nnpe)
    dN = np.zeros(nnpe)
    for i in range(nnpe):
        for k in range(nnpe):
            if k == i:
                continue
            term = 1.0 / (nodes[i] - nodes[k])
            for j in range(nnpe):
                if j != i and j != k:
                    term *= (xi - nodes[j]) / (nodes[i] - nodes[j])
            dN[i] += term
    return dN


def gauss_rule(nq):
    return np.polynomial.legendre.leggauss(nq)


def evaluate_function(expr, x):
    # Assignment requires evaluation with NumPy and x; only use trusted inputs.
    result = np.asarray(eval(expr, {"np": np, "x": x}), dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Nonfinite function evaluation at x={x}: {expr}")
    return result


def scalar_function(expr, x):
    value = evaluate_function(expr, float(x))
    if value.size != 1:
        raise ValueError(f"Function must evaluate to a scalar at x={x}")
    return float(value)


def build_computational_mesh(x_end, end_connectivity, nnpe):
    x_all = list(map(float, x_end))
    full_connectivity = []
    for left, right in end_connectivity:
        xL, xR = x_end[left], x_end[right]
        interior = []
        for k in range(1, nnpe - 1):
            interior.append(len(x_all))
            x_all.append(xL + (xR - xL) * k / (nnpe - 1))
        full_connectivity.append([left, *interior, right])
    return np.asarray(x_all, dtype=float), full_connectivity


def element_matrices(xe, nnpe, xi_q, w_q, p_expr, f_expr):
    ke = np.zeros((nnpe, nnpe))
    fe = np.zeros(nnpe)
    for xi, w in zip(xi_q, w_q):
        N = shape_functions(xi, nnpe)
        dN_dxi = shape_derivatives(xi, nnpe)
        J = float(dN_dxi @ xe)
        if J <= 0:
            raise ValueError("Nonpositive element Jacobian")
        B = dN_dxi / J
        x = float(N @ xe)
        p = scalar_function(p_expr, x)
        f = scalar_function(f_expr, x)
        ke += np.outer(B, B) * p * J * w
        fe += N * f * J * w
    return ke, fe


def validate_inputs(x_end, connectivity, nnpe, nq, p_expr, f_expr, sources, bcs):
    if type(nnpe) is not int or nnpe not in (2, 3, 4):
        raise ValueError("nodes per element must be 2, 3, or 4")
    if type(nq) is not int or not 1 <= nq <= 5:
        raise ValueError("num quadrature points must be an integer from 1 to 5")
    if len(x_end) < 2 or not np.all(np.isfinite(x_end)) or len(set(x_end)) != len(x_end):
        raise ValueError("End-node coordinates must be distinct finite values")
    if not connectivity:
        raise ValueError("At least one element is required")
    n = len(x_end)
    for edge in connectivity:
        if (len(edge) != 2 or any(type(i) is not int or i < 0 or i >= n for i in edge)
                or x_end[edge[1]] <= x_end[edge[0]]):
            raise ValueError(f"Invalid left-to-right connectivity: {edge}")
    # Verify a continuous 1D chain covering the domain, regardless of input ordering.
    ordered = sorted((x_end[i], x_end[j]) for i, j in connectivity)
    if (not np.isclose(ordered[0][0], min(x_end)) or
            not np.isclose(ordered[-1][1], max(x_end)) or
            any(not np.isclose(a[1], b[0]) for a, b in zip(ordered, ordered[1:]))):
        raise ValueError("Elements must form a continuous nonoverlapping domain")
    if len(bcs) != 2 or any(len(bc) != 2 or bc[0] not in ('essential', 'natural')
                                or not np.isfinite(float(bc[1])) for bc in bcs):
        raise ValueError("Exactly two valid boundary conditions are required")
    for source in sources:
        if (len(source) != 2 or not all(np.isfinite(float(z)) for z in source)
                or not min(x_end) <= float(source[0]) <= max(x_end)):
            raise ValueError(f"Invalid point source: {source}")
    for x in np.linspace(min(x_end), max(x_end), 5):
        scalar_function(p_expr, x)
        scalar_function(f_expr, x)


def add_point_sources(fe, xL, xR, nnpe, sources):
    for xp, strength in sources:
        xi = 2.0 * (xp - xL) / (xR - xL) - 1.0
        fe += shape_functions(xi, nnpe) * strength


def assemble(x_all, full_connectivity, nnpe, nq, p_expr, f_expr, sources, end_connectivity):
    K = np.zeros((len(x_all), len(x_all)))
    F = np.zeros(len(x_all))
    xi_q, w_q = gauss_rule(nq)
    owned = [[] for _ in full_connectivity]
    # First matching element owns a source, including at shared nodes.
    for xp, strength in sources:
        for e, (left, right) in enumerate(end_connectivity):
            xL, xR = x_all[left], x_all[right]
            if xL - 1e-12 <= xp <= xR + 1e-12:
                owned[e].append((xp, strength))
                break
        else:
            raise ValueError(f"Point source {xp} is not inside an element")
    for e, gdofs in enumerate(full_connectivity):
        idx = np.asarray(gdofs, dtype=int)
        xe = x_all[idx]
        ke, fe = element_matrices(xe, nnpe, xi_q, w_q, p_expr, f_expr)
        add_point_sources(fe, xe[0], xe[-1], nnpe, owned[e])
        F[idx] += fe
        K[np.ix_(idx, idx)] += ke
    return K, F


def solve_partitioned(K, F, essential_dofs, prescribed_values):
    if not essential_dofs:
        raise ValueError("At least one essential BC is required for a unique solution")
    all_dofs = np.arange(len(F))
    essential = np.asarray(essential_dofs, dtype=int)
    free = np.setdiff1d(all_dofs, essential)
    d = np.zeros(len(F))
    d[essential] = prescribed_values
    if len(free):
        Kff = K[np.ix_(free, free)]
        Kfe = K[np.ix_(free, essential)]
        d[free] = np.linalg.solve(Kff, F[free] - Kfe @ d[essential])
    return d


def element_end_derivatives(xe, de, nnpe):
    values = []
    for xi in (-1.0, 1.0):
        dN_dxi = shape_derivatives(xi, nnpe)
        J = float(dN_dxi @ xe)
        values.append(float((dN_dxi / J) @ de))
    return values


def run():
    x_end = read_literal('nodal coordinates.txt')
    connectivity = read_literal('connectivity array.txt')
    nnpe = read_literal('nodes per element.txt')
    nq = read_literal('num quadrature points.txt')
    p_expr = read_literal('p function.txt')
    f_expr = read_literal('f function.txt')
    sources = read_literal('point sources.txt')
    bcs = read_literal('boundary conditions.txt')
    validate_inputs(x_end, connectivity, nnpe, nq, p_expr, f_expr, sources, bcs)
    x_all, full_conn = build_computational_mesh(x_end, connectivity, nnpe)
    K, F = assemble(x_all, full_conn, nnpe, nq, p_expr, f_expr, sources, connectivity)
    left_dof = int(np.argmin(x_end))
    right_dof = int(np.argmax(x_end))
    essential_dofs, prescribed_values = [], []
    for dof, (bc_type, value), sign in zip((left_dof, right_dof), bcs, (-1, 1)):
        if bc_type == 'essential':
            essential_dofs.append(dof)
            prescribed_values.append(float(value))
        else:
            F[dof] += sign * scalar_function(p_expr, x_all[dof]) * float(value)
    d = solve_partitioned(K, F, essential_dofs, prescribed_values)
    residual = K @ d - F
    reactions = [float(residual[dof]) if bc[0] == 'essential' else 0.0
                 for dof, bc in zip((left_dof, right_dof), bcs)]
    derivatives = []
    for gdofs in full_conn:
        idx = np.asarray(gdofs, dtype=int)
        derivatives.append(element_end_derivatives(x_all[idx], d[idx], nnpe))
    outputs = {
        'nodal solution.txt': [float(z) for z in d[:len(x_end)]],
        'reactions.txt': reactions,
        'element derivatives.txt': derivatives,
    }
    for filename, values in outputs.items():
        Path(filename).write_text(repr(values) + '\n')
    print('Solved', len(full_conn), 'elements with', nnpe, 'nodes/element and', nq, 'Gauss points.')
    print('Max free-DOF residual:', max((abs(residual[i]) for i in range(len(d))
                                       if i not in essential_dofs), default=0.0))
    print('Wrote:', ', '.join(outputs))
    return outputs


if __name__ == '__main__':
    run()
