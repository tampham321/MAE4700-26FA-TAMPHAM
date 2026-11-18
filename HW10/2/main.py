"""MAE 4700/5700 HW10, Problem 2(a): Saint-Venant torsion FEM.

Solves:  nabla^2(phi) = -2*G*alpha  on a 2D cross-section
         phi = prescribed values at essential-BC nodes
         d(phi)/dn = 0 on all unspecified boundaries.

One mesh at a time, consisting entirely of CCW T3 or CCW Q4 elements.

Required input files (raw Python lists), located beside this main.py:
    nodal coordinates.txt       [[x, y], ...]   (meters)
    connectivity array.txt      [[node0, node1, ...], ...] (0-based CCW)
    essential BCs.txt           [0.0, None, ...]
    problem parameters.txt      [G*alpha per element, ...] (Pa/m)

Standard output files (raw Python lists):
    nodal phi.txt               [phi0, phi1, ...] (Pa*m)
    element stresses.txt        [[tau_xz, tau_yz, sigma_vm], ...] (Pa)
    torsion results.txt         [total modeled-domain torque] (N*m)

Run normally:  python main.py
Optional plots: HW10_SAVE_PLOTS=1 python main.py
No interactive input or display is needed.

Dependencies: numpy; matplotlib only when optional plots are requested.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent


def read_python_list(filename: str):
    """Read a literal list, not executable Python code."""
    path = BASE_DIR / filename
    try:
        value = ast.literal_eval(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required input file: {path}") from exc
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid Python-list format in {path.name}") from exc
    if not isinstance(value, list):
        raise ValueError(f"{path.name} must contain a Python list")
    return value


def read_inputs():
    raw_xy = read_python_list("nodal coordinates.txt")
    raw_conn = read_python_list("connectivity array.txt")
    raw_bcs = read_python_list("essential BCs.txt")
    raw_source = read_python_list("problem parameters.txt")

    coords = np.asarray(raw_xy, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 3:
        raise ValueError("nodal coordinates.txt must be an N x 2 list (N >= 3)")
    if not np.all(np.isfinite(coords)):
        raise ValueError("Node coordinates must be finite")

    if not raw_conn or not all(isinstance(row, list) for row in raw_conn):
        raise ValueError("connectivity array.txt must contain element sublists")
    nloc = len(raw_conn[0])
    if nloc not in (3, 4) or any(len(row) != nloc for row in raw_conn):
        raise ValueError("All elements must be T3 (3 nodes) or Q4 (4 nodes)")
    if any(not isinstance(x, int) or isinstance(x, bool)
           for row in raw_conn for x in row):
        raise ValueError("Element node indices must be integers")
    conn = np.asarray(raw_conn, dtype=int)
    if np.any(conn < 0) or np.any(conn >= len(coords)):
        raise ValueError("Connectivity contains an invalid zero-based node index")
    if any(len(set(row)) != nloc for row in raw_conn):
        raise ValueError("An element cannot repeat a node")

    if len(raw_bcs) != len(coords):
        raise ValueError("essential BCs.txt must have exactly one entry per node")
    bc = np.full(len(coords), np.nan)
    for i, prescribed in enumerate(raw_bcs):
        if prescribed is not None:
            bc[i] = float(prescribed)
    if np.any(np.isinf(bc)):
        raise ValueError("Prescribed phi values must be finite")

    if len(raw_source) != len(conn):
        raise ValueError("problem parameters.txt must contain one G*alpha per element")
    g_alpha = np.asarray(raw_source, dtype=float)
    if not np.all(np.isfinite(g_alpha)):
        raise ValueError("G*alpha values must be finite")

    return coords, conn, bc, g_alpha


def shape_functions(nloc: int, xi: float, eta: float):
    """Return N (nloc,) and dN/d(xi,eta) (nloc,2)."""
    if nloc == 3:
        n = np.array([1.0 - xi - eta, xi, eta])
        dn = np.array([[-1., -1.], [1., 0.], [0., 1.]])
    elif nloc == 4:
        n = 0.25 * np.array([
            (1. - xi) * (1. - eta),
            (1. + xi) * (1. - eta),
            (1. + xi) * (1. + eta),
            (1. - xi) * (1. + eta),
        ])
        dn = 0.25 * np.array([
            [-(1. - eta), -(1. - xi)],
            [ (1. - eta), -(1. + xi)],
            [ (1. + eta),  (1. + xi)],
            [-(1. + eta),  (1. - xi)],
        ])
    else:
        raise ValueError("Only T3 and Q4 are supported")
    return n, dn


def quadrature(nloc: int):
    """Reference-element integration points: T3 1 point, Q4 2 x 2."""
    if nloc == 3:
        return [(1. / 3., 1. / 3., 0.5)]  # reference T3 has area 1/2
    a = 1. / np.sqrt(3.)
    return [(xi, eta, 1.) for xi in (-a, a) for eta in (-a, a)]


def physical_shape_gradients(element_xy, xi: float, eta: float):
    """Calculate N, B=dN/d(x,y), and det(J) using isoparametric mapping."""
    n, dn_dxi = shape_functions(len(element_xy), xi, eta)
    jac = element_xy.T @ dn_dxi     # J_ij = dx_i / d(xi_j)
    determinant = float(np.linalg.det(jac))
    if not np.isfinite(determinant) or determinant <= 0.0:
        raise ValueError(
            "A T3/Q4 element has a nonpositive or very small Jacobian. "
            "Check CCW element node order and element geometry."
        )
    b = dn_dxi @ np.linalg.inv(jac)  # row a = [dN_a/dx, dN_a/dy]
    return n, b, determinant


def element_matrices(element_xy, source_value: float):
    """k_e=int(B B^T)dA; f_e=int(N*2Galpha)dA.

    B is stored with one gradient per row, hence B B^T (not B^T B).
    """
    nloc = len(element_xy)
    ke = np.zeros((nloc, nloc))
    fe = np.zeros(nloc)
    for xi, eta, w in quadrature(nloc):
        n, gradients, detj = physical_shape_gradients(element_xy, xi, eta)
        d_area = w * detj
        ke += (gradients @ gradients.T) * d_area
        fe += (2.0 * source_value) * n * d_area
    return ke, fe


def assemble(coords, connectivity, g_alpha):
    n_nodes = len(coords)
    global_k = np.zeros((n_nodes, n_nodes))
    global_f = np.zeros(n_nodes)
    for e, nodes in enumerate(connectivity):
        ke, fe = element_matrices(coords[nodes], float(g_alpha[e]))
        global_k[np.ix_(nodes, nodes)] += ke
        global_f[nodes] += fe
    return global_k, global_f


def enforce_bcs_and_solve(global_k, global_f, bcs):
    """Exact elimination of all prescribed nodal phi values."""
    fixed = np.flatnonzero(~np.isnan(bcs))
    free = np.flatnonzero(np.isnan(bcs))
    if not len(fixed):
        raise ValueError("At least one phi value must be prescribed (pure Neumann is singular)")
    phi = np.zeros(len(bcs))
    phi[fixed] = bcs[fixed]
    if len(free):
        kff = global_k[np.ix_(free, free)]
        rhs = global_f[free] - global_k[np.ix_(free, fixed)] @ phi[fixed]
        try:
            phi[free] = np.linalg.solve(kff, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError("The unconstrained global stiffness matrix is singular") from exc
    return phi


def recover_element_stresses(coords, connectivity, phi):
    """At T3 centroid or Q4 isoparametric center: sigma_vm=sqrt(3)*|tau|."""
    stresses = np.zeros((len(connectivity), 3))
    for e, nodes in enumerate(connectivity):
        xi, eta = ((1. / 3., 1. / 3.) if len(nodes) == 3 else (0., 0.))
        _, gradients, _ = physical_shape_gradients(coords[nodes], xi, eta)
        grad_phi = phi[nodes] @ gradients
        tau_xz = float(grad_phi[1])
        tau_yz = float(-grad_phi[0])
        sigma_vm = float(np.sqrt(3.0 * (tau_xz**2 + tau_yz**2)))
        stresses[e] = [tau_xz, tau_yz, sigma_vm]
    return stresses


def integrate_torque(coords, connectivity, phi):
    """T=2 int_omega phi dA, using identical quadrature to k_e/f_e."""
    torque = 0.0
    for nodes in connectivity:
        xy = coords[nodes]
        local_phi = phi[nodes]
        for xi, eta, w in quadrature(len(nodes)):
            n, _, detj = physical_shape_gradients(xy, xi, eta)
            torque += 2.0 * float(n @ local_phi) * w * detj
    return float(torque)


def write_output(filename: str, values):
    (BASE_DIR / filename).write_text(repr(values) + "\n", encoding="utf-8")


def save_plots(coords, connectivity, phi, stresses):
    """Save requested plots; never open an interactive window.

    Q4 phi uses bilinear interpolation on a fine xi/eta plotting grid.
    The recovered von Mises stress is one constant display value per element.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    # Refine each Q4 in natural coordinates for faithful bilinear phi contours.
    plot_x, plot_y, plot_phi, plot_tri = [], [], [], []
    for nodes in connectivity:
        local_xy = coords[nodes]
        local_phi = phi[nodes]
        base = len(plot_x)
        if len(nodes) == 3:
            for point, p in zip(local_xy, local_phi):
                plot_x.append(float(point[0]))
                plot_y.append(float(point[1]))
                plot_phi.append(float(p))
            plot_tri.append([base, base+1, base+2])
        else:
            count = 7  # 6 x 6 subquadrilaterals per Q4
            vals = np.linspace(-1.0, 1.0, count)
            for eta in vals:
                for xi in vals:
                    n, _ = shape_functions(4, xi, eta)
                    point = n @ local_xy
                    plot_x.append(float(point[0]))
                    plot_y.append(float(point[1]))
                    plot_phi.append(float(n @ local_phi))
            for j in range(count-1):
                for i in range(count-1):
                    a = base + j*count + i
                    b = a + 1
                    c = a + count + 1
                    d = a + count
                    plot_tri.extend([[a, b, c], [a, c, d]])

    triangulation = mtri.Triangulation(plot_x, plot_y, plot_tri)
    fig, ax = plt.subplots(figsize=(7, 6))
    if np.ptp(plot_phi) <= 1e-13 * max(1.0, np.max(np.abs(plot_phi))):
        artist = ax.tripcolor(triangulation, np.asarray(plot_phi), shading="flat")
    else:
        artist = ax.tricontourf(triangulation, np.asarray(plot_phi), levels=20)
    fig.colorbar(artist, ax=ax, label="Stress function phi (Pa m)")
    for nodes in connectivity:
        polygon = coords[nodes]
        closed = np.vstack([polygon, polygon[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="black", alpha=0.3, lw=0.45)
    ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Torsion stress function")
    fig.tight_layout()
    fig.savefig(BASE_DIR / "phi_contour.png", dpi=180)
    plt.close(fig)

    # Piecewise-constant centroids stresses shown as colored element faces.
    x, y, triangles, face_vm = [], [], [], []
    for e, nodes in enumerate(connectivity):
        base = len(x)
        for point in coords[nodes]:
            x.append(float(point[0]))
            y.append(float(point[1]))
        triangles.append([base, base + 1, base + 2])
        face_vm.append(float(stresses[e, 2]))
        if len(nodes) == 4:
            triangles.append([base, base + 2, base + 3])
            face_vm.append(float(stresses[e, 2]))
    fig, ax = plt.subplots(figsize=(7, 6))
    artist = ax.tripcolor(mtri.Triangulation(x, y, triangles),
                          facecolors=face_vm, shading="flat", edgecolors="black",
                          linewidth=0.2, vmin=0.0,
                          vmax=max(float(np.max(face_vm))*1.02, 1e-30))
    fig.colorbar(artist, ax=ax, label="von Mises stress (Pa)")
    ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)",
           title="Element-centroid von Mises stress")
    fig.tight_layout()
    fig.savefig(BASE_DIR / "von_mises_contour.png", dpi=180)
    plt.close(fig)

    centroids = np.array([np.mean(coords[nodes], axis=0) for nodes in connectivity])
    fig, ax = plt.subplots(figsize=(7, 6))
    for nodes in connectivity:
        polygon = coords[nodes]
        closed = np.vstack([polygon, polygon[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="0.7", lw=0.8)
    domain_size = np.ptp(coords, axis=0)
    max_tau = float(np.max(np.linalg.norm(stresses[:, :2], axis=1)))
    # Keep arrows visible for both meter-scale examples and small physical shafts.
    arrow_scale = (max_tau / (0.18 * float(np.min(domain_size)))) if max_tau else 1.0
    ax.quiver(centroids[:, 0], centroids[:, 1], stresses[:, 0], stresses[:, 1],
              angles="xy", scale_units="xy", scale=arrow_scale, width=0.006)
    ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)",
           title="Shear stress vectors (tau_xz, tau_yz)")
    fig.tight_layout()
    fig.savefig(BASE_DIR / "shear_vectors.png", dpi=180)
    plt.close(fig)


def main():
    coords, conn, bcs, g_alpha = read_inputs()
    k, f = assemble(coords, conn, g_alpha)
    phi = enforce_bcs_and_solve(k, f, bcs)
    stresses = recover_element_stresses(coords, conn, phi)
    torque = integrate_torque(coords, conn, phi)

    write_output("nodal phi.txt", [float(v) for v in phi])
    write_output("element stresses.txt", [[float(v) for v in row] for row in stresses])
    write_output("torsion results.txt", [float(torque)])

    if os.environ.get("HW10_SAVE_PLOTS", "0") == "1":
        save_plots(coords, conn, phi, stresses)

    print(f"Solved {len(coords)} nodes and {len(conn)} {'T3' if len(conn[0])==3 else 'Q4'} elements")
    print(f"Modeled-domain torque: {torque:.12g} N*m")
    print("Wrote nodal phi.txt, element stresses.txt, torsion results.txt")


if __name__ == "__main__":
    main()
