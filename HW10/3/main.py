"""Steady 2D conduction, T3/Q4 elements, Robin edges and nodal sources.
Run from the directory containing the six assignment input files.
Units for the supplied example: cm, W, degrees C; unit out-of-plane length.
"""
import ast
from pathlib import Path
import warnings
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, MatrixRankWarning


def shape(n, xi, eta):
    """Return shape functions and derivatives (rows: xi, eta)."""
    if n == 3:
        return np.array([1-xi-eta, xi, eta]), np.array([[-1., 1., 0.], [-1., 0., 1.]])
    signs = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    a, b = signs.T
    return (1+a*xi)*(1+b*eta)/4, np.array([a*(1+b*eta), b*(1+a*xi)])/4


def quadrature(n):
    if n == 3:
        return [(1/3, 1/3, 0.5)]
    g = 1/np.sqrt(3)
    return [(xi, eta, 1.) for xi in [-g,g] for eta in [-g,g]]


def gradients(xy, xi, eta):
    N, D = shape(len(xy), xi, eta)
    J = D @ xy  # J rows correspond to derivatives wrt natural coordinates
    det = np.linalg.det(J)
    if det <= 0:
        raise ValueError('Element has nonpositive Jacobian; use CCW, valid elements.')
    return N, np.linalg.solve(J, D), det


def solve(nodes, elements, conductivity, prescribed, convection, sources):
    """Assemble K*T=F; return temperatures, centroid fluxes, convection loss."""
    nodes = np.asarray(nodes, float)
    elements_raw = np.asarray(elements)
    if elements_raw.ndim != 2 or elements_raw.shape[1] not in (3,4):
        raise ValueError('Use a uniform T3 or Q4 mesh.')
    if not np.issubdtype(elements_raw.dtype, np.integer):
        raise ValueError('Connectivity must contain integers.')
    elements = elements_raw.astype(int)
    nn = len(nodes)
    kappa = np.asarray(conductivity, float)
    F = np.asarray(sources, float).copy()
    if nodes.shape != (nn,2) or len(prescribed) != nn or F.shape != (nn,):
        raise ValueError('Invalid coordinates, BC or source dimensions.')
    if kappa.shape != (len(elements),) or np.any(kappa <= 0):
        raise ValueError('Supply one positive conductivity per element.')
    if elements.min() < 0 or elements.max() >= nn:
        raise ValueError('Connectivity node number out of range.')
    rows, cols, data = [], [], []
    def assemble(ids, ke):
        rows.extend(np.repeat(ids, len(ids)).tolist())
        cols.extend(np.tile(ids, len(ids)).tolist())
        data.extend(ke.ravel().tolist())
    edge_counts = {}
    for ids, k in zip(elements, kappa):
        if len(set(ids)) != len(ids):
            raise ValueError('Repeated node in element.')
        xy = nodes[ids]
        ke = np.zeros((len(ids),len(ids)))
        for xi, eta, weight in quadrature(len(ids)):
            _, B, det = gradients(xy, xi, eta)
            ke += k * B.T @ B * det * weight
        assemble(ids, ke)
        for a,b in zip(ids, np.roll(ids,-1)):
            edge = tuple(sorted((int(a),int(b))))
            edge_counts[edge] = edge_counts.get(edge,0)+1
    seen = set()
    for a,b,h,ambient in convection:
        if int(a) != a or int(b) != b or not (0 <= a < nn and 0 <= b < nn):
            raise ValueError('Invalid convection node numbers.')
        a,b = int(a),int(b)
        edge = tuple(sorted((a,b)))
        if edge_counts.get(edge) != 1 or edge in seen or h < 0:
            raise ValueError('Convection must reference a unique exterior edge, h >= 0.')
        seen.add(edge)
        length = np.linalg.norm(nodes[b]-nodes[a])
        # Exact integration of a straight, linear two-node edge.
        assemble([a,b], h*length/6*np.array([[2.,1.],[1.,2.]]))
        F[[a,b]] += h*ambient*length/2
    K = coo_matrix((data,(rows,cols)),shape=(nn,nn)).tocsr()
    fixed = np.array([i for i,t in enumerate(prescribed) if t is not None],int)
    free = np.array([i for i,t in enumerate(prescribed) if t is None],int)
    T = np.zeros(nn)
    T[fixed] = [prescribed[i] for i in fixed]
    if len(free):
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            T[free] = spsolve(K[free][:,free], F[free]-K[free][:,fixed]@T[fixed])
    if not np.all(np.isfinite(T)):
        raise ValueError('Singular system or invalid input.')
    flux = []
    for ids,k in zip(elements,kappa):
        xi = eta = 1/3 if len(ids)==3 else 0.
        _,B,_ = gradients(nodes[ids],xi,eta)
        flux.append((-k*B@T[ids]).tolist())
    loss = sum(h*np.linalg.norm(nodes[int(b)]-nodes[int(a)])*
               ((T[int(a)]+T[int(b)])/2-ambient) for a,b,h,ambient in convection)
    return T, np.array(flux), float(loss)


def read_inputs(directory):
    names = ['nodal coordinates.txt','connectivity array.txt','problem parameters.txt',
             'essential BCs.txt','convection BCs.txt','point sources.txt']
    return [ast.literal_eval((Path(directory)/name).read_text()) for name in names]


def main():
    workdir = Path.cwd()
    inputs = read_inputs(workdir)
    nodes, elements, _, _, convection, sources = inputs
    T, flux, convection_loss = solve(*inputs)
    (workdir/'nodal temperatures.txt').write_text(repr(T.tolist())+'\n')
    (workdir/'element heat fluxes.txt').write_text(repr(flux.tolist())+'\n')

    total_source = float(np.sum(sources))
    balance_error = abs(convection_loss-total_source) / max(abs(total_source), 1e-30) * 100
    kind = 'T3' if len(elements[0]) == 3 else 'Q4'
    print(f'Solved {len(nodes)} nodes with {len(elements)} {kind} elements.')
    print(f'Total nodal source: {total_source:.12g} W/cm')
    print(f'Top convection loss: {convection_loss:.12g} W/cm')
    print(f'Source/convection difference: {balance_error:.6g}%')
    print('Wrote nodal temperatures.txt and element heat fluxes.txt.')


if __name__ == '__main__':
    main()
