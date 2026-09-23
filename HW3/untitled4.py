#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 23:08:31 2026

@author: tampham
"""

import numpy as np
import ast
import matplotlib.pyplot as plt
from pathlib import Path


def read_input_file(filename):
    with open(filename, "r") as f:
        return ast.literal_eval(f.read())


def write_output_file(filename, values):
    with open(filename, "w") as f:
        f.write(str(values) + "\n")


# Converts extremely small numerical round-off errors to zero
def clean_near_zero(values, tolerance=1e-8):
    values = np.asarray(values, dtype=float)
    values[np.abs(values) < tolerance] = 0.0
    return values


def element_geometry(x1, x2):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    delta = x2 - x1
    L = np.linalg.norm(delta)

    if L <= 0.0:
        raise ValueError("Truss element has zero length")

    n = delta / L
    return L, n


def element_operator(x1, x2):
    L, n = element_geometry(x1, x2)
    B = np.concatenate((-n, n))
    return L, n, B


def element_stiffness(x1, x2, k):
    _, _, B = element_operator(x1, x2)
    return k * np.outer(B, B)


def element_thermal_force(x1, x2, k, alpha, dT):
    L, _, B = element_operator(x1, x2)

    return k * L * alpha * dT * B


def node_dofs(node, ndim):
    first = node * ndim
    return np.arange(first, first + ndim, dtype=int)


def element_dofs(nodes, ndim):
    i, j = nodes

    return np.concatenate((
        node_dofs(i, ndim),
        node_dofs(j, ndim)
    ))


def assemble_global_stiffness(
        num_nodes,
        ndim,
        coordinates,
        connectivity,
        k_values):

    ndof = ndim * num_nodes

    K = np.zeros((ndof, ndof))

    if len(k_values) != len(connectivity):
        raise ValueError(
            "Number of element stiffnesses must match "
            "number of elements"
        )

    for e, nodes in enumerate(connectivity):

        x1 = np.array(
            coordinates[nodes[0]],
            dtype=float
        )

        x2 = np.array(
            coordinates[nodes[1]],
            dtype=float
        )

        ke = element_stiffness(
            x1,
            x2,
            k_values[e]
        )

        gdofs = element_dofs(
            nodes,
            ndim
        )

        K[np.ix_(gdofs, gdofs)] += ke

    return K


def assemble_thermal_force(
        num_nodes,
        ndim,
        coordinates,
        connectivity,
        k_values,
        alpha_values,
        dT_values):

    F_thermal = np.zeros(
        ndim * num_nodes
    )

    if len(alpha_values) != len(connectivity):
        raise ValueError(
            "Number of thermal expansion coefficients "
            "must match number of elements"
        )

    if len(dT_values) != len(connectivity):
        raise ValueError(
            "Number of temperature changes "
            "must match number of elements"
        )

    for e, nodes in enumerate(connectivity):

        x1 = np.asarray(
            coordinates[nodes[0]],
            dtype=float
        )

        x2 = np.asarray(
            coordinates[nodes[1]],
            dtype=float
        )

        fth = element_thermal_force(
            x1,
            x2,
            k_values[e],
            alpha_values[e],
            dT_values[e]
        )

        gdofs = element_dofs(
            nodes,
            ndim
        )

        F_thermal[gdofs] += fth

    return F_thermal


def build_force_vector(
        num_nodes,
        ndim,
        loads):

    external_nodal_forces = np.array(
        loads,
        dtype=float
    )

    if external_nodal_forces.shape != (
        num_nodes,
        ndim
    ):
        raise ValueError(
            "Force file dimension does not match "
            "nodal coordinates"
        )

    return external_nodal_forces.reshape(-1)


def solve_system(
        K,
        F_total,
        prescribed_displacements,
        num_nodes,
        ndim):

    constrained = []
    prescribed_values = []

    if len(prescribed_displacements) != num_nodes:
        raise ValueError(
            "Number of displacement BC rows must "
            "equal number of nodes"
        )

    for node, values in enumerate(
        prescribed_displacements
    ):

        if len(values) != ndim:
            raise ValueError(
                "Displacement BC dimension does not "
                "match nodal coordinates"
            )

        for direction, value in enumerate(values):

            gdof = node * ndim + direction

            if value is not None:

                constrained.append(gdof)

                prescribed_values.append(
                    float(value)
                )

    constrained = np.array(
        constrained,
        dtype=int
    )

    prescribed_values = np.array(
        prescribed_values,
        dtype=float
    )

    all_dofs = np.arange(
        ndim * num_nodes
    )

    free = np.setdiff1d(
        all_dofs,
        constrained
    )

    d = np.zeros(
        ndim * num_nodes
    )

    d[constrained] = prescribed_values

    K_ff = K[np.ix_(
        free,
        free
    )]

    K_fe = K[np.ix_(
        free,
        constrained
    )]

    F_f = F_total[free]

    d_free = np.linalg.solve(
        K_ff,
        F_f - K_fe @ prescribed_values
    )

    d[free] = d_free

    full_residual = K @ d - F_total

    reactions = np.zeros_like(F_total)

    reactions[constrained] = (
        full_residual[constrained]
    )

    return d, reactions, free


def recover_internal_forces(
        coordinates,
        connectivity,
        k_values,
        d,
        ndim,
        alpha_values=None,
        dT_values=None):

    thermal_enabled = (
        alpha_values is not None
        and dT_values is not None
    )

    internal_forces = []

    for e, nodes in enumerate(connectivity):

        gdofs = element_dofs(
            nodes,
            ndim
        )

        de = d[gdofs]

        x1 = np.array(
            coordinates[nodes[0]],
            dtype=float
        )

        x2 = np.array(
            coordinates[nodes[1]],
            dtype=float
        )

        L, _, B = element_operator(
            x1,
            x2
        )

        axial_extension = B @ de

        thermal_extension = 0.0

        if thermal_enabled:
            thermal_extension = (
                L
                * alpha_values[e]
                * dT_values[e]
            )

        force = (
            k_values[e]
            * (
                axial_extension
                - thermal_extension
            )
        )

        internal_forces.append(
            float(force)
        )

    return internal_forces


def recover_element_strains(
        coordinates,
        connectivity,
        d,
        ndim,
        alpha_values=None,
        dT_values=None):

    thermal_enabled = (
        alpha_values is not None
        and dT_values is not None
    )

    total_strains = []
    thermal_strains = []
    mechanical_strains = []

    for e, nodes in enumerate(connectivity):

        gdofs = element_dofs(
            nodes,
            ndim
        )

        de = d[gdofs]

        x1 = np.asarray(
            coordinates[nodes[0]],
            dtype=float
        )

        x2 = np.asarray(
            coordinates[nodes[1]],
            dtype=float
        )

        L, _, B = element_operator(
            x1,
            x2
        )

        total_extension = B @ de

        total_strain = (
            total_extension / L
        )

        thermal_strain = 0.0

        if thermal_enabled:
            thermal_strain = (
                alpha_values[e]
                * dT_values[e]
            )

        mechanical_strain = (
            total_strain
            - thermal_strain
        )

        total_strains.append(
            float(total_strain)
        )

        thermal_strains.append(
            float(thermal_strain)
        )

        mechanical_strains.append(
            float(mechanical_strain)
        )

    return (
        total_strains,
        thermal_strains,
        mechanical_strains
    )


def calculate_stresses(
        internal_forces,
        area=1e-4):

    stresses = []

    for force in internal_forces:

        stress = force / area

        stress_MPa = (
            stress / 1e6
        )

        stresses.append(
            float(stress_MPa)
        )

    return stresses


def check_solution(
        K,
        F_total,
        d,
        reactions,
        free_dofs,
        num_nodes,
        ndim):

    if np.allclose(
        K,
        K.T
    ):
        print(
            "Stiffness matrix symmetry check: PASS"
        )
    else:
        print(
            "Stiffness matrix symmetry check: FAIL"
        )

    residual = K @ d - F_total

    if np.allclose(
        residual[free_dofs],
        0.0
    ):
        print(
            "Free DOF residual check: PASS"
        )
    else:
        print(
            "Free DOF residual check: FAIL"
        )

    F_nodes = F_total.reshape(
        (num_nodes, ndim)
    )

    R_nodes = reactions.reshape(
        (num_nodes, ndim)
    )

    net_component_force = np.sum(
        F_nodes + R_nodes,
        axis=0
    )

    if np.allclose(
        net_component_force,
        0.0
    ):
        print(
            "Global equilibrium check: PASS"
        )
    else:
        print(
            "Global equilibrium check: FAIL"
        )

    print(
        "Net total RHS + reaction force by component:"
    )

    print(
        net_component_force
    )


def plot_deformed_shape_2d(
        coordinates,
        connectivity,
        d,
        scale=100):

    coordinates = np.array(
        coordinates,
        dtype=float
    )

    displacements = d.reshape(
        (-1, 2)
    )

    deformed_coordinates = (
        coordinates
        + scale * displacements
    )

    plt.figure()

    for e, element in enumerate(
        connectivity
    ):

        node1 = element[0]
        node2 = element[1]

        x_original = [
            coordinates[node1, 0],
            coordinates[node2, 0]
        ]

        y_original = [
            coordinates[node1, 1],
            coordinates[node2, 1]
        ]

        if e == 0:
            plt.plot(
                x_original,
                y_original,
                '--',
                label="Original"
            )
        else:
            plt.plot(
                x_original,
                y_original,
                '--'
            )

    for e, element in enumerate(
        connectivity
    ):

        node1 = element[0]
        node2 = element[1]

        x_deformed = [
            deformed_coordinates[node1, 0],
            deformed_coordinates[node2, 0]
        ]

        y_deformed = [
            deformed_coordinates[node1, 1],
            deformed_coordinates[node2, 1]
        ]

        if e == 0:
            plt.plot(
                x_deformed,
                y_deformed,
                label="Deformed"
            )
        else:
            plt.plot(
                x_deformed,
                y_deformed
            )

    plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        label="Original nodes"
    )

    plt.scatter(
        deformed_coordinates[:, 0],
        deformed_coordinates[:, 1],
        label="Deformed nodes"
    )

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.title(
        "Original and Deformed Truss"
    )

    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.show()


def main():

    coordinates = read_input_file(
        "nodal_coordinates.txt"
    )

    connectivity = read_input_file(
        "connectivity_array.txt"
    )

    k_values = read_input_file(
        "element_stiffnesses.txt"
    )

    loads = read_input_file(
        "external_nodal_forces.txt"
    )

    prescribed_displacements = read_input_file(
        "displacement_BCs.txt"
    )

    alpha_file = Path(
        "thermal expansion coeff.txt"
    )

    dT_file = Path(
        "temperature change.txt"
    )

    thermal = (
        alpha_file.exists()
        or dT_file.exists()
    )

    if thermal and not (
        alpha_file.exists()
        and dT_file.exists()
    ):
        raise ValueError(
            "Both thermal input files are required "
            "when thermal behavior is requested."
        )

    coords_array = np.array(
        coordinates,
        dtype=float
    )

    if coords_array.ndim != 2:
        raise ValueError(
            "Nodal coordinates must be a list "
            "of coordinate lists"
        )

    num_nodes, ndim = coords_array.shape

    if ndim not in (2, 3):
        raise ValueError(
            "Only 2D and 3D trusses are supported"
        )

    if len(connectivity) == 0:
        raise ValueError(
            "Connectivity array is empty"
        )

    if len(k_values) != len(connectivity):
        raise ValueError(
            "Number of element stiffnesses must "
            "match number of elements"
        )

    print(
        f"\nDetected {ndim}D truss "
        f"with {num_nodes} nodes."
    )

    if thermal:

        alpha_values = read_input_file(
            str(alpha_file)
        )

        dT_values = read_input_file(
            str(dT_file)
        )

        if len(alpha_values) != len(connectivity):
            raise ValueError(
                "thermal expansion coeff.txt must "
                "contain one value per element"
            )

        if len(dT_values) != len(connectivity):
            raise ValueError(
                "temperature change.txt must "
                "contain one value per element"
            )

        alpha_values = np.asarray(
            alpha_values,
            dtype=float
        )

        dT_values = np.asarray(
            dT_values,
            dtype=float
        )

        print(
            "Thermal loading detected."
        )

    else:

        alpha_values = np.zeros(
            len(connectivity)
        )

        dT_values = np.zeros(
            len(connectivity)
        )

        print(
            "No thermal loading detected."
        )

    K = assemble_global_stiffness(
        num_nodes,
        ndim,
        coordinates,
        connectivity,
        k_values
    )

    F_external = build_force_vector(
        num_nodes,
        ndim,
        loads
    )

    if thermal:

        F_thermal = assemble_thermal_force(
            num_nodes,
            ndim,
            coordinates,
            connectivity,
            k_values,
            alpha_values,
            dT_values
        )

    else:

        F_thermal = np.zeros_like(
            F_external
        )

    F_total = (
        F_external
        + F_thermal
    )

    d, reactions, free_dofs = solve_system(
        K,
        F_total,
        prescribed_displacements,
        num_nodes,
        ndim
    )

    internal_forces = recover_internal_forces(
        coordinates,
        connectivity,
        k_values,
        d,
        ndim,
        alpha_values,
        dT_values
    )

    (
        total_strains,
        thermal_strains,
        mechanical_strains
    ) = recover_element_strains(
        coordinates,
        connectivity,
        d,
        ndim,
        alpha_values,
        dT_values
    )

    stresses = calculate_stresses(
        internal_forces,
        area=1e-4
    )

    # Clean up tiny floating-point round-off errors
    d = clean_near_zero(d)
    reactions = clean_near_zero(reactions)
    internal_forces = clean_near_zero(internal_forces)
    total_strains = clean_near_zero(total_strains)
    thermal_strains = clean_near_zero(thermal_strains)
    mechanical_strains = clean_near_zero(mechanical_strains)
    stresses = clean_near_zero(stresses)

    check_solution(
        K,
        F_total,
        d,
        reactions,
        free_dofs,
        num_nodes,
        ndim
    )

    nodal_displacements = (
        d.reshape(
            (num_nodes, ndim)
        )
    )

    reaction_values = (
        reactions.reshape(
            (num_nodes, ndim)
        )
    )

    write_output_file(
        "nodal_displacements.txt",
        nodal_displacements.tolist()
    )

    write_output_file(
        "reaction_forces.txt",
        reaction_values.tolist()
    )

    write_output_file(
        "internal_forces.txt",
        internal_forces
    )

    print("\nNodal displacements:")
    print(
        nodal_displacements.tolist()
    )

    print("\nReaction forces:")
    print(
        reaction_values.tolist()
    )

    print("\nInternal element forces:")
    print(
        internal_forces
    )

    print("\nStresses in each element (MPa):")
    print(
        stresses
    )

    if thermal:

        print(
            "\nTotal strain in each element:"
        )

        print(
            total_strains
        )

        print(
            "\nThermal strain in each element:"
        )

        print(
            thermal_strains
        )

        print(
            "\nMechanical strain in each element:"
        )

        print(
            mechanical_strains
        )

    if ndim == 2:

        plot_deformed_shape_2d(
            coordinates,
            connectivity,
            d,
            scale=100
        )


if __name__ == "__main__":
    main()