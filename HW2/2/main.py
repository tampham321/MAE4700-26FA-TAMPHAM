

import numpy as np
import ast
import matplotlib.pyplot as plt


def read_input_file(filename):

    with open(filename, "r") as f:
        return ast.literal_eval(f.read())


def write_output_file(filename, values):

    with open(filename, "w") as f:
        f.write(str(values) + "\n")

def element_geometry(x1, x2):

    delta = x2 - x1

    L = np.linalg.norm(delta)

    if L <= 0.0:
        raise ValueError("Truss element has zero length")

    c = delta[0] / L
    s = delta[1] / L

    return L, c, s

def element_stiffness(x1, x2, k):

    # Get element length and direction cosines
    L, c, s = element_geometry(x1, x2)

    # 2D truss element stiffness matrix
    ke = k * np.array([
        [ c**2,  c*s, -c**2, -c*s],
        [ c*s,  s**2, -c*s, -s**2],
        [-c**2, -c*s,  c**2,  c*s],
        [-c*s, -s**2,  c*s,  s**2]
    ])

    return ke

def element_dofs(nodes):

    i, j = nodes

    return np.array([
        2*i,
        2*i + 1,
        2*j,
        2*j + 1
    ], dtype=int)


def assemble_global_stiffness(num_nodes,
                              coordinates,
                              connectivity,
                              k_values):

    # Two DOFs per node
    ndof = 2 * num_nodes

    K = np.zeros((ndof, ndof))

    # Loop through every element
    for e in range(len(connectivity)):

        nodes = connectivity[e]

        # Get coordinates of the two nodes
        x1 = np.array(
            coordinates[nodes[0]],
            dtype=float
        )

        x2 = np.array(
            coordinates[nodes[1]],
            dtype=float
        )

        # Calculate element stiffness matrix
        ke = element_stiffness(
            x1,
            x2,
            k_values[e]
        )

        # Get global DOF numbers
        gdofs = element_dofs(nodes)

        # Add element stiffness to global stiffness matrix
        for a in range(4):

            A = gdofs[a]

            for b in range(4):

                B = gdofs[b]

                K[A, B] += ke[a, b]

    return K


def build_force_vector(num_nodes, loads):

    # Convert:
    #
    # [[Fx,Fy],
    #  [Fx,Fy],
    #  ...]
    #
    # into:
    #
    # [F0x,F0y,F1x,F1y,...]

    external_nodal_forces = np.array(
        loads,
        dtype=float
    )

    F = external_nodal_forces.reshape(-1)

    return F

def solve_system(K,
                 F,
                 prescribed_displacements,
                 num_nodes):

    constrained = []
    prescribed_values = []

    # Find constrained DOFs
    for node, pair in enumerate(
            prescribed_displacements):

        for direction, value in enumerate(pair):

            gdof = 2 * node + direction

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

    # All DOFs
    all_dofs = np.arange(
        2 * num_nodes
    )

    # Free DOFs
    free = np.setdiff1d(
        all_dofs,
        constrained
    )

    # Complete displacement vector
    d = np.zeros(
        2 * num_nodes
    )

    # Insert prescribed displacements
    d[constrained] = prescribed_values

    # Partition stiffness matrix
    K_ff = K[np.ix_(
        free,
        free
    )]

    K_fe = K[np.ix_(
        free,
        constrained
    )]

    F_f = F[free]

    # Solve:
    #
    # K_ff d_f =
    # F_f - K_fe d_e

    d_free = np.linalg.solve(
        K_ff,
        F_f - K_fe @ prescribed_values
    )

    # Put free displacements into full vector
    d[free] = d_free

    # Calculate residual
    full_residual = K @ d - F

    # Reactions only occur at constrained DOFs
    reactions = np.zeros_like(F)

    reactions[constrained] = (
        full_residual[constrained]
    )

    return d, reactions, free


def recover_internal_forces(coordinates,
                            connectivity,
                            k_values,
                            d):

    internal_forces = []

    # Loop through every element
    for e, nodes in enumerate(connectivity):

        # Global DOFs for element
        gdofs = element_dofs(nodes)

        # Element displacement vector:
        #
        # [u1x, u1y, u2x, u2y]

        de = d[gdofs]

        # Coordinates
        x1 = np.array(
            coordinates[nodes[0]],
            dtype=float
        )

        x2 = np.array(
            coordinates[nodes[1]],
            dtype=float
        )

        # Direction cosines
        L, c, s = element_geometry(
            x1,
            x2
        )

        # Axial extension
        axial_extension = np.array([
            -c,
            -s,
             c,
             s
        ]) @ de

        # Axial internal force
        #
        # Positive = tension
        # Negative = compression

        force = (
            k_values[e]
            * axial_extension
        )

        internal_forces.append(force)

    return internal_forces



def calculate_stresses(internal_forces):

    # Cross-sectional area:
    # 1 cm^2 = 1e-4 m^2

    area = 1e-4

    stresses = []

    for force in internal_forces:

        # Stress in Pa
        stress = force / area

        # Convert Pa -> MPa
        stress_MPa = stress / 1e6

        stresses.append(stress_MPa)

    return stresses


def check_solution(K,
                   F,
                   d,
                   reactions,
                   free_dofs):

    # -----------------------------------------------------
    # Stiffness matrix symmetry
    # -----------------------------------------------------

    if np.allclose(K, K.T):

        print(
            "Stiffness matrix symmetry check: PASS"
        )

    else:

        print(
            "Stiffness matrix symmetry check: FAIL"
        )

    # -----------------------------------------------------
    # Free DOF residual
    # -----------------------------------------------------

    residual = K @ d - F

    if np.allclose(
            residual[free_dofs],
            0.0):

        print(
            "Free DOF residual check: PASS"
        )

    else:

        print(
            "Free DOF residual check: FAIL"
        )

    # -----------------------------------------------------
    # Global equilibrium
    # -----------------------------------------------------

    if np.allclose(
            np.sum(F) + np.sum(reactions),
            0.0):

        print(
            "Global equilibrium check: PASS"
        )

    else:

        print(
            "Global equilibrium check: FAIL"
        )

def plot_deformed_shape(coordinates,
                        connectivity,
                        d,
                        scale=100):

    # Convert coordinates to NumPy array
    coordinates = np.array(
        coordinates,
        dtype=float
    )

    # Convert displacement vector into:
    #
    # [[ux, uy],
    #  [ux, uy],
    #  ...]

    displacements = d.reshape(
        (-1, 2)
    )

    # Magnify displacement for visualization
    deformed_coordinates = (
        coordinates
        + scale * displacements
    )

    plt.figure()

    # -----------------------------------------------------
    # Plot original truss
    # -----------------------------------------------------

    for e, element in enumerate(connectivity):

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

    # -----------------------------------------------------
    # Plot deformed truss
    # -----------------------------------------------------

    for e, element in enumerate(connectivity):

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

    # -----------------------------------------------------
    # Plot original nodes
    # -----------------------------------------------------

    plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        label="Original nodes"
    )

    # -----------------------------------------------------
    # Plot deformed nodes
    # -----------------------------------------------------

    plt.scatter(
        deformed_coordinates[:, 0],
        deformed_coordinates[:, 1],
        label="Deformed nodes"
    )

    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Read input files
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Number of nodes
    # -----------------------------------------------------

    num_nodes = len(coordinates)

    # -----------------------------------------------------
    # Assemble global stiffness matrix
    # -----------------------------------------------------

    K = assemble_global_stiffness(
        num_nodes,
        coordinates,
        connectivity,
        k_values
    )

    # -----------------------------------------------------
    # Build global force vector
    # -----------------------------------------------------

    F = build_force_vector(
        num_nodes,
        loads
    )

    # -----------------------------------------------------
    # Solve system
    # -----------------------------------------------------

    d, reactions, free_dofs = solve_system(
        K,
        F,
        prescribed_displacements,
        num_nodes
    )

    # -----------------------------------------------------
    # Calculate internal forces
    # -----------------------------------------------------

    internal_forces = recover_internal_forces(
        coordinates,
        connectivity,
        k_values,
        d
    )

    # -----------------------------------------------------
    # Calculate stresses
    # -----------------------------------------------------

    stresses = calculate_stresses(
        internal_forces
    )

    # -----------------------------------------------------
    # Verification
    # -----------------------------------------------------

    check_solution(
        K,
        F,
        d,
        reactions,
        free_dofs
    )

    # -----------------------------------------------------
    # Convert displacement and reaction arrays
    # -----------------------------------------------------

    nodal_displacements = d.reshape(
        (num_nodes, 2)
    )

    reaction_pairs = reactions.reshape(
        (num_nodes, 2)
    )

    # -----------------------------------------------------
    # Write required output files
    # -----------------------------------------------------

    write_output_file(
        "nodal_displacements.txt",
        nodal_displacements.tolist()
    )

    write_output_file(
        "reaction_forces.txt",
        reaction_pairs.tolist()
    )

    write_output_file(
        "internal_forces.txt",
        internal_forces
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\nNodal displacements:")
    print(
        nodal_displacements.tolist()
    )

    print("\nReaction forces:")
    print(
        reaction_pairs.tolist()
    )

    print("\nInternal element forces:")
    print(
        internal_forces
    )

    print("\nStresses in each element (MPa):")
    print(
        stresses
    )

    # -----------------------------------------------------
    # Plot deformed shape
    # -----------------------------------------------------

    plot_deformed_shape(
        coordinates,
        connectivity,
        d,
        scale=100
    )


if __name__ == "__main__":
    main()
