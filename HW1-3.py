import numpy as np
import ast


def read_input_file(filename):
    with open(filename, "r") as f:
        return ast.literal_eval(f.read())


def element_stiffness(k):
    return k * np.array([
        [1.0, -1.0],
        [-1.0, 1.0]
    ])



def assemble_global_stiffness(num_nodes, connectivity, k_values):

    K = np.zeros((num_nodes, num_nodes))

    for e in range(len(connectivity)):

        nodes = connectivity[e]
        ke = element_stiffness(k_values[e])

        # Local-to-global mapping
        for local_row in range(2):
            global_row = nodes[local_row]

            for local_col in range(2):
                global_col = nodes[local_col]

                K[global_row, global_col] += ke[local_row, local_col]

    return K


def build_force_vector(num_nodes, loads):

    F = np.array(loads, dtype=float)

    return F

def solve_system(K, F, prescribed_displacements):

    # Find constrained and free nodes
    constrained = []
    free = []

    for i in range(len(prescribed_displacements)):

        if prescribed_displacements[i] is None:
            free.append(i)
        else:
            constrained.append(i)

    # Create displacement vector
    u = np.zeros(len(F))

    # Put known displacements into u
    for i in constrained:
        u[i] = prescribed_displacements[i]

    # Partition the system
    K_ff = K[np.ix_(free, free)]
    K_fe = K[np.ix_(free, constrained)]

    F_f = F[free]
    u_e = u[constrained]


    F_modified = F_f - K_fe @ u_e

    u_f = np.linalg.solve(K_ff, F_modified)

    # Put solved displacements into full vector
    u[free] = u_f

    # Calculate reactions
    reactions = K @ u - F

    return u, reactions, free


def recover_element_forces(u, connectivity, k_values):

    internal_forces = []

    for e in range(len(connectivity)):

        node1 = connectivity[e][0]
        node2 = connectivity[e][1]

        # Element displacement difference
        extension = u[node2] - u[node1]

        # Internal force
        force = k_values[e] * extension

        internal_forces.append(force)

    return internal_forces

def check_solution(K, F, u, reactions, free_dofs):

    # Check that K is symmetric
    if np.allclose(K, K.T):
        print("Stiffness matrix symmetry check: PASS")
    else:
        print("Stiffness matrix symmetry check: FAIL")

    # Check residual at free nodes
    residual = K @ u - F

    if np.allclose(residual[free_dofs], 0.0):
        print("Free DOF residual check: PASS")
    else:
        print("Free DOF residual check: FAIL")

    # Check global equilibrium
    if np.allclose(np.sum(F) + np.sum(reactions), 0.0):
        print("Global equilibrium check: PASS")
    else:
        print("Global equilibrium check: FAIL")


def main():

    # Read inputs
    connectivity = read_input_file("connectivity_array.txt")
    k_values = read_input_file("element_stiffnesses.txt")
    loads = read_input_file("external_nodal_forces.txt")
    prescribed_displacements = read_input_file("displacement_BCs.txt")

    # Number of nodes
    num_nodes = len(loads)

    # Assemble K
    K = assemble_global_stiffness(
        num_nodes,
        connectivity,
        k_values
    )

    # Build F
    F = build_force_vector(num_nodes, loads)

    # Solve
    u, reactions, free_dofs = solve_system(
        K,
        F,
        prescribed_displacements
    )

    # Calculate element forces
    internal_forces = recover_element_forces(
        u,
        connectivity,
        k_values
    )

    # Verification
    check_solution(
        K,
        F,
        u,
        reactions,
        free_dofs
    )

    # Write outputs
    with open("nodal_displacements.txt", "w") as f:
        f.write(str(u.tolist()))

    with open("reaction_forces.txt", "w") as f:
        f.write(str(reactions.tolist()))

    with open("internal_forces.txt", "w") as f:
        f.write(str(internal_forces))

    # Display results
    print("\nNodal displacements:")
    print(u.tolist())

    print("\nReaction forces:")
    print(reactions.tolist())

    print("\nInternal element forces:")
    print(internal_forces)


if __name__ == "__main__":
    main()
