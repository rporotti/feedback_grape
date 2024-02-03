import tensorflow as tf


def qubit_cavity_control(rho, ctrl, complex_fields, subindex, substeps, H_cav_qb, id_sigma_m, id_sigma_p, sx, a_sigma_p,
                         a_dag_sigma_m):
    if complex_fields:
        tot_ctrl = tf.cast(ctrl[:, subindex * 4 + 0], dtype="complex128") + 1j * tf.cast(
            ctrl[:, subindex * 4 + 1], dtype="complex128")
        U2 = tf.linalg.expm(
            -1j / substeps * (tf.tensordot(tot_ctrl, id_sigma_p, axes=0) + tf.tensordot(
                tf.math.conj(tot_ctrl), id_sigma_m, axes=0)))
    else:
        U2 = tf.linalg.expm(
            -1j / substeps * tf.tensordot(
                tf.cast(ctrl[:, subindex * 2 + 0], dtype="complex128"), sx,
                axes=0))

    rho = tf.matmul(U2, tf.matmul(rho, U2, adjoint_b=True))
    if complex_fields:
        tot_ctrl = tf.cast(ctrl[:, subindex * 4 + 2], dtype="complex128") + 1j * tf.cast(
            ctrl[:, subindex * 4 + 3], dtype="complex128")
        U3 = tf.linalg.expm(-1j / substeps * (
                tf.tensordot(tot_ctrl, a_sigma_p, axes=0) + tf.tensordot(tf.math.conj(tot_ctrl),
                                                                         a_dag_sigma_m,
                                                                         axes=0)))
    else:
        U3 = tf.linalg.expm(
            -1j / substeps * tf.tensordot(
                tf.cast(ctrl[:, subindex * 2 + 1], dtype="complex128"),
                H_cav_qb, axes=0))

    rho = tf.matmul(U3, tf.matmul(rho, U3, adjoint_b=True))
    return rho

def snap_evolution(rho, ctrl, complex_fields, subindex, N_snap, shift, a, a_dag, batch_size, N_cavity):
    if complex_fields:
        tot_ctrl = tf.cast(ctrl[:, subindex * (2 + N_snap) + 0] \
                           + shift[:, subindex * (2 + N_snap) + 0], dtype="complex128") \
                   + 1j * tf.cast(ctrl[:, subindex * (2 + N_snap) + 1] \
                                  + shift[:, subindex * (2 + N_snap) + 1], dtype="complex128")

        D = tf.linalg.expm(
            -1j * (tf.tensordot(tot_ctrl, a, axes=0) + tf.tensordot(tf.math.conj(tot_ctrl),
                                                                    a_dag, axes=0)))
    else:
        D = tf.linalg.expm(
            -1j * tf.tensordot(tf.cast(ctrl[:, subindex * (1 + N_snap) + 0] \
                                       + shift[:, subindex * (1 + N_snap) + 0],
                                       dtype="complex128"), a + a_dag, axes=0))

    phis_1 = tf.cast(ctrl[:, subindex * (2 + N_snap) + 2:(subindex + 1) * (2 + N_snap)],
                     dtype="complex128")

    phis_2 = tf.zeros([batch_size, N_cavity - N_snap], dtype='complex128')
    phis = tf.concat([phis_1, phis_2], axis=1)

    SNAP = tf.linalg.diag(tf.math.exp(1j * phis))

    D_SNAP_D_dag = tf.matmul(D, tf.matmul(SNAP, D, adjoint_b=True))

    rho = tf.matmul(D_SNAP_D_dag, tf.matmul(rho, D_SNAP_D_dag, adjoint_b=True))
    return rho