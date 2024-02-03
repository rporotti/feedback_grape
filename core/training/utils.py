import itertools

import numpy as np
import tensorflow as tf


def measure_system(F_meas, rho, goal, generator, measure_op="sy", theta=None, fake_measure_dict=None):
    """

    Parameters
    ----------
    F_meas: tf.Tensor
        The measurement strength. It has shape [batch_size]
    rho:
        The density matrix to which apply the measurement


    Returns
    -------
    rho: tf.Tensor
        The density matrix after the measurement.
    meas:
        The measured result.
    log_probs:
        The log probabilities of the measurement.




    """
    # In the paper F_meas is denoted gamma and theta is denoted delta

    N_cavity = int(rho.shape[2] / 2)
    batch_size = int(rho.shape[0])
    F_meas = tf.cast(F_meas, dtype="complex128")

    if len(F_meas.shape) == 1:
        F_meas = F_meas[:, None]
        theta = theta[:, None]
    # Operator needed for the measurement
    foo = tf.cast(tf.repeat(tf.range(N_cavity), 2), dtype='complex128')

    # Choose value of theta in order to select the correct operator
    if measure_op == "sy":
        theta = np.pi / 2
    if measure_op == "sx":
        theta = 0
    if measure_op == "both":
        pass
    measure_term = (1 - tf.ones(batch_size, dtype="complex128")[:, None]) * np.pi / 4 + tf.cast(theta / 2,
                                                                                                dtype="complex128")
    # Compute update density matrix for a particular measure, in order
    # to compute the probability of a measure, but don't update yet
    # the density matrix
    op = tf.linalg.diag(tf.cos(F_meas * foo + measure_term))
    rho_temp = tf.matmul(op, tf.matmul(rho, op))
    prob = tf.reduce_sum(tf.linalg.diag_part(rho_temp), axis=1)

    # Generate a random msmt with the correct statistic
    cond = tf.cast((generator.uniform([rho.shape[0]], dtype="float64") < tf.stop_gradient(tf.abs(prob))),
                   dtype="int8")
    scaled = tf.subtract(tf.math.scalar_mul(2, cond), 1)
    measure = tf.cast(scaled, dtype="complex128")

    if fake_measure_dict is not None:
        if goal == "purity":
            repeat = int(fake_measure_dict["max_steps"] + 1)

            lst = (np.array(list(map(list, itertools.product([0, 1], repeat=repeat)))) * 2 - 1)

            x_input = np.flip(((lst))[:, :(fake_measure_dict["current_index"] + 1)], axis=1)
            measure = x_input[:, 0]

            fake_measure_dict["current_index"] += 1
        if goal == "fidelity":
            repeat = int(fake_measure_dict["max_steps"])
            lst = (np.array(list(map(list, itertools.product([0, 1], repeat=repeat)))) * 2 - 1)

            x_input = np.flip(((lst))[:, :(fake_measure_dict["current_index"] + 1)], axis=1)

            measure = x_input[:, 0]
            fake_measure_dict["current_index"] += 1
    measure_term = (1 - measure[:, None]) * np.pi / 4 + tf.cast(theta / 2, dtype="complex128")
    # Now, really compute the update and then measure again
    op = tf.linalg.diag(tf.cos(F_meas * foo + measure_term))
    rho = tf.matmul(op, tf.matmul(rho, op))
    prob = tf.reduce_sum(tf.linalg.diag_part(rho), axis=1)
    rho = rho / prob[:, None, None]
    measure = tf.reshape(tf.math.real(measure), (rho.shape[0], 1))

    # Begin to accumulate the lnP terms
    log_probs = tf.math.log(tf.abs(prob))
    return rho, measure, log_probs

@tf.function
def compute_fidelity(rho, rho_target):
    """

    Parameters
    ----------
    rho: tf.Tensor
        The current density matrix of the system.
    rho_target: tf.Tensor
        The target density matrix of the system.


    Returns
    -------
    fidelity: tf.Tensor
        The fidelity of the input density matrix.
    """
    if rho.shape[1:]!=rho_target.shape:
        N_cavity = int(rho.shape[1] / 2)
        _rho = tf.linalg.einsum("bikjk->bij", tf.reshape(rho, (rho.shape[0], N_cavity, 2, N_cavity, 2)))
    else:
        _rho = rho
    fidelity = tf.cast(tf.linalg.einsum("bii->b", tf.matmul(_rho, rho_target)), dtype="float64")

    return fidelity

