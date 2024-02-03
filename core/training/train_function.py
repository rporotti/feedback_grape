import numpy as np
import tensorflow as tf

from .utils import measure_system, compute_fidelity
from ..evolution.decay import decay_step
from ..evolution.unitary import qubit_cavity_control, snap_evolution


def train_step(rho_init, rho_target, max_steps, N_cavity,
               generator, N_snap=1, type_unitary="qubit-cavity",
               F_1_0=None, parameters=None, control=True,
               type_input="measure", substeps=None,
               gamma_m=0.0, ronge_kutta_steps_m=1, ronge_kutta_steps=1,
               gamma=0.0, with_logP_term=True, mode="network",
               feedback=True, discount_factor=1.0,
               just_last_fidelity=True, goal="fidelity",
               measure_op="sy", controls_array=None,
               fake_measure=None, ctrl_deviation=None, double_measure=False,
               complex_fields=False, clock=False, memory=True, stochastic_ctrl=False, system="JC"):
    """
    Compute the gradients with respect to the input parameters.
    Parameters
    ----------
    rho_init : tf.Tensor
        Initial density matrix of the system. Must have shape [batch_size, N_cavity, N_cavity]
    rho_target: tf.Tensor
        Target density matrix of the system. Must have shape [N_cavity, N_cavity]
    max_steps: int
        Maximum number of steps to take in the optimization. Each steps includes a measurement,
        a $\sigma_x$ gate, and a coupling between cavity and qubit.
    N_cavity: int
        Size of the Hilbert space of the cavity.
    F_1_0: None or tf.Tensor
        First measurement of the control sequence. It must be separated from the others
        since it does not depend on any previous measurement.
    mode: str
        Can be 'lookup' or 'network'.
    parameters: tf.Tensor:
        If mode='lookup', a tensor of shape [max_steps, 2**max_steps, 3].
        Contains all of the learnable parameters.
        If mode='network', a neural network that can have different types of inputs (see type_input),
        but the output should have 3 real values.
    control: bool
        If True, the control sequence is used.
    type_input: str
        Type of input to the network. Can be 'measure' or 'state'.
    substeps: None or int
        If not None, every control step is divided in substeps.
    gamma: float
        Decay rate. Must be used in combination with substeps.
    with_logP_term: bool
        If True, the logarithm of the probability is added to the loss.
    feedback: bool
        If True, the feedback is used.
    discount_factor: float
        Discount factor for the reward.
    just_last_fidelity: bool
        If True, only the last fidelity is optimized. If False, the sum of all the fidelities is optimized.
    goal: str
        Can be 'fidelity' or 'purity'. Target to optimize
    measure_op: str
         If "sy" the measure is the y-Pauli matrix. If "both" the measure is a superposition of y- and
         x- Pauli matrices. If "non-demolition" the measure is fixed in this case F_1_0 is chosen such that
         the target state is an eigenstate of the measure.

    Returns
    -------
    grads: tf.Tensor
        Gradients of the loss with respect to the parameters.
    loss: tf.Tensor
        Loss of the optimization.
    final_fidelity:
        Final fidelity of the optimization.
    rho: tf.Tensor
        Final density matrix of the system.
    """
    if not substeps:
        substeps = 1
    batch_size = rho_init.shape[0]
    if not complex_fields:
        ctrl_num_unitary = (1 + N_snap) * substeps
    else:
        ctrl_num_unitary = (2 + N_snap) * substeps

    if not feedback or measure_op == "non-demolition":
        ctrl_num = ctrl_num_unitary
    else:
        ctrl_num = ctrl_num_unitary + 2

    if mode == "lookup" or mode == "lookup_array" or mode == "array":
        F = parameters
    if mode == "network":
        network = parameters
    if fake_measure is True:
        fake_measure_dict = {"current_index": 0, "max_steps": max_steps}
    else:
        fake_measure_dict = None

    # Define quantum operators
    # Qubit
    sigma_p = np.array([[0. + 0.j, 1. + 0.j], [0. + 0.j, 0. + 0.j]])
    sigma_m = np.transpose(sigma_p)
    sx = np.array([[0. + 0.j, 1. + 0.j], [1. + 0.j, 0. + 0.j]])
    i_sy = sigma_p - sigma_m

    # Cavity
    n = np.arange(N_cavity - 1)
    a = np.zeros([N_cavity, N_cavity])
    a[n, n + 1] = np.sqrt(n + 1)
    a_dag = np.transpose(a)
    a_dag_a = np.matmul(a_dag, a)

    # Convert them to Tensorflow
    id_sigma_p = tf.constant(np.kron(np.eye(N_cavity), sigma_p), dtype='complex128')
    id_sigma_m = tf.constant(np.kron(np.eye(N_cavity), sigma_m), dtype='complex128')
    if system == "JC":
        sx = id_sigma_p + id_sigma_m
    a_sigma_p = tf.constant(np.kron(a, sigma_p), dtype='complex128')
    a_dag_sigma_m = tf.constant(np.kron(a_dag, sigma_m), dtype='complex128')
    H_cav_qb = a_sigma_p + a_dag_sigma_m
    if type_unitary == "SNAP":
        a_dag = tf.constant(a_dag, dtype='complex128')
        a = tf.constant(a, dtype='complex128')
        a_dag_a = tf.matmul(a_dag, a)
    else:
        a = tf.constant(np.kron(a, np.eye(2)), dtype='complex128')
        a_dag = tf.constant(np.kron(a_dag, np.eye(2)), dtype='complex128')
        a_dag_a = tf.constant(np.kron(a_dag_a, np.eye(2)), dtype='complex128')

    discount_factor = tf.constant(discount_factor, dtype="float64")

    with tf.GradientTape(persistent=True) as tape:
        if parameters is not None:
            if mode == "lookup" or mode == "lookup_array":
                tape.watch(F)

            if mode == "network":
                # Reset the state of the LSTM network (needed at the beginning of a trajectory)
                network.reset_states()

        rho = rho_init

        if stochastic_ctrl:

            shift = tf.einsum('i,j->ij', tf.stop_gradient(generator.normal([rho.shape[0]], dtype="float64")),
                              ctrl_deviation)

        #            shift_1=ctrl_deviation[0]*((tf.random.uniform([rho.shape[0]], dtype="float64")-1/2))
        #            shift_2=ctrl_deviation[1]*((tf.random.uniform([rho.shape[0]], dtype="float64")-1/2))
        else:
            if np.shape(ctrl_deviation) != ctrl_num_unitary:
                if ctrl_deviation is not None:
                    print('Warning: ctrl_deviation has wrong dimensions')
                ctrl_deviation = tf.zeros([ctrl_num_unitary], dtype='float64')

            shift = tf.einsum('i,j->ij', tf.ones([rho.shape[0]], dtype='float64'), ctrl_deviation)

        ##############################################################################################
        #####    Decay step before measurement  ######################################################
        ##############################################################################################

        if gamma_m > 0.:
            rho = decay_step(rho, gamma_m, ronge_kutta_steps_m, a, a_dag, a_dag_a)

        if feedback:
            if measure_op != "non-demolition":
                tape.watch(F_1_0)
            if measure_op == "both" or "non-demolition":
                theta = F_1_0[1]
            else:
                theta = None
            rho, measure, log_probs = measure_system(F_1_0[0], rho, goal, generator, measure_op, theta,
                                                     fake_measure_dict)
            if double_measure:
                rho, measure, log_probs_temp = measure_system(F_1_0[0], rho, goal, generator, measure_op, theta,
                                                              fake_measure_dict)
                log_probs = log_probs + log_probs_temp
        else:
            log_probs = tf.zeros(batch_size)[:, None]
            measure = tf.zeros(batch_size)[:, None]

        fidelity = tf.constant(np.zeros(batch_size), dtype="float64")
        # add comment what is comb?
        comb = tf.constant(np.zeros(batch_size), dtype="int32")

        def body(index, rho, log_probs, measure, fidelity, comb):
            # Predict the next control sequence
            if control:
                if controls_array is None:
                    if mode == "network":
                        if type_input == "state":
                            rho_split = tf.stack([tf.math.real(rho), tf.math.imag(rho)], axis=1)
                            ctrl = network(rho_split)
                        if type_input == "measure":
                            if not memory:
                                network.reset_states()
                            if clock:
                                x_input = tf.concat([tf.cast(tf.reshape(measure, (batch_size, 1, 1)), dtype='float64'), \
                                                     tf.cast(index, dtype='float64') * tf.ones([batch_size, 1, 1],
                                                                                               dtype='float64')],
                                                    axis=2)
                            else:
                                x_input = tf.reshape(measure, (batch_size, 1, 1))
                            ctrl = network(x_input)
                    if mode == "lookup":
                        rescaled = tf.cast((measure + 1) / 2, dtype="float64")
                        if feedback:
                            power = tf.round(2 ** tf.cast(index, dtype="float64"))

                            appo = tf.cast(tf.round(rescaled[:, 0] * power), dtype="int32")
                            comb = tf.cast(comb + appo, dtype=tf.int32)
                            ctrl = tf.gather(tf.gather(F, index), comb)
                        else:
                            ctrl = tf.gather(tf.gather(F, index), comb)
                else:

                    ctrl = tf.reshape(controls_array[index], [1, ctrl_num])
            else:
                ctrl = tf.constant(np.zeros((batch_size, ctrl_num), dtype='float64'))

            subindex = tf.constant(0, dtype="int32")

            def sub_body(subindex, rho, ctrl):
                rho_start = rho
                if gamma > 0.0:
                    rho = decay_step(rho, gamma, ronge_kutta_steps, a, a_dag, a_dag_a)

                if goal == "fidelity" or goal == "both":
                    if system == "JC":
                        if type_unitary == "qubit-cavity":
                            rho = qubit_cavity_control(rho, ctrl, complex_fields, subindex, substeps, H_cav_qb,
                                                       id_sigma_m,
                                                       id_sigma_p, sx, a_sigma_p,
                                                       a_dag_sigma_m)

                        elif type_unitary == "SNAP":
                            rho = snap_evolution(rho, ctrl, complex_fields, subindex, N_snap, shift, a, a_dag,
                                                 batch_size, N_cavity)
                    elif system == "qubits":
                        tot_ctrl = tf.cast(ctrl[:, 0:2], dtype="complex128") * tf.cast(1 + shift, dtype="complex128")
                        U2 = tf.einsum('i,jk->ijk', tf.math.cos(tot_ctrl[:, 0] / 2), tf.eye(2, dtype='complex128'))
                        U2 += 1j * tf.einsum('i,jk->ijk', tf.math.sin(tot_ctrl[:, 0] / 2) * tf.math.sin(tot_ctrl[:, 1]),
                                             sx)
                        U2 += tf.einsum('i,jk->ijk', tf.math.sin(tot_ctrl[:, 0] / 2) * tf.math.cos(tot_ctrl[:, 1]),
                                        i_sy)

                        rho = tf.matmul(U2, tf.matmul(rho, U2, adjoint_b=True))

                subindex = subindex + 1
                return subindex, rho, ctrl

            def sub_condition(subindex, rho, ctrl):
                return subindex < substeps

            if substeps > 1:
                subindex, rho, ctrl = tf.while_loop(sub_condition,
                                                    sub_body,
                                                    [subindex, rho, ctrl])
            else:
                subindex, rho, ctrl = sub_body(subindex, rho, ctrl)

            if feedback:
                if (goal == "purity" and index < max_steps) or (goal == "fidelity" and index < max_steps - 1):
                    if measure_op == "both":
                        theta = ctrl[:, ctrl_num_unitary + 1]
                    elif measure_op == "sy":
                        theta = None
                    if gamma_m > 0.:
                        rho = decay_step(rho, gamma_m, ronge_kutta_steps_m, a, a_dag, a_dag_a)
                    if measure_op == "both" or measure_op == "sy":
                        rho, measure, log_probs_temp = measure_system(ctrl[:, ctrl_num_unitary], rho, goal, generator,
                                                                      measure_op, theta, fake_measure_dict)
                    else:
                        rho, measure, log_probs_temp = measure_system(F_1_0[0], rho, goal, generator, measure_op,
                                                                      F_1_0[1], fake_measure_dict)
                        if double_measure:
                            log_probs = log_probs + log_probs_temp
                            rho, measure, log_probs_temp = measure_system(F_1_0[0], rho, goal, generator, measure_op,
                                                                          F_1_0[1], fake_measure_dict)

                    log_probs = log_probs + log_probs_temp

            if not just_last_fidelity:
                fid_partial = compute_fidelity(rho, rho_target)
                fidelity = fidelity + tf.pow(discount_factor, tf.cast(max_steps - index, dtype="float64")) * fid_partial
            index = index + 1

            return index, rho, log_probs, measure, fidelity, comb

        def condition(index, rho, log_probs, measure, fidelity, comb):
            return index < max_steps

        index = tf.constant(0, dtype="int32")
        args_loop = [index, rho, log_probs, measure, fidelity, comb]
        index, rho, log_probs, measure, fidelity, comb = tf.while_loop(condition,
                                                                       body,
                                                                       args_loop)

        if goal == "fidelity":
            final_fidelity = compute_fidelity(rho, rho_target)
            if just_last_fidelity:
                R = final_fidelity
            else:
                R = fidelity

        if goal == "purity":
            final_purity = tf.abs(tf.linalg.trace(rho ** 2))
            R = final_purity

        if goal == "both":
            final_fidelity = compute_fidelity(rho, rho_target)
            final_purity = tf.abs(tf.linalg.trace(rho ** 2))
            R = final_fidelity + final_purity

        loss1 = tf.reduce_mean(-R)
        if not feedback:
            loss = loss1
        if feedback and with_logP_term:
            loss2 = tf.reduce_mean(log_probs * tf.stop_gradient(-R))
            loss = loss1 + loss2
    # Define parameters to optimize
    if mode == "network":
        params = network.trainable_variables
    if mode == "lookup":
        params = [F]
    if mode == "none":
        params = [controls_array]
    if feedback:
        params += [F_1_0]

    grads = tape.gradient(loss, params)

    # Add noise to the gradients
    # t = tf.cast(epoch, grads[0].dtype )
    # variance = 0.01 / ((1 + t) ** 0.55)
    # grads = [ grad + tf.random.normal(grad.shape, mean=0.0, stddev=tf.math.sqrt(variance), dtype=grads[0].dtype) for grad in grads ]
    # epoch = epoch + 1

    if goal == "fidelity":
        return grads, -loss1 / max_steps, final_fidelity, rho
    if goal == "purity":
        return grads, loss, final_purity, rho
    if goal == "both":
        return grads, loss, final_fidelity, final_purity, rho
