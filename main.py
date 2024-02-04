import os
import sys

SEED = 1
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from core.create_state import create_batched_tensor_state, create_cavity_state, batch_state, create_qubit_state
from core.training.train_function import train_step

tf.keras.utils.set_random_seed(SEED)


def main(
    N_cavity = 10,
    batch_size = 10,
    gradient_steps=10,
    type_unitary="qubit-cavity",
    system="JC",
    substeps = 1,
    gamma = 0.0,
    feedback = True,
    control = True,
    goal = "fidelity",
    measure_op = "both",
    max_steps = 5,
    mode = "lookup",
    type_input = "state",
    input = "thermal",
    n_average = 1,
    kind_state_output = "four_legged_kitten",
    test=True,
    return_fidelity=True,
    complex_fields=False,
    double_measure=False,
    clock=False,
    N_snap=1
):
    ###### Initial state ######
    # Cavity state can be an integer or a numpy array with the same size as the Hilbert space (it will be normalized)
    if input=="thermal":
        rho_init = create_batched_tensor_state(N_cavity=N_cavity,
                                               batch_size=batch_size,
                                               qubit_state=[1, 0],
                                               # cavity_state=0,
                                               thermal=True, n_average=n_average
                                               )
    elif input=="SNAP":
        cavity_state = np.zeros(N_cavity)
        cavity_state[0] = 1
        rho = create_cavity_state(N_cavity=N_cavity,
                                  cavity_state=cavity_state, threshold=1e-2)
        rho_init = batch_state(rho, batch_size)
    elif input=="qubits":
        _, rho_init = create_qubit_state(batch_size, thermal=True)
    else:

        rho_init = create_batched_tensor_state(N_cavity=N_cavity,
                                               batch_size=batch_size,
                                               qubit_state=[1, 0],
                                               cavity_state=0,
                                               )

    ###### Target State #######
    # Cavity state can be an integer or a numpy array with the same size as the Hilbert space (it will be normalized)
    if kind_state_output=="four_legged_kitten":
        rho_target = create_batched_tensor_state(
            N_cavity=N_cavity,
            batch_size=1,
            qubit_state=None,
            # cavity_state=cavity_state,
            # thermal=True, n_average=n_av,
            # kitten=True,
            kind_state=kind_state_output,
            alpha=np.sqrt(2)
        )
    elif kind_state_output=="GKP":
        cavity_state = np.zeros(N_cavity, dtype='complex')
        cavity_state[0] = 0
        cavity_state[1] = 1
        cavity_state[0] = 0
        delta = 0.15
        rho_target = create_cavity_state(N_cavity=N_cavity,
                                         cavity_state=cavity_state, special_state='GKP', delta=delta)
    elif kind_state_output=="qubits":
        rho_target, _ = create_qubit_state(batch_size)
    else:
        rho_target = create_batched_tensor_state(
            N_cavity=N_cavity,
            batch_size=1,
            qubit_state=None,
            cavity_state=3,
        )

    ###### Define the neural network ######
    F_1_0 = tf.Variable(np.array([np.pi / 2, 0]))
    if mode == "network":
        if type_input == "state":
            input_shape = (None, 2, N_cavity * 2, N_cavity * 2)
            network = tf.keras.Sequential(
                [tf.keras.layers.Flatten(),
                 tf.keras.layers.Dense(30, input_shape=input_shape, activation='tanh', dtype="float64"),
                 tf.keras.layers.Dense(30, activation='tanh', dtype="float64"),
                 tf.keras.layers.Dense(4, dtype="float64", bias_initializer=tf.keras.initializers.Constant(np.pi))]
            )

            network.build(input_shape)
        if type_input == "measure":
            if input=="SNAP":
                if complex_fields == False:
                    ctrl_num_unitary = substeps * (1 + N_snap)
                else:
                    ctrl_num_unitary = substeps * (2 + N_snap)
                if feedback == False or measure_op == "non-demolition":
                    ctrl_num = ctrl_num_unitary
                else:
                    ctrl_num = ctrl_num_unitary + 2
                add = 0
                if clock:
                    add = 1
                network = tf.keras.Sequential(
                    [
                        tf.keras.layers.GRU(30, batch_input_shape=(batch_size, 1, 1 + add),
                                            stateful=True),
                        tf.keras.layers.Dense(30, activation='relu', dtype="float64"),
                        tf.keras.layers.Dense(30, activation='relu', dtype="float64"),
                        tf.keras.layers.Dense(ctrl_num, dtype="float64",
                                              bias_initializer=tf.keras.initializers.Constant(0.1))]
                )
            else:
                network = tf.keras.Sequential(
                    [
                        tf.keras.layers.GRU(30, batch_input_shape=(batch_size, 1, 1),
                                            stateful=True),
                        tf.keras.layers.Dense(4, dtype="float64", bias_initializer=tf.keras.initializers.Constant(np.pi / 2))]
                )
        parameters = network
        variables = network.trainable_variables + [F_1_0]
    if mode == "lookup":
        F = []
        for i in range(1, max_steps + 1):
            appo = np.random.rand(2 ** i, 4) * np.pi
            F.append(tf.Variable(appo))
        F = tf.ragged.stack(F)
        F = tf.Variable(F.to_tensor())
        parameters = F
        variables = [F] + [F_1_0]

    # optimizer = tf.optimizers.Adam(learning_rate=1E-3, clipvalue=0.5, clipnorm=1.0) working!!!!
    optimizer = tf.optimizers.Adam(learning_rate=1E-2, clipvalue=0.5, clipnorm=1.0)
    generator = tf.random.Generator.from_non_deterministic_state()
    train_function = tf.function(train_step)
    fids = np.zeros((batch_size, gradient_steps))
    for optim_step in range(gradient_steps):
        grads, loss, fidelity, rho = train_function(rho_init=rho_init,
                                                   rho_target=rho_target,
                                                   max_steps=max_steps,
                                                   N_cavity=N_cavity,
                                                   type_unitary=type_unitary,
                                                    system=system,
                                                    F_1_0=F_1_0,
                                                   mode=mode,
                                                   goal=goal,
                                                   generator=generator,
                                                   parameters=parameters,
                                                   type_input=type_input,
                                                   measure_op=measure_op,
                                                   substeps=substeps,
                                                   gamma=gamma,
                                                   feedback=feedback,
                                                    complex_fields=complex_fields,
                                                    double_measure=double_measure,
                                                    clock=clock,
                                                    N_snap=N_snap
                                                   )
        if control:
            optimizer.apply_gradients(zip(grads, variables))
        fids[:, optim_step] = fidelity.numpy()
        if not test:
            print(f"Epoch {optim_step} - Loss: {loss}, Fidelity: {fidelity.numpy().mean()}")
    # plt.plot(fids.mean(axis=0))
    # plt.show()
    if return_fidelity:
        return fidelity.numpy()[-1]
    else:
        return True

if __name__=="__main__":
    gradient_steps = 10  # number of epochs
    substeps = 1  # Whether to subdivide a control step in substeps (default at 1)
    gamma = 0.0  # Decay rate
    type_unitary = "qubit-cavity" # qubit-cavity or SNAP
    system = "JC" # JC or qubits
    feedback = True  # Whether to measure the system
    control = True  # Whether to control the system
    goal = "fidelity"  # Can be "fidelity" or "purity"
    measure_op = "both"  # Which operator to measure with
    max_steps = 1  # Number of controls/measurements
    mode = "lookup"  # "network" or "lookup"
    type_input = "state"  # "measure" or "state", input of the density matrix
    n_average = 1
    kind_state_output = "four_legged_kitten"
    main(
        gradient_steps=gradient_steps,
        substeps = substeps,
        gamma = gamma,
        type_unitary=type_unitary,
        system=system,
        feedback = feedback,
        control = control,
        goal = goal,
        measure_op = measure_op,
        max_steps = max_steps,
        mode = mode,
        type_input = type_input,
        input = "thermal",
        n_average = n_average,
        kind_state_output = kind_state_output,
        test=False,
        return_fidelity=True,
    )