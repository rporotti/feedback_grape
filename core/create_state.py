import numpy as np
import tensorflow as tf
from scipy import special


def create_batched_tensor_state(N_cavity,
                                batch_size,
                                qubit_state=[1, 0],
                                cavity_state=0,
                                thermal=False,
                                kitten=False,
                                kitten_four_leg=False,
                                kind_state=None,
                                vodoo=False,
                                n_average=1,
                                method="analytic",
                                alpha=1):
    """

    :param N_cavity:
    :param batch_size:
    :param qubit_state:
    :param cavity_state:
    :param thermal:
    :param n_average:
    :return:
    """
    if kind_state is not None:
        if kind_state == "four_legged_kitten":
            kitten_four_leg = True
        if kind_state == "vodoo":
            vodoo = True
        if kind_state == "kitten":
            kitten = True

    if thermal:
        if n_average == 0.0:
            n_average = 1E-7
        n_average = np.ones(batch_size, dtype="complex128") * n_average
        beta = np.log(1.0 / n_average + 1.0)
        diags = np.exp(-beta[:, None] * np.arange(N_cavity))
        b = (diags / diags.sum(axis=1)[:, None])
        rho_cavity = tf.linalg.diag(b)
    elif kitten:

        sqrtn = np.sqrt(np.arange(N_cavity, dtype=complex))
        sqrtn[0] = 1  # Get rid of divide by zero warning

        data = alpha / sqrtn
        data[0] = np.exp(-abs(alpha) ** 2 / 2.0)
        out1 = np.cumprod(data)

        alpha = -alpha
        data = alpha / sqrtn
        data[0] = np.exp(-abs(alpha) ** 2 / 2.0)
        out2 = np.cumprod(data)

        out = out1 + out2
        out /= np.linalg.norm(out)

        diags = np.zeros((batch_size, N_cavity, N_cavity), dtype="complex128")
        diags[:] = np.outer(out, np.transpose(np.conj(out)))
        rho_cavity = tf.constant(diags, dtype="complex128")
    elif kitten_four_leg:
        if method == "operator":
            x = np.zeros(N_cavity)
            x[0] = 1
            n = np.arange(N_cavity - 1)
            a = np.zeros([N_cavity, N_cavity])
            a[n, n + 1] = np.sqrt(n + 1)
            a_dag = np.transpose(a)
            out = np.zeros(N_cavity, dtype=complex)
            for _alpha in [alpha, -alpha, 1j * alpha, -1j * alpha]:
                D = tf.linalg.expm(_alpha * a_dag - np.conj(_alpha) * a)
                # print(out.shape, tf.tensordot(D, x).shape)
                out += np.dot(D, x)

        if method == "analytic":
            out = np.zeros(N_cavity, dtype=complex)
            n = np.arange(N_cavity)
            for _alpha in [alpha, -alpha, 1j * alpha, -1j * alpha]:
                data = np.exp(-(abs(_alpha) ** 2) / 2.0) * (_alpha ** (n)) / \
                       np.array([np.prod(np.sqrt(np.arange(1, x + 1))) for x in n])
                out += data

        out /= np.linalg.norm(out)

        diags = np.zeros((batch_size, N_cavity, N_cavity), dtype="complex128")
        diags[:] = np.outer(out, np.transpose(np.conj(out)))
        rho_cavity = tf.constant(diags, dtype="complex128")
    elif vodoo:

        out = np.zeros(N_cavity, dtype=complex)
        n = np.arange(N_cavity)
        for _alpha in [alpha, np.exp(1j * 2 * np.pi / 3) * alpha, np.exp(-1j * 2 * np.pi / 3) * alpha]:
            data = np.exp(-(abs(_alpha) ** 2) / 2.0) * (_alpha ** (n)) / \
                   np.array([np.prod(np.sqrt(np.arange(1, x + 1))) for x in n])
            out += data

        out /= np.linalg.norm(out)

        diags = np.zeros((batch_size, N_cavity, N_cavity), dtype="complex128")
        diags[:] = np.outer(out, np.transpose(np.conj(out)))
        rho_cavity = tf.constant(diags, dtype="complex128")


    else:
        diagonal = np.zeros(N_cavity, dtype="complex128")
        diags = np.zeros((batch_size, N_cavity, N_cavity), dtype="complex128")
        if isinstance(cavity_state, int):
            diagonal[cavity_state] = 1
        else:
            diagonal = cavity_state
        diagonal /= np.linalg.norm(diagonal)
        diags[:] = np.outer(diagonal, np.transpose(np.conj(diagonal)))

        # norm = np.sum(diags, axis=1)
        # diags = diags / norm[:, None]
        rho_cavity = tf.constant(diags, dtype="complex128")

    if qubit_state is None:
        return tf.cast(rho_cavity[0], dtype="complex128")
    else:
        # Generate qubit state
        q = np.array(qubit_state, dtype="complex128") * np.ones(batch_size)[:, None]
        norm = np.sqrt(np.sum(q ** 2, axis=1))
        q = q / norm[:, None]
        rho_qubit = tf.linalg.diag(q)

        # Tensor product
        rho = tf.cast(
            tf.reshape(tf.einsum('bij,bkl->bikjl', rho_cavity, rho_qubit), (batch_size, N_cavity * 2, N_cavity * 2)),
            dtype="complex128")
        return rho


def create_cavity_state(N_cavity,
                        cavity_state=0,
                        special_state=None, alpha=1, delta=1, threshold=0.01):
    """

    :param N_cavity:

    :param special state: none, kitten, kitten_four_leg, vodoo, GKP and others
    
    :alpha: amplitude of building block coherent state
    
    :delta: delta of gkp code state
    
    :threshold: tolerated infidelity because of cut-off Hilbert space
 
    :return: rho (density matrix)
    """
    if isinstance(cavity_state, int):
        temp = np.zeros(N_cavity)
        temp[cavity_state] = 1
        psi = tf.constant(temp, dtype='complex128')
    if special_state != None:

        def create_coerent_state(N_cavity, _alpha):
            n = np.arange(N_cavity)
            return (_alpha ** (n)) / \
                np.array([np.prod(np.sqrt(np.arange(1, x + 1))) for x in n])

        def create_superposition(N_cavity, list_alpha):
            psi = np.zeros(N_cavity, dtype='complex128')
            for _alpha in list_alpha:
                psi += create_coerent_state(N_cavity, _alpha)

            return psi

        if special_state == 'kitten':

            list_alpha = [alpha, -alpha]

            psi = create_superposition(N_cavity, list_alpha)
            psi /= np.sqrt(4 * np.cosh(abs(alpha) ** 2))
        elif special_state == 'odd_kitten':
            psi = create_coerent_state(N_cavity, alpha)
            psi -= create_coerent_state(N_cavity, -alpha)

            psi /= np.sqrt(4 * np.sinh(abs(alpha) ** 2))

        elif special_state == 'kitten_four_leg':
            list_alpha = [alpha, -alpha, 1j * alpha, -1j * alpha]
            psi = create_superposition(N_cavity, list_alpha)
            psi /= np.sqrt(8 * (np.cos(abs(alpha) ** 2) + np.cosh(abs(alpha) ** 2)))

        elif special_state == 'kitten_four_leg_p_+':

            psi = create_coerent_state(N_cavity, alpha)

            psi -= create_coerent_state(N_cavity, -alpha)
            psi = tf.cast(psi, dtype='complex128')
            psi += 1j * create_coerent_state(N_cavity, 1j * alpha)
            psi -= 1j * create_coerent_state(N_cavity, -1j * alpha)
            psi /= np.sqrt(8 * (-np.sin(abs(alpha) ** 2) + np.sinh(abs(alpha) ** 2)))
        elif special_state == 'kitten_four_leg_p_-':

            psi = create_coerent_state(N_cavity, alpha)

            psi -= create_coerent_state(N_cavity, -alpha)
            psi = tf.cast(psi, dtype='complex128')
            psi -= 1j * create_coerent_state(N_cavity, 1j * alpha)
            psi += 1j * create_coerent_state(N_cavity, -1j * alpha)
            psi /= np.sqrt(8 * (np.sin(abs(alpha) ** 2) + np.sinh(abs(alpha) ** 2)))

        elif special_state == 'kitten_four_leg_d':

            psi = create_coerent_state(N_cavity, alpha)

            psi += create_coerent_state(N_cavity, -alpha)
            psi = tf.cast(psi, dtype='complex128')
            psi -= create_coerent_state(N_cavity, 1j * alpha)
            psi -= create_coerent_state(N_cavity, -1j * alpha)
            print()
            psi /= np.sqrt(8 * (-np.cos(abs(alpha) ** 2) + np.cosh(abs(alpha) ** 2)))


        elif special_state == 'vodoo':

            list_alpha = [alpha, np.exp(1j * 2 * np.pi / 3) * alpha, np.exp(-1j * 2 * np.pi / 3) * alpha]
            psi = create_superposition(N_cavity, list_alpha)
            psi /= np.sqrt(3 * np.exp(-(abs(alpha) ** 2 / 2)) * (np.exp((3 * abs(alpha) ** 2) / 2) + \
                                                                 2 * np.cos((np.sqrt(3) * abs(alpha) ** 2) / 2)))


        elif special_state == 'GKP':

            n = np.arange(N_cavity)

            n_peak = 10 * int(1 / delta) + 1
            positions = np.sqrt(2) * np.sqrt(np.pi) * (np.arange(n_peak) - (n_peak - 1) / 2)

            squeezed_states = np.zeros((N_cavity, n_peak))

            for nn in range(N_cavity):
                if nn == 0:
                    fact = 1.
                else:
                    fact *= nn

                squeezed_states[nn] = special.hermite(nn)(positions) / np.sqrt(np.sqrt(np.pi) * fact)
            squeezed_states *= np.exp(-n[:, None] * delta ** 2) / np.sqrt(2. ** n[:, None])
            #            squeezed_states/=np.sqrt(2.**n[:,None])
            squeezed_states *= np.exp(-positions[None, :] ** 2 / 2.)

            psi = np.sum(squeezed_states, axis=1)
        norm = np.linalg.norm(psi)
        if special_state == 'vodoo' or special_state == 'kitten_four_leg' or special_state == 'kitten' or special_state == 'odd_kitten' or 'kitten_four_leg_d' or 'kitten_four_leg_p_-' or 'kitten_four_leg_p_+':

            print('the infidelity due to Hilbert space cut-off is', np.abs(1 - norm))
            if norm < 1 - threshold:
                print('Hilbert space is too small')
        psi = tf.constant(psi, dtype='complex128') / norm






    else:
        psi = tf.constant(cavity_state, dtype='complex128')
        psi /= np.linalg.norm(psi)
    rho = tf.constant(np.outer(psi, np.conj(psi)), dtype='complex128')

    return rho


def batch_state(rho, batch_size):
    return tf.einsum('i,jk->ijk', np.ones(batch_size, dtype="complex128"), rho)


# def create_quantum_error_correction(N_cavity,generator,alpha):
#    rho=create_cavity_state(N_cavity,special_state='kitten_four_leg',alpha):
#    rho_batch=batch_state(rho,batch_size)
#    tf.linalg.einsum('b,b'generator.uniform([batch_size])*rho_batch)

# finish to code this    

def create_qubit_state(batch_size, qubit_state='up',
                       thermal=False,
                       p_excited=1):
    """

    :param N_cavity:
    :param batch_size:
    :param qubit_state:
    :param cavity_state:
    :param thermal:
    :param n_average:
    :return:
    """

    if thermal:

        diags = np.array([1 - p_excited, p_excited])
        rho = tf.cast(tf.linalg.diag(diags), dtype="complex128")
        rho_batched = tf.einsum('i,jk->ijk', np.ones(batch_size, dtype="complex128"), rho)


    else:
        diagonal = np.zeros(2, dtype="complex128")

        if qubit_state[0] == 'u':
            diagonal[1] = 1
        elif qubit_state[0] == 'd':
            diagonal[0] = 1
        else:
            diagonal = qubit_state
        diagonal /= np.linalg.norm(diagonal)

        rho = tf.cast(np.outer(diagonal, np.transpose(np.conj(diagonal))), dtype="complex128")

        rho_batched = tf.einsum('i,jk->ijk', np.ones(batch_size, dtype="complex128"), rho)

        # norm = np.sum(diags, axis=1)
        # diags = diags / norm[:, None]

    return rho, rho_batched
