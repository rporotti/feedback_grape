import tensorflow as tf

def decay_step(rho, gamma, ronge_kutta_steps, a, a_dag, a_dag_a):
    def lindbladian(rho):
        return (
                tf.cast(gamma / ronge_kutta_steps, dtype="complex128")
                * (
                        tf.matmul(a, tf.matmul(rho, a_dag))
                        - 0.5 * (tf.matmul(a_dag_a, rho) + tf.matmul(rho, a_dag_a))
                )
        )

    index = tf.constant(0, dtype="int32")

    def ronge_kutta_step(index, rho):

        k1 = lindbladian(rho)
        k2 = lindbladian(rho + k1 / 2)
        k3 = lindbladian(rho + k2 / 2)
        k4 = lindbladian(rho + k3)
        rho += (k1 + 2 * k2 + 2 * k3 + k4) / 6

        index += 1
        return index, rho

    def ronge_kutta_condition(index, rho):
        return index < ronge_kutta_steps

    if ronge_kutta_steps > 1:
        index, rho = tf.while_loop(ronge_kutta_condition,
                                   ronge_kutta_step, [index, rho])
    else:
        index, rho = ronge_kutta_step(index, rho)
    return rho
