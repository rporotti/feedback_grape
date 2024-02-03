import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from core.create_state import create_batched_tensor_state, create_cavity_state
from core.training.train_function import train_step


class StrategyTreePlot:
    def __init__(self, gs=None, fig=None):
        """
        Initialize a Strategy Tree plot (use plot afterwards).
        
        Parameters
        ----------
        
        ax:
            the matplotlib axis object to plot into
        
        Other parameters can be set by setting the following variables
        after initialization:
        
        self.path_color='darkcyan'
        self.state_circle_radius=0.1
        self.state_circle_color='darkcyan'
        self.state_circle_fill_color='white'
        self.measure_circle_radius=0.05
        self.measure_circle_color='darkcyan'
        self.measure_circle_fill_color='darkcyan'
        self.measure_box_fill_color='rebeccapurple'
        self.measure_box_text_color='white'
        self.measure_text=self.simple_measure_text
        self.measure_box_fontsize=4
        self.control_box_fill_color='crimson'
        self.control_box_text_color='white'
        self.control_text=self.simple_measure_text
        self.control_box_fontsize=4
        self.box_shift=-0.2
        self.pi_magic_tolerance=1e-2
        self.box_translate=-0.1
        """
        if gs:
            ax = fig.add_subplot(gs)
        else:
            scale = 0.4
            if fig:
                ax = fig.add_subplot(111)
            else:
                fig, ax = plt.subplots(figsize=(8 * scale, 3 * scale), dpi=400)
        self.ax = ax
        self.fig = fig
        self.path_color = 'darkcyan'
        self.state_circle_radius = 0.1
        self.state_circle_color = 'darkcyan'
        self.state_circle_fill_color = 'white'
        self.measure_circle_radius = 80
        self.measure_circle_color = 'darkcyan'
        self.measure_circle_fill_color = 'darkcyan'
        self.measure_box_fill_color = 'rebeccapurple'
        self.measure_box_text_color = 'white'
        self.measure_text = self.simple_measure_text

        self.measure_box_fontsize = 10
        self.measure_circle_fontsize = 8
        self.node_circle_radius = 15
        self.control_box_fill_color = 'crimson'
        self.control_box_text_color = 'white'
        self.control_text = self.simple_control_text
        self.control_box_fontsize = 10
        self.box_shift = -0.2
        self.box_translate = -0.03
        self.pi_magic_tolerance = 0.1e-1
        self.pi_magic_tolerance_coupling = 0.1e-1
        self.fidelity_translate = 0.6

    def pi_magic_coupling(self, number):
        """
        Takes any number as input and compares
        it against (m/n)*pi as well as pi/sqrt(n). If it finds a match according
        to some relative tolerance (stored inside pi_magic_tolerance), they
        will indicate that via a string like "3/4" (for pi*3/4) or "1/s3" (for pi/sqrt(3)). 
        Otherwise returns the number.
        """

        for n in range(1, 8):

            for m in np.arange(-10, 10, 1):

                if m != 0:

                    if np.abs(number / (m / np.sqrt(n)) - 1) < self.pi_magic_tolerance_coupling / np.abs(m):
                        return r'$\frac{{{}}}{{\sqrt{}}}$'.format(m, n)

        if np.abs(number) < 1E-1:
            return "0"

        return f"{np.round((number), 1)}"

    def pi_magic(self, number):
        """
        Takes any number as input and compares
        it against (m/n)*pi as well as pi/sqrt(n). If it finds a match according
        to some relative tolerance (stored inside pi_magic_tolerance), they
        will indicate that via a string like "3/4" (for pi*3/4) or "1/s3" (for pi/sqrt(3)). 
        Otherwise returns the number.
        """

        for n in range(1, 17):
            for m in range(-17, 17):
                if m != 0:
                    if np.abs(number / (m / n) - 1) < self.pi_magic_tolerance:

                        if n == 1:
                            if np.abs(m) == 1:
                                return r"1"
                        elif n == 2 and m == 4:
                            return "0"
                        else:
                            return f"{m}/{n}"

        if np.abs(number) < 1E-1:
            return "0"
        else:

            # return f"{np.round((((number)-1/2)%2)-(1/2), 1)}"
            return f"{np.round(number, 2)}"

    def simple_measure_text(self, measure_choice):
        num1 = self.pi_magic(measure_choice[0])
        num2 = self.pi_magic(measure_choice[1])

        if num1 == "0":
            return f"No meas"
        return f"{num1}|{num2}"

    def simple_control_text(self, control_choice):
        num1 = self.pi_magic(control_choice[0])
        num2 = self.pi_magic_coupling(control_choice[1])

        if num1 == "1":
            num1 = "X"
        if "sqrt" in num2:
            numerator = num2.split("frac")[1].split("}")[0]
            denominator = num2.split("sqrt")[1].split("}")[0]

            num2 = r"$\updownarrow{{{}}}$".format(denominator)
        return f"{num1}|{num2}"

    def plot_path(self, measure_choices, control_choices, measurement_results, fidelity, goal, spacing="double"):
        """
        Plot a single path from the tree.
        
        Parameters
        ----------

        measure_choices:
            shape = [num_time_steps,num_measurement_parameters]
        control_choices:
            shape = [num_time_steps,num_control_parameters]
        measurement_results:
            shape = [num_time_steps]
            assumed +1 or -1
        """

        def measure_circle(x, y, measure=None, fill=False):
            #             self.ax.add_patch(plt.Circle((x,y),radius=self.state_circle_radius,
            #                              edgecolor=self.state_circle_color,
            #                             facecolor=(self.state_circle_color if fill else self.state_circle_fill_color),
            #                             zorder=10))
            self.ax.scatter(x, y, c=self.state_circle_fill_color, edgecolor=self.state_circle_color,
                            s=self.measure_circle_radius, zorder=20)
            if measure:
                self.ax.text(x, y - 0.01, int(measure), color="black",
                             fontsize=self.measure_circle_fontsize,
                             horizontalalignment='center', verticalalignment="center", zorder=30)

        def node_circle(x, y, fill=True):
            #             self.ax.add_patch(plt.Circle((x,y),radius=self.measure_circle_radius,
            #                              edgecolor=self.measure_circle_color,
            #                             facecolor=self.measure_circle_fill_color,
            #                             zorder=10))
            self.ax.scatter(x, y, c=self.measure_circle_fill_color, edgecolor=self.measure_circle_fill_color,
                            s=self.node_circle_radius)

        def measure_box(x, y, text):
            self.ax.text(x + 0.5, y, text, color=self.measure_box_text_color,
                         fontsize=self.measure_box_fontsize,
                         horizontalalignment='center', fontweight=1000,
                         bbox=dict(facecolor=self.measure_box_fill_color,
                                   edgecolor='none', boxstyle='round,pad=.2'), zorder=10)

        def control_box(x, y, text):
            self.ax.text(x + 0.5, y, text, color=self.control_box_text_color,
                         fontsize=self.control_box_fontsize,
                         horizontalalignment='center', fontweight=1000,
                         bbox=dict(facecolor=self.control_box_fill_color,
                                   edgecolor='none', boxstyle='round,pad=.2'), zorder=10)

        num_time_steps = measure_choices.shape[0]
        x, y = 0, 0

        for time_step in range(num_time_steps):
            if x == 0:
                node_circle(x, y, fill=True)

            self.ax.plot([x, x + 1], [y, y], color=self.path_color)
            y_box = y
            if self.display_multiple_control_values:
                while (x, y_box) in self.text_slot_occupied:
                    y_box += self.box_shift
            if measurement_results[time_step] == 0:
                measure_box(x, y_box + self.box_translate + 0.01, self.measure_text(measure_choices[time_step]))
            else:
                measure_box(x, y_box + self.box_translate, self.measure_text(measure_choices[time_step]))
            if self.display_multiple_control_values:
                self.text_slot_occupied[(x, y_box)] = True
            if goal == "purity":
                y_new = y + measurement_results[time_step] / (2 ** time_step)
            else:
                if spacing == "double":
                    y_new = y + measurement_results[time_step] / (2 ** time_step)
                if spacing == "single":
                    y_new = y + measurement_results[time_step] / (time_step + 1)
            node_circle(x + 1, y)

            if goal == "fidelity":
                self.ax.plot([x + 1, x + 1, x + 2], [y, y_new, y_new], color=self.path_color)
                control_box(x + 1, y_new + self.box_translate,
                            self.control_text(control_choices[time_step]))
                x = x + 2
                y = y_new
                measure_circle(x - 1, y, measurement_results[time_step])
                if time_step == num_time_steps - 1:
                    measure_circle(x - 1, y, measurement_results[time_step])
                    node_circle(x, y, fill=True)
            #                     self.ax.text(x+self.fidelity_translate, y-0.02, r"F: {{{}}}".format(np.round(fidelity, 2)), color="black",
            #                                  fontsize=self.measure_box_fontsize,
            #                                  horizontalalignment='center', verticalalignment="center", zorder=10)
            if goal == "purity":
                if time_step < num_time_steps - 1:
                    self.ax.plot([x + 1, x + 1, x + 2], [y, y_new, y_new], color=self.path_color)
                    # state_circle(x,y, measurement_results[time_step])

                if time_step == num_time_steps - 1:
                    # y_new = y+measurement_results[time_step]/(time_step+10)
                    self.ax.plot([x + 1, x + 1, x + 1.2], [y, y_new, y_new], color=self.path_color)
                    # state_circle(x,y, measurement_results[time_step])
                    # state_circle(x+1,y, fill=True)
                    # self.ax.text(x+1+self.fidelity_translate, y, f"Pur: {np.round(fidelity, 2)}", color="black",
                    #             fontsize=self.measure_box_fontsize,
                    #             horizontalalignment='center', verticalalignment="center", zorder=10)
                x = x + 1
                y = y_new

    def load_info_and_plot(self, folder, steps, seed, name, n_average, how_many=False, steps_to_plot=3, save=False,
                           mode="network", rho_target=None, filename=None, N_cavity=None):
        if N_cavity is None:
            N_cavity = 10
        else:
            N_cavity = N_cavity
        max_steps = steps
        batch_size = 2 ** (max_steps + 1)  # number of trajectories

        rho_init = create_batched_tensor_state(N_cavity=N_cavity,
                                               batch_size=batch_size,
                                               qubit_state=[0, 1],
                                               # cavity_state=0,
                                               thermal=True, n_average=n_average
                                               )
        if rho_target is None:
            cavity_state = np.zeros(N_cavity)
            for x in name.split("state")[1].split("_"):
                cavity_state[int(x)] = 1

            rho_target = create_cavity_state(N_cavity=N_cavity,
                                             cavity_state=cavity_state)
        if mode == "network":
            network = tf.keras.Sequential(
                [
                    tf.keras.layers.GRU(30, batch_input_shape=(batch_size, 1, 1),
                                        stateful=True),
                    tf.keras.layers.Dense(4, dtype="float64", bias_initializer=tf.keras.initializers.Constant(np.pi))]
            )
            network.load_weights(f"{folder}/network_{name}_nav_{n_average}_steps{steps}_seed{seed}.h5")
            parameters = network
            F_1_0 = tf.Variable(np.load(f"{folder}/F_1_0_{name}_nav_{n_average}_steps{steps}_seed{seed}.npy"))
        if mode == "lookup":
            if filename is None:
                F = tf.Variable(np.load(f"{folder}/{name}_steps{steps}_casetable_seed{seed}_lookup.npy"))
                F_1_0 = tf.Variable(np.load(f"{folder}/F_1_0_{name}_nav_{n_average}_steps{steps}_seed{seed}.npy"))
            else:
                F = tf.Variable(np.load(f"{folder}/{filename}_lookup.npy"))
                F_1_0 = tf.Variable(np.load(f"{folder}/F_1_0_{filename}.npy"))
            parameters = F

        _, _, _, _, info = train_step(
            rho_init=rho_init,
            rho_target=rho_target,
            max_steps=max_steps,
            N_cavity=N_cavity,
            F_1_0=F_1_0,
            mode=mode,
            goal="fidelity",
            parameters=parameters,
            type_input="measure",
            measure_op="both",
            fake_measure=True,
            control=True,
            log=True,
            new_method=True)

        if how_many:
            arr1inds = np.exp(info["log_probs"][-1]).argsort()[::-1][:how_many]
        else:
            arr1inds = np.arange(batch_size)
        info["measurements"] = info["measurements"][:, arr1inds]
        info["controls"] = info["controls"][:, arr1inds]
        info["fidelities"] = info["fidelities"][:, arr1inds]
        info["log_probs"] = np.array(info["log_probs"])[:, arr1inds]

        if steps_to_plot > steps:
            steps_to_plot = steps

        # batch_size = how_many
        measurement_results = info["measurements"][:steps_to_plot].T
        batchsize = info["measurements"].shape[1]

        num_time_steps = steps_to_plot
        measure_choices = np.zeros([batchsize, num_time_steps, 2])
        measure_choices[:, 0, :] = [F_1_0[0], F_1_0[1]]  # special values
        measure_choices[:, 1:, 0] = info["controls"][:steps_to_plot - 1, :, 0].T  # shape [batchsize,num_time_steps-1]
        measure_choices[:, 1:, 1] = info["controls"][:steps_to_plot - 1, :, -1].T  # shape [batchsize,num_time_steps-1]

        # measure_choices[:, :, 0] %= np.pi
        # measure_choices[:, :, 0] *= 2
        # measure_choices[:, :, 1] %= 2*np.pi

        control_choices = np.zeros([batchsize, num_time_steps, 2])
        control_choices[:, :, 0] = info["controls"][:steps_to_plot, :, 1].T  # shape [batchsize,num_time_steps-1]
        control_choices[:, :, 1] = info["controls"][:steps_to_plot, :, 2].T  # shape [batchsize,num_time_steps-1]
        # control_choices[:, :, 0] %= 2*np.pi

        measure_choices[:, :, 0] = ((((measure_choices[:, :, 0] / np.pi)) % 1))
        measure_choices[:, :, 1] = ((((measure_choices[:, :, 1] / np.pi)) % 2))

        control_choices[:, :, 0] = ((((control_choices[:, :, 0] / np.pi)) % 1)) * 2
        control_choices[:, :, 1] = (control_choices[:, :, 1] / np.pi) * 2

        fidelities = info["fidelities"][-1]

        self.plot(measure_choices,
                  control_choices,
                  measurement_results,
                  fidelities,
                  display_multiple_control_values=False,
                  spacing=("single" if how_many else "double"),
                  save=save, path=f"pdfs/tree_{name}_nav_{n_average}_steps{steps}_seed{seed}.pdf")

        return info

    def plot(self, measure_choices, control_choices, measurement_results, fidelities=None, spacing="double",
             display_multiple_control_values=False, goal="fidelity", save=False, path=None):
        """
        Plot a decision tree, where measure_choices come before each split,
        representing a measurement, and control choices come after, representing
        the control.
        
        The line leading towards the filled circle carries the measurement parameters 
        (purple box), the line leading towards the open circle carries the control 
        parameters for the respective branch (red box).
        
        The filled circle should be understood as the measurement. 
        "+1" or "-1" measurement results go up or down, respectively.
        
        This works by providing a full batch of trajectories, each with their
        measurement_results (+1 or -1 in each step).

        Note: The measure and control parameters are converted into strings
        by the methods measure_text and control_text. You can also set them
        to your own functions, which take a vector of numbers as input and produce
        a string as return value. The predefined methods access the
        helper method pi_magic, which takes any value as input and compares
        it against (m/n)*pi as well as pi/sqrt(n). If it finds a match according
        to some relative tolerance (stored inside pi_magic_tolerance), they
        will indicate that via a string like "3/4" (for pi*3/4) or "/s3" (for pi/sqrt(3)).
        
        Parameters
        ----------

        measure_choices:
            shape = [batchsize,num_time_steps,num_measurement_parameters]
        control_choices:
            shape = [batchsize,num_time_steps,num_control_parameters]
        measurement_results:
            shape = [batchsize,num_time_steps]
        display_multiple_control_values:
            If True, print multiple control values below each other.
            (will not be necessary if only msmts are stochastic)
            
        Note: If for some reason you are treating the first measurement choice
        differently, you still need to insert it properly into the array passed
        to the method here!
        """

        self.display_multiple_control_values = display_multiple_control_values
        batchsize = measure_choices.shape[0]
        if self.display_multiple_control_values:
            self.text_slot_occupied = {}  # will keep track of where text boxes have been put
        for idx in range(batchsize):
            self.plot_path(measure_choices[idx], control_choices[idx], measurement_results[idx], fidelities[idx],
                           goal=goal, spacing=spacing)
        # self.ax.set_aspect('equal')
        self.ax.axis('off')
        plt.tight_layout()
        if save:
            self.fig.savefig(path, format="pdf", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close('all')
