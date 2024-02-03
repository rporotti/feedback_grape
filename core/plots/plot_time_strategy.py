import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

matplotlib.rcParams["figure.dpi"] = 300
from PyPDF2 import PdfFileMerger
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from core.create_state import create_batched_tensor_state, create_cavity_state
from core.training.utils import compute_fidelity
from core.training.train_function import train_step
import seaborn as sns

my_cmap = sns.color_palette("rocket", as_cmap=True)
# wigner_cmap = sns.color_palette("vlag", as_cmap=True)
wigner_cmap = sns.diverging_palette(255, 13, sep=1, s=100, as_cmap=True)


def plot(folder, steps, seed, name, n_average, filename=None, gamma=None, rho_target=None, cavity_state=None,
         just_one=False, plot_control=True, fig=None, gs=None, save=True, return_info=False, index=False,
         fake_measure=False, N_control_steps=1, N_cavity=10, rho_init=None, batch_size=None, timesteps=None,
         control=True, feedback=True, type_input="measure", compl_controls=False, epsilon=1E-10, lim_y=False,
         mode="network", new_method=False):
    if gamma is None:
        gamma = 0.0

    substeps = 1

    max_steps = steps
    if batch_size is None:
        batch_size = 2 ** (max_steps + 1)  # number of trajectories

    pdfs = ["pdfs/" + "{:04d}".format(traj) + ".pdf" for traj in range(batch_size)]
    for file in pdfs:
        if os.path.exists(file):
            os.remove(file)
    if rho_init is None:

        if cavity_state is not None:
            rho_init = create_batched_tensor_state(N_cavity=N_cavity,
                                                   batch_size=batch_size,
                                                   qubit_state=[0, 1],
                                                   cavity_state=cavity_state,
                                                   # thermal=True, n_average=n_average
                                                   )
        else:
            rho_init = create_batched_tensor_state(N_cavity=N_cavity,
                                                   batch_size=batch_size,
                                                   qubit_state=[0, 1],
                                                   # cavity_state=0,
                                                   thermal=True, n_average=n_average
                                                   )
    if rho_target == None:
        cavity_state = np.zeros(N_cavity)
        for x in name.split("state")[1].split("_"):
            cavity_state[int(x)] = 1

        rho_target = create_cavity_state(N_cavity=N_cavity,
                                         cavity_state=cavity_state)

    if compl_controls:
        output_neurons = 6
    else:
        output_neurons = 4
    if mode == "network":
        if type_input == "state":
            feedback = False
            input_shape = (None, 2, N_cavity * 2, N_cavity * 2)
            if N_control_steps == 1:
                network = tf.keras.Sequential(
                    [tf.keras.layers.Flatten(),
                     tf.keras.layers.Dense(30, input_shape=input_shape, activation='tanh', dtype="float64"),
                     tf.keras.layers.Dense(30, activation='tanh', dtype="float64"),
                     tf.keras.layers.Dense(output_neurons, dtype="float64",
                                           bias_initializer=tf.keras.initializers.Constant(np.pi))]
                )
            else:
                network = tf.keras.Sequential(
                    [tf.keras.layers.Flatten(),
                     tf.keras.layers.Dense(30, input_shape=input_shape, activation='tanh', dtype="float64"),
                     tf.keras.layers.Dense(30, activation='tanh', dtype="float64"),
                     tf.keras.layers.Dense(output_neurons * N_control_steps, dtype="float64",
                                           bias_initializer=tf.keras.initializers.Constant(np.pi)),
                     tf.keras.layers.Reshape((N_control_steps, output_neurons))]
                )

            network.build(input_shape)
        if type_input == "measure":

            if N_control_steps == 1:
                network = tf.keras.Sequential(
                    [
                        tf.keras.layers.GRU(30, batch_input_shape=(batch_size, 1, 1),
                                            stateful=True),
                        tf.keras.layers.Dense(output_neurons, dtype="float64",
                                              bias_initializer=tf.keras.initializers.Constant(np.pi / 2))]
                )

            else:
                network = tf.keras.Sequential(
                    [
                        tf.keras.layers.GRU(30, batch_input_shape=(batch_size, 1, 1),
                                            stateful=True),
                        tf.keras.layers.Dense(output_neurons * N_control_steps, dtype="float64",
                                              bias_initializer=tf.keras.initializers.Constant(np.pi / 2)),
                        tf.keras.layers.Reshape((N_control_steps, output_neurons))

                    ]
                )
            network.compile()

        #     network.load_weights(f"{folder}/network_{name}_nav_{n_average}_steps{steps}_seed{seed}.h5")
        #     F_1_0 = tf.Variable(np.load(f"{folder}/F_1_0_{name}_nav_{n_average}_steps{steps}_seed{seed}.npy"))
        if filename:
            network.load_weights(f"{folder}/network_{filename}.h5")
            F_1_0 = tf.Variable(np.load(f"{folder}/F_1_0_{filename}.npy"))

        else:
            network.load_weights(f"{folder}/network_{name}_nav_{n_average}_steps{steps}_seed{seed}.h5")
            F_1_0 = tf.Variable(np.load(f"{folder}/F_1_0_{name}_nav_{n_average}_steps{steps}_seed{seed}.npy"))
        parameters = network

    if mode == "lookup":
        if filename:
            try:
                F = tf.Variable(np.load(f"{folder}/{filename}_lookup.npy"))
                F_1_0 = tf.Variable(np.load(f"{folder}/F_1_0_{filename}.npy"))
            except:
                F = tf.Variable(np.load(f"{folder}/lookup_{filename}"))
                F_1_0 = tf.Variable(np.load(f"{folder}/F_1_0_{filename}"))
        else:
            F = tf.Variable(np.load(f"{folder}/{name}_steps{steps}_casetable_seed{seed}_lookup.npy"))
            F_1_0 = tf.Variable(np.load(f"{folder}/F_1_0_{name}_nav_{n_average}_steps{steps}_seed{seed}.npy"))
        parameters = F
    # gs0 = matplotlib.gridspec.GridSpec(nrows=2, ncols=3, )

    if timesteps is not None:
        max_steps = timesteps

    #     info = plot_strategy(network=network,
    #         rho_init=rho_init,
    #         rho_target=rho_target,
    #         max_steps=max_steps,
    #         N_cavity=10,
    #         F_1_0=F_1_0,
    #         mode="network",
    #         goal="fidelity" ,
    #         parameters=network,
    #         type_input="measure",
    #         measure_op="both",
    #         fake_measure=True,
    #         mask=False,
    #         mask_cutoff=1,
    #         ticks=True, borders=True,
    #         #save_path="strategy.pdf",
    #         #main_grid = main_grid[:, 1], fig=fig,
    #         return_info=True)

    if gamma == 0.0 and compl_controls == False:

        grads, loss, fidelity, rho, info = train_step(
            rho_init=rho_init,
            rho_target=rho_target,
            max_steps=max_steps,
            N_cavity=N_cavity,
            F_1_0=F_1_0,
            mode=mode,
            goal="fidelity",
            gamma=gamma,
            substeps=substeps,
            parameters=parameters,
            feedback=feedback,
            type_input=type_input,
            measure_op="both",
            fake_measure=fake_measure,
            epsilon=epsilon,
            control=control,
            log=True,
            new_method=new_method)
    elif (compl_controls == True or gamma > 0.0):

        grads, loss, fidelity, rho, info = train_step(
            rho_init=rho_init,
            rho_target=rho_target,
            max_steps=max_steps,
            N_cavity=N_cavity,
            F_1_0=F_1_0,
            mode=mode,
            goal="fidelity",
            gamma=gamma,
            substeps=substeps,
            parameters=network,
            N_control_steps=N_control_steps,
            feedback=feedback,
            type_input=type_input,
            measure_op="both",
            complex_controls=compl_controls,
            fake_measure=fake_measure,
            control=control,
            epsilon=epsilon,
            log=True,
            new_method=new_method)

    if return_info:
        return info
    #     if substeps>1:
    #         indices = [0, 1]+ list(x+idx for idx, x in enumerate(np.arange(substeps*2+2, len(info["rhos"]), substeps*2 ))) + list(np.array([[x+idx, x+1+idx] for idx, x in enumerate(range(2, len(info["rhos"]), substeps*2))]).flatten())
    #         indices = sorted(indices)[:-3]

    #         info["rhos"] = info["rhos"][indices]
    #         info["purities"] = info["purities"][indices]
    #         info["fidelities"] = info["fidelities"][indices]

    if index is not False:

        arr1inds = [index]

    else:
        if just_one:
            arr1inds = np.exp(info["log_probs"][-1]).argsort()[::-1][:1]
        else:
            arr1inds = np.exp(info["log_probs"][-1]).argsort()[::-1]

    n = np.arange(N_cavity - 1)
    a = np.zeros([N_cavity, N_cavity])
    a[n, n + 1] = np.sqrt(n + 1)

    a_dag = np.transpose(a)
    a_dag_a = np.matmul(a_dag, a)

    for idx, traj in enumerate(arr1inds):

        # plt.style.use('bmh')
        plt.style.use('bmh')

        #     info["controls"][:, :, 0] = (info["controls"][:, :, 0]/np.pi)%1
        #     info["controls"][:, :, 1] = (info["controls"][:, :, 1]/np.pi)%1
        #     info["controls"][:, :, 2] = (info["controls"][:, :, 2])/np.pi
        #     info["controls"][:, :, 3] = (info["controls"][:, :, 3]/np.pi)%1
        if plot_control:
            rows = 5
            height_ratios = [1, 0.25, 1, 0.025, 0.5]
            shift = 0
        else:
            rows = 4
            height_ratios = [5, 2, 0.025, 2]
            shift = -1
        if gs:

            gs0 = matplotlib.gridspec.GridSpecFromSubplotSpec(rows, 1,
                                                              subplot_spec=gs,
                                                              hspace=0.1,
                                                              height_ratios=height_ratios)
            ax = []
            for i in range(rows):
                ax.append(fig.add_subplot(gs0[i, 0]))

        else:
            fig, ax = plt.subplots(rows, 1, figsize=(8, 6),
                                   sharex=True,
                                   gridspec_kw={"height_ratios": height_ratios,
                                                # "hspace":0.3
                                                })
        ax[3 + shift].set_visible(False)
        fig.patch.set_alpha(0.)

        colors = ["#e63946", "#2a9d8f", "#e9c46a", "#540b0e"]
        lw = 3
        ### Convert them to Tensorflow
        t_max = len(info["rhos"])
        probabilities = np.zeros((t_max, N_cavity))
        qubit_av = np.zeros((t_max, 2))
        lim = 0.01
        n_av = np.zeros(t_max)
        fidelity = np.zeros(t_max)
        purity = np.zeros(t_max)
        for t in range(t_max):
            rho = info["rhos"][t, traj]
            qubit = tf.linalg.einsum("kikj->ij", tf.reshape(rho, (N_cavity, 2, N_cavity, 2)))
            qubit_av[t] = np.abs(np.diag(qubit))[::-1]
            rho_cav = tf.linalg.einsum("ikjk->ij", tf.reshape(rho, (N_cavity, 2, N_cavity, 2)))
            probabilities[t] = np.abs(np.diag(rho_cav))
            n_av[t] = np.abs(np.trace(rho_cav * a_dag_a))
            probabilities[probabilities <= lim] = lim
            qubit_av[qubit_av <= lim] = lim
            fidelity[t] = compute_fidelity(rho[None], rho_target, epsilon=1E-12)
            purity[t] = tf.abs(tf.linalg.trace(tf.matmul(rho, rho)))
        print(print(info["measurements"][:, traj]))
        n_av[0] = n_average
        im = ax[0].imshow(probabilities.T, interpolation=None, origin="lower", aspect="auto", cmap=my_cmap,
                          norm=matplotlib.colors.LogNorm(vmin=lim, vmax=1), rasterized=True)
        ax[1].imshow(qubit_av.T, interpolation=None, origin="lower", aspect="auto", cmap=my_cmap,
                     norm=matplotlib.colors.LogNorm(vmin=lim, vmax=1), rasterized=True)

        y0 = ax[1].get_position().bounds[1]
        x0 = ax[1].get_position().bounds[0]
        width = ax[1].get_position().bounds[2]
        height1 = ax[0].get_position().bounds[3]
        height2 = ax[1].get_position().bounds[3]

        titles = [r"$F$", r"$\sigma_x$", r"$H_{cav,qb}$", r"$\theta$"]
        cax = fig.add_axes([x0 + width + 0.005, y0, 0.006, height1 + height2])
        cbar = plt.colorbar(im, ax=ax[:2], cax=cax, fraction=0.1, aspect=40, pad=0.01, extend='min')
        cbar.set_ticks(np.logspace(-2, 0, 3, base=10, endpoint=True))
        # cbar.set_label("Probability", rotation=270, labelpad=10)
        cbar.ax.tick_params(labelsize=7)
        labels = [r"$10^{{-{}}}$".format(i) for i in range(3)[::-1]]
        labels[-1] = r"1"
        cbar.set_ticklabels(labels)
        if plot_control:
            if gamma > 0.0:
                print(t_max)
                controls = np.zeros((t_max - 1, batch_size, 4)) * np.nan
                # F
                controls[1, :, 0] = ((((F_1_0[0] / np.pi) - 1 / 2) % 1) - 1 / 2) * 2
                if max_steps > 1:
                    controls[5:-1:4, :, 0] = ((((info["controls"][:-1, :, 0] / np.pi) - 1 / 2) % 1) - (1 / 2)) * 2

                # Theta
                controls[1, :, -1] = (((F_1_0[-1] / np.pi) - 1 / 2) % 1) - 1 / 2
                if max_steps > 1:
                    controls[5:-1:4, :, -1] = (((info["controls"][:-1, :, -1] / np.pi) - 1 / 2) % 1) - (1 / 2)

                controls[2::4, :, 1] = ((((info["controls"][:, :, 1] / np.pi) - 1 / 2) % 1) - (1 / 2)) * 2  # sigmax
                controls[3::4, :, 2] = (info["controls"][:, :, 2] / np.pi * 2)  # H_cav_qb

            else:
                controls = np.zeros((t_max - 1, batch_size, 4)) * np.nan

                # F
                controls[0, :, 0] = ((((F_1_0[0] / np.pi) - 1 / 2) % 1) - 1 / 2) * 2
                if max_steps > 1:
                    controls[3:-1:3, :, 0] = ((((info["controls"][:-1, :, 0] / np.pi) - 1 / 2) % 1) - (1 / 2)) * 2

                # Theta
                controls[0, :, -1] = (((F_1_0[-1] / np.pi) - 1 / 2) % 1) - 1 / 2
                if max_steps > 1:
                    controls[3:-1:3, :, -1] = (((info["controls"][:-1, :, -1] / np.pi) - 1 / 2) % 1) - (1 / 2)

                controls[1::3, :, 1] = ((((info["controls"][:, :, 1] / np.pi) - 1 / 2) % 1) - (1 / 2)) * 2  # sigmax
                controls[2::3, :, 2] = (info["controls"][:, :, 2] / np.pi * 2)  # H_cav_qb

        #         ax[2].hlines((np.pi/2/np.pi), 0, 1, lw=lw, color=colors[0])
        #         ax[2].hlines(0, 0, 1, lw=lw, color=colors[3])
        #         ax[2].vlines(1, (np.pi/2/np.pi), info["controls"][0, traj, 0]/np.pi%1, lw=lw, color=colors[0])
        #         ax[2].vlines(1, 0, (info["controls"][0, traj, 3]/np.pi), lw=lw, color=colors[3])
        if plot_control:
            ax_twin = ax[2].twinx()
            for i in [0, 1, 3]:

                #             ax[2].step(np.arange(1, max_steps+1), ((info["controls"][:, traj, i])/np.pi), color=colors[i], lw=lw, where="post", label=titles[i])
                #             ax[2].hlines(((info["controls"][-1, traj, i])/np.pi), max_steps, max_steps+1, lw=lw, color=colors[i])

                # ax[2].stairs((controls[:, traj, i]/np.pi)%1, [0]+list(np.arange(0, max_steps*2+1, 2)+1), baseline=None, color=colors[i], lw=lw,  label=titles[i])
                for j in range(len(controls)):
                    if not np.isnan(controls[j, traj, i]):
                        line = ax[2].hlines(controls[j, traj, i], j, j + 1, color=colors[i], lw=lw)
                line.set_label(titles[i])
                line = ax_twin.hlines([], 0, 0, color=colors[i], lw=lw)  # trick
                line.set_label(titles[i])

            for i in [2]:
                for j in range(len(controls)):
                    if not np.isnan(controls[j, traj, i]):
                        line = ax_twin.hlines(controls[j, traj, i], j, j + 1, color=colors[i], lw=lw)
                line.set_label(titles[i])

            #         for i in [1, 2]:

            # #             ax[2].step(np.arange(1, max_steps+1), ((info["controls"][:, traj, i])/np.pi), color=colors[i], lw=lw, where="post", label=titles[i])
            # #             ax[2].hlines(((info["controls"][-1, traj, i])/np.pi), max_steps, max_steps+1, lw=lw, color=colors[i])

            #             #ax[2].stairs((controls[:, traj, i]/np.pi)%1, [0]+list(np.arange(0, max_steps*2+1, 2)+1), baseline=None, color=colors[i], lw=lw,  label=titles[i])
            #             for j in range(len(controls)):
            #                 if not np.isnan(controls[j, traj, i]):
            #                     line= ax_twin.hlines(controls[j, traj, i], j, j+1,color=colors[i], lw=lw)
            #             line.set_label(titles[i])

            #         ax[2].legend(loc='lower left',
            #                       ncol=4, fancybox=True, shadow=True, columnspacing=1.0, handletextpad=0.3)
            #         ax_twin.legend(loc='lower right',
            #                       ncol=4, fancybox=True, shadow=True, columnspacing=1.0, handletextpad=0.3)

            ax_twin.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                           ncol=4, fancybox=True, shadow=True, columnspacing=1.0, handletextpad=0.3)

        if not just_one:
            ax[0].set_title(
                f'Fidelity: {np.round(info["fidelities"][-1][traj], 3)}, prob: {np.round(np.exp(info["log_probs"][-1][traj]), 3)}')
        print(traj, f'Fidelity: {info["fidelities"][-1][traj]}, prob: {np.exp(info["log_probs"][-1][traj])}', end="\r")

        #         for i in [1, 2]:
        # #             ax[3].step(np.arange(1, max_steps+1), (info["controls"][:, traj, i])/np.pi*2, lw=lw, where="post", label=titles[i], color=colors[i])
        # #             ax[3].hlines((info["controls"][-1, traj, i])/np.pi*2, max_steps, max_steps+1, lw=lw, color=colors[i])
        #             #ax[3].stairs((controls[:, traj, i]/np.pi)*2, [0]+list(np.arange(0, max_steps*2+1, 2)+1), baseline=None, color=colors[i], lw=lw,  label=titles[i])

        #             for j in range(len(controls)):
        #                 if not np.isnan(controls[j, traj, i]):
        #                     line = ax[2].hlines(controls[j, traj, i], j, j+1,color=colors[i], lw=lw)
        #             line.set_label(titles[i])

        if plot_control:

            for i in np.arange(1, 8):
                for j in range(1, t_max, 3):
                    # ax_twin.hlines(3/np.sqrt(i), j+1, j+2, lw=0.75, color="red", linestyle="dashed", alpha=1)
                    # ax_twin.hlines(2/np.sqrt(i), j+1, j+2, lw=0.75, color="blue", linestyle="dashed", alpha=1)
                    ax_twin.hlines(1 / np.sqrt(i), j + 1, j + 2, lw=0.75, color="black", linestyle="dashed", alpha=1)

        # ax[3].hlines(-1/np.sqrt(i), 0, max_steps+1, color="gray", linestyle="dashed")

        colors = ["#5BC0EB", "#81b29a", "#90be6d"]

        fidelity[fidelity > 1.0] = 1

        # ax[0].stairs(n_av, np.arange(len(n_av)+1)-1/2, lw=lw, color=colors[2], baseline=None,label=r"$\langle n\rangle$")
        #         l = ax[0].legend()
        #         for text in l.get_texts():
        #             text.set_color("white")

        #         for i in [2, 3]:
        #             ax[i].grid(axis='y')

        ax[0].set_xlim(-0.5, max_steps * 2 + 2.5)
        ax[1].set_xlim(-0.5, max_steps * 2 + 2.5)
        ax[0].set(yticks=(np.arange(0, lim_y + 1, 1)),
                  yticklabels=[r"$|{{{}}}\rangle_c$".format(x) for x in np.arange(0, lim_y + 1, 1)],
                  xticks=np.arange(0, t_max, 1) - 1 / 2,
                  xticklabels=[int(idx / 3) if x % 3 == 0 else "" for idx, x in enumerate(np.arange(t_max))]
                  )
        if lim_y:
            ax[0].set_ylim(-1 / 2, lim_y + 1 / 2)
        ax[1].set(yticks=([0, 1]), yticklabels=[r"$|0\rangle_q$", r"$|1\rangle_q$"])
        # ax[2].set_xlim(-0.5, max_steps+1.5)
        # ax[2].set(yticks=([-1/2, 0, 1/2]), yticklabels=[ r"$-\pi/2$", 0, r"$\pi/2$"])
        ax[2].set(yticks=([-1, -1 / 2, 0, 1 / 2, 1]), yticklabels=[r"$-\pi$", r"$-\pi/2$", 0, r"$-\pi/2$", r"$\pi$"])
        # ax[3].set(yticks=([0, 1/2, 1]), yticklabels=[ 0, r"$\pi/2$", r"$\pi$"])
        ax[4 + shift].set_ylim(-0.05, 1.05)
        epsilon = 0.1
        if plot_control:
            ax[2].set_ylim(-1 - epsilon, 1 + epsilon)
            ax_twin.set(yticks=([0, 1, 2]), yticklabels=[0, r"$\pi$", r"$2\pi$"])
            ax_twin.grid(axis='y')
            ax_twin.set_ylim(-epsilon, 2 + epsilon)
            ax[2].set_ylabel(r"$F$, $\sigma_{x}$, $\theta$")
            ax_twin.set_ylabel(r"$H_{cav,qb}$")

        else:
            [plt.setp(ax[i].get_xticklabels(), visible=False) for i in range(3)]
        ax[4 + shift].set_ylim(-0.05, 1.1)
        # ax[0].set_xlim(-0.5, t_max-1.5)
        ax[-1].set_xlabel("Steps", labelpad=0.05)

        ax[0].set_ylabel(r"Cavity")
        ax[1].set_ylabel(r"Qubit")

        if gamma == 0.0:
            xticklabels = np.zeros(t_max, dtype="str")
            xticklabels[1::3] = np.arange(max_steps + 1)
            ax[0].set(xticks=np.arange(0, t_max, 1) - 1 / 2, xticklabels=xticklabels)
            ax[1].set(xticks=np.arange(0, t_max, 1) - 1 / 2, xticklabels=xticklabels)
            ax[4 + shift].set(xticks=np.arange(0, t_max, 1) - 1 / 2, xticklabels=xticklabels)
            ax[4 + shift].tick_params(pad=10)
        ax[4 + shift].set_xlim(-0.5, t_max - 1.5)

        #         ax[4+shift].legend(loc='lower right',
        #                       ncol=1, fancybox=True, shadow=True, columnspacing=1.0, handletextpad=0.3)
        if gamma == 0.0:
            # ax[4+shift].vlines(np.arange(1, t_max-1, 3)-1/2,-0.05, 1.05, linestyle="dashed", color="red")

            #             for idx_meas, i in enumerate(np.arange(1, t_max-1, 3)-1/2):

            #                 ax[4+shift].text(i, (info["measurements"][idx_meas, traj]+1)/2, int(info["measurements"][idx_meas, traj]), color="w",
            #                                  bbox=dict(boxstyle="circle", fc="#54AE32", ec="k", linewidth=2),
            #                                  horizontalalignment='center', verticalalignment="center", zorder=30,fontsize=7        )
            #             for idx_X, i in enumerate(np.arange(2, t_max-1, 3)-1/2):

            #                 ax[1].text(i, -1/2, r"$\hat \sigma_x$", color="w",fontsize=7,
            #                                  bbox=dict(boxstyle="round", fc="#A62A17", ec="k", linewidth=2, pad=0.5),

            #                                  horizontalalignment='center', verticalalignment="center", zorder=30)
            #             for idx_cav, i in enumerate(np.arange(3, t_max-1, 3)-1/2):

            #                 ax[1].text(i, 1.75, r"$\hat H_{cav,qb}$", color="w",fontsize=7,
            #                                  bbox=dict(boxstyle="round", fc="#3274B5", ec="k", linewidth=2, pad=0.5),

            #                                  horizontalalignment='center', verticalalignment="center", zorder=50, clip_on=False)

            ax[0].grid(axis='y')
            ax[1].grid(axis='y')

        ax[4 + shift].stairs(fidelity, np.arange(len(fidelity) + 1) - 1 / 2, lw=lw, color=colors[0], baseline=None,
                             label=r"$\mathrm{Fidelity}$")
        ax[4 + shift].stairs(purity, np.arange(len(purity) + 1) - 1 / 2, lw=lw, color=colors[1], baseline=None,
                             label=r"$\mathrm{Purity}$")
        #         if gamma==0.0:
        #             ax[4+shift].text(4, purity[5]*0.95, r"$\mathrm{Purity}$", va="top", color=colors[1])
        #             ax[4+shift].text(5, fidelity[7]*1.05, r"$\mathrm{Fidelity}$", va="bottom", color=colors[0])
        # ax[4+shift].legend()

        if timesteps is not None:
            [axes.grid(False) for axes in ax]

        if save:
            fig.savefig("pdfs/" + "{:04d}".format(idx) + ".pdf", format="pdf", dpi=300, bbox_inches='tight',
                        pad_inches=0)
        if just_one:
            fig.show()
        else:
            plt.close('all')

    if not just_one:
        merger = PdfFileMerger()

        for pdf in pdfs[:10]:
            merger.append(pdf)

        merger.write(f"pdfs/result_{name}_nav_{n_average}_steps{steps}_seed{seed}.pdf")
        merger.close()

        pdfs = ["pdfs/" + "{:04d}".format(traj) + ".pdf" for traj in range(batch_size)]
        for file in pdfs:
            if os.path.exists(file):
                os.remove(file)
