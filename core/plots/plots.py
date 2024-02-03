import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from core.training.train_function import train_step


def plot_strategy(network, rho_init, rho_target, max_steps,
                  N_cavity, F_1_0, mode, goal,
                  parameters, type_input, measure_op,
                  fake_measure=True, mask=False, mask_cutoff=None,
                  borders=False, ticks=True, save_path=None,
                  main_grid=None, fig=None, return_info=False):
    batch_size = rho_init.shape[0]
    if goal == "purity":
        scale = 4
        fidelities = np.zeros((max_steps + 2, batch_size))
        purities = np.zeros((max_steps + 2, batch_size))
        # fig = plt.figure(figsize=(1.5*scale, 2*scale), constrained_layout=True)
    if goal == "fidelity":
        scale = 2
        fidelities = np.zeros((max_steps + 2, batch_size))
        purities = np.zeros((max_steps + 2, batch_size))
        # fig = plt.figure(figsize=(1.5*scale, 5*scale), constrained_layout=True)
    ctrls = np.zeros((max_steps + 1, batch_size, 4))

    grads, loss, fidelity, rho, info = train_step(
        rho_init=rho_init,
        rho_target=rho_target,
        max_steps=max_steps,
        N_cavity=N_cavity,
        F_1_0=F_1_0,
        mode=mode,
        goal=goal,
        parameters=network,
        type_input=type_input,
        measure_op=measure_op,
        fake_measure=fake_measure,
        log=True)
    if return_info:
        return info

    # print(np.repeat(, ).shape)

    probs = np.exp(info["log_probs"])
    fidelities[0] = info["fidelities"][0]
    purities[0] = info["purities"][0]
    ctrls[0, :, 0] = F_1_0[0]
    ctrls[0, :, -1] = F_1_0[1]

    nums = []

    for bit in range(max_steps + 1):
        nums.append([])
        if fake_measure:

            for i in range(batch_size):
                string = "".join([str(int((x + 1) / 2)) for x in info["measurements"][:bit + 1, i]])
                num = 0
                # print(string)
                for idx, s in enumerate(string):
                    num += int(s) * 2 ** idx
                nums[-1].append(num)
            nums[-1] = np.array(nums[-1]) * 2 ** (max_steps - bit)
            # nums[-1] = np.arange(batch_size)
        else:
            nums[-1] = np.arange(batch_size)

    for bit in range(max_steps):
        fidelities[bit + 1] = info["fidelities"][bit + 1][nums[bit]]
        purities[bit + 1] = info["purities"][bit + 1][nums[bit]]
        ctrls[bit + 1] = info["controls"][bit][nums[bit]]
        probs[bit + 1] = probs[bit + 1][nums[bit + 1]]
    fidelities[-1] = info["fidelities"][-1][nums[-1]]
    purities[-1] = info["purities"][-1][nums[-1]]
    # probs[-1] = probs[-1][nums[-1]]

    if mask:
        mask_array = np.ones((max_steps + 2, batch_size)) * 0.98
        mask_array[0] = np.nan
        for bit in range(max_steps):
            cutoff = sorted(probs[bit])[-mask_cutoff]
            mask_array[bit + 1][probs[bit] >= cutoff] = np.nan
        cutoff = sorted(probs[bit])[-mask_cutoff]
        mask_array[-1][probs[-1] >= cutoff] = np.nan

    # Plot
    fidelity_and_purity = [purities, fidelities]

    # gs0 = fig.add_gridspec(2, 1, height_ratios=[2,4])

    gs0 = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1,
                                                      subplot_spec=main_grid,
                                                      hspace=0.05,
                                                      height_ratios=[2, 4])

    bits = max_steps

    if goal == "fidelity":
        numbers1 = np.arange(2)
        numbers2 = np.arange(4)
        height_ratios = [bits + 1, bits + 1, bits, bits]
        order = [0, 3, 1, 2]
    if goal == "purity":
        numbers1 = np.arange(1)
        numbers2 = [0, 3]
        order = [0, 3]
        height_ratios = [bits + 1, bits + 1]

    cmaps = ["copper", "copper"]
    titles = ["Purity", "Fidelity"]
    gs01 = matplotlib.gridspec.GridSpecFromSubplotSpec(len(numbers1), 1, subplot_spec=gs0[0], hspace=0.05)

    axes = [[], []]
    for idx in numbers1:
        ax = fig.add_subplot(gs01[idx, 0])
        ax.axes.xaxis.set_visible(False)
        axes[0].append(ax)
        ax.set_ylabel("Bit")

        im = ax.imshow(np.rot90(fidelity_and_purity[idx], 2),
                       extent=(0, 2 ** (bits + 1), 0, bits + 2),
                       interpolation=None,
                       origin="lower",
                       vmin=0, vmax=1,
                       cmap=cmaps[idx],
                       aspect="auto")
        if mask:
            ax.imshow(np.rot90(mask_array, 2), cmap="gray", extent=(0, 2 ** (bits + 1), 0, bits + 2),
                      interpolation=None, vmin=0, vmax=1,
                      origin="lower", aspect="auto")
        ax.set(yticks=np.arange(1, max_steps + 2) - 1 / 2, yticklabels=np.arange(1, max_steps + 2)[::-1])
        # ax.set_title(titles[idx])

    cbar = plt.colorbar(im, ax=axes[0], fraction=0.1, aspect=40, pad=0.01)

    cbar.set_ticks([0, 1])
    cbar.ax.set_yticklabels([0, 1])

    top = matplotlib.cm.get_cmap('Oranges_r', 128)
    bottom = matplotlib.cm.get_cmap('Oranges', 128)

    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    newcmp = matplotlib.colors.ListedColormap(newcolors, name='OrangeBlue')
    matplotlib.cm.register_cmap(name='OrangeBlue', cmap=newcmp)

    titles = ["Msmt strength", "Msmt axis", r"$\sigma_x$", r"$H_{cav,qb}$"]
    cmaps = ["OrangeBlue", "OrangeBlue", "OrangeBlue", "OrangeBlue"]

    gs02 = matplotlib.gridspec.GridSpecFromSubplotSpec(int(len(numbers2)), 1,
                                                       subplot_spec=gs0[1],

                                                       height_ratios=height_ratios)
    for idx, j in enumerate(numbers2):

        ax = fig.add_subplot(gs02[idx, 0])
        if j > 0:
            ax.set_yticks(np.arange(bits) + 1 / 2)
            ax.set(yticklabels=[""] * bits)
        axes[1].append(ax)

        ax.axes.xaxis.set_visible(False)
        if idx <= 1:
            to_plot = ctrls[:, :, order[idx]]
            extent = (0, 2 ** (bits + 1), 0, bits + 1)
            ax.set_yticks(np.arange(bits) + 1 / 2)
            ax.set(yticklabels=(np.arange(bits) + 1)[::-1])

        if idx > 1:
            to_plot = ctrls[1:, :, order[idx]]
            extent = (0, 2 ** (bits + 1), 0, bits)

            ax.set_yticks(np.arange(bits) + 1 / 2)
            ax.set(yticklabels=(np.arange(bits) + 1)[::-1])

        to_plot = np.rot90(to_plot, 2)
        to_plot = to_plot / np.pi
        to_plot = to_plot % 1

        im = ax.imshow(to_plot,
                       extent=extent,
                       interpolation=None,
                       origin="lower",
                       cmap=cmaps[j],
                       vmin=0, vmax=1,
                       aspect="auto")
        if mask:
            if idx <= 1:
                ax.imshow(np.rot90(mask_array[:-1], 2), cmap="gray", extent=(0, 2 ** (bits + 1), 0, bits + 1),
                          interpolation=None, vmin=0, vmax=1,
                          origin="lower", aspect="auto")
            if idx > 1:
                ax.imshow(np.rot90(mask_array[1:-1], 2), cmap="gray", extent=(0, 2 ** (bits + 1), 0, bits),
                          interpolation=None, vmin=0, vmax=1,
                          origin="lower", aspect="auto")
        ax.set_title(titles[idx])
        # plt.colorbar(im, ax=ax, location='bottom')
        cbar = plt.colorbar(im, ax=ax, fraction=0.1, aspect=20, pad=0.01)
        cbar.set_ticks([0, 1])
        cbar.ax.set_yticklabels([0, r"$\pi$"])

        ax.set_ylabel("Bit")
    if ticks:

        axes[-1][-1].set(xticks=np.arange(2 ** (bits + 1)) + 1 / 2, xticklabels=[""] * 2 ** (bits + 1))
    else:
        axes[0][-1].axes.xaxis.set_visible(False)
        axes[-1][-1].axes.xaxis.set_visible(False)

    bits = max_steps + 1
    if borders:
        for line in range(len(axes)):
            rows = np.arange(1, max_steps + 2)
            if line == 0:
                start = 0
            if line == 1:
                start = -1
            for ax in axes[line]:
                for row in rows:
                    for j in range(0, 2 ** bits, 2 ** (bits - row)):
                        rect = matplotlib.patches.Rectangle((j, bits - row + start),
                                                            2 ** (bits - row),
                                                            1,
                                                            linewidth=0.5,
                                                            # linestyle="dashed",
                                                            edgecolor='k',
                                                            snap=False,
                                                            facecolor='none')
                        ax.add_patch(rect)

    if save_path:
        fig.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight', pad_inches=0)
