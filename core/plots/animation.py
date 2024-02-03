import os
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import matplotlib
import itertools
matplotlib.rcParams["figure.dpi"] = 200
plt.rcParams['animation.ffmpeg_path'] = '/zeropoint/u/system/soft/SLE_15/packages/x86_64/ffmpeg/4.4.0/bin/ffmpeg'
import matplotlib.path as mpath
import matplotlib.patches as mpatches

import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from core.create_state import create_batched_tensor_state, create_cavity_state
from core.training.train_function_old import train_step
from core.plots.plots import plot_strategy
import core.plots.plot_time_strategy as plot_time_strategy
import core.plots.plot_time_strategy as plot_time_strategy
from core.plots.strategytree import StrategyTreePlot
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython import display
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

import seaborn as sns
my_cmap = sns.color_palette("rocket", as_cmap=True)
#wigner_cmap = sns.color_palette("vlag", as_cmap=True)
wigner_cmap = sns.diverging_palette(255, 13, sep=1, s=100,as_cmap=True)




def animation(info, substeps=5, N_wigner=500, length=None, figsize=(8,3)):
    states = []
    substeps=substeps
    wigners = []
    states_fitted = []
    wigners_fitted = []

    N_wigner = N_wigner
    X = np.linspace(-6, 6, N_wigner)
    rhos = info["rhos"][:, 0]
    ctrls = np.repeat(np.abs(info["controls"][:, 0, 2]),3)
    #phases = (np.repeat(np.arctan(np.abs(info["controls"][:, 0, 5])/np.abs(info["controls"][:, 0, 2])   ),2)+np.pi/2)/np.pi
    controls_fitted = []
    phases_fitted=[]
    count = 0
    controls=[]
    for i in range(len(rhos)):


        rho = rhos[i]

        if np.any(rho!=rhos[i-1]):

            rho_part_cav = tf.linalg.einsum("ikjk->ij", tf.reshape(rho, (int(rho.shape[1]/2), 2, int(rho.shape[1]/2), 2))).numpy()
            rho_part_qubit = tf.linalg.diag_part(tf.linalg.einsum("kikj->ij", tf.reshape(rho, (int(rho.shape[1]/2), 2, int(rho.shape[1]/2), 2)))).numpy()
            W = qt.wigner(qt.Qobj(rho_part_cav), X, X)


            states.append(np.real(rho_part_qubit)[::-1])
            wigners.append(W)


            if i>0:
                print
                for substep in range(substeps):
                    wigners_fitted.append(wigners[-2]+(wigners[-1]-wigners[-2])/substeps*substep)
                    states_fitted.append(states[-2]+(states[-1]-states[-2])/substeps*substep)


    for count in range(len(ctrls)):
            for substep in range(substeps):
                val = (ctrls[count-1]+(ctrls[count]-ctrls[count-1])/substeps*substep)
                controls_fitted.append(val)
                #phases_fitted.append((phases[count-1]+(phases[count]-phases[count-1])/substeps*substep-2)*5)
    controls_fitted = np.array(controls_fitted)
    #phases_fitted = np.array(phases_fitted)
    
    
   
    controls_fitted = (controls_fitted - np.min(controls_fitted))/np.ptp(controls_fitted)
    #phases_fitted = (phases_fitted - np.min(phases_fitted))/np.ptp(phases_fitted)
    controls_fitted*=10
    wigners = wigners_fitted
    states = states_fitted
    
    def to_vec(state):
        arr = np.array([
                qt.expect(qt.sigmax(), qt.Qobj(state)),
                qt.expect(qt.sigmay(), qt.Qobj(state)),
                qt.expect(qt.sigmaz(), qt.Qobj(state))     
            ])

        return arr/np.linalg.norm(arr)
    def create_circular_mask(h, w, center=None, radius=None):

        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask

    # Pass the ffmpeg path
    
    
    
    fig = plt.figure(figsize=figsize, dpi=300)


    ax1 = fig.add_subplot(1, 3, 1, projection='3d', elev=0, azim=0
                         )
    ax2 = fig.add_subplot(1, 3, 3)
    ax = fig.add_subplot(1, 3, 2)

    ax1.set_facecolor("white")
    nrm = matplotlib.colors.Normalize(-wigners[0].max(), wigners[0].max())
    w, h = W.shape
    center = (int(w/2), int(h/2))
    radius = int(N_wigner/2)
    mask = create_circular_mask(h, w, center=center, radius=radius)


    masked_img = wigners[0].copy()
    masked_img[~mask] = np.nan
    masked_img[np.abs(masked_img)<5E-3] = 0.0
    im = ax2.imshow(masked_img,
                            cmap=wigner_cmap, 
                            origin="lower", 
                            aspect="equal",
                            norm=nrm)
    ax1.axis('off')
    ax2.axis('off')
    ax.axis('off')
    x = np.linspace(0,7*np.pi,151)
    y = np.sin(x)
    #color = matplotlib.cm.magma(phases[0])
    color = "k"
    line,=ax.plot(x,y, color=color,lw=0.01, zorder=30)
    verts = np.array([[0,2],[0,-2],[2,0],[0,2]]).astype(float)*1.5
    verts[:,0] += x[-1]
    path = mpath.Path(verts)
    patch1 = mpatches.PathPatch(path, fc=color, ec=color,zorder=100)
    ax.add_patch(patch1)

    verts = np.array([[0,2],[0,-2],[-2,0],[0,2]]).astype(float)*1.5
    verts[:,0] += x[0]
    path = mpath.Path(verts)
    patch2 = mpatches.PathPatch(path, fc=color, ec=color,zorder=100)
    ax.add_patch(patch2)
    ax.set_aspect("equal",'datalim')
    ax.relim()
    ax.autoscale_view()
    ax.set_xlim(-np.pi, 8*np.pi)



    b = qt.Bloch(axes=ax1)
    b.frame_alpha=0.0
    #b.ylabel = ["", ""]
    #b.xlabel = ["", ""]
    b.sphere_alpha=0.5

    # ax.set_xlim(0, 2*np.pi)
    # ax.set_ylim(-1.1, 1.1)
    if length:
        frames=length
    else:
        frames = len(wigners)
    vectors = [to_vec(qt.Qobj(states[frame_num])) for frame_num in range(frames)]
    vectors = np.array(vectors)
    pnts = np.array([vectors[:, 0], vectors[:, 1], vectors[:, 2]])
    
    def animate(frame_num):
        print(f"{np.round(frame_num/frames*100, 2)}%", end="\r")
        b.clear()
        #print(states[frame_num])
        b.add_vectors(vectors[frame_num])
        #print(pnts[:(frame_num+1)])
        #b.add_points(pnts[:, :(frame_num+1)], 'l')
        b.make_sphere()
        W = wigners[frame_num]
        nrm = matplotlib.colors.Normalize(-W.max(), W.max())
        masked_img = W.copy()
        masked_img[~mask] = np.nan
        masked_img[np.abs(masked_img)<5E-3] = 0.0
        im = ax2.imshow(masked_img,
                                cmap=wigner_cmap, 
                                origin="lower", 
                                aspect="equal",
                                norm=nrm)
        x = np.linspace(0,7*np.pi,151)
        line.set_data(x, np.sin(x))


        line.set_linewidth(controls_fitted[frame_num])
        #line.set_color(matplotlib.cm.magma(phases_fitted[frame_num]))
        #patch1.set_color(matplotlib.cm.magma(phases_fitted[frame_num]))
        #patch2.set_color(matplotlib.cm.magma(phases_fitted[frame_num]))
        return [line]


    fig.tight_layout()
    fig.patch.set_alpha(0.)
    anim = FuncAnimation(fig, animate, frames=frames, 
                         interval=100, blit=True, repeat=False

    )
    return anim
    # video = anim.to_html5_video()
    # html = display.HTML(video)
    # display.display(html)
    # plt.close()


