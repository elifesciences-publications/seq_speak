"""
Plotting functions for replay smln rslts.
"""
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d as smooth

from disp import set_font_size, set_n_x_ticks, set_n_y_ticks


def heat_maps(rslt, epoch=None, cmap='viridis', sct_sz=25):
    """
    Plot heatmaps showing:
        1. W_PC_G values at start of trial.
        2. W_PC_G values at replay trigger.
        3. # spks per PC within detection wdw.
        4. Firing order of first spikes.
    """
    # Potentiation profile
    pcs = np.nonzero(rslt.ntwk.types_rcr == 'PC')[0]

    sgm = rslt.ntwk.sgm[pcs]

    ## get corresponding place fields
    pfx_pcs = rslt.ntwk.pfxs[pcs]
    pfy_pcs = rslt.ntwk.pfys[pcs]

    ## make plots
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    
    v_min = 1
    v_max = rslt.p['SGM_MAX']
    sgm_ticks = np.linspace(v_min, v_max, 5)

    im = ax.scatter(pfx_pcs, pfy_pcs, c=sgm, s=sct_sz, vmin=v_min, vmax=v_max, cmap='hot')

    ax.set_title('Trajectory-induced LTP-IE')

    ## colorbar nonsense
    divider = make_axes_locatable(ax)
    c_ax = divider.append_axes('right', '5%', pad=0.05)
    cb = fig.colorbar(im, cax=c_ax, ticks=sgm_ticks)
    c_ax.set_ylabel('$\sigma$')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    set_n_x_ticks(ax, 5, -rslt.s_params['BOX_W']/2, rslt.s_params['BOX_W']/2)
    set_n_y_ticks(ax, 5, -rslt.s_params['BOX_H']/2, rslt.s_params['BOX_H']/2)
    ax.set_facecolor((.7, .7, .7))
    set_font_size(ax, 20)

    set_font_size(c_ax, 20)

    figs = [fig]
    axss = [ax]

    # Spike statistics
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)

    ## detection wdw
    if epoch is None:
        start = rslt.trg[0]['T']
        end = start + rslt.s_params['metrics']['WDW']
    else:
        start = epoch[0]
        end = epoch[1]

    t_mask = (start <= rslt.ts) & (rslt.ts < end)

    ## PC mask and PFs
    pc_mask = rslt.ntwk.types_rcr == 'PC'

    pfxs_pc = rslt.ntwk.pfxs[pc_mask]
    pfys_pc = rslt.ntwk.pfys[pc_mask]

    ## PC spk cts within detection window
    spks_wdw_pc = rslt.spks[t_mask][:, pc_mask]
    spk_ct_wdw_pc = spks_wdw_pc.sum(0)

    ## discrete colormap for showing spk cts
    c_map_tmp = plt.cm.jet
    c_map_list = [c_map_tmp(i) for i in range(c_map_tmp.N)]
    c_map_list[0] = (0., 0., 0., 1.)
    c_map = c_map_tmp.from_list('spk_ct', c_map_list, c_map_tmp.N)

    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mpl.colors.BoundaryNorm(bounds, c_map.N)

    im_0 = axs[0].scatter(
        pfxs_pc, pfys_pc, c=spk_ct_wdw_pc, s=sct_sz, cmap=c_map, norm=norm)
    divider_0 = make_axes_locatable(axs[0])
    c_ax_0 = divider_0.append_axes('right', size='5%', pad=0.05)

    cb_0 = fig.colorbar(im_0, cax=c_ax_0, ticks=range(6))
    cb_0.set_ticklabels([0, 1, 2, 3, 4, '>4'])

    axs[0].set_aspect('equal')
    set_n_x_ticks(axs[0], 5, -rslt.s_params['BOX_W']/2, rslt.s_params['BOX_W']/2)
    set_n_y_ticks(axs[0], 5, -rslt.s_params['BOX_H']/2, rslt.s_params['BOX_H']/2)
    axs[0].set_facecolor((.7, .7, .7))

    axs[0].set_xlabel('X (m)')
    axs[0].set_ylabel('Y (m)')
    axs[0].set_title('detection wdw spk ct')

    
    for ax in [axs[0], cb_0.ax]:
        set_font_size(ax, 20)

    # PC spiking order
    if np.any(spk_ct_wdw_pc):

        ## black bkgd for all PCs
        axs[1].scatter(pfxs_pc, pfys_pc, c='k', s=25, zorder=-1)

        ## color PCs according to timing of first spike
        spk_mask = spk_ct_wdw_pc > 0
        spk_order = np.argmax(spks_wdw_pc[:, spk_mask], 0)
        spk_order = np.argsort(spk_order).argsort()
        
        v_min = spk_order.min()
        v_max = spk_order.max()

        im_1 = axs[1].scatter(
            pfxs_pc[spk_mask], pfys_pc[spk_mask], c=spk_order, s=sct_sz,
            vmin=v_min, vmax=v_max, cmap=cmap, zorder=0)

        divider_1 = make_axes_locatable(axs[1])
        c_ax_1 = divider_1.append_axes('right', size='5%', pad=0.05)

        cb_1 = fig.colorbar(im_1, cax=c_ax_1, ticks=[v_min, v_max])
        cb_1.set_ticklabels(['first', 'last'])

        axs[1].set_aspect('equal')
        
        set_n_x_ticks(axs[1], 5, -rslt.s_params['BOX_W']/2, rslt.s_params['BOX_W']/2)
        set_n_y_ticks(axs[1], 5, -rslt.s_params['BOX_H']/2, rslt.s_params['BOX_H']/2)
        
        axs[1].set_xlabel('X (m)')
        axs[1].set_ylabel('Y (m)')
        axs[1].set_title('Replay spike order')
        for ax in [axs[1], cb_1.ax]:
            set_font_size(ax, 20)
    else:
        axs[1].set_title('No PC spks')
        set_font_size(axs[1], 20)
    axs[1].set_facecolor((.7, .7, .7))

    figs.append(fig)
    axss.append(axs)
    
    return figs, axss
        

# PC (excitatory) and INH raster/spike rate plot funcs
def plot_pc(rslt, smoothness=1):
    fig, axs = plt.subplots(2, 1, figsize=(15, 6), sharex=True, tight_layout=True)
    
    # get spks
    spks_pc = rslt.spks[:, :rslt.p['N_PC']]
    
    # raster
    t_idxs_spks_pc, nrn_spks_pc = spks_pc.nonzero()
    t_spks_pc = t_idxs_spks_pc * rslt.dt
    
    axs[0].scatter(t_spks_pc, nrn_spks_pc, s=5, c='k')
    axs[0].set_ylim(-1, rslt.p['N_PC'])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Neuron')
    axs[0].set_title('PC activity')
    
    # inh population average
    axs[1].plot(rslt.ts, smooth(spks_pc.sum(axis=1) / (rslt.dt * rslt.p['N_PC']), smoothness), c='k', lw=3)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Average spike rate')
    
    for ax in axs:
        ax.set_xlim(0, rslt.ts[-1])
        set_font_size(ax, 16)
    
    return fig, axs


def plot_inh(rslt, smoothness=1):
    fig, axs = plt.subplots(2, 1, figsize=(15, 6), sharex=True, tight_layout=True)
    
    # get spks
    spks_inh = rslt.spks[:, -rslt.p['N_INH']:]
    
    # raster
    t_idxs_spks_inh, nrn_spks_inh = spks_inh.nonzero()
    t_spks_inh = t_idxs_spks_inh * rslt.dt
    
    axs[0].scatter(t_spks_inh, nrn_spks_inh, s=5, c='r')
    axs[0].set_ylim(-1, rslt.p['N_INH'])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Neuron')
    axs[0].set_title('Inhibitory activity')
    
    # inh population average
    axs[1].plot(rslt.ts, smooth(spks_inh.sum(axis=1) / (rslt.dt * rslt.p['N_INH']), smoothness), c='r', lw=3)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Average spike rate')
    
    for ax in axs:
        ax.set_xlim(0, rslt.ts[-1])
        set_font_size(ax, 16)
    
    return fig, axs

    
# selective raster plot
def raster(rslt, xys, colors, cmap, nearest, epoch, trg_plt, y_lim, y_ticks, n_t_ticks=None, fig_size=(15, 4), title=None):
    """
    Generate a raster plot of spikes from a smln.
    
    :param xys: list of (x, y) locs to plot spks from nearby cells for
    :param nearest: # of cells per (x, y)
    :param epoch: 'replay', 'wdw', 'trj', or 'full', specifying which epoch
        to make raster for (replay, detection window, trajectory, or full smln)
    """
    # get ordered idxs of PCs to plot
    ## get pfs
    pc_mask = rslt.ntwk.types_rcr == 'PC'
    pfxs = rslt.ntwk.pfxs[pc_mask]
    pfys = rslt.ntwk.pfys[pc_mask]
    
    ## loop through (x, y) pairs and add idxs of nearest PCs
    ### pc_c_dict_0 uses original pc idxs, pc_c_dict_1 uses simplified pc idxs
    pc_idxs, pc_c_dict_0, pc_c_dict_1 = get_idxs_nearest(xys, pfxs, pfys, nearest, colors) 
    
    # get all spks for selected PCs
    spks_pc_chosen = rslt.spks[:, pc_idxs]
    
    # get desired time window
    if epoch == 'replay':
        start = 0
        end = rslt.schedule['D_SMLN']
    elif isinstance(epoch, tuple):
        start = epoch[0]
        end = epoch[1]
    
    t_mask = (start <= rslt.ts) & (rslt.ts < end)
    t_start = rslt.ts[t_mask][0]
    
    spk_t_idxs, pcs = spks_pc_chosen[t_mask].nonzero()
    spk_ts = spk_t_idxs * rslt.s_params['DT'] + t_start
    
    # make plots
    fig, ax = plt.subplots(1, 1, figsize=fig_size, tight_layout=True)
    
    ## spks
    c = [pc_c_dict_1[pc] for pc in pcs]
    ax.scatter(spk_ts, pcs, c=c, s=30, vmin=0, vmax=1, cmap=cmap, lw=.5, edgecolor='k')
    
    ## replay trigger
    for trg, (y, marker) in zip(rslt.trg, trg_plt):
        ax.scatter(trg['T'], y, marker=marker, s=100, c='k')
    
    ax.set_xlim(start, end)
    ax.set_ylim(y_lim)
    ax.set_yticks(y_ticks)
    
    if n_t_ticks is not None:
        set_n_x_ticks(ax_0, n_t_ticks)
    
    ax.set_xlabel('t (s)')
    ax.set_ylabel('Neuron')
    if title is not None:
        ax.set_title(title)
    
    ax.set_facecolor((.9, .9, .9))
    
    set_font_size(ax, 20)
        
    return fig, ax

    
def get_idxs_nearest(xys, pfxs, pfys, nearest, colors):
    """
    Get ordered idxs of place fields nearest to a
    sequence of (x, y) points.
    """
    idxs = []
    c_dict_0 = {}
    c_dict_1 = []
    
    for xy, color in zip(xys, colors):
        # get dists of all PFs to (x, y)
        dx = pfxs - xy[0]
        dy = pfys - xy[1]
        d = np.sqrt(dx**2 + dy**2)
        
        # add idxs of closest neurons to list
        pcs = list(d.argsort()[:nearest])
        idxs.extend(pcs)
        
        for pc in pcs:
            c_dict_0[pc] = color
            
        c_dict_1.extend(len(pcs)*[color])
        
    return idxs, c_dict_0, np.array(c_dict_1)