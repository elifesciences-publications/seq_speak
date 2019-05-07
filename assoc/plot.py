"""
Plotting functions for replay smln rslts.
"""
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d as smooth

from disp import set_font_size, set_n_x_ticks, set_n_y_ticks, set_colors


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


def get_idxs_nearest_simple(xys, pfxs, pfys, nearest):
    """
    Get ordered idxs of place fields nearest to a
    sequence of (x, y) points.
    """
    idxs = []
    
    for xy in xys:
        # get dists of all PFs to (x, y)
        dx = pfxs - xy[0]
        dy = pfys - xy[1]
        d = np.sqrt(dx**2 + dy**2)
        
        # add idxs of closest neurons to list
        pcs = list(d.argsort()[:nearest])
        idxs.append(pcs)
        
    return idxs


def ltp_ie_profile(rslt, ax, sct_sz=25):
    """Heatmap showing LTP-IE-potentiated cells."""
    pcs = np.nonzero(rslt.ntwk.types_rcr == 'PC')[0]

    sgm = rslt.ntwk.sgm[pcs]

    ## get corresponding place fields
    pfx_pcs = rslt.ntwk.pfxs[pcs]
    pfy_pcs = rslt.ntwk.pfys[pcs]

    ## make plots
    v_min = 1
    v_max = rslt.p['SGM_MAX']
    sgm_ticks = np.linspace(v_min, v_max, 3)

    im = ax.scatter(
        pfx_pcs, pfy_pcs, c=sgm, s=sct_sz, vmin=v_min, vmax=v_max, cmap='hot')

    ax.set_title('LTP-IE Profile')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)', labelpad=-12)
    ax.set_aspect('equal')
    set_n_x_ticks(ax, 5, -rslt.s_params['BOX_W']/2, rslt.s_params['BOX_W']/2)
    set_n_y_ticks(ax, 5, -rslt.s_params['BOX_H']/2, rslt.s_params['BOX_H']/2)
    ax.set_facecolor((.7, .7, .7))
    set_font_size(ax, 20)

    return ax


def spike_seq(rslt, ax, fig, epoch, cmap='gist_rainbow', sct_sz=25):
    """Plot spike count and order during specified epoch."""
    ## detection wdw
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

    # PC spiking order
    if np.any(spk_ct_wdw_pc):

        ## black bkgd for all PCs
        ax.scatter(pfxs_pc, pfys_pc, c='k', s=25, zorder=-1)

        ## color PCs according to timing of first spike
        spk_mask = spk_ct_wdw_pc > 0
        spk_order = np.argmax(spks_wdw_pc[:, spk_mask], 0)
        spk_order = np.argsort(spk_order).argsort()
        
        v_min = spk_order.min()
        v_max = spk_order.max()

        im = ax.scatter(
            pfxs_pc[spk_mask], pfys_pc[spk_mask], c=spk_order, s=sct_sz,
            vmin=v_min, vmax=v_max, cmap=cmap, zorder=0)

        divider = make_axes_locatable(ax)
        c_ax = divider.append_axes('right', size='5%', pad=0.05)

        cb = fig.colorbar(im, cax=c_ax, ticks=[v_min, v_max])
        cb.set_ticklabels(['first', 'last'])

        ax.set_aspect('equal')
        
        set_n_x_ticks(ax, 5, -rslt.s_params['BOX_W']/2, rslt.s_params['BOX_W']/2)
        set_n_y_ticks(ax, 5, -rslt.s_params['BOX_H']/2, rslt.s_params['BOX_H']/2)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)', labelpad=-12)
        c_ax.set_ylabel('Spike order', labelpad=-25)
        ax.set_title('Spikes ({} to {} s).'.format(*epoch))
        for ax_ in [ax, cb.ax]:
            set_font_size(ax_, 20)
    else:
        ax.set_title('No PC spks')
        set_font_size(ax, 20)
    ax.set_facecolor((.7, .7, .7))

    return ax, c_ax


def raster_with_pc_inh(
        rslt, xys, colors, cmap, nearest, epoch, trg_plt, y_lim, y_ticks,
        smoothness=1, n_t_ticks=None, fig_size=(15, 9), title=None):
    """
    Make raster plots of PCs specified by place fields, along with full PC/INH rasters/rate traces.
    
    :param xys: list of (x, y) locs to plot spks from nearby cells for
    :param nearest: # of cells per (x, y)
    :param epoch: 'replay', 'wdw', 'trj', or 'full', specifying which epoch
        to make raster for (replay, detection window, trajectory, or full smln)
    """
    fig, axs = plt.subplots(3, 1, figsize=fig_size, tight_layout=True)
    
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
    
    ## spks
    c = [pc_c_dict_1[pc] for pc in pcs]
    axs[0].scatter(spk_ts, pcs, c=c, s=30, vmin=0, vmax=1, cmap=cmap, lw=.5, edgecolor='k')
    
    ## replay trigger
    for trg, (y, marker) in zip(rslt.trg, trg_plt):
        axs[0].scatter(trg['T'], y, marker=marker, s=100, c='k')
    
    axs[0].set_xlim(start, end)
    axs[0].set_ylim(y_lim)
    axs[0].set_yticks(y_ticks)
    
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Neuron')
    axs[0].set_title('Spike sequences')
    if title is not None:
        axs[0].set_title(title)
    
    set_font_size(axs[0], 16)
        
    # PCs
    ## get spks
    spks_pc = rslt.spks[:, :rslt.p['N_PC']]
    
    ## raster
    t_idxs_spks_pc, nrn_spks_pc = spks_pc.nonzero()
    t_spks_pc = t_idxs_spks_pc * rslt.dt
    
    axs[1].scatter(t_spks_pc, nrn_spks_pc, s=5, c='k')
    
    # population firing rate
    axs[2].plot(rslt.ts, smooth(spks_pc.sum(axis=1) / (rslt.dt * rslt.p['N_PC']), smoothness), c='k', lw=3)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('PC spike rate (Hz)')
    axs[2].set_title('Population spike rates')
    
    # INHs
    # get spks
    spks_inh = rslt.spks[:, -rslt.p['N_INH']:]
    
    # raster
    t_idxs_spks_inh, nrn_spks_inh = spks_inh.nonzero()
    t_spks_inh = t_idxs_spks_inh * rslt.dt
    
    axs[1].scatter(t_spks_inh, -(1 + nrn_spks_inh), s=5, c='r')
    axs[1].set_yticks([-rslt.p['N_INH']/2, rslt.p['N_PC']/2])
    axs[1].set_yticklabels(['INH', 'PC'])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_title('Full raster')
    
    for tick_label, color in zip(axs[1].get_yticklabels(), ['r', 'k']):
        tick_label.set_color(color)
    
    # inh population average
    ax_2_twin = axs[2].twinx()
    ax_2_twin.plot(rslt.ts, smooth(spks_inh.sum(axis=1) / (rslt.dt * rslt.p['N_INH']), smoothness), c='r', lw=2)
    ax_2_twin.set_ylabel('INH spike rate (Hz)')
    
    axs[2].set_zorder(ax_2_twin.get_zorder() + 1)
    axs[2].patch.set_visible(False)
    
    set_colors(ax_2_twin, 'r')
    
    for ax in list(axs[1:]) + [ax_2_twin]:
        ax.set_xlim(0, rslt.ts[-1])
        set_font_size(ax, 16)
     
    return fig, axs
