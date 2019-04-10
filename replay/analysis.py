import numpy as np
from scipy.ndimage import gaussian_filter1d as smooth
from scipy import stats

from aux import get_segments
from replay.smln import dist_to_trj


cc = np.concatenate


def remove_short_gaps(starts, ends, min_gap):
    """Remove short gaps from list of starts and end times denoting events."""
    gaps = starts[1:] - ends[:-1]  # get gap times (n-1 gaps for n starts and ends)
    mask = gaps >= min_gap  # find all sufficiently long gaps (len n-1)
    
    # convert to proper length mask to select corrected starts and ends
    return starts[cc([[True], mask])], ends[cc([mask, [True]])]


def remove_short_evts(starts, ends, min_evt):
    """Remove short events from list of starts and end times denoting events."""
    mask = (ends - starts) >= min_evt
    return starts[mask], ends[mask]


def get_evts(rslt, a_params):
    """Return start and end times of candidate replay events."""
    # get PC firing rates
    ## PC spks
    spks_pc = rslt.spks[:, :rslt.p['N_PC']]
    
    ## smoothed instantaneous firing rate avg'd over PCs
    fr_pc = smooth(spks_pc.sum(axis=1) / (rslt.dt * rslt.p['N_PC']), a_params['SMOOTH_FR'])
    
    # get start and end time idxs when PC FR is above threshold
    starts, ends = get_segments(fr_pc >= a_params['EVT_DTCN_TH'])
    
    # convert to time
    starts = starts.astype(float) * rslt.dt
    ends = ends.astype(float) * rslt.dt

    # remove too-short gaps btwn events
    if len(starts) > 0:
        starts, ends = remove_short_gaps(starts, ends, a_params['MIN_GAP_DUR'])
    
    # remove too-short events
    if len(starts) > 0:
        starts, ends = remove_short_evts(starts, ends, a_params['MIN_EVT_DUR'])
    
    # remove all events that start before min start time
    if len(starts):
        mask = starts > a_params['MIN_START']
        starts = starts[mask]
        ends = ends[mask]
        
    # remove final event if it hits end of smln
    if len(ends) and ends[-1] >= rslt.ts[-1]:
        starts = starts[:-1]
        ends = ends[:-1]
        
    return starts, ends
    

def get_fr_trj_ntrj(rslt, start, end, a_params):
    """Check whether event exhibits "blowup" behavior."""
    # get spks during candidate replay event
    spks_evt = rslt.spks[(start <= rslt.ts) & (rslt.ts < end), :]
    
    # get mask over trj and non-trj PCs
    pc_mask = rslt.ntwk.types_rcr == 'PC'
    
    sgm_cutoff = .5 * (1 + rslt.p['SGM_MAX'])
    
    trj_mask = (rslt.ntwk.sgm * pc_mask.astype(float)) > sgm_cutoff
    ntrj_mask = (~trj_mask) & pc_mask

    # get trj-PC spks
    spks_trj = spks_evt[:, trj_mask]
    fr_trj = (spks_trj.sum(0) / (end - start)).mean()
    
    # get non-trj-PC spks
    spks_ntrj = spks_evt[:, ntrj_mask]
    fr_ntrj = (spks_ntrj.sum(0) / (end - start)).mean()
    
    # return trj-PC and non-trj-PC firing rates
    return fr_trj, fr_ntrj


def get_pos_t_corr(rslt, start, end, a_params):
    """Check whether event exhibits one-way propagation."""
    # get evt spks
    t_mask = (start <= rslt.ts) & (rslt.ts < end)
    spks_evt = rslt.spks[t_mask, :]
    
    # get trj-PC spks
    ## make masks
    pc_mask = rslt.ntwk.types_rcr == 'PC'
    
    sgm_cutoff = .5 * (1 + rslt.p['SGM_MAX'])
    
    trj_mask = (rslt.ntwk.sgm * pc_mask.astype(float)) > sgm_cutoff
    ntrj_mask = (~trj_mask) & pc_mask
    
    ## apply masks
    spks_trj = spks_evt[:, trj_mask]
    spks_ntrj = spks_evt[:, ntrj_mask]
    
    ## get mask over trj PCs that spiked
    trj_spk_mask = trj_mask & (rslt.spks[t_mask].sum(0) > 0.5)
    
    ## order nrns by place field location along trj
    pfxs = rslt.ntwk.pfxs
    pfys = rslt.ntwk.pfys
    
    ### get pos of spk'ing trj PCs along trj
    pf_dists, pf_order = dist_to_trj(
        pfxs[trj_spk_mask], pfys[trj_spk_mask], rslt.trj['x'], rslt.trj['y'])
    
    ## get order spks occurred in
    spk_order = spks_evt[:, trj_spk_mask].argmax(0)
    
    ## return correlation btwn spk order and pos along trj
    return stats.spearmanr(pf_order, spk_order)[0]

    
def get_metrics(rslt, a_params):
    """
    Compute metrics from network simulation run:
        (1) evt_ct: spontaneous event frequency
        (2) evt_dur: avg spontaneous event duration
        (3) class: blowup, replay (unidirectional), or other
        (4) speed: virtual replay speed
    """
    metrics = {}
    
    # get candidate replay event start and end times
    starts, ends = get_evts(rslt, a_params)
    
    # calc event freq
    metrics['evt_ct'] = int(len(starts))
    
    # calc mean event dur
    metrics['evt_dur'] = np.mean(ends - starts) if len(starts) else -1
    
    ## calc avg event stats
    frs_trj = []
    frs_ntrj = []
    pos_t_corrs = []
    
    for start, end in zip(starts, ends):
        # trj- and non-trj-PC firing rates
        fr_trj, fr_ntrj = get_fr_trj_ntrj(rslt, start, end, a_params)
        frs_trj.append(fr_trj)
        frs_ntrj.append(fr_ntrj)
        
        # position-spk-time rank corr
        pos_t_corr = get_pos_t_corr(rslt, start, end, a_params)
        pos_t_corrs.append(np.abs(pos_t_corr))
    
    metrics['fr_trj'] = np.mean(frs_trj) if len(starts) else -1
    metrics['fr_ntrj'] = np.mean(frs_ntrj) if len(starts) else -1
    
    # count num one-way replay events
    metrics['one_way_ct'] = int(np.sum(np.array(pos_t_corrs) > a_params['POS_T_CORR_TH']))
    
    return metrics


def decode_trj(rslt, start, end, wdw, min_spks_wdw=10):
    """Decode trajectory from spike sequence and place fields."""
    t = []
    xy_hat = []
    
    pc_mask = rslt.ntwk.types_rcr == 'PC'
    pfxs_pc = rslt.ntwk.pfxs[pc_mask]
    pfys_pc = rslt.ntwk.pfys[pc_mask]
    
    for wdw_start in np.arange(start, end, wdw):
        wdw_end = wdw_start + wdw
        t.append(.5 * (wdw_start + wdw_end))
        
        # get mask for this moving wdw
        t_mask = (wdw_start <= rslt.ts) & (rslt.ts < (wdw_end))
        
        # get spks during current window
        spks_wdw = rslt.spks[t_mask, :][:, pc_mask]
        
        if spks_wdw.sum() < min_spks_wdw:
            xy_hat.append(np.nan)
        else:
            # get idxs of spking pcs
            pc_idxs = np.nonzero(spks_wdw)[1]
            
            # get median of place fields for spiking PCs
            x_hat = np.nanmedian(pfxs_pc[pc_idxs])
            y_hat = np.nanmedian(pfys_pc[pc_idxs])
            
            xy_hat.append([x_hat, y_hat])
            
    return np.array(t), np.array(xy_hat)
