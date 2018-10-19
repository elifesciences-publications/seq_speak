from copy import deepcopy
import numpy as np
from scipy.sparse import csc_matrix
import time

from aux import lognormal_mu_sig, sgmd
from replay import cxn
from ntwk import LIFNtwk, join_w

cc = np.concatenate


def run(p, s_params):
    """
    Run smln and return rslt.
    
    :param p: dict of model params
    :param s_params: dict of smln params
    """
    # prepare smln
    prep_start = time.time()
    
    ## build trajectory
    trj = build_trj(s_params)
    schedule = s_params['schedule']
    
    ## adjust schedule
    t = np.arange(0, schedule['D_SMLN'], s_params['DT'])
    
    ## build ntwk
    ntwk = build_ntwk(p, s_params)
    
    ## get apx. real-valued mask ("veil") over trj nrns;
    ## values are >= 0 and correspond to apx. scale factors on
    ## corresponding G->PC weights minus 1
    trj_veil = get_trj_veil(trj, ntwk, p, s_params)
    
    ## approximate G->PC weights if desired
    ntwk = apx_ws_up(ntwk, trj_veil)
        
    spks_up, i_ext = build_stim(t, trj, ntwk, p, s_params, schedule)
    
    prep_end = time.time()
    prep_time = prep_end - prep_start
    
    # run smln
    run_start = time.time()
    
    rslt = ntwk.run(spks_up=spks_up, dt=s_params['DT'], i_ext=i_ext)
    run_end = time.time()
    
    run_time = run_end - run_start
    
    # consolidate smln rslt
    rslt.ntwk = ntwk
    rslt.schedule = schedule
    
    rslt.p = p
    rslt.s_params = s_params
    
    rslt.trj = trj
    rslt.trj_veil = trj_veil
    
    metrics, success = get_metrics(rslt, s_params)
    
    rslt.metrics = metrics
    rslt.success = success
    
    rslt.prep_time = prep_time
    rslt.run_time = run_time
   
    return rslt

 
def build_ntwk(p, s_params):
    """
    Construct a network object from the model and
    simulation params.
    """
    np.random.seed(s_params['RNG_SEED'])
    
    # set membrane properties
    n = p['N_PC'] + p['N_INH']
    
    t_m = cc(
        [np.repeat(p['T_M_PC'], p['N_PC']), np.repeat(p['T_M_INH'], p['N_INH'])])
    e_l = cc(
        [np.repeat(p['E_L_PC'], p['N_PC']), np.repeat(p['E_L_INH'], p['N_INH'])])
    v_th = cc(
        [np.repeat(p['V_TH_PC'], p['N_PC']), np.repeat(p['V_TH_INH'], p['N_INH'])])
    v_r = cc(
        [np.repeat(p['V_R_PC'], p['N_PC']), np.repeat(p['V_R_INH'], p['N_INH'])])
    t_rp = cc(
        [np.repeat(p['T_R_PC'], p['N_PC']), np.repeat(p['T_R_INH'], p['N_INH'])])
    
    # set latent nrn positions
    lb = [-s_params['BOX_W']/2, -s_params['BOX_H']/2]
    ub = [s_params['BOX_W']/2, s_params['BOX_H']/2]
    
    # sample evenly spaced place fields
    ## E cells
    pfxs_e, pfys_e = cxn.apx_lattice(lb, ub, p['N_PC'], randomize=True)
    ## I cells
    pfxs_i, pfys_i = cxn.apx_lattice(lb, ub, p['N_INH'], randomize=True)
    
    ## join E & I place fields
    pfxs = cc([pfxs_e, pfxs_i])
    pfys = cc([pfys_e, pfys_i])
    
    # make upstream ws
    if p['W_PC_PL'] > 0:
        w_pc_pl_flat = np.random.lognormal(
            *lognormal_mu_sig(p['W_PC_PL'], p['S_PC_PL']), p['N_PC'])
    else:
        w_pc_pl_flat = np.zeros(p['N_PC'])
    
    if p['W_PC_G'] > 0:
        w_pc_g_flat = np.random.lognormal(
            *lognormal_mu_sig(p['W_PC_G'], p['S_PC_G']), p['N_PC'])
    else:
        w_pc_g_flat = np.zeros(p['N_PC'])
    
    ws_up_temp = {
        'E': {
            ('PC', 'PL'): np.diag(w_pc_pl_flat),
            ('PC', 'G'): np.diag(w_pc_g_flat),
        },
    }
    
    targs_up = cc([np.repeat('PC', p['N_PC']), np.repeat('INH', p['N_INH'])])
    srcs_up = cc([np.repeat('PL', p['N_PC']), np.repeat('G', p['N_PC'])])
    
    ws_up = join_w(targs_up, srcs_up, ws_up_temp)
    
    # make rcr ws
    w_pc_pc = cxn.make_w_pc_pc(pfxs[:p['N_PC']], pfys[:p['N_PC']], p)
    
    w_inh_pc = cxn.make_w_inh_pc(
        pfxs_inh=pfxs[-p['N_INH']:],
        pfys_inh=pfys[-p['N_INH']:],
        pfxs_pc=pfxs[:p['N_PC']],
        pfys_pc=pfys[:p['N_PC']],
        p=p)
    
    w_pc_inh = cxn.make_w_pc_inh(
        pfxs_pc=pfxs[:p['N_PC']],
        pfys_pc=pfys[:p['N_PC']],
        pfxs_inh=pfxs[-p['N_INH']:],
        pfys_inh=pfys[-p['N_INH']:],
        p=p)
    
    ws_rcr_temp = {
        'E': {
            ('PC', 'PC'): w_pc_pc,
            ('INH', 'PC'): w_inh_pc,
        },
        'I': {
            ('PC', 'INH'): w_pc_inh,
        },
    }
    targs_rcr = cc([np.repeat('PC', p['N_PC']), np.repeat('INH', p['N_INH'])])
    
    ws_rcr = join_w(targs_rcr, targs_rcr, ws_rcr_temp)
    
    # make ntwk
    ntwk = LIFNtwk(
        t_m=t_m,
        e_l=e_l,
        v_th=v_th,
        v_r=v_r,
        t_r=t_rp,
        es_syn={'E': p['E_E'], 'I': p['E_I']},
        ts_syn={'E': p['T_E'], 'I': p['T_I']},
        ws_up=ws_up,
        ws_rcr=ws_rcr)
    
    ntwk.pfxs = pfxs
    ntwk.pfys = pfys
    
    ntwk.types_up = srcs_up
    ntwk.types_rcr = targs_rcr
    
    ntwk.n_pc = p['N_PC']
    ntwk.n_inh = p['N_INH']
    ntwk.n_g = p['N_PC']
    ntwk.n_inp = p['N_PC']
    ntwk.n_rcr = p['N_PC'] + p['N_INH']
    ntwk.n_up = 2 * p['N_PC']
    
    ntwk.types_up_slc = {
        'PL': slice(0, p['N_PC']),
        'G': slice(p['N_PC'], 2*p['N_PC'])
    }
    ntwk.types_rcr_slc = {
        'PC': slice(0, p['N_PC']),
        'INH': slice(p['N_PC'], p['N_PC'] + p['N_INH'])
    }
    
    return ntwk


def build_trj(s_params):
    """
    Build trajectory.
    """
    # calc trj segs
    segs_x = []
    segs_y = []
    
    for (x_0, y_0), (x_1, y_1) in zip(s_params['TRJ'][:-1], s_params['TRJ'][1:]):
        
        path_len = np.sqrt((x_1-x_0)**2 + (y_1-y_0)**2)
        dur = path_len / s_params['SPD']
        
        # num timesteps
        n_t = int(round(dur/s_params['DT'])) - 1
        
        # x and y segs
        segs_x.append(np.linspace(x_0, x_1, n_t, endpoint=False))
        segs_y.append(np.linspace(y_0, y_1, n_t, endpoint=False))
        
    # make full x, y seq from segs
    x = np.concatenate(segs_x)
    y = np.concatenate(segs_y)
    
    t = np.arange(len(x), dtype=float) * s_params['DT']
    spd = s_params['SPD'] * np.ones(len(t))
    
    return {'x': x, 'y': y, 't': t, 'spd': spd}
 
    
def get_trj_veil(trj, ntwk, p, s_params):
    """
    Return a "veil" (positive real-valued mask) over cells in the ntwk
    with place fields along the trajectory path.
    """
    # compute scale factor for all PCs
    ## get distance to trj
    d = dist_to_trj(ntwk.pfxs, ntwk.pfys, trj['x'], trj['y'])[0]
    
    # sensory-driven firing rates as function of d
    r = p['R_MAX'] * np.exp(-.5*(d**2)/(p['L_PL']**2))
    sgm = 1 + (p['SGM_MAX'] - 1) * sgmd(p['B_SGM']*(r - p['R_SGM']))
    veil = sgm - 1
    
    return veil
    
    
def dist_to_trj(pfxs, pfys, x, y):
    """
    Compute distance of static points (pfxs, pfys) to trajectory (x(t), y(t)).
    
    :return: dists to nearest pts, idxs of nearest pts
    """
    # get dists to all pts along trj
    dx = np.tile(pfxs[None, :], (len(x), 1)) - np.tile(x[:, None], (1, len(pfxs)))
    dy = np.tile(pfys[None, :], (len(y), 1)) - np.tile(y[:, None], (1, len(pfys)))
    
    d = np.sqrt(dx**2 + dy**2)
    
    # return dists of cells to nearest pts on trj
    return np.min(d, 0), np.argmin(d, 0)

   
def apx_ws_up(ntwk, trj_veil):
    """
    Replace G->PC weights with apxns expected following
    initial sensory input.
    """
    scale = trj_veil[ntwk.types_rcr == 'PC'] + 1
    
    ws_up_e_dense = np.array(ntwk.ws_up['E'].todense())
    
    n_pc = ntwk.n_pc
    
    mask = np.zeros((ntwk.n, 2*n_pc), dtype=bool)
    mask[:n_pc, n_pc:] = np.eye(n_pc, dtype=bool)
    
    ws_up_e_dense[mask] *= scale
    
    ntwk.ws_up['E'] = csc_matrix(ws_up_e_dense)
    
    return ntwk

       
def build_stim(t, trj, ntwk, p, s_params, schedule):
    """
    Put together upstream spk and external current inputs
    according to stimulation params and schedule.
    """
    np.random.seed(s_params['RNG_SEED'])
    
    # initialize upstream spks array
    spks_up = np.zeros((len(t), 2*p['N_PC']), int)
    
    # fill in replay epoch Gate inputs
    spks_up += spks_up_from_g(t, ntwk, p, s_params, schedule)
    
    # initialize external current array
    i_ext = np.zeros((len(t), p['N_PC'] + p['N_INH']))
    
    # add replay trigger
    i_ext += i_ext_trg(t, ntwk, p, s_params, schedule)
    
    return spks_up, i_ext


def spks_up_from_g(t, ntwk, p, s_params, schedule):
    """
    Add G --> PC spks to upstream spk array.
    """
    spks_up = np.zeros((len(t), 2*p['N_PC']), int)
    
    # replay epoch
    spks_up[:, p['N_PC']:] += np.random.poisson(
        p['R_G'] * s_params['DT'], (len(t), p['N_PC']))
        
    return spks_up


def i_ext_trg(t, ntwk, p, s_params, schedule):
    """
    Add replay trigger to external current stim.
    """
    i_ext = np.zeros((len(t), p['N_PC'] + p['N_INH']))
    
    # get mask over cells to trigger to induce replay
    ## compute distances to trigger center
    trg_mask = get_trg_mask_pc(ntwk, p, s_params)
    
    ## get time mask
    t_mask = (schedule['T_TRG'] <= t) \
        & (t < (schedule['T_TRG'] + p['D_TRG']))
    
    ## add in external trigger
    i_ext[np.outer(t_mask, trg_mask)] = p['A_TRG']
    
    return i_ext


def get_trg_mask_pc(ntwk, p, s_params):
    dx = ntwk.pfxs - s_params['X_TRG']
    dy = ntwk.pfys - s_params['Y_TRG']
    d = np.sqrt(dx**2 + dy**2)
    
    ## get mask
    trg_mask = (d < p['R_TRG']) & (ntwk.types_rcr == 'PC')
    return trg_mask


def get_metrics(rslt, s_params):
    """
    Compute basic metrics from smln rslt quantifying replay properties.
    """
    m = s_params['metrics']
    mask_pc = rslt.ntwk.types_rcr == 'PC'
    
    # get masks over trj & non-trj PCs
    trj_mask = (rslt.trj_veil * mask_pc) > (m['MIN_SCALE_TRJ'] - 1)
    non_trj_mask = (~trj_mask) & mask_pc
    
    # get t_mask for detection window
    start = rslt.schedule['T_TRG']
    end = start + m['WDW']
    t_mask = (start <= rslt.ts) & (rslt.ts < end)
    
    # get spk cts for trj/non-trj cells during detection window
    spks_wdw = rslt.spks[t_mask]
    spks_trj = spks_wdw[:, trj_mask]
    spks_non_trj = spks_wdw[:, non_trj_mask]
    
    # get fraction of trj/non-trj cells that spiked
    frac_spk_trj = np.mean(spks_trj.sum(0) > 0)
    frac_spk_non_trj = np.mean(spks_non_trj.sum(0) > 0)
    
    # get avg spk ct of spiking trj cells
    avg_spk_ct_trj = spks_trj.sum(0)[spks_trj.sum(0) > 0].mean()
    
    # check conditions for successful replay 
    if (frac_spk_trj >= m['MIN_FRAC_SPK_TRJ']) \
            and (frac_spk_trj >= (frac_spk_non_trj*m['TRJ_NON_TRJ_SPK_RATIO'])) \
            and (avg_spk_ct_trj < m['MAX_AVG_SPK_CT_TRJ']):
        success = True
    else:
        success = False
        
    metrics = {
        'frac_spk_trj': frac_spk_trj,
        'frac_spk_non_trj': frac_spk_non_trj,
        'avg_spk_ct_trj': avg_spk_ct_trj,
        'success': success,
    }
    
    return metrics, success
