from copy import deepcopy
import numpy as np
from scipy.sparse import csc_matrix
import os
import time


# CONNECTIVITY

def join_w(targs, srcs, ws):
    """
    Combine multiple weight matrices specific to pairs of populations
    into a single, full set of weight matrices (one per synapse type).
    
    :param targs: dict of boolean masks indicating targ cell classes
    :param srcs: dict of boolean masks indicating source cell classes
    :param ws: dict of inter-population weight matrices, e.g.:
        ws = {
            'AMPA': {
                ('EXC', 'EXC'): np.array([[...]]),
                ('INH', 'EXC'): np.array([[...]]),
            },
            'GABA': {
                ('EXC', 'INH'): np.array([[...]]),
                ('INH', 'INH'): np.array([[...]]),
            }
        }
        note: keys given as (targ, src)
    
    :return: ws_full, a dict of full ws, one per synapse
    """
    # convert targs/srcs to dicts if given as arrays
    if not isinstance(targs, dict):
        targs_ = deepcopy(targs)
        targs = {
            cell_type: targs_ == cell_type for cell_type in set(targs_)
        }
    if not isinstance(srcs, dict):
        srcs_ = deepcopy(srcs)
        srcs = {
            cell_type: srcs_ == cell_type for cell_type in set(srcs_)
        }
        
    # make sure all targ/src masks have same shape
    targ_shapes = [mask.shape for mask in targs.values()]
    src_shapes = [mask.shape for mask in srcs.values()]
    
    if len(set(targ_shapes)) > 1:
        raise Exception('All targ masks must have same shape.')
        
    if len(set(src_shapes)) > 1:
        raise Exception('All targ masks must have same shape.')
        
    n_targ = targ_shapes[0][0]
    n_src = src_shapes[0][0]
    
    # make sure weight matrix dimensions match sizes
    # of targ/src classes
    for syn, ws_ in ws.items():
        for (targ, src), w_ in ws_.items():
            if not w_.shape == (targs[targ].sum(), srcs[src].sum()):
                raise Exception(
                    'Weight matrix for {}: ({}, {}) does not match '
                    'dimensionality specified by targ/src masks.')
        
    # loop through synapse types
    dtype = list(list(ws.values())[0].values())[0].dtype
    ws_full = {}
    
    for syn, ws_ in ws.items():
        
        w = np.zeros((n_targ, n_src), dtype=dtype)
        
        # loop through population pairs
        for (targ, src), w_ in ws_.items():
            
            # get mask of all cxns from src to targ
            mask = np.outer(targs[targ], srcs[src])
            
            assert mask.sum() == w_.size
            
            w[mask] = w_.flatten()
            
        ws_full[syn] = w
        
    return ws_full


# NETWORK CLASS AND FUNCTIONS

class LIFNtwk(object):
    """
    Network of leaky integrate-and-fire (LIF) neurons.
    All parameters should be given in SI units
    (i.e., time constants in seconds, potentials in volts).
    This simulation uses exponential
    synapses for all synapse types.

    In all weight matrices, rows index target, cols index source.
    
    :param t_m: membrane time constant (or 1D array)
    :param e_l: leak reversal potential (or 1D array)
    :param v_th: firing threshold potential (or 1D array)
    :param v_r: reset potential (or 1D array)
    :param t_r: refractory time
    :param es_syn: synaptic reversal potentials (dict with keys naming
        synapse types, e.g., 'AMPA', 'NMDA', ...)
    :param ts_syn: synaptic time constants (dict)
    :param ws_rcr: recurrent synaptic weight matrices (dict with keys
        naming synapse types)
    :param ws_up: input synaptic weight matrices from upstream inputs (dict)
    :param sparse: whether to convert weight matrices to sparse matrices for
        more efficient processing
    """
    
    def __init__(self,
            t_m, e_l, v_th, v_r, t_r,
            es_syn=None, ts_syn=None, ws_up=None, ws_rcr=None, 
            sparse=True):
        """Constructor."""

        # validate arguments
        if es_syn is None:
            es_syn = {}
        if ts_syn is None:
            ts_syn = {}
        if ws_up is None:
            ws_up = {}
        if ws_rcr is None:
            ws_rcr = {}

        self.syns = list(es_syn.keys())
        
        # check weight matrices have correct dims
        shape_rcr = list(ws_rcr.values())[0].shape
        shape_up = list(ws_up.values())[0].shape
        
        self.n = shape_rcr[1]

        # fill in unspecified weight matrices with zeros
        for syn in self.syns:
            if syn not in ws_rcr:
                ws_rcr[syn] = np.zeros(shape_rcr)
            if syn not in ws_up:
                ws_up[syn] = np.zeros(shape_up)
        
        # check syn. dicts have same keys
        if not set(es_syn) == set(ts_syn) == set(ws_rcr) == set(ws_up):
            raise ValueError(
                'All synaptic dicts ("es_syn", "ts_syn", '
                '"ws_rcr", "ws_inp") must have same keys.'
            )

        if not all([w.shape[0] == w.shape[1] == self.n for w in ws_rcr.values()]):
            raise ValueError('All recurrent weight matrices must be square.')

        # check input matrices' have correct dims
        self.n_up = list(ws_up.values())[0].shape[1]

        if not all([w.shape[0] == self.n for w in ws_up.values()]):
            raise ValueError(
                'Upstream weight matrices must have one row per neuron.')

        if not all([w.shape[1] == self.n_up for w in ws_up.values()]):
            raise ValueError(
                'All upstream weight matrices must have same number of columns.')

        # make sure v_r is actually an array
        if isinstance(v_r, (int, float, complex)):
            v_r = v_r * np.ones(self.n)
            
        # store network params
        self.t_m = t_m
        self.e_l = e_l
        self.v_th = v_th
        self.v_r = v_r
        self.t_r = t_r
        
        self.es_syn = es_syn
        self.ts_syn = ts_syn
        
        if sparse:
            ws_rcr = {syn: csc_matrix(w) for syn, w in ws_rcr.items()}
            ws_up = {syn: csc_matrix(w) for syn, w in ws_up.items()}
            
        self.ws_rcr = ws_rcr
        self.ws_up = ws_up
        
    def run(self, spks_up, dt, vs_0=None, gs_0=None, i_ext=None, store=None, report_every=None):
        """
        Run a simulation of the network.

        :param spks_up: upstream spiking inputs (rows are time points, 
            cols are neurons) (should be non-negative integers)
        :param dt: integration time step for dynamics simulation
        :param vs_0: initial vs
        :param gs_0: initial gs (dict of 1-D arrays)
            are time points, cols are neurons)

        :return: network response object
        """

        # validate arguments
        if vs_0 is None:
            vs_0 = self.e_l * np.ones(self.n)
        if gs_0 is None:
            gs_0 = {syn: np.zeros(self.n) for syn in self.syns}
            
        if type(spks_up) != np.ndarray or spks_up.ndim != 2:
            raise TypeError('"inps_upstream" must be a 2D array.')

        if not spks_up.shape[1] == self.n_up:
            raise ValueError(
                'Upstream input size does not match size of input weight matrix.')

        if not vs_0.shape == (self.n,):
            raise ValueError(
                '"vs_0" must be 1-D array with one element per neuron.')

        if not all([gs.shape == (self.n,) for gs in gs_0.values()]):
            raise ValueError(
                'All elements of "gs_0" should be 1-D array with '
                'one element per neuron.')

        if store is None:
            store = {}
        
        if 'vs' not in store:
            store['vs'] = np.float64
        if 'spks' not in store:
            store['spks'] = bool
        if 'gs' not in store:
            store['gs'] = np.float64
           
        for key, val in store.items():
            
            if key == 'vs':
                assert val in (None, float, np.float16, np.float64)
            elif key == 'gs':
                assert val in (None, float, np.float16, np.float64)
            elif key == 'spks':
                assert val in (None, bool)
            
        # prepare smln
        ts = np.arange(len(spks_up)) * dt
                  
        # initialize membrane potentials, conductances, and refractory counters
        vs_prev = vs_0.copy()
        spks_prev = np.zeros(vs_0.shape, dtype=bool)
        gs_prev = {syn: gs_0[syn].copy() for syn in self.syns}
        rp_ctrs = np.zeros(self.n)
                  
        # allocate space for slmn results and store initial values
        sim_shape = (len(ts), self.n)
        
        vs = None
        spks = None
        gs = None
        
        if (i_ext is None):
            i_ext = np.zeros(len(ts))
        
        if store['vs'] is not None:
            vs = np.nan * np.zeros(sim_shape, dtype=store['vs'])
            vs[0, :] = vs_prev.copy()
                  
        if store['spks'] is not None:
            spks = np.zeros(sim_shape, dtype=bool)
            spks[0, :] = spks_prev.copy()
                  
        if store['gs'] is not None:
            gs = {
                syn: np.nan * np.zeros(sim_shape, dtype=store['gs'])
                for syn in self.syns
            }
                  
            for syn in self.syns:
                gs[syn][0, :] = gs_0[syn].copy()
                  
        # run simulation
        smln_start_time = time.time()
        last_update = time.time()
        
        for step in range(1, len(ts)):

            ## update dynamics
            for syn in self.syns:
                
                # calculate new conductances for all synapse types
                w_up = self.ws_up[syn]
                w_rcr = self.ws_rcr[syn]
                t_syn = self.ts_syn[syn]

                # calculate upstream and recurrent inputs to conductances
                inps_up = w_up.dot(spks_up[step])
                inps_rcr = w_rcr.dot(spks_prev)

                # decay conductances and add any positive inputs
                dg = -(dt/t_syn) * gs_prev[syn] + inps_up + inps_rcr
                gs_prev[syn] = gs_prev[syn] + dg
             
            # calculate current input resulting from synaptic conductances
            ## note: conductances are relative, so is_g are in volts
            is_g = [
                gs_prev[syn] * (self.es_syn[syn] - vs_prev)
                for syn in self.syns
            ]
            
            # get total input current
            is_all = np.sum(is_g, axis=0) + i_ext[step]
            
            # update membrane potential
            dvs = -(dt/self.t_m) * (vs_prev - self.e_l) + is_all
            vs_prev = vs_prev + dvs
            
            # force refractory neurons to reset potential
            vs_prev[rp_ctrs > 0] = self.v_r[rp_ctrs > 0]
            
            # identify spks
            spks_prev = vs_prev >= self.v_th
                  
            # reset membrane potentials of spiking neurons
            vs_prev[spks_prev] = self.v_r[spks_prev]
            
            # set refractory counters for spiking neurons
            rp_ctrs[spks_prev] = self.t_r[spks_prev]
            # decrement refractory counters for all neurons
            rp_ctrs -= dt
            # adjust negative refractory counters up to zero
            rp_ctrs[rp_ctrs < 0] = 0
                 
            # store vs
            if store['vs'] is not None:
                vs[step] = vs_prev.copy()
            
            # store spks
            if store['spks'] is not None:
                spks[step] = spks_prev.copy()

            # store conductances
            if store['gs'] is not None:
                for syn in self.syns:
                    gs[syn][step] = gs_prev[syn].copy()
                  
            if report_every is not None:
                if time.time() > last_update + report_every:
                    
                    print('{0}/{1} steps completed after {2:.3f} s...'.format(
                        step + 1, len(ts), time.time() - smln_start_time))
                    
                    last_update = time.time()
               
        # return NtwkResponse object
        return NtwkResponse(
            ts=ts, vs=vs, spks=spks, spks_up=spks_up, dt=dt, i_ext=i_ext, e_l=self.e_l, v_th=self.v_th,
            gs=gs, ws_rcr=self.ws_rcr, ws_up=self.ws_up)

    
class NtwkResponse(object):
    """
    Class for storing network response parameters.

    :param ts: timestamp vector
    :param vs: membrane potentials
    :param spks: spk times
    :param gs: syn-dict of conductances
    :param ws_rcr: syn-dict of recurrent weight matrices
    :param ws_up: syn-dict upstream weight matrices
    :param cell_types: array-like of cell-types
    :param cs: spk ctr variables for each cell
    :param ws_plastic: syn-dict of time-courses of plastic weights
    :param masks_plastic: syn-dict of masks specifying which weights the plastic
        ones correspond to
    :param pfcs: array of cell place field centers
    """

    def __init__(self, ts, vs, spks, spks_up, dt, i_ext, e_l, v_th, gs, ws_rcr, ws_up, cell_types=None, pfcs=None):
        """Constructor."""
        # check args
        if (cell_types is not None) and (len(cell_types) != vs.shape[1]):
            raise ValueError(
                'If "cell_types" is provided, all cells must have a type.')
            
        self.ts = ts
        self.vs = vs
        self.spks = spks
        self.spks_up = spks_up
        self.dt = dt
        self.i_ext = i_ext
        self.e_l = e_l
        self.v_th = v_th
        self.gs = gs
        self.ws_rcr = ws_rcr
        self.ws_up = ws_up
        self.cell_types = cell_types
        
        self.dt = np.mean(np.diff(ts))
        self.fs = 1 / self.dt
   
    @property
    def n(self):
        """Number of neurons."""
        return self.vs.shape[1]
