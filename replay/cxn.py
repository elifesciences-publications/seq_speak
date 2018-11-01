"""
Functions for creating structured synaptic weight matrices
for sequence replay simulations.
"""
import numpy as np

from aux import lognormal_mu_sig

cc = np.concatenate


def apx_lattice(lb, ub, n, randomize):
    """
    Arrange n points on an apx. lattice within a rectangle.
    (Apx. b/c rectangle dims may not evenly divide n.)
    """
    lb_x, lb_y = lb
    ub_x, ub_y = ub

    r_x = ub_x - lb_x
    r_y = ub_y - lb_y

    # get apx factors of n
    n_x = np.sqrt((r_x/r_y) * n)
    n_y = n/n_x

    # get # pts per row
    n_rows = int(np.round(n_y))
    n_pts = [len(row) for row in np.array_split(np.arange(n), n_rows)]

    # evenly distribute n_pts so that largest rows are not clumped at top
    if len(set(n_pts)) > 1:
        
        ## split into groups of same n_pts
        gp_0, gp_1 = [[ii for ii in n_pts if ii == i] for i in set(n_pts)]

        if len(gp_1) > len(gp_0):
            gp_0, gp_1 = gp_1, gp_0

        ## assign float "t" to each n_pt
        n_0 = len(gp_0)
        n_1 = len(gp_1)

        ts_0 = [k * (n - 1) / (n_0 - 1) for k in range(n_0)]
        ts_1 = [k * (n - 1) / (n_1 + 1) for k in range(1, n_1 + 1)]

        ts = cc([ts_0, ts_1])

        ## sort n_pts according to ts
        n_pts = cc([gp_0, gp_1])[np.argsort(ts)]
    
    # assign (x, y) positions
    ys_row = np.linspace(lb_y, ub_y, n_rows+2)[1:-1]

    xs = []
    ys = []

    ## add group of positions for each row
    for y_row, n_pts_ in zip(ys_row, n_pts):

        xs_ = list(np.linspace(lb_x, ub_x, n_pts_ + 2)[1:-1])
        ys_ = list(np.repeat(y_row, len(xs_)))

        xs.extend(xs_)
        ys.extend(ys_)
        
    xs = np.array(xs)
    ys = np.array(ys)
    
    if randomize:
        shuffle = np.random.permutation(n)
        
        xs = xs[shuffle]
        ys = ys[shuffle]

    return xs, ys


def _w_pc_pc_vs_d(d, p):
    """Return distance-dependent portion of w_pc_pc computation."""
    assert np.all(d >= 0)
    
    # decrease weights as squared exp of dist
    w = p['W_PC_PC'] * np.exp(-d**2/(2*p['L_PC_PC']**2))
    
    # set all weights below min weight th to 0
    w[w < p['W_MIN_PC_PC']] = 0
    
    return w


def make_w_pc_pc(pfxs, pfys, p):
    """Make PC-PC weight mat w/ weight increasing w/ proxim."""
    n_pc = p['N_PC']
    
    # build distance matrix
    dx = np.tile(pfxs[None, :], (n_pc, 1)) - np.tile(pfxs[:, None], (1, n_pc))
    dy = np.tile(pfys[None, :], (n_pc, 1)) - np.tile(pfys[:, None], (1, n_pc))
    d = np.sqrt(dx**2 + dy**2)
    
    # build weight matrix
    return _w_pc_pc_vs_d(d, p)
    

def _w_inh_pc_vs_d(d, p):
    """Return distance-dependent portion of w_inh_pc computation."""
    assert np.all(d >= 0)
    
    return np.zeros(d.shape)
    
    
def make_w_inh_pc(pfxs_inh, pfys_inh, pfxs_pc, pfys_pc, p):
    """
    Make proximally biased PC->INH weight matrix.
    """
    n_inh = p['N_INH']
    n_pc = p['N_PC']
    
    # build distance matrix
    dx = np.tile(pfxs_pc[None, :], (n_inh, 1)) \
        - np.tile(pfxs_inh[:, None], (1, n_pc))
    dy = np.tile(pfys_pc[None, :], (n_inh, 1)) \
        - np.tile(pfys_inh[:, None], (1, n_pc))
    d = np.sqrt(dx**2 + dy**2)
    
    # build weight matrix
    return _w_inh_pc_vs_d(d, p)
    
    
def _w_pc_inh_vs_d(d, p):
    """Return distance-dependent portion of w_pc_inh computation."""
    assert np.all(d >= 0)
   
    return np.zeros(d.shape)


def make_w_pc_inh(pfxs_pc, pfys_pc, pfxs_inh, pfys_inh, p):
    """
    Make center-surround structured INH->PC weight matrix.
    """
    n_pc = p['N_PC']
    n_inh = p['N_INH']
    
    # build distance matrix
    dx = np.tile(pfxs_inh[None, :], (n_pc, 1)) \
        - np.tile(pfxs_pc[:, None], (1, n_inh))
    dy = np.tile(pfys_inh[None, :], (n_pc, 1)) \
        - np.tile(pfys_pc[:, None], (1, n_inh))
    d = np.sqrt(dx**2 + dy**2)
    
    # build weight matrix
    return _w_pc_inh_vs_d(d, p)
