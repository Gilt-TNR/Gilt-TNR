import GiltTNR3D
import numpy as np
import functools as fct
import operator as opr
import itertools as itt
import logging
from ncon import ncon

version = "0.1"

def gilttnr_step_impurity(A, log_fact, Rps_pure, ws_pure, As_deed_pure,
                          log_facts_pure, pars, start_at_7=False):
    """ Apply Gilt-TNR, with known and preconstructed Rp and w tensors,
    to an impurity. Note that this is strictly speaking not "allowed",
    in the sense that the the w and Rp tensors were constructed for
    environments that had no impurities in them, and inserting them in a
    different environment is not guaranteed to cause only a small error.
    """
    if not start_at_7:
        A = gilt(A, Rps_pure[1], 1)
        A = coarsegrain(As_deed_pure[1][0], A, 0, ws_pure[1], 1, pars)
        A, log_fact = normalize(A, log_fact, log_facts_pure[0], 0)

        A = gilt(A, Rps_pure[2], 0)
        A = coarsegrain(A, As_deed_pure[2][4], 0, ws_pure[2], 2, pars)
        A, log_fact = normalize(A, log_fact, log_facts_pure[1], 4)

        A = gilt(A, Rps_pure[3], 4)
        A = coarsegrain(As_deed_pure[3][7], A, 7, ws_pure[3], 3, pars)
        A, log_fact = normalize(A, log_fact, log_facts_pure[2], 7)
    else:
        A = gilt(A, Rps_pure[1], 7)
        A = coarsegrain(A, As_deed_pure[1][6], 7, ws_pure[1], 1, pars)
        A, log_fact = normalize(A, log_fact, log_facts_pure[0], 6)

        A = gilt(A, Rps_pure[2], 6)
        A = coarsegrain(As_deed_pure[2][2], A, 2, ws_pure[2], 2, pars)
        A, log_fact = normalize(A, log_fact, log_facts_pure[1], 2)

        A = gilt(A, Rps_pure[3], 2)
        A = coarsegrain(A, As_deed_pure[3][1], 2, ws_pure[3], 3, pars)
        A, log_fact = normalize(A, log_fact, log_facts_pure[2], 1)
    return A, log_fact


def gilt(A, Rps, as_which):
    tensor_list = [A]
    index_list = [[-1,-2,-3,-4,-5,-6]]
    ARps = Rps[as_which]
    for j, Rp in enumerate(ARps):
        if Rp is None:
            continue
        k = -index_list[0][j]
        index_list[0][j] = k
        tensor_list.append(Rp)
        index_list.append([k,-k])
    A = ncon(tensor_list, index_list)
    return A


def coarsegrain(A1, A2, as_which, ws, typenumber, pars):
    rotation = "x" if typenumber==3 else "z" if typenumber==2 else "y"
    indexperm = GiltTNR3D.indexperms[rotation]
    if indexperm != tuple(range(6)):
        A1, A2 = [A.transpose(indexperm) for A in (A1, A2)]

    tensorperm = GiltTNR3D.tensorperms[rotation]
    inv_tensorperm = GiltTNR3D.invert_permutation(tensorperm)
    as_which = inv_tensorperm[as_which]

    w0, w1, w2, w3 = ws[as_which]
    for w in (w0, w1, w2, w3):
        if w is None:
            msg = ("Something went wrong:"
                   " impurity would need to generate a w.")
            raise RuntimeError(msg)
    A, w0, w1, w2, w3 = GiltTNR3D.coarse_grain_pair(A1, A2, pars, w0=w0,
                                                    w1=w1, w2=w2, w3=w3)

    inv_indexperm = GiltTNR3D.invert_permutation(indexperm)
    if indexperm != tuple(range(6)):
        A = A.transpose(inv_indexperm)
    return A


def normalize(A, log_fact, log_facts_pure, with_which):
    log_fact += log_facts_pure[with_which]
    m = A.norm()
    if m != 0:
        A /= m
        log_fact += np.log(m)
    return A, log_fact

