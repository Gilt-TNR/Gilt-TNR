import numpy as np
import logging
import itertools as itt
from ncon import ncon

version = 1.0

# Threshold for when the recursive iteration of Gilt is considered to
# have converged.
convergence_eps = 1e-2

def gilttnr_step(A, log_fact, pars, **kwargs):
    """
    Apply a full step of Gilt-TNR to a lattice made of tensors A.
    A full step means a transformation that takes the square lattice to
    a square lattice (no 45deg tilt), and changes the lattice spacing by
    a factor of 2.

    Arguments:
    A: The tensor to be coarse-grained. No symmetries are assumed for A.
    log_fact: A scalar factor, such that A*np.exp(log_fact) is the
              physical tensor.
    pars: A dictionary of various parameters that the algorithm takes,
          see below.
    **kwargs: Additional keyword arguments may be given to override the
              parameters in pars. The original dictionary is not
              modified.

    Returns:
    A', log_fact'
    such that the coarse-grained physical tensor is A'*np.exp(log_fact').

    The Gilt-TNR algorithm takes the following parameters:
    gilt_eps:
    The threshold for how small singular values are considered "small
    enough" in Gilt, which determines the amount of truncation done.

    gilt_print_envspec and gilt_print_envspec_recursive:
    Whether to print the environment spectra in the logs. The _recursive
    determines the same thing, but specifically for the repeated
    applications of Gilt on the same leg.  In other words, if
    gilt_print_envspec=True but gilt_print_envspec_recursive=False, then
    the environment spectrum is only printed when we first start Gilting
    a leg.

    cg_chis:
    An iterable of integers, that lists the possible bond dimensions
    to which TRG is allowed to truncate.

    cg_eps:
    A threshold for the truncation error in TRG.
    The bond dimension used in the truncated SVD of TRG is the smallest
    one from cg_chis, such that the truncation error is below cg_eps.
    If this isn't possible, then the largest chi in cg_chis is used.

    verbosity:
    Determines the amount of output the algorithm prints out.
    """
    pars = update_pars(pars, **kwargs)
    verbose = pars["verbosity"] > 1

    # Normalize the tensor A, to keep its elements near unity or below.
    m = A.norm()
    if m != 0:
        A /= m
        log_fact += np.log(m)

    # Apply Gilt.
    if pars["gilt_eps"] >= 0:
        A1, A2 = gilt_plaq(A, A, pars)
    else:
        A1, A2, = A, A

    # Apply TRG to the checker-board lattice made of A1 and A2, and then
    # once to the homogeneous lattice of As.
    A, log_fact = trg(A1, A2, log_fact, pars)
    A, log_fact = trg(A, A, log_fact, pars)

    # The lattice has been rotated by 90deg in the process, so rotate
    # back.
    A = A.transpose((3,0,1,2))

    retval = (A, log_fact)
    return retval


# # # # # # # # # # # # # # # # # # TRG # # # # # # # # # # # # # # # # # # #


def trg(A1, A2, log_fact, pars, **kwargs):
    """
    Apply the TRG algorithm to a checker-board lattice of tensors A1 and
    A2:
                  |                          |               \      /
      |         --B1           |             C1--    \ /      B2--C2
    --A1--   ->     \    ,   --A2--  ->     /    ,    A'  =   |   |
      |             B2--       |        --C2         / \      C1--B1
                    |                     |                  /      \

    Arguments:
    A1, A2: The tensors to coarse-grain.
    log_fact: A scalar factor, such that the physical tensors are
              A1*np.exp(log_fact) and A1*np.exp(log_fact).
    pars: A dictionary of parameters for TRG.
    **kwargs: Keyword arguments may be used to override the parameters
              in pars.

    Returns:
    A', log_fact'
    such that the coarse-grained physical tensor is A'*np.exp(log_fact').

    TRG algorithm takes the following parameters:

    cg_chis:
    An iterable of integers, that lists the possible bond dimensions
    to which TRG is allowed to truncate.

    cg_eps:
    A threshold for the truncation error in TRG.
    The bond dimension used in the truncated SVD of TRG is the smallest
    one from cg_chis, such that the truncation error is below cg_eps.
    If this isn't possible, then the largest chi in cg_chis is used.

    verbosity:
    Determines the amount of output the algorithm prints out.
    """
    pars = update_pars(pars, **kwargs)
    verbose = pars["verbosity"] > 1
    if verbose:
        status_print("TRG splitting,")
    # Split the tensor A1 along a diagonal using an SVD A1=USV.
    # Absorb the singular values into the unitaries, so that B1 =
    # U*sqrt(S) and B2 = sqrt(S)V.
    # Use a truncated SVD, where the error threshold pars["cg_eps"] and
    # allowed bond dimensions to truncate to pars["cg_chis"].
    B1, B2, err = A1.split([0,1], [2,3], chis=pars["cg_chis"],
                           eps=pars["cg_eps"], return_rel_err=True)
    if verbose:
        chi = type(B1).flatten_dim(B1.shape[2])
        status_print("TRG splitting, done.",
                     "Error = {:.3e}".format(err),
                     "chi = {}".format(chi))

    if verbose:
        status_print("TRG splitting,")
    # Split A2 like A1 was split, but along a different diagonal.
    C1, C2, err = A2.split([2,1], [0,3], chis=pars["cg_chis"],
                           eps=pars["cg_eps"], return_rel_err=True)
    if verbose:
        chi = type(C1).flatten_dim(C1.shape[2])
        status_print("TRG splitting, done.",
                     "Error = {:.3e}".format(err),
                     "chi = {}".format(chi))

    # Contract the square of four pieces together to form the new
    # coarse-grained tensor.
    if verbose:
        status_print("TRG contracting,")
    A = ncon((B2, C2, B1, C1),
             ([-1,11,1], [-2,11,2], [10,2,-3], [10,1,-4]))
    if verbose:
        status_print("TRG contracting, done.")

    # Two tensors we combined into one, so the log_fact is doubled.
    log_fact *= 2
    return A, log_fact


# # # # # # # # # # # # # # # # # # Misc # # # # # # # # # # # # # # # # # # #

print_pad = 36

def status_print(pre, *args, indent=0):
    """
    Pretty, consistent formatting of output prints, with the same amount
    of indent on each line.
    """
    arg_str = ", ".join(["{}"]*len(args))
    pre_str = " "*indent + "{:<" + str(print_pad-indent) + "}"
    status_str =  pre_str + arg_str
    status_str = status_str.format(pre, *args)
    logging.info(status_str)
    return


def print_envspec(S):
    """
    Print the environment spectrum S in the output.
    Only at most a hundred values are printed. If the spectrum has l
    values and l>100, only every ceil(l/100)th value is printed.
    """
    l = len(S)
    step = int(np.ceil(l/100))
    envspeclist = sorted(S.to_ndarray(), reverse=True)
    envspeclist = envspeclist[0:-1:step]
    envspeclist = np.array(envspeclist)
    msg = "The environment spectrum, with step {} in {}".format(step, l)
    logging.info(msg)
    logging.info(envspeclist)
    return


def update_pars(pars, **kwargs):
    if kwargs:
        pars = pars.copy()
        pars.update(kwargs)
    return pars


# # # # # # # # # # # # # # # # # Gilt # # # # # # # # # # # # # # # # # #

def gilt_plaq(A1, A2, pars):
    """
    Apply the Gilt algorithm to a plaquette.
     |   |
    -A2--A1-
     |   |
    -A1--A2-
     |   |
    All four legs around the plaquette are truncated.
    It is assumed that this plaquette forms a unit cell that tiles the
    whole lattice, meaning the same truncation can be applied to the
    plaquettes diagonally connected to this one.

    Arguments:
    A1, A2: The tensors at the corners of the plaquette.
    pars: A dictionary of parameters for Gilt.
    **kwargs: Keyword arguments can be used to override values in pars.

    Returns:
    A1', A2'
    the new tensors at the corners of the plaquette, with truncated legs.
    """
    # Store the bond dimensions of the original tensors, so that we can
    # keep track of how they keep changing.
    orig_shape = type(A1).flatten_shape(A1.shape)
    # Each of the four legs, labeled by the directions S, E, W, N is
    # truncated, until an attempt to truncate them yields no progress.
    # The following booleans keep track of this.
    # If any leg is truncated, even the ones that we weren't able to
    # truncate more before or tried again.
    done_legs = {i: False for i in {"S", "N", "E", "W"}}
    # Note that we simply sum up the truncation errors caused by the
    # individual truncations, which typically overestimates the total
    # truncation error.
    gilt_error = 0
    verbose = pars["verbosity"] > 1
    if verbose:
        status_print("Applying Gilt,")
    for leg in itt.cycle("SNEW"):
        A1, A2, done, err = apply_gilt(A1, A2, pars, where=leg)
        done_legs[leg] = done
        gilt_error += err
        if all(done_legs.values()):
            break
        if verbose:
            shape = type(A1).flatten_shape(A1.shape)
            status_print("Applying Gilt,",
                         "Error = {:.3e}".format(gilt_error),
                         "shape = {} [{}]"
                         .format(shape, orig_shape))
    if verbose:
        status_print("Applying Gilt, done.")

    return A1, A2


def apply_gilt(A1, A2, pars, where=None):
    """
    Apply Gilt to a single leg around a plaquette. Which leg, is
    specified by the argument 'where'.
    """
    U, S = get_envspec(A1, A2, pars, where=where)
    S /= S.sum()
    if pars["gilt_print_envspec"]:
        print_envspec(S)
    Rp, insertionerr = optimize_Rp(U, S, pars)
    spliteps = pars["gilt_eps"]*1e-3
    Rp1, s, Rp2, spliterr = Rp.split(0, 1, eps=spliteps, return_rel_err=True,
                                     return_sings=True)
    global convergence_eps
    if (s-1).abs().max() < convergence_eps:
        done = True
    else:
        done = False

    err = insertionerr + spliterr
    A1, A2 = apply_Rp(A1, A2, Rp1, Rp2, where=where)
    return A1, A2, done, err


def get_envspec(A1, A2, pars, where=None):
    """
    Obtain the environment spectrum S and the singular vectors U of
    the environment.
    Rather by a costly SVD of E, this is done by eigenvalue decomposing
    EE^\dagger.
    """
    SW = ncon((A1, A1.conjugate()), ([1,-1,-11,2], [1,-2,-12,2]))
    NW = ncon((A2, A2.conjugate()), ([1,2,-1,-11], [1,2,-2,-12]))
    NE = ncon((A1, A1.conjugate()), ([-11,1,2,-1], [-12,1,2,-2]))
    SE = ncon((A2, A2.conjugate()), ([-1,-11,1,2], [-2,-12,1,2]))

    if where=="S":
        nconlist = (SW, NW, NE, SE)
    elif where=="W":
        nconlist = (NW, NE, SE, SW)
    elif where=="N":
        nconlist = (NE, SE, SW, NW)
    elif where=="E":
        nconlist = (SE, SW, NW, NE)
    else:
        raise(ValueError("Unknown direction: {}.".format(where)))
    E = ncon(nconlist,
             ([1,2,-1,-11], [3,4,1,2], [5,6,3,4], [-2,-12,5,6]))

    S, U = E.eig([0,1], [2,3], hermitian=True)
    # S are all non-negative, taking abs is just for numerical errors
    # around zero.
    S = S.abs().sqrt()
    return U, S


def apply_Rp(A1, A2, Rp1, Rp2, where=None):
    """
    Absorb the matrices created by Gilt to the tensors A1 and A2.
    """
    if where=="S":
        A1 = ncon((A1, Rp1), ([-1,-2,3,-4], [3,-3]))
        A2 = ncon((A2, Rp2), ([1,-2,-3,-4], [-1,1]))
    elif where=="W":
        A2 = ncon((A2, Rp1), ([-1,-2,-3,4], [4,-4]))
        A1 = ncon((A1, Rp2), ([-1,2,-3,-4], [-2,2]))
    elif where=="N":
        A1 = ncon((A1, Rp1), ([1,-2,-3,-4], [1,-1]))
        A2 = ncon((A2, Rp2), ([-1,-2,3,-4], [-3,3]))
    elif where=="E":
        A2 = ncon((A2, Rp1), ([-1,2,-3,-4], [2,-2]))
        A1 = ncon((A1, Rp2), ([-1,-2,-3,4], [-4,4]))
    else:
        raise(ValueError("Unknown direction: {}.".format(where)))
    return A1, A2


def optimize_Rp(U, S, pars, **kwargs):
    """
    Given the environment spectrum S and the singular vectors U, choose
    t' and build the matrix R' (called tp and Rp in the code).
    Return also the truncation error caused in inserting this Rp into
    the environment.
    """
    pars = update_pars(pars, **kwargs)
    t = ncon(U, [1,1,-1])
    S = S.flip_dir(0)   # Necessary for symmetry preserving tensors only.

    C_err_constterm = (t*S).norm()
    def C_err(tp):
        nonlocal t, S, C_err_constterm
        diff = t-tp
        diff = diff*S
        err = diff.norm()/C_err_constterm
        return err

    # The following minimizes ((t-tp)*S).norm_sq() + gilt_eps*tp.norm_sq()
    gilt_eps = pars["gilt_eps"]
    if gilt_eps != 0:
        ratio = S/gilt_eps
        weight = ratio**2/(1+ratio**2)
        tp = t.multiply_diag(weight, 0, direction="left")
    else:
        tp = t
    Rp = build_Rp(U, tp)

    # Recursively keep absorbing Rp into U, and repeating the procedure
    # to build a new Rp, until the leg can not be truncated further.
    spliteps = gilt_eps*1e-3
    u, s, v = Rp.svd(0, 1, eps=spliteps)
    # If the singular value spectrum of the Rp matrix that was last
    # created was essentially flat, we are done.
    global convergence_eps
    done_recursing = (s-1).abs().max() < convergence_eps
    if not done_recursing:
        ssqrt = s.sqrt()
        us = u.multiply_diag(ssqrt, 1, direction="right")
        vs = v.multiply_diag(ssqrt, 0, direction="left")
        Uuvs = ncon((U, us, vs), ([1,2,-3], [1,-1], [-2,2])) 
        UuvsS = Uuvs.multiply_diag(S, 2, direction="left")
        Uinner, Sinner = UuvsS.svd([0,1], [2])[0:2]
        Sinner /= Sinner.sum()
        if pars["gilt_print_envspec"] and pars["gilt_print_envspec_recursive"]:
            print_envspec(Sinner)
        Rpinner = optimize_Rp(Uinner, Sinner, pars)[0]
        Rp = ncon((Rpinner, us, vs), ([1,2], [-1,1], [2,-2]))

    err = C_err(tp)
    return Rp, err
    

def build_Rp(U, tp):
    Rp = ncon((U.conjugate(), tp), ([-1,-2,1], [1]))
    return Rp

