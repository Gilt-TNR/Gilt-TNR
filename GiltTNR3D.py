import numpy as np
import warnings
import functools as fct
import operator as opr
import itertools as itt
import logging
from tntools.ncon_sparseeig import ncon_sparsesvd
from ncon import ncon
from scipy.sparse.linalg.eigen.arpack.arpack import (
    ArpackNoConvergence, ArpackError
)

version = "0.1"

# Threshold for when the recursive iteration of Gilt is considered to
# have converged.
convergence_eps = 1e-2

def gilttnr_step(As, log_facts, pars, **kwargs):
    """
    Apply a full step of Gilt-TNR to a cubical lattice.
    The lattice consists of a unit cube that is repeated:
       A3+------+ A7
         |`.    |`.
         | A4+------+ A8                z
         |   |  |   |               y   |
      A2 +---|--+A6 |                `. |
          `. |   `. |                  `o----x
         A1 `+------+ A5
    A full step of Gilt-TNR consists of three substeps, in each of which
    Gilt is first applied to remove local correlations and reduce the
    bond dimension, and HOTRG is then used to coarse-grain along one of
    the three axes of the lattice, reducing the linear dimension along
    that direction by a factor of 2.

    The Gilt procedure applied during each three substeps is always the
    same. There are 8 unique tensors in the network, there
    are in total 24 unique legs. Each of these legs is an edge for four
    different cubes, each of which consists of the tensors A1 to A8, but
    organized in a different manner. Each of these four cubes can be used
    as the neighborhood of the leg, when applying Gilt. A combination of
    a leg and cube that is its neighborhood is called a culg (cube-leg).
    There are in total 24*4=96 culgs, meaning 96 unique places in which
    to try to apply Gilt.

    During a single substep, when Gilt is applied, either some or all of
    these culgs maybe "visited", i.e., Gilt is attempted on them. See
    the parameters of the algorithm below for details.

    In addition to culgs, Gilt can also be applied using faces of cubes
    (squares) as the neighborhoods. The procedure is much like Gilt
    applied to square lattice. This is much cheaper, but can not remove
    all types of local correlations, for it is blind to correlations
    that transcend a single face and wrap around the whole cube. Its
    often a good idea to precede the full Gilt, applied to cubes, with
    Gilt applied to squares, to reduce the bond dimensions.

    Arguments:
    As: The tensors to be coarse-grained, in an iterable with either 8
    or 4 elements. If there are only 4 elements, one-site translation
    symmetry along the x-axis is assumed. No spatial symmetries
    (rotations, reflections) are assumed for any of the As.
    log_facts: A list of scalar factors, such that
    A[i]*np.exp(log_facts[i]) are the physical tensors.
    pars: A dictionary of various parameters that the algorithm takes,
    see below.
    **kwargs: Additional keyword arguments may be given to override the
    parameters in pars. The original dictionary is not modified.

    Returns:
    As', log_facts',
    ws_dict, Rps_dict, gilted_As_dict, cged_As_dict, log_facts_dict

    As' and log_facts' include the actual tensors, such the physical
    tensors are As'[i] * np.exp(log_facts'[i]). The other return values
    provide supplementary data, that may be useful when dealing with
    for instance impurity systems. Each of them is a dictionary,
    for which the keys are 1, 2 and 3, corresponding to the three
    substeps.
    ws_dict includes the isometries (called w) that perform the HOTRG
    coarse-graining.
    Rps_dict includes the Rp matrices that Gilt creates.
    gilted_As_dict includes the A tensors after Gilt but before
    coarse-graining. 
    cged_As_dict includes the tensors after coarse-graining, but before
    the next round of Gilt.
    log_facts_dict includes the log_facts for cged_As_dict.


    The Gilt-TNR 3D algorithm takes the following parameters:
    gilt_squares:
    Boolean for whether to apply Gilt with squares as the neighborhoods.

    gilt_cubes:
    Boolean for whether to apply Gilt with cubes as the neighborhoods.

    gilt_eps_cubes and gilt_eps_squares:
    The threshold for how small singular values are considered "small
    enough" in Gilt, which determines the amount of truncation done.
    One is for when applying Gilt with cubes as the neighborhoods,
    the other for squares.

    gilt_split:
    Boolean for whether to, when applying Gilt to cubes, use a sparse
    SVD of parts of the environment to try to reduce the cost of
    contracting the environment.

    gilt_split_factor:
    If gilt_split is True, then we assume that
    gilt_split_factor*(D**0.5) singular values in the sparse SVD could
    be non-zero, where D is the total number of singular values.
    See code for details.

    gilt_split_dynamic:
    Boolean for whether to adjust gilt_split_factor during the run.

    gilt_split_dynamic_eps:
    A threshold for how small a singular value needs to be to be
    considered zero, when using gilt_split_dynamic.

    gilt_split_dynamic_max_factor:
    Maximum gilt_split_factor allowed, when using gilt_split_dynamic.

    gilt_hastyquit:
    Whether to quit the Gilt procedure for the cubes early on, without
    even trying all the culgs, if it seems that no more truncations are
    probably possible.

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
    # We often call coarse-graining CG in the code.

    pars = update_pars(pars, **kwargs)
    As = list(As)
    ws_dict = {}
    Rps_dict = {}
    gilted_As_dict = {}
    log_facts_dict = {}
    cged_As_dict = {}

    log_facts_dict[0] = tuple(log_facts)

    # Apply Gilt and coarse-graining along all the three different axes.
    for i in [1,2,3]:
        As, Rps = gilt_step(As, pars)
        Rps_dict[i] = Rps
        gilted_As_dict[i] = As

        As, ws = coarsegrain_step(As, pars, i)
        cged_As_dict[i] = As
        ws_dict[i] = ws

        As, log_facts = normalize_As(As, log_facts, i)
        log_facts_dict[i] = tuple(log_facts)

    retval = (As, log_facts,
              ws_dict, Rps_dict, gilted_As_dict, cged_As_dict, log_facts_dict)
    return retval


# A dictionary for which pairs of tensor have been combined during which
# CG.
normalization_pairs = {1: ((0,1), (3,2), (4,5), (6,7)),
                       2: ((0,4), (1,5), (2,6), (3,7)),
                       3: ((3,0), (2,1), (6,5), (7,4))}

def normalize_As(As, log_facts, typenumber):
    pairlist = normalization_pairs[typenumber]
    for n, m in pairlist:
        log_facts[n] += log_facts[m]
        log_facts[m] = log_facts[n]
    for i, A in enumerate(As):
        m = A.norm()
        if m != 0:
            As[i] /= m
            log_facts[i] += np.log(m)
    return As, log_facts


def gilt_step(As, pars):
    # If there are only 4 tensors, assume translation symmetry along
    # the x-axis.
    if len(As) == 4:
        As = As + As

    if pars["gilt_eps_squares"] >= 0 or pars["gilt_eps_cubes"] >= 0:
        if pars["verbosity"] > 0:
            status_print("Applying Gilt, starting.")
            orig_shps = [get_A_shape(As, None, i) for i in range(8)]

        res = gilt_As(As, pars)
        As, gilt_error = res[0], res[1]
        Rps = res[2]

        if pars["verbosity"] > 0:
            gilt_status_print(As, gilt_error, orig_shps, Rps=Rps)
    else:
        Rps = [[None]*6 for i in range(8)]

    return As, Rps


#==========================PERMUTATIONS=============================
#===================================================================

#       A3+------+ A7
#         |`.    |`.
#         | A4+------+ A8                z
#         |   |  |   |               y   |
#      A2 +---|--+A6 |                `. |
#          `. |   `. |                  `o----x
#         A1 `+------+ A5

# Dictionaries for permuting the eight tensors and their indices under
# various rotations. The keys are the names of the rotations, which mark
# the position of the index to which Gilt is being performed.
# The default position is FN (front north).
indexperms = {}
tensorperms = {}

# Elementary rotations by 90 degrees right-handedly around the three
# different axes.
indexperms["id"] = (0,1,2,3,4,5)
indexperms["x"]  = (0,5,2,4,1,3)
indexperms["y"]  = (3,0,1,2,4,5)
indexperms["z"]  = (5,1,4,3,0,2)
tensorperms["id"] = (0,1,2,3,4,5,6,7)
tensorperms["x"]  = (3,0,1,2,7,4,5,6)
tensorperms["y"]  = (4,5,1,0,7,6,2,3)
tensorperms["z"]  = (1,5,6,2,0,4,7,3)

def combine_permutations(p1, p2):
    """ p2 is applied first, then p1. """
    p = tuple(map(p2.__getitem__, p1))
    return p

def invert_permutation(p):
    inv = [0]*len(p)
    for i, e in enumerate(p):
        inv[e] = i
    return inv

# A function that helps building the dictionaries by combining already
# existing permutations. The permutations in sequence are combined so
# that the first one gets applied first.
def set_perms(indexperms, tensorperms, name, sequence):
    indexperm = tuple(range(6))
    tensorperm = tuple(range(8))
    for pname in sequence:
        ip = indexperms[pname]
        tp = tensorperms[pname]
        indexperm = combine_permutations(ip, indexperm)
        tensorperm = combine_permutations(tp, tensorperm)
    indexperms[name] = indexperm
    tensorperms[name] = tensorperm
    return indexperm, tensorperm

set_perms(indexperms, tensorperms, "FN", ("id",))
set_perms(indexperms, tensorperms, "FW", ("y", "FN"))
set_perms(indexperms, tensorperms, "FS", ("y", "FW"))
set_perms(indexperms, tensorperms, "FE", ("y", "FS"))

set_perms(indexperms, tensorperms, "BN", ("x", "FN"))
set_perms(indexperms, tensorperms, "BW", ("FW", "BN"))
set_perms(indexperms, tensorperms, "BS", ("FS", "BN"))
set_perms(indexperms, tensorperms, "BE", ("FE", "BN"))

set_perms(indexperms, tensorperms, "MNW", ("z", "FN"))
set_perms(indexperms, tensorperms, "MSW", ("FW", "MNW"))
set_perms(indexperms, tensorperms, "MSE", ("FS", "MNW"))
set_perms(indexperms, tensorperms, "MNE", ("FE", "MNW"))

# Permutations for doing the 8 different cubes, that are the
# neighborhoods in different culgs.
cubeperms = {}
cubeperms["id"]  = (0,1,2,3,4,5,6,7)  # North-back-west
cubeperms["nbw"] = cubeperms["id"]    # North-back-west
cubeperms["nbe"] = (4,5,6,7,0,1,2,3)  # North-back-east
cubeperms["nfw"] = (1,0,3,2,5,4,7,6)  # North-front-west
cubeperms["nfe"] = (5,4,7,6,1,0,3,2)  # North-front-east
cubeperms["sbw"] = (3,2,1,0,7,6,5,4)  # South-back-west
cubeperms["sbe"] = (7,6,5,4,3,2,1,0)  # South-back-east
cubeperms["sfw"] = (2,3,0,1,6,7,4,5)  # South-front-west
cubeperms["sfe"] = (6,7,4,5,2,3,0,1)  # South-front-east


def permute_As(As, leg="id", cube="id", inverse=False, Rps=None):
    """ Permute the list of tensors As according to the given cube or
    rotation.
    """
    global cubeperms, indexperms, tensorperms
    cubeperm = cubeperms[cube]
    indexperm = indexperms[leg]
    tensorperm = tensorperms[leg]
    if inverse:
        cubeperm = invert_permutation(cubeperm)
        indexperm = invert_permutation(indexperm)
        tensorperm = invert_permutation(tensorperm)
    if cubeperm != tuple(range(8)):
        As = [As[i] for i in cubeperm]
    if indexperm != tuple(range(6)):
        As = [A.transpose(indexperm) for A in As]
    if tensorperm != tuple(range(8)):
        As = [As[i] for i in tensorperm]

    # All modifications to Rp are done in place.
    # There was some reason why this was necessary, but now that I'm
    # reading it, I can't remember it.
    if Rps is not None:
        Rpspermed = Rps
        if cubeperm != tuple(range(8)):
            Rpspermed = [Rpspermed[i] for i in cubeperm]
        if tensorperm != tuple(range(8)):
            Rpspermed = [Rpspermed[i] for i in tensorperm]
        if indexperm != tuple(range(6)):
            for i in range(8):
                ARps = Rpspermed[i]
                ARpspermed = [ARps[i] for i in indexperm]
                Rpspermed[i] = ARpspermed
        for i in range(8):
            Rps[i] = Rpspermed[i]
    return As


#=========================COARSE GRAINING===========================
#===================================================================

# Illustrations of the different types:

def coarsegrain_step(As, pars, typenumber):
    """ Perform an HOTRG coarse-graining along one of the three axes.
    Which axis, is specified by typenumber:

     type1:                type2:                    type3:
        |                                             `. A
     ---+---                `. |   `. |              ---`+---
        |`. |                A`+------+                  |`.
         ---+---               | `.   | `.               |
           A|                                         `. |
                                                     ---`+---
                                                          `.
    """
    # The philosophy here is to always do the coarse-graining the same
    # way, but rotate the whole system before and after as necessary.
    rotation = "x" if typenumber==3 else "z" if typenumber==2 else "y"
    As = permute_As(As, leg=rotation)

    if pars["verbosity"] > 0:
        status_print("Coarse-graining, type{}.".format(typenumber))

    As, ws = coarsegrain_cube_y(As, pars)

    # Revert the rotation.
    As = permute_As(As, leg=rotation, inverse=True)
    return As, ws


def coarsegrain_cube_y(As, pars):
    """ Coarse-grain the cube of tensors As along the y-axis using
    HOTRG. Returns a new cubeful of tensors as As, half of which are
    copies of each other same, because of the newfound translation
    symmetry along y.
    """
    verb = pars["verbosity"]
    A1, w1b0, w1b1, w1b2, w1b3 = coarse_grain_pair(As[0], As[1], pars)
    A2 = A1.copy()
    if verb > 0:
        logging.info('-----')
    A8, w8b0, w8b1, w8b2, w8b3 = coarse_grain_pair(As[7], As[6], pars)
    A7 = A8.copy()
    if verb > 0:
        logging.info('-----')
    A4, w4b0, w4b1, w4b2, w4b3 = coarse_grain_pair(As[3], As[2], pars,
                                                   w0=w8b2.conjugate(),
                                                   w1=w1b3.conjugate(),
                                                   w2=w8b0.conjugate(),
                                                   w3=w1b1.conjugate())
    A3 = A4.copy()
    if verb > 0:
        logging.info('-----')
    A5, w5b0, w5b1, w5b2, w5b3 = coarse_grain_pair(As[4], As[5], pars,
                                                   w0=w1b2.conjugate(),
                                                   w1=w8b3.conjugate(),
                                                   w2=w1b0.conjugate(),
                                                   w3=w8b1.conjugate())
    A6 = A5.copy()
    if verb > 0:
        logging.info('-----')
    As = [A1, A2, A3, A4, A5, A6, A7, A8]

    # Store the isometries that did the coarse-graining in a dictionary,
    # according to which place in the network they occupy.
    # The first key is one of the two tensors that were used for
    # creating this isometry, the next one is the position with respect
    # to this tensor, counting from the negative x-axis clock-wise.
    ws = {k: [None]*4 for k in (0,3,4,7)}
    ws[0][0] = w1b0
    ws[0][1] = w1b1
    ws[0][2] = w1b2
    ws[0][3] = w1b3

    ws[3][0] = w4b0
    ws[3][1] = w4b1
    ws[3][2] = w4b2
    ws[3][3] = w4b3

    ws[4][0] = w5b0
    ws[4][1] = w5b1
    ws[4][2] = w5b2
    ws[4][3] = w5b3

    ws[7][0] = w8b0
    ws[7][1] = w8b1
    ws[7][2] = w8b2
    ws[7][3] = w8b3

    return As, ws


def cg_build_w(A, B, i, pars):
    """ Given to tensors A, B, build the ith HOTRG isometry. i is the
    position around the pair of tensors, and can take values from 0 to 3,
    0 corresponding to being on the negative x-axis side of A and B, 1
    corresponding to being on the positive z-axis side of A and B, and
    clock-wise onwards from there. A truncation error is also returned.
    """
    # If no truncation is done, use the identity as w.
    dimA = type(A).flatten_dim(A.shape[i])
    dimB = type(B).flatten_dim(B.shape[i])
    maxchi = max(pars["cg_chis"])
    if maxchi >= dimA*dimB:
        # All this qhape and dirs stuff is only accommodate for
        # symmetric tensors.
        if A.qhape is not None and B.qhape is not None:
            eyeA = type(A).eye(A.shape[i], qim=A.qhape[i])
            eyeB = type(B).eye(B.shape[i], qim=B.qhape[i])
        else:
            eyeA = type(A).eye(A.shape[i])
            eyeB = type(B).eye(B.shape[i])
        if A.dirs is not None and A.dirs[i] == -1:
            eyeA = eyeA.transpose()
        if B.dirs is not None and B.dirs[i] == -1:
            eyeB = eyeB.transpose()
        eye = ncon((eyeA, eyeB), ([-1,-11], [-2,-12]))
        if A.dirs is None:
            newdir = None
        else:
            newdir = A.dirs[i]
        w = eye.join_indices([0,1], dirs=[newdir])
        err = 0
    else:
        env = cg_build_w_env(A, B, i)
        # TODO do we want break_degenerate or not?
        S, U, err = env.eig([0,1], [2,3],
                            chis=pars["cg_chis"], eps=pars["cg_eps"],
                            hermitian=True, return_rel_err=True,
                            break_degenerate=True)
        w = U.conjugate().transpose((2,0,1))
    return w, err


def cg_build_w_env(A, B, i):
    """ Build the environment for constructing the ith HOTRG isometry
    for A and B.
    """
    A_indices1 = [1,2,3,4,5,-11]
    A_indices2 = [1,2,3,4,5,-12]
    A_indices1[i] = -1
    A_indices2[i] = -2
    A2 = ncon((A, A.conjugate()), (A_indices1, A_indices2))

    B_indices1 = [1,2,3,4,-11,6]
    B_indices2 = [1,2,3,4,-12,6]
    B_indices1[i] = -1
    B_indices2[i] = -2
    B2 = ncon((B, B.conjugate()), (B_indices1, B_indices2))

    env = ncon((A2, B2), ([-1,-11,1,2], [-2,-12,1,2]))
    return env


def cg_apply_ws(A, B, w0, w1, w2, w3, pars):
    """ Contract the tensors A and B with HOTRG isometries w1 to w3, to
    create the new, coarse-grained tensor.
    """
    # We manually loop over one of the legs to lower the memory cost.
    vects = []
    # Generate the vectors to sum over in this manual loop,
    # for non-symmetric tensors:
    if A.qhape is None:
        dim = A.shape[5]
        for j in range(dim):
            vect = type(A).zeros([dim])
            vect[j] = 1.
            vects.append(vect)
    # and for symmetric tensors:
    else:
        qim = A.qhape[5]
        dim = A.shape[5]
        direction = A.dirs[5]
        for i, q in enumerate(qim):
            qdim = dim[i]
            for j in range(qdim):
                vect = type(A).zeros([dim], qhape=[qim], dirs=[-direction],
                                     charge=-direction*q, invar=True)
                vect[(q,)][j] = 1.
                vects.append(vect)
    # Compute the networks with the middle leg replaced with
    # vect \otimes vect, and sum them all up.
    result = None
    for vect in vects:
        Ared = ncon((A, vect), ([-1,-2,-3,-4,-5,6], [6]))
        Bred = ncon((B, vect.conjugate()), ([-1,-2,-3,-4,5,-6], [5]))
        term = ncon((Ared, Bred,
                     w0, w1, w2, w3),
                    ([1,2,11,12,-5], [13,14,3,4,-6],
                     [-1,1,13], [-2,2,14], [-3,11,3], [-4,12,4]))
        if result is None:
            result = term
        else:
            result += term
    return result


def coarse_grain_pair(A, B, pars, print_indent=3,
                      w0=None, w1=None, w2=None, w3=None):
    """ Coarse-grain the tensors A and B using HOTRG. The isometries can be
    provided as keyword arguments. If they are not, they are generated
    by SVDing the respective environments.
    """
    verb = pars["verbosity"]
    ws = [w0, w1, w2, w3]
    for i in range(4):
        if ws[i] is None:
            if verb > 1:
                status_print("Building w{}, starting.".format(i),
                             indent=print_indent)
            w, w_error = cg_build_w(A, B, i, pars)
            ws[i] = w
            if verb > 1:
                chi = type(w).flatten_dim(w.shape[0])
                status_print("Building w{}, done.".format(i),
                             "Error = {:.3e}".format(w_error),
                             "chi = {}".format(chi),
                             indent=print_indent)
        elif verb > 1:
            w = ws[i]
            chi = type(w).flatten_dim(w.shape[0])
            status_print("w{} provided".format(i),
                         " "*17,
                         "chi = {}".format(chi),
                         indent=print_indent)

    if verb > 1:
        status_print("Contracting A's and w's, starting.",
                     indent=print_indent)
    w0, w1, w2, w3 = ws
    A_new = cg_apply_ws(A, B, w0, w1, w2, w3, pars)
    if verb > 1:
        status_print("Contracting A's and w's, done.",
                     indent=print_indent)
    return A_new, w0, w1, w2, w3



#==============================Gilt=================================
#===================================================================

#--- Part 1: Iterating over legs and environments. ---#

# Terminology:
# A culg is a combination of a cube (one of the eight
# different cubes we consider when building environments) and a leg
# (meaning the leg/edge of a cube).
#
# Cubes are labeled by lowercase letter combinations, such as "sfe",
# which stands for south-front-east. This reference cube:
#   A3+------+ A7
#     |`.    |`.
#     | A4+------+A8                     z
#     |   |  |   |                   y   |
#   A2+---|--+ A6|                    `. |
#      `. |   `. |                      `o----x
#      A1`+------+A5
# is called nbw, or north-back-west. North is towards the positive z-axis,
# east is towards the positive x-axis, and back is towards the positive
# y-axis.
#
# Legs of cubes are labeled by uppercase letter combinations such as FN
# for front-north or MSE for mid-south-east. In the above cube, front
# refers to the face of A4, A8, A1 and A5, and the opposite face is
# called back. The four legs around these are north, west, east and
# south. The legs that are diagonal in the above picture are called the
# middle/mid legs, and they are further specified by northeast,
# southwest, etc.
#
# All this labeling was developed with the Gilts in mind that uses cubes
# as environments. The Gilt that uses squares as environments uses the
# same labeling for culgs, although that's not the most natural
# description in that case.
#
# For any given culg, building the environment and at least trying to
# construct an Rp matrix that would perform a Gilt, is called a visit. If
# a culg has not been visited yet, or if the last time it was visited
# truncation was still succesfully happening, this culg is said to be
# not-done. Otherwise it's said to be done.
#
# Culgs are always visited in groups. The groups are called x+, x-, y+,
# y-, z+ and z-. x/y/z refer to axes. The A1 tensor in the above nbw
# cube is considered the origin. Each culg is in a group based on which
# axis the leg is parallel to, and whether the instance of this leg
# (keep in mind that the unit cube repeats) that is closest to A1 is on
# the positive (+) or negative (-) side along this axis. So as an
# example, the culg between A4 for A8 in the above picture is called
# (nbw, FN) and it belongs to the culg group x+.
#
# The following piece of code constructs these culg groups and a
# dictionary that holds their names.

leggroups = (("FN", "BN", "BS", "FS"),      # Along the x-axis.
             ("MNW", "MSW", "MSE", "MNE"),  # Along the y-axis.
             ("BW", "BE", "FW", "FE"))      # Along the z-axis.
cubegroups = (("nbw", "nfw", "sbw", "sfw"),  # All west
              ("nbe", "nfe", "sbe", "sfe"),  # All east
              ("nbe", "nbw", "sbe", "sbw"),  # All back
              ("nfe", "nfw", "sfe", "sfw"),  # All front
              ("nbe", "nbw", "nfw", "nfe"),  # All north
              ("sbe", "sbw", "sfw", "sfe"))  # All south
cube_culggroups = (tuple(itt.product(cubegroups[0], leggroups[0])),
                   tuple(itt.product(cubegroups[1], leggroups[0])),
                   tuple(itt.product(cubegroups[2], leggroups[1])),
                   tuple(itt.product(cubegroups[3], leggroups[1])),
                   tuple(itt.product(cubegroups[4], leggroups[2])),
                   tuple(itt.product(cubegroups[5], leggroups[2])))

# When Gilting with squares each leg only has one environment (unlike
# when doing cubes, where each leg is the leg for four different culgs).
# Thus we only need to consider two cubes that are diagonal to each
# other.
square_culggroups = (tuple(("nbw", d) for d in leggroups[0]),
                     tuple(("sfe", d) for d in leggroups[0]),
                     tuple(("nbw", d) for d in leggroups[1]),
                     tuple(("sfe", d) for d in leggroups[1]),
                     tuple(("nbw", d) for d in leggroups[2]),
                     tuple(("sfe", d) for d in leggroups[2]))

# Dictionaries for the names of these groups.
culggroup_names = {cube_culggroups[0]: "x+",
                   cube_culggroups[1]: "x-",
                   cube_culggroups[2]: "y+",
                   cube_culggroups[3]: "y-",
                   cube_culggroups[4]: "z+",
                   cube_culggroups[5]: "z-",
                   square_culggroups[0]: "x+",
                   square_culggroups[1]: "x-",
                   square_culggroups[2]: "y+",
                   square_culggroups[3]: "y-",
                   square_culggroups[4]: "z+",
                   square_culggroups[5]: "z-"}
# and for membership of individual culgs.
culg_groupnames = {}
for group, name in culggroup_names.items():
    for c in group:
        culg_groupnames[c] = name


def gilt_As(As, pars):
    """ Perform Gilt on the lattice defined by As. Returns as well the Rp
    matrices created and a rought estimeate of the error caused (that is
    usually overestimated.)
    """
    global cube_culggroups, square_culggroups
    gilt_error = 0.
    # Create a list for the Rp matrices to be created during this.
    # There are 8 elements in this list, one for in tensor in As.
    # Each element is itself a list, with six Rp matrices, one for each
    # leg. The legs are numbered as
    # 5   1
    #  `. |   
    # 0---+---2
    #     |`.   
    #     3  4
    # Note that the list comprehension makes sure that all the lists are
    # independent objects.
    Rps = [[None]*6 for i in range(8)]

    if pars["gilt_eps_squares"] >= 0:
        pars_squares = pars.copy()
        pars_squares["gilt_eps"] = pars_squares["gilt_eps_squares"]
        As, err = gilt_culggroups(As, square_culggroups, pars_squares,
                                  do_squares=True, Rps=Rps)
        gilt_error += err

    if pars["gilt_eps_cubes"] >= 0:
        pars_cubes = pars.copy()
        pars_cubes["gilt_eps"] = pars_cubes["gilt_eps_cubes"]
        As, err = gilt_culggroups(As, cube_culggroups, pars_cubes,
                                  do_squares=False, Rps=Rps)
        gilt_error += err

    return As, gilt_error, Rps


def gilt_culggroups(As, culggroups, pars, do_squares=False, Rps=None,
                    **kwargs):
    """
    Iterate over the different culgs, performing Gilt on each of them.
    The logic of the iteration is explained in the code/comments.
    The goal is to do as few visits as possible to save computational
    time, while still trying to make sure each culg gets truncated
    as much as possible. Depeding on the parameters, not all culgs may
    be visited. If for instance some of the legs in the same leg group
    don't seem to be truncatable, we may assume that the remaining ones
    are as well.
    """
    pars = update_pars(pars, **kwargs)
    allculgs = set(fct.reduce(opr.add, culggroups, ()))
    # Store the original dimensions of the tensors, so that we can
    # keep track of how they have been truncated.
    orig_shps = [get_A_shape(As, Rps, i) for i in range(8)]
    # Dones is a dictionary of culg: done-status.
    dones = reset_dones(allculgs)
    alldone = False
    gilt_error = 0.
    total_visits = 0
    # Counts the number of visits during which no truncation was done.
    useless_visits = 0  

    # groupdones is a list of the done-statuses picked up during
    # Gilting of the latest group.
    groupdones = []
    while not alldone:
        if all(groupdones):
            # We made no progress with the current group, so move on to
            # the next one.
            group = next_culggroup(As, Rps, culggroups, dones)
        groupdones = []
        # Sort the culgs according to the bond dimension of the leg we
        # are trying to truncate, starting with the thickest. This tries
        # to ensure that we first try squeezing the leg that should be
        # the easiest to squeeze.
        sorted_group = sorted(group,
                              key=lambda x: get_culg_dimension(As, Rps, x),
                              reverse=True)

        for culg in sorted_group:
            if dones[culg]:
                # Maybe the group was already partially done when we
                # started. In this case, skip the ones that are done.
                continue
            # Do the actual Gilt.
            As, done, err = gilt_culg(As, culg, pars, square=do_squares,
                                      Rps=Rps)
            total_visits += 1
            dones[culg] = done
            groupdones.append(done)
            gilt_error += err

            if not done:
                # If some Gilt was done, we should revisit all legs.
                dones = reset_dones(allculgs, dones=dones)
            else:
                useless_visits += 1

            if pars["verbosity"] > 1:
                gilt_status_print(As, gilt_error, orig_shps, Rps=Rps,
                                  total_visits=total_visits,
                                  useless_visits=useless_visits,
                                  culg=culg, do_squares=do_squares,
                                  dones=dones, culggroups=culggroups)

            alldone = stopping_criterion(As, Rps, culggroups, dones, pars)
            min_donecount = culggroups_mindonecount(culggroups, dones)
            this_donecount = culggroup_donecount(group, dones)
            if alldone or (this_donecount > min_donecount):
                # Either
                # 1) We are all done with the whole Gilting.
                # 2) Some culggroup has less legs done than this one.
                # In this case we shouldn't keep pushing on with this
                # group, but go back to selecting a new group. Note that
                # if a culg that is done is not one of the ones with the
                # thickest bonds, we may return to doing the same group
                # again, but starting with the thickest bond.
                break
    return As, gilt_error


def stopping_criterion(As, Rps, groups, dones, pars):
    """ Should we stop this whole Gilting? """
    if pars["gilt_hastyquit"]:
        stop = all(culggroup_thinnestdone(As, Rps, g, dones) for g in groups)
    else:
        stop = all(dones.values())
    return stop


def culggroup_donecount(group, dones):
    """ How many culgs in this group are done? """
    return sum(dones[l] for l in group)


def culggroups_mindonecount(groups, dones):
    """ What's the smallest donecount for any of the culg groups? """
    return min(culggroup_donecount(g, dones) for g in groups)


def culggroup_thinnestdone(As, Rps, group, dones):
    """ Is at least one of the culgs with the smallest bond dimension in
    this group done?
    """
    pairs = tuple((get_culg_dimension(As, Rps, l), l) for l in group)
    mindim = min(pairs)[0]
    anydone = any(dones[l] for dim, l in pairs if dim == mindim)
    return anydone


def culggroup_thickestdonecount(As, Rps, group, dones):
    """ Start counting from the thickest culg of the group, and count
    how many done ones are found before the first undone.
    """
    pairs = sorted(((get_culg_dimension(As, Rps, l), dones[l], l)
                    for l in group),
                   reverse=True)
    count = len(tuple(itt.takewhile(lambda p: p[1], pairs)))
    return count


def reset_dones(culgs, dones=None):
    if dones is None:
        dones = {}
        for l in culgs:
            dones[l] = False
    else:
        dones = type(dones).fromkeys(dones, False)
    return dones


def next_culggroup(As, Rps, groups, dones):
    """ The next culggroup to be visited is chosen as follows:
    First choose possible candidates. If the thickest culg in each group
    is done, candidate groups are ones that are not fully done. If
    there's a group that has its thickest culgs not-done, all such
    groups are candidates. Among the candidates, the next group is the
    one where the sum of the computational cost of Gilting its
    culgs is the smallest.
    """
    costgroups = []
    for g in groups:
        donecount = culggroup_thickestdonecount(As, Rps, g, dones)
        comp_cost = sum(get_culg_cost(As, Rps, c) for c in g)
        thickest_dimension = max(get_culg_dimension(As, Rps, c) for c in g)
        cost = (donecount, comp_cost, -thickest_dimension)
        costgroups.append((cost, g))
    next_group= min(costgroups)[1]
    return next_group


def gilt_culg(As, culg, pars, square=False, Rps=None):
    """ Perform Gilt on the given culg. Either using the cube or a face
    as the environment, depending on the keyword argument square. The
    nested list of Rp matrices should be provided as a keyword argument
    if they are to be kept track of: The list is edited in-place."""
    # The philosophy here is to always perform the Gilt the same way,
    # but first permute the tensors into a suitable order according to
    # the culg, and then revert the permutation.
    cube, leg = culg
    # Permute to correct cube.
    As = permute_As(As, cube=cube, Rps=Rps)

    if square:
        As, done, err = apply_gilt_squares(As, pars, leg=leg, Rps=Rps)
    else:
        As, done, err = apply_gilt_cubes(As, pars, leg=leg, Rps=Rps)

    # Reverse permutation.
    As = permute_As(As, cube=cube, inverse=True, Rps=Rps)
    return As, done, err


#--- Utility functions for part 1. ---#

# Some string(s) that will be formatted again and again.
# TODO note that the group names are hard coded, because it's hard to
# make the printed bond dimensions be determined dynamically from the
# groups. This may cause confusion if at some point the groups are
# changed.
shape_str_line =  "\n{}  "
shape_str_line += "   {:2,d} [{:2,d}]"*6
shape_str_template =  "\nbonds:"
shape_str_template += "   x+        x-        y+        y-        z+        z-"
shape_str_template += shape_str_line*4


def get_A_dim(As, Rps, A_number, leg_number):
    Atype = type(As[A_number])
    try:
        res = Atype.flatten_dim(Rps[A_number][leg_number].shape[1])
    except (AttributeError, TypeError):
        res = Atype.flatten_dim(As[A_number].shape[leg_number])
    return res


def get_A_shape(As, Rps, A_number):
    res = [get_A_dim(As, Rps, A_number, i) for i in range(6)]
    return res


def gilt_status_print(As, gilt_error, orig_shps, Rps=None, total_visits=None,
                      useless_visits="?", do_squares=None, culg=None,
                      dones=None, culggroups=None):
    """ Print out a nice message that shows the current status of the
    ongoing Gilt.
    """
    shps = [get_A_shape(As, Rps, i) for i in range(8)]
    sqr_vs_cube_str = ("" if do_squares is None else
                       "squares" if do_squares else
                       "cubes")
    culg_str = ("" if culg is None else
                "last culg: {:3}/{:3} ({})"
                .format(*culg, culg_groupnames[culg]))
    total_visits_str = ("" if total_visits is None else
                        "\nuseless/total visits: {:3,d}/{:3,d}"
                        .format(useless_visits, total_visits))
    if dones is not None and culggroups is not None:
        dones_str = "\ndone:"
        for g in culggroups:
            count = sum(dones[l] for l in g)
            total_count = len(g)
            dones_str += "  {:2,d}/{:2,d}   ".format(count, total_count)
    else:
        dones_str = ""
    s = shps
    o = orig_shps
    shape_str = shape_str_template.format(
        "A1",
        s[0][2], o[0][2], s[0][0], o[0][0], s[0][5], o[0][5],
        s[0][4], o[0][4], s[0][1], o[0][1], s[0][3], o[0][3],
        "A3",
        s[2][2], o[2][2], s[2][0], o[2][0], s[2][4], o[2][4],
        s[2][5], o[2][5], s[2][3], o[2][3], s[2][1], o[2][1],
        "A6",
        s[5][0], o[5][0], s[5][2], o[5][2], s[5][4], o[5][4],
        s[5][5], o[5][5], s[5][1], o[5][1], s[5][3], o[5][3],
        "A8",
        s[7][0], o[7][0], s[7][2], o[7][2], s[7][5], o[7][5],
        s[7][4], o[7][4], s[7][3], o[7][3], s[7][1], o[7][1]
    )
    status_print("Applying Gilt {},".format(sqr_vs_cube_str),
                 "Error = {:.3e}".format(gilt_error),
                 total_visits_str, culg_str, dones_str, shape_str)
    return None


def get_culg_dimension(As, Rps, culg):
    """ What is the bond dimension of the to-be-Gilted leg in this culg?
    """
    c, d = culg
    cubeperm = cubeperms[c]
    tensorperm = tensorperms[d]
    indexperm = indexperms[d]
    A_number = cubeperm[tensorperm[3]]
    dim = get_A_dim(As, Rps, A_number, indexperm[2])
    return dim


def get_culg_cost(As, Rps, culg):
    """ A pretty good estimate of the leading order cost of constructing
    the environment for this culg using the parameter gilt_split.
    """
    c, d = culg
    # To avoid actually permuting tensor elements we do this manually
    # instead of calling permute_As.
    cubeperm = cubeperms[c]
    tensorperm = tensorperms[d]
    indexperm = indexperms[d]
    T = type(As[0])
    As = [As[cubeperm[i]] for i in tensorperm]
    NW_dim = get_A_dim(As, Rps, 3, indexperm[5])
    NE_dim = get_A_dim(As, Rps, 7, indexperm[5])
    SW_dim = get_A_dim(As, Rps, 0, indexperm[5])
    SE_dim = get_A_dim(As, Rps, 4, indexperm[5])
    BW_dim = get_A_dim(As, Rps, 1, indexperm[1])
    BE_dim = get_A_dim(As, Rps, 5, indexperm[1])
    FW_dim = get_A_dim(As, Rps, 0, indexperm[1])
    FE_dim = get_A_dim(As, Rps, 4, indexperm[1])
    N_dim = NW_dim**2 * NE_dim**2
    S_dim = SW_dim**2 * SE_dim**2
    B_dim = BW_dim**2 * BE_dim**2
    F_dim = FW_dim**2 * FE_dim**2
    cost = N_dim*B_dim + B_dim*S_dim + S_dim*F_dim
    return cost


#--- Part 2: Building the environment and applying Rp. ---#


def update_Rps(Rps, A_number, leg_number, Rp):
    """ Update the M matrix on a given leg of a given A.
    If there is already a matrix here, the two are multiplied together.
    """
    old_Rp = Rps[A_number][leg_number]
    if old_Rp is None:
        new_Rp = Rp
    else:
        new_Rp = ncon((old_Rp, Rp), ([-1,1], [1,-2]))
    Rps[A_number][leg_number] = new_Rp
    return


def apply_gilt_squares(As, pars, leg=None, Rps=None):
    """ Apply Gilt to the given leg of As, using a 2D square(s) as the
    neighborhood. The list of Rp matrices is updated in-place. Note that
    by the time we hit this function, the As have already been permuted
    according to which cube we are looking at, although the rotation
    according to which leg is being squeezed has not been done yet.
    """
    # Permute so that the leg that is about to get Gilted is the FN leg.
    As = permute_As(As, leg=leg, Rps=Rps)
    all_done = True
    total_err = 0

    As, done, err = apply_gilt_squares_FN(As, pars, "horz_across", Rps=Rps)
    all_done = all_done and done
    total_err += err
    As, done, err = apply_gilt_squares_FN(As, pars, "vert_across", Rps=Rps)
    all_done = all_done and done
    total_err += err

    # Invert the permutation done in the beginning.
    As = permute_As(As, leg=leg, inverse=True, Rps=Rps)
    return As, all_done, total_err


def apply_gilt_squares_FN(As, pars, envtype, Rps=None):
    # Construct the environment, depending on which type has been
    # chosen.
    if envtype=="horz_across":
        BW = ncon((As[2], As[2].conjugate()),
                  ([1,2,-11,4,-1,3], [1,2,-12,4,-2,3]))
        BE = ncon((As[6], As[6].conjugate()),
                  ([-11,1,2,4,-1,3], [-12,1,2,4,-2,3]))
        MW = ncon((As[3], As[3].conjugate()),
                  ([1,2,-1,4,-111,-11], [1,2,-2,4,-112,-12]))
        ME = ncon((As[7], As[7].conjugate()),
                  ([-1,1,2,4,-111,-11], [-2,1,2,4,-112,-12]))
        FW = ncon((As[2], As[2].conjugate()),
                  ([1,2,-11,4,3,-1], [1,2,-12,4,3,-2]))
        FE = ncon((As[6], As[6].conjugate()),
                  ([-11,1,2,4,3,-1], [-12,1,2,4,3,-2]))
        env = ncon((BW, BE,
                    MW, ME,
                    FW, FE),
                   ([5,6,1,2], [9,10,1,2],
                    [-1,-11,5,6,7,8], [-2,-12,9,10,11,12],
                    [7,8,3,4], [11,12,3,4]))

    elif envtype=="vert_across":
        NW = ncon((As[0], As[0].conjugate()),
                  ([1,2,-11,-1,3,4], [1,2,-12,-2,3,4]))
        NE = ncon((As[4], As[4].conjugate()),
                  ([-11,2,1,-1,3,4], [-12,2,1,-2,3,4]))
        MW = ncon((As[3], As[3].conjugate()),
                  ([1,-11,-1,-111,2,3], [1,-12,-2,-112,2,3]))
        ME = ncon((As[7], As[7].conjugate()),
                  ([-1,-11,1,-111,2,3], [-2,-12,1,-112,2,3]))
        SW = ncon((As[0], As[0].conjugate()),
                  ([1,-1,-11,2,3,4], [1,-2,-12,2,3,4]))
        SE = ncon((As[4], As[4].conjugate()),
                  ([-11,-1,1,2,3,4], [-12,-2,1,2,3,4]))
        env = ncon((NW, NE,
                    MW, ME,
                    SW, SE),
                   ([5,6,1,2], [9,10,1,2],
                    [-1,-11,5,6,7,8], [-2,-12,9,10,11,12],
                    [7,8,3,4], [11,12,3,4]))

    # Apply Gilt to this environment.
    As[7], As[3], done, err = apply_gilt_FNenv(env, As[7], As[3],
                                               pars, Rps=Rps)
    return As, done, err


def apply_gilt_FNenv(env, Aright, Aleft, pars, Rps=None):
    S, U = env.eig([0,1], [2,3], hermitian=True)
    S = S.abs().sqrt()
    S /= S.sum()
    if pars["gilt_print_envspec"]:
        print_envspec(S)
    Rp, roterr = optimize_Rp(U, S, pars)
    spliteps = pars["gilt_eps"]*1e-3
    Rp1, s, Rp2, spliterr = Rp.split(0, 1, eps=spliteps, return_rel_err=True,
                                     return_sings=True)
    Rp2 = Rp2.transpose()
    global convergence_eps
    if (s-1).abs().max() < convergence_eps:
        done = True
    else:
        done = False

    err = roterr + spliterr
    Aleft = ncon((Aleft, Rp1), ([-1,-2,3,-4,-5,-6], [3,-3]))
    Aright = ncon((Aright, Rp2), ([1,-2,-3,-4,-5,-6], [1,-1]))
    if Rps is not None:
        update_Rps(Rps, 3, 2, Rp1)
        update_Rps(Rps, 7, 0, Rp2)

    return Aright, Aleft, done, err


def apply_gilt_cubes(As, pars, leg=None, Rps=None):
    """ Apply Gilt to the leg in question, for the cube consisting of
    As.
    """
    # Permute and rotate the tensors, so that they are arranged as if
    # leg="FN".
    As = permute_As(As, leg=leg, Rps=Rps)

    #COST: 8chi^9
    FNW = ncon((As[3], As[3].conjugate()),
               ([1,2,-1,-11,3,-111], [1,2,-2,-12,3,-112]))
    FNE = ncon((As[7], As[7].conjugate()),
               ([-111,1,2,-11,3,-1], [-112,1,2,-12,3,-2]))
    FSW = ncon((As[0], As[0].conjugate()),
               ([1,-11,-1,2,3,-111], [1,-12,-2,2,3,-112]))
    FSE = ncon((As[4], As[4].conjugate()),
               ([-111,-11,1,2,3,-1], [-112,-12,1,2,3,-2]))
    BNW = ncon((As[2], As[2].conjugate()),
               ([1,2,-111,-11,-1,3], [1,2,-112,-12,-2,3]))
    BNE = ncon((As[6], As[6].conjugate()),
               ([-1,1,2,-11,-111,3], [-2,1,2,-12,-112,3]))
    BSW = ncon((As[1], As[1].conjugate()),
               ([1,-11,-111,2,-1,3], [1,-12,-112,2,-2,3]))
    BSE = ncon((As[5], As[5].conjugate()),
               ([-1,-11,1,2,-111,3], [-2,-12,1,2,-112,3]))

    if pars["gilt_split"]:
        # Use fancy splittings to try to contract the environment faster.
        env = build_gilt_split_cube(FSW, FSE, FNW, FNE, BSW, BSE, BNW, BNE,
                                    pars)
    if not pars["gilt_split"] or env is None:
        # If env is None, then splitting was attempted but aborted.
        # Just contract the environment.
        # Cost: O(chi^12)
        env = ncon((FSW, FNW,
                    FNE, FSE,
                    BSW, BNW,
                    BNE, BSE),
                   ([3,4,1,2,5,6], [-1,-11,1,2,9,10],
                    [13,14,15,16,-2,-12], [19,20,15,16,3,4],
                    [5,6,7,8,21,22], [9,10,7,8,11,12],
                    [11,12,17,18,13,14], [21,22,17,18,19,20]),
                   order=([5,6,17,18,15,16,13,14,19,20,21,22,3,4,11,12,7,8,9,
                           10,1,2]))

    As[7], As[3], done, err = apply_gilt_FNenv(env, As[7], As[3], pars,
                                               Rps=Rps)

    # Invert the permutations done in the beginning.
    As = permute_As(As, leg=leg, inverse=True, Rps=Rps)
    return As, done, err


def build_gilt_split_cube(FSW, FSE, FNW, FNE, BSW, BSE, BNW, BNE, pars):
    """ Build the environment from the cube, but instead of doing a
    straight-forward contraction of the environment, contract parts of,
    split those parts into pieces with an SVD, and contract the pieces.
    This is in many cases faster.
    """
    FS = ncon((FSW, FSE), ([1,2,-11,-12,-1,-2], [-3,-4,-13,-14,1,2]))
    BS = ncon((BSW, BSE), ([-11,-12,-1,-2,1,2], [1,2,-3,-4,-13,-14]))
    BN = ncon((BNW, BNE), ([-1,-2,-11,-12,1,2], [1,2,-13,-14,-3,-4]))

    shp_left = [BN.shape[i] for i in range(4)]
    shp_right = [FS.shape[i] for i in range(4,8)]
    if BN.qhape is not None:
        qhp_left = [BN.qhape[i] for i in range(4)]
    else:
        qhp_left = None
    if FS.qhape is not None:
        qhp_right = [FS.qhape[i] for i in range(4,8)]
    else:
        qhp_right = None
    if BN.dirs is not None:
        dirs_left = [BN.dirs[i] for i in range(4)]
    else:
        dirs_left = None
    if FS.dirs is not None:
        dirs_right = [FS.dirs[i] for i in range(4,8)]
    else:
        dirs_right = None

    FS = FS.join_indices([0,1,2,3], [4,5,6,7], dirs=[1,-1])
    BS = BS.join_indices([0,1,2,3], [4,5,6,7], dirs=[1,-1])
    BN = BN.join_indices([0,1,2,3], [4,5,6,7], dirs=[1,-1])

    # To create the environment, FS, BS and BN should be multiplied
    # together as matrices: X = FS*BS*BN. We try to SVD the product
    # matri X, without ever even constructing it. This is done by
    # constructing a linear function that implements X*v for vectors v,
    # and using sparse SVD techniques (essentially fancier power
    # methods) to do the SVD. This relies on the assumption that most of
    # the singular values of X are negligbly small.

    # Maximum number of singular values.
    Smaxdim = min(type(BN).flatten_dim(BN.shape[0]),
                  type(BN).flatten_dim(BN.shape[1]),
                  type(FS).flatten_dim(FS.shape[0]),
                  type(FS).flatten_dim(FS.shape[1]))
    # If the maximum number of singular values is small enough, just
    # revert to the usual full contraction, by returning None.
    if Smaxdim <= 512:  # TODO hard constant.
        return None

    # The tricky question is, how many singular values do we need to do,
    # to get all the non-zero ones? We don't know exactly, but what we
    # do, is assume that the answer is of the form
    # constant*sqrt(Smaxdim). The constant is set by the parameter
    # gilt_split_factor. If in addition the parameter gilt_split_dynamic
    # is True, then during the run we then keep adjusting the
    # constant. We do an SVD, if the smallest singular values are not
    # small enough (smaller than gilt_split_dynamic_eps), we need to
    # increase the constant and redo the SVD. If the smallest singular
    # values are small enough, and in fact many of them are really
    # small, we decrease the constant for the next time we come to this
    # function. If the constant ever grows too big (bigger than
    # glit_split_dynamic_max_factor), so that doing the "sparse" SVD
    # based contraction isn't faster anymore, we revert to the full
    # contraction.
    exponent = 0.5
    while True:
        n_sings = round((Smaxdim)**exponent * pars["gilt_split_factor"])
        try:
            U, S, V = ncon_sparsesvd((BN, BS, FS),
                                     ([-1,2], [2,3], [3,-4]),
                                     left_inds=[0], right_inds=[1],
                                     matvec_order=[4,3,2],
                                     rmatvec_order=[1,2,3],
                                     matmat_order=[4,3,2],
                                     chis=[n_sings], truncate=False)
        except (ArpackNoConvergence, ArpackError,
                ValueError, LookupError) as err:
            # These types of errors should be reported, but it's more
            # useful to fall back to the full SVD than to simply crash.
            # Probable causes are bugs related to corner cases and bad
            # luck with convergence.
            msg = ("In gilt_split's ncon_sparsesvd, the following exception"
                   " occured:"
                   "\n{}:, {}"
                   "\nSwitching to full contraction.").format(type(err), err)
            warnings.warn(msg)
            return None
        except:
            # Any other generic error we reraise.
            raise

        Sratio = S.min() / S.sum()
        cutsufficient = Sratio < pars["gilt_split_dynamic_eps"]
        max_factor = pars["gilt_split_dynamic_max_factor"]
        if not cutsufficient:
            # We cut too much in svds and we didn't have to.
            if pars["gilt_split_dynamic"]:
                # We should increase split_factor and retry.
                padding_factor = 1.3  # TODO hard constant
                if hasattr(S, "sects"):
                    chi = 0
                    for sect in S.sects.values():
                        chi = max(len(sect), chi)
                else:
                    chi = len(S)
                chi = max(round(chi*padding_factor), chi+1)
                new_factor = chi/(Smaxdim)**exponent
                if new_factor > max_factor:
                    # The whole sparse thing doesn't make sense any more
                    # if the factor is this big.
                    return None
                else:
                    pars["gilt_split_factor"] = new_factor
            else:
                # Can't do anything about it, so just lament and
                # move on.
                msg = ("gilt_split_factor possibly too small, "
                       "min(S)/sum(S) = {}."
                       .format(Sratio))
                warnings.warn(msg)
                break
        else:
            if pars["gilt_split_dynamic"]:
                # We should decrease split_factor for the next step.
                padding_factor = 1.1  # TODO hard constant
                Ssum = S.sum()
                if hasattr(S, "sects"):
                    chi = 0
                    for sect in S.sects.values():
                        sectbigs = sect/Ssum > pars["gilt_split_dynamic_eps"]
                        chi = max(np.count_nonzero(sectbigs), chi)
                else:
                    chi = np.count_nonzero(
                        S/Ssum > pars["gilt_split_dynamic_eps"]
                    )
                chi = max(round(chi*padding_factor), chi+1)
                new_factor = chi/(Smaxdim)**exponent
                pars["gilt_split_factor"] = min(new_factor, max_factor)
            # In any case, we are good, so move on.
            break

    US = U.multiply_diag(S, 1, direction="right")
    US = US.split_indices(0, shp_left, qims=qhp_left, dirs=dirs_left)
    V = V.split_indices(1, shp_right, qims=qhp_right, dirs=dirs_right)

    cube = ncon((US, V,
                 FNW, FNE),
                ([10,11,21,22,1], [1,13,14,23,24],
                 [-1,-11,13,14,10,11], [21,22,23,24,-2,-12]))
    return cube


#--- Part 3: Building an optimal Rp, given an environment. ---#


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
    ratio = S/gilt_eps
    weight = ratio**2/(1+ratio**2)
    tp = t.multiply_diag(weight, 0, direction="left")
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


#--- Part 4: Utility functions. ---#


print_pad = 36


def status_print(pre, *args, indent=0):
    """ Uniform formating for various status messages. """
    arg_str = ", ".join(["{}"]*len(args))
    pre_str = " "*indent + "{:<" + str(print_pad-indent) + "}"
    status_str =  pre_str + arg_str
    status_str = status_str.format(pre, *args)
    logging.info(status_str)
    return


def print_envspec(S):
    """ Print out the environment spectrum S. """
    l = len(S)
    step = int(np.ceil(l/100))
    envspeclist = sorted(S.to_ndarray(), reverse=True)
    envspeclist = envspeclist[0:-1:step]
    envspeclist = np.array(envspeclist)
    msg = "The environment spectrum, with step {} in {}".format(step, l)
    logging.info(msg)
    logging.info(envspeclist)


def update_pars(pars, **kwargs):
    if kwargs:
        pars = pars.copy()
        pars.update(kwargs)
    return pars


