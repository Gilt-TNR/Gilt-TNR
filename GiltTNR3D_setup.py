import GiltTNR3D
import GiltTNR3D_impurity
import logging

version = GiltTNR3D.version

parinfo = {
    "iter_count": {
        "default": 0,
        "idfunc":  lambda dataname, pars: True
    },

    # Parameters for disentangling.
    "gilt_eps_cubes": {
        "default": 1e-4,
        "idfunc":  (lambda dataname, pars:
                    pars["gilt_eps_cubes"] >= 0
                    and not bool(pars["gilt_eps_cubes_list"]))
    },
    "gilt_eps_cubes_list": {
        "default": None,
        "idfunc":  lambda dataname, pars: bool(pars["gilt_eps_cubes_list"])
    },
    "gilt_eps_squares": {
        "default": 1e-4,
        "idfunc":  (lambda dataname, pars:
                    pars["gilt_eps_squares"] >= 0
                    and not bool(pars["gilt_eps_squares_list"]))
    },
    "gilt_eps_squares_list": {
        "default": None,
        "idfunc":  lambda dataname, pars: bool(pars["gilt_eps_squares_list"])
    },
    "gilt_split": {
        "default": True,
        "idfunc":  lambda dataname, pars: True
    },
    "gilt_split_factor": {
        "default": 1.,
        "idfunc":  lambda dataname, pars: True
    },
    "gilt_split_dynamic": {
        "default": True,
        "idfunc":  lambda dataname, pars: True
    },
    "gilt_split_dynamic_eps": {
        "default": 1e-8,
        "idfunc":  lambda dataname, pars: pars["gilt_split_dynamic"]
    },
    "gilt_split_dynamic_max_factor": {
        "default": 2.,
        "idfunc":  lambda dataname, pars: pars["gilt_split_dynamic"]
    },
    "gilt_hastyquit": {
        "default": False,
        "idfunc":  lambda dataname, pars: True
    },
    "gilt_print_envspec": {
        "default": False,
        "idfunc":  lambda dataname, pars: False
    },
    "gilt_print_envspec_recursive": {
        "default": False,
        "idfunc":  lambda dataname, pars: False
    },

    # Parameters for the coarse-graining.
    "cg_chis": {
        "default": [1,2,3,4,5,6],
        "idfunc":  lambda dataname, pars: True
    },
    "cg_eps": {
        "default": 1e-3,
        "idfunc":  lambda dataname, pars: True
    },

    # Other parameters
    "verbosity": {
        "default": 10,
        "idfunc":  lambda dataname, pars: False
    },
}


def generate(dataname, *args, pars=dict(), filelogger=None):
    infostr = ("{}"
               "\nGenerating {} with GiltTNR3D (version {})."
               "\niter_count = {}"
               .format("="*70, dataname, version, pars["iter_count"]))
    logging.info(infostr)
    if filelogger is not None:
        # Only print the dictionary into the log file, not in stdout.
        dictstr = ""
        for k,v in sorted(pars.items()):
           dictstr += "\n%s = %s"%(k, v)
        filelogger.info(dictstr)

    iter_count = pars["iter_count"]
    if pars["gilt_eps_cubes_list"]:
        pars = pars.copy()
        try:
            pars["gilt_eps_cubes"] = pars["gilt_eps_cubes_list"][iter_count-1]
        except IndexError:
            pars["gilt_eps_cubes"] = pars["gilt_eps_cubes_list"][-1]
    if pars["gilt_eps_squares_list"]:
        pars = pars.copy()
        try:
            pars["gilt_eps_squares"] = pars["gilt_eps_squares_list"][iter_count-1]
        except IndexError:
            pars["gilt_eps_squares"] = pars["gilt_eps_squares_list"][-1]

    if dataname == "As":
        res = generate_As(*args, pars=pars)
    elif dataname == "As_impure":
        res = generate_As_impure(*args, pars=pars)
    else:
        raise ValueError("Unknown dataname: {}".format(dataname))
    return res


def generate_As(*args, pars=dict()):
    As, log_facts = args[0][0], args[0][1]
    # TODO The array should be the default form of returning log_facts for
    # initialtensors, but that would break backwards compatibility.
    if log_facts == 0:
        log_facts = [0]*8
    res = GiltTNR3D.gilttnr_step(As, log_facts, pars)
    return res


def generate_As_impure(*args, pars=dict()):
    A_impure, log_fact_impure = args[0]
    ws_pure, Ms_pure, As_deed_pure, log_facts_pure = (
        args[1][2], args[1][3], args[1][4], args[1][6]
    )
    odd_iter = (pars["iter_count"] % 2) == 1
    A_impure, log_fact_impure = GiltTNR3D.gilttnr_step_impurity(
        A_impure, log_fact_impure,
        Ms_pure, ws_pure, As_deed_pure, log_facts_pure, pars,
        start_at_7=odd_iter
    )
    return A_impure, log_fact_impure


def prereq_pairs(dataname, pars):
    if dataname == "As":
        res = prereq_pairs_As(pars)
    elif dataname == "As_impure":
        res = prereq_pairs_As_impure(pars)
    else:
        raise ValueError("Unknown dataname: {}".format(dataname))
    return res


def prereq_pairs_As(pars):
    prereq_pars = pars.copy()
    prereq_pars["iter_count"] -= 1
    res = [("As", prereq_pars)]
    return res


def prereq_pairs_As_impure(pars):
    prereq_pars1 = pars.copy()
    prereq_pars2 = prereq_pars1.copy()
    prereq_pars1["iter_count"] -= 1
    res = [("As_impure", prereq_pars1)]  # The previous impurity
    res += [("As", prereq_pars2)]  # The Ms, ws, etc. from the pure step.
    return res


