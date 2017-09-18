import GiltTNR2D
import logging

version = GiltTNR2D.version

parinfo = {
    "iter_count": {
        "default": 0,
        "idfunc":  lambda dataname, pars: True
    },

    # Parameters for disentangling.
    "gilt_eps": {
        "default": 1e-6,
        "idfunc":  lambda dataname, pars: not bool(pars["gilt_eps_list"])
    },
    "gilt_eps_list": {
        "default": None,
        "idfunc":  lambda dataname, pars: bool(pars["gilt_eps_list"])
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
               "\nGenerating {} with Gilt-TNR2D (version {})."
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
    if pars["gilt_eps_list"]:
        pars = pars.copy()
        try:
            pars["gilt_eps"] = pars["gilt_eps_list"][iter_count-1]
        except IndexError:
            pars["gilt_eps"] = pars["gilt_eps_list"][-1]

    if dataname == "A":
        res = generate_A(*args, pars=pars)
    else:
        raise ValueError("Unknown dataname: {}".format(dataname))
    return res


def generate_A(*args, pars=dict()):
    A, log_fact = args[0]
    res = GiltTNR2D.gilttnr_step(A, log_fact, pars)
    return res


def prereq_pairs(dataname, pars):
    if dataname == "A":
        res = prereq_pairs_A(pars)
    else:
        raise ValueError("Unknown dataname: {}".format(dataname))
    return res


def prereq_pairs_A(pars):
    prereq_pars = pars.copy()
    prereq_pars["iter_count"] -= 1
    res = [("A", prereq_pars)]
    return res

