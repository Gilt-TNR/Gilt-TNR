import numpy as np
import sys
import os
import warnings
import logging
import logging.config
import configparser
import datetime
from tntools import modeldata, datadispenser, multilineformatter
from tntools.yaml_config_parser import parse_argv
from tntools.ncon_sparseeig import ncon_sparseeig
from ncon import ncon

tools_path = os.path.dirname(multilineformatter.__file__)
logging.config.fileConfig(tools_path + "/logging_default.conf")
filename = os.path.basename(__file__).replace(".py", "")
np.set_printoptions(precision=10)


parinfo = {
    "iters": {
        "default": range(1, 6),
    },
    "print_scaldims": {
        "default": 50,
    },
    "print_spectra": {
        "default": True,
    },
    "print_free_energy": {
        "default": True,
    },
    "debug": {
        "default": False
    },
    "database": {
        "default": "data/GiltTNR2D/"
    }
}


def parse():
    pars = parse_argv(sys.argv)
    pars["algorithm"] = "GiltTNR2D"
    return pars


def apply_default_pars(pars, parinfo):
    for k, v in parinfo.items():
        if k not in pars:
            pars[k] = v["default"]
    return


def set_filehandler(logger, logfilename, pars):
    os.makedirs(os.path.dirname(logfilename), exist_ok=True)
    filehandler = logging.FileHandler(logfilename, mode='w')
    if pars["debug"]:
        filehandler.setLevel(logging.DEBUG)
    else:
        filehandler.setLevel(logging.INFO)
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(tools_path + '/logging_default.conf')
    fmt = parser.get('formatter_default', 'format')
    datefmt = parser.get('formatter_default', 'datefmt')
    formatter = multilineformatter.MultilineFormatter(fmt=fmt, datefmt=datefmt)
    filehandler.setFormatter(formatter)
    rootlogger.addHandler(filehandler)
    return


def get_free_energy(A, log_fact, pars, iter_count):
    Z = ncon(A, [1,2,1,2]).value()
    log_Z = np.log(Z) + log_fact
    F = -log_Z/pars["beta"]
    f = F/(2*4**(iter_count))
    return f


def get_A_spectrum(A):
    es = A.svd([0,1], [2,3])[1]
    es = es.to_ndarray()
    es /= np.max(es)
    es = -np.sort(-es)
    return es


def get_scaldims(A, pars):
    logging.info("Diagonalizing the transfer matrix.")
    # The cost of this scales as O(chi^6).
    transmat = ncon((A, A), [[3,-101,4,-1], [4,-102,3,-2]])
    es = transmat.eig([0,1], [2,3], hermitian=False)[0]
    # Extract the scaling dimensions from the eigenvalues of the
    # transfer matrix.
    es = es.to_ndarray()
    es = np.abs(es)
    es = -np.sort(-es)
    es[es==0] += 1e-16  # Ugly workaround for taking the log of zero.
    log_es = np.log(es)
    log_es -= np.max(log_es)
    log_es /= -np.pi
    return log_es


if __name__ == "__main__":
    pars = parse()
    apply_default_pars(pars, parinfo)
    datadispenser.update_default_pars("A", pars, iter_count=max(pars["iters"]))

    if pars["debug"]:
        warnings.filterwarnings('error')

    datetime_str = datetime.datetime.strftime(datetime.datetime.now(),
                                             '%Y-%m-%d_%H-%M-%S')
    title_str = ('{}_{}_beta{}_cgchis{}_deeps{}'
                 .format(filename, datetime_str, pars["beta"],
                         pars["cg_chis"][-1], pars["gilt_eps"]))
    logfilename = "logs/{}.log".format(title_str)
    rootlogger = logging.getLogger()
    set_filehandler(rootlogger, logfilename, pars)

    # - Infoprint -
    infostr = "\n{}\n".format("="*70)
    infostr += "Running {} with the following parameters:".format(filename)
    for k,v in sorted(pars.items()):
        infostr += "\n%s = %s"%(k, v)
    logging.info(infostr)

    dbname = pars["database"]

    for it in pars["iters"]:
        logging.info("\nIteration {}".format(it))
        res = datadispenser.get_data(
            dbname, "A", pars, iter_count=it
        )
        A, log_fact = res[0], res[1]

        if pars["print_spectra"]:
            es = get_A_spectrum(A)
            msg = "Spectrum of A:\n{}".format(es[:30])
            logging.info(msg)

        if pars["print_free_energy"]:
            f = get_free_energy(A, log_fact, pars, it)
            exact_f = modeldata.get_free_energy(pars)
            f_error = np.abs(f - exact_f)/exact_f
            msg = ("Free energy per site: {} ({}, off by {:.4e})"
                   .format(f, exact_f, f_error))
            logging.info(msg)

        if pars["print_scaldims"] > 0:
            scaldims = get_scaldims(A, pars)
            scaldims = scaldims[:pars["print_scaldims"]]
            msg = "Scaldims:\n{}".format(scaldims)
            logging.info(msg)

