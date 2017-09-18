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
from ncon import ncon

tools_path = os.path.dirname(multilineformatter.__file__)
logging.config.fileConfig(tools_path + "/logging_default.conf")
filename = os.path.basename(__file__).replace(".py", "")
np.set_printoptions(precision=10)

parinfo = {
    "print_spectra": {
        "default": True,
    },
    "print_free_energy": {
        "default": True,
    },
    "iters": {
        "default": range(1, 6),
    },
    "debug": {
        "default": False
    },
    "database": {
        "default": "data/GiltTNR3D/"
    }
}


def parse():
    pars = parse_argv(sys.argv)
    pars["algorithm"] = "GiltTNR3D"
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


def get_A_spectrum(As):
    es = As[0].svd([0,1,5], [2,3,4])[1]
    es = es.to_ndarray()
    es /= np.max(es)
    es = -np.sort(-es)
    return es


def get_free_energy(As, log_facts, pars, iter_count):
    Z = ncon((As[0], As[4],
              As[1], As[5]),
             ([1,100,3,100,4,5], [3,102,1,102,10,11],
              [6,101,8,101,5,4], [8,103,6,103,11,10]),
             order=([100,101,102,103,10,11,6,8,1,3,4,5]))
    log_fact = log_facts[0] + log_facts[1] + log_facts[4] + log_facts[5]
    Z = Z.value()
    logZ = np.abs(np.log(Z)+log_fact)
    F = -logZ/pars["beta"]
    f = F/(8**(iter_count+1))
    return f


if __name__ == "__main__":
    pars = parse()
    apply_default_pars(pars, parinfo)
    datadispenser.update_default_pars("As", pars, iter_count=max(pars["iters"]))

    if pars["debug"]:
        warnings.filterwarnings('error')

    datetime_str = datetime.datetime.strftime(datetime.datetime.now(),
                                             '%Y-%m-%d_%H-%M-%S')
    title_str = '{}_{}'.format(filename, datetime_str)
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
        res = datadispenser.get_data(
            dbname, "As", pars, iter_count=it
        )
        As, log_facts = res[0], res[1]
        if log_facts == 0:
            log_facts = [0]*8

        if pars["print_spectra"]:
            es = get_A_spectrum(As)
            msg = "Spectrum of A1:\n{}".format(es[:30])
            logging.info(msg)

        if pars["print_free_energy"]:
            f = get_free_energy(As, log_facts, pars, it)
            logging.info('Free energy per site: {}'.format(f))

