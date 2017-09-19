import numpy as np
import sys
import os
import warnings
import logging
import logging.config
import configparser
import datetime
import GiltTNR2D
from tntools import datadispenser, multilineformatter
from tntools.yaml_config_parser import parse_argv

tools_path = os.path.dirname(multilineformatter.__file__)
logging.config.fileConfig(tools_path + "/logging_default.conf")
filename = os.path.basename(__file__).replace(".py", "")
np.set_printoptions(precision=10)

parinfo = {
    "iters": {
        "default": range(1, 6),
    },
    "debug": {
        "default": False
    },
    "database": {
        "default": "data/GiltTNR2D/"
    },
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


if __name__ == "__main__":
    pars = parse()
    apply_default_pars(pars, parinfo)
    datadispenser.update_default_pars("A", pars, iter_count=max(pars["iters"]))

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
        data = datadispenser.get_data(dbname, "A", pars, iter_count=it)
        A = data[0]
        U, S = GiltTNR2D.get_envspec(A, A, pars, where="N")
        S /= S.sum()
        GiltTNR2D.print_envspec(S)

