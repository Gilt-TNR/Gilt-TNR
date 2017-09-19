import numpy as np
import sys
import os
import warnings
import logging
import logging.config
import configparser
import datetime
import GiltTNR3D
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
        "default": "data/GiltTNR3D/"
    },
    "envtype": {
        "default": "cube"
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
        data = datadispenser.get_data(dbname, "As", pars, iter_count=it)
        As = data[0]
        if pars["envtype"] == "cube":
            env = GiltTNR3D.build_gilt_cube_env(As, pars)
        elif pars["envtype"] == "square":
            env = GiltTNR3D.build_gilt_square_env(As, pars, "horz_across")
        else:
            msg = "Unknown envtype {}.".format(pars["envtype"])
            raise ValueError(msg)
        S, U = env.eig([0,1], [2,3], hermitian=True)
        S = S.abs().sqrt()
        S /= S.sum()
        GiltTNR3D.print_envspec(S)

