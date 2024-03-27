import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import glob as glob

from larch import Group
from larch.fitting import param, param_group
from larch.math.utils import index_of
from larch.xafs import feffrunner, pre_edge, autobk, xftf, xftr, feffpath
from larch.xafs.feffit import feffit_transform, feffit_dataset, feffit, feffit_report


#EXAFS_dir = "./Processed_EXAFS"
#FEFF_dir = "./FEFF"

# Utilities

def get_temp(key):
    temp = key.split("_")[-1]
    
    if temp =="RT":
        return 25
    
    return int(temp)


def fit_param(data: list[Group] = None, **p):
    keys = p.keys()
    fit_param_get = {}
    for k in keys:
        if p[k]["global"]:
            fit_param_get[k] = param(p[k]["initial"], vary=p[k]["vary"])
        else:
            for i,group in enumerate(data):
                fit_param_get[f"{k}_{group.temp}"] = param(p[k]["initial"], vary=p[k]["vary"])
    pars = param_group(**fit_param_get)
    return pars


def path_param(data, feff_folder, fitpath_num=None, rules=None):
    # for now, only consider the first path
    if rules is None:
        keys = dict()

    keys = rules.keys()
    print(feff_folder)
    feff_pathes = feff_path(feff_folder)

    paths = []
    if rules["SO2"]["vary"] is True and rules["N"]["vary"] is True:
        raise ValueError("you fixed N and SO2. It is not physical!")
    if rules["SO2"]["vary"] is False and rules["N"]["vary"] is False:
        raise ValueError(
            "N and SO2 cannot be changed at the same time. It is not physical!"
        )
    for i, group in enumerate(data):
        if rules["SO2"]["global"] is True and rules["N"]["global"] is True:
            s02 = "N*SO2"
        if rules["SO2"]["global"] is False and rules["N"]["global"] is False:
            s02 = f"N_{group.temp}*SO2_{group.temp}"
        if rules["SO2"]["global"] is True and rules["N"]["global"] is False:
            s02 = f"N_{group.temp}*SO2"
        if rules["SO2"]["global"] is False and rules["N"]["global"] is True:
            s02 = f"N*SO2_{group.temp}"
        if rules["dele"]["global"] is True:
            dele = "dele"
        if rules["dele"]["global"] is False:
            dele = f"dele_{group.temp}"
        if rules["ss2"]["global"] is True:
            ss2 = "ss2"
        if rules["ss2"]["global"] is False:
            ss2 = f"ss2_{group.temp}"
        if rules["delr"]["global"] is True:
            delr = "delr"
        if rules["delr"]["global"] is False:
            delr = f"delr_{group.temp}"
        if rules["th"]["global"] is True:
            th = "th"
        if rules["th"]["global"] is False:
            th = f"th_{group.temp}"

        if fitpath_num is None:
            fitpath_num = 0

        paths.append(
            feffpath(
                f"{feff_folder}/{feff_pathes['file'][fitpath_num]}",
                s02=s02,
                degen=1,
                e0=dele,
                sigma2=ss2,
                deltar=delr,
                third=th,
            )
        )
    return paths


def feff_path(feff_folder):
    print(feff_folder + "/files.dat")
    with open(feff_folder + "/files.dat", "r") as f:
        for line in f:
            if "file        sig2   amp ratio    deg    nlegs  r effective" in line:
                break
                
        return pd.read_csv(f,
                           sep=" +",
                           names=["file", "sig2", "amp ratio", "deg", "nlegs", "r effective"],        
                          )


def feff_rules(parm_dict):
    rules = dict()
    for key in parm_dict.keys():
        rules[key] = {
            "vary": parm_dict[key]["vary"],
            "global": parm_dict[key]["global"],
        }
    return rules


def run_fit(
    data,
    parm_dict,
    feff_folder,
    fitpath_num,
    fit_range_param,
    write=False,
    save_name="fit.out",
):
    rules = feff_rules(parm_dict)
    pars = fit_param(data, **parm_dict)
    paths = path_param(data, feff_folder, fitpath_num=fitpath_num, rules=rules)
    trans = feffit_transform(**fit_range_param)
    dset = []
    for i in range(len(data)):
        dset.append(feffit_dataset(data=data[i], pathlist=[paths[i]], transform=trans))
    out = feffit(pars, dset, fix_unused_variables=False)
    report = feffit_report(out)
    if write is True:
        write_report(save_name, report)
    return dset, out, report, paths


def write_report(filename, out):
    "write report to file"

    with open(filename, "w") as f:
        f.write(out)


def plot_fitting_results(data, dset, d_range=[0, -1]):
    for i in range(len(dset[d_range[0] : d_range[1]])):
        mod = dset[i].model
        dat = dset[i].data
        data_chik = dat.chi * dat.k**2
        model_chik = mod.chi * mod.k**2
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(dat.k, data_chik, label=data[d_range[0] + i].temp)
        ax[0].plot(mod.k, model_chik)
        ax[0].plot(dat.k, dat.kwin)
        ax[0].legend(frameon=False)
        ax[1].plot(dat.r, dat.chir_mag)
        ax[1].plot(mod.r, mod.chir_mag)
        ax[1].plot(mod.r, mod.rwin)
        ax[0].set_xlim(0, 20)
        ax[0].legend(frameon=False)
        ax[1].set_xlim(0, 6)
        plt.show()


# Input


def read_data(fname: str, regex=None, skip=0, assignment_dict=None, pre_edge_kws=None, autobk_kws=None, xftf_kws=None, manual_metadata=None) -> Group:
    """
    Read and preprocess the data from a file.

    Parameters:
    - fname: Filename to read.
    - regex: Regular expression to extract info from the filename. Optional if manual_metadata is provided.
    - skip: Number of rows to skip at the start of the file.
    - assignment_dict: Dictionary mapping Group attributes to match positions. Used with regex.
    - pre_edge_kws: Dictionary of keyword arguments for the pre_edge function.
    - autobk_kws: Dictionary of keyword arguments for the autobk function.
    - xftf_kws: Dictionary of keyword arguments for the xftf function.
    - manual_metadata: Optional. A dictionary of metadata to use instead of extracting from the filename.
    """
    data = np.loadtxt(fname, unpack=True, skiprows=skip)
    group = Group()
    group.energy = data[0]
    group.mu = data[1]

    if manual_metadata:
        for key, value in manual_metadata.items():
            setattr(group, key, value)
    else:
        if regex and assignment_dict:
            match = re.findall(regex, fname)
            if not match:
                print(f"No match found for filename: {fname}")
                return None
            match = match[0]

            # Dynamically assign attributes to the group based on assignment_dict
            for attribute, position in assignment_dict.items():
                value = match[position]
                if 'int' in attribute:
                    value = int(value)
                    attribute = attribute.replace('_int', '')
                setattr(group, attribute, value)

    # Process the group with Larch functions
    if pre_edge_kws:
        pre_edge(group, **pre_edge_kws)
    if autobk_kws:
        autobk(group, **autobk_kws)
    if xftf_kws:
        xftf(group, **xftf_kws)

    return group


def read_multiple_spectra(
    files,
    regex=None,
    skip=0,
    assignment_dict=None,
    pre_edge_kws=None,
    autobk_kws=None,
    xftf_kws=None
) -> list[Group]:
    """Read and preprocess multiple spectra from a file pattern."""

    files.sort()

    # Pass the additional parameters to read_data
    groups = [read_data(
        fname,
        regex=regex,
        skip=skip,
        assignment_dict=assignment_dict,
        pre_edge_kws=pre_edge_kws,
        autobk_kws=autobk_kws,
        xftf_kws=xftf_kws
    ) for fname in files]

    return groups
