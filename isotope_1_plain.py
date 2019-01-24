### This python code calculates isotope info from raw mass spectrometry data
# v1: plain
# Xin Guan (github.com/x-guan)

# basics
import numpy as np
import pandas as pd
import datetime

###############################################################################
### determine monoisotopic m/z and charge from structure
# function - CalMz

# parse ion structures
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdmolops
# input: acutumine.sdf
# return: monoiso_mz

# input structure from a sdf file
structure = "acutumine.sdf"
mol = Chem.SDMolSupplier(structure)[0]

# determine properties of mol
monoiso_mz = rdMolDescriptors.CalcExactMolWt(mol)
charge = rdmolops.GetFormalCharge(mol)


###############################################################################
### determine mz_windows
# function - CalMzWindow

# input
C13_NEUTRON = 13.0033548378 - 12
H2_NEUTRON = 2.01410177811 - 1.00782503224
neutron_num = 5
ppm=100
# return: mz_windows

# initialize output matrix
mz_window = np.zeros(shape=(neutron_num+1, 2), dtype='float')
row_idxs = range(neutron_num+1)

# calc ppm factor
ppm_low = 1-ppm/1e6
ppm_high = 1+ppm/1e6

# loop through rows
for i in row_idxs:
    mz_tmp = np.array((monoiso_mz + i*C13_NEUTRON, monoiso_mz + i*H2_NEUTRON))
    mz_window[i, :] = np.min(mz_tmp * ppm_low), np.max(mz_tmp * ppm_high)

###############################################################################
### calculate mean_mz, total_i, and mid
# function - CalMid

# parse mzml files
from pyteomics import mzml

# input
rt_window = (5, 5.5)  # from 5 to 5.5 min
mzml_file = "mca_1_l_d2o.mzML"
# return: mean_mz, total_i, mid

# import mzml file
mz_dt = mzml.read(mzml_file)

# identify rt window
rt_min, rt_max = tuple(rt_window)
# print("m/z window is from", rt_min, "to", rt_max, "min")

# initialize output matrix of mean mz and intensities
n_rows = mz_windows.shape[0]
m_span = range(n_rows)

# initialize lists to store relevant raw data points
mzs = [[] for m in m_span]
intensities = [[] for m in m_span]

# loop through scans, keeping data points in mz_windows and rt_window only
for spec in mz_dt:
    rt = spec['scanList']['scan'][0]['scan start time']
    # print(rt)
    if rt >= rt_min and rt <= rt_max:
        # get raw scan data
        these_mzs = spec['m/z array']
        # print(these_mzs)
        these_intensities = spec['intensity array']
        # print(these_intensities)

        # index into mz_windows to find relevant data in scan
        index_mat = np.searchsorted(these_mzs, mz_windows)
        # print(index_mat)
        start = index_mat[:, 0]
        # print(start)
        stop = index_mat[:, 1]
        # print(end)

        for m in m_span:
            # if scan has no mz values of interest, skip it
            if start[m] != stop[m]:
                mzs[m].extend(list(these_mzs[start[m]:stop[m]]))
                # print(mzs)
                intensities[m].extend(list(these_intensities[start[m]:stop[m]]))
                # print(intensities)
mean_mz = np.asarray([np.average(mzs[m], weights=intensities[m]) for m in m_span])
# print("mean_mz: \n", mean_mz)
total_i = np.asarray([np.sum(intensities[m]) for m in m_span])
# print("total_i: \n", total_i)
mid = total_i / total_i.sum()  # .sum()???
# print("mid: \n", mid)

###############################################################################
### preprare output
today = datetime.datetime.today()
m = np.arange(max_neutrons + 1)
C13_theo_mz = monoiso_mz + m*C13_NEUTRON
H2_theo_mz = monoiso_mz + m*H2_NEUTRON

out_df = pd.DataFrame({'m': m,
                       'mzml_file': mzml_file,
                       'structure': structure,
                       'analysis_date': today,
                       'C13_theo_mz': C13_theo_mz,
                       'H2_theo_mz': H2_theo_mz,
                       'mean_mz': mean_mz,
                       'raw_intensity': total_i,
                       'mid': mid
                      })
print("out_df: \n", out_df)
out_df.to_csv("acutumine.csv")
