### This python code calculates isotope info from raw mass spectrometry data
# v3: executable in the command line
# Xin Guan (github.com/x-guan)

# basics
import numpy as np
import pandas as pd
import datetime
import argparse

# global variables
C13_NEUTRON = 13.0033548378 - 12
H2_NEUTRON = 2.01410177811 - 1.00782503224

# CalMz function
def CalMz(structure):
    # parse ion structures
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import rdmolops
    # input: acutumine.sdf
    # return: monoiso_mz

    # input structure from a sdf file
    # structure = "acutumine.sdf"   # in "main" function
    mol = Chem.SDMolSupplier(structure)[0]

    # determine properties of mol
    monoiso_mz = rdMolDescriptors.CalcExactMolWt(mol)
    charge = rdmolops.GetFormalCharge(mol)
    out_mz = {"mol": mol, "monoiso_mz": monoiso_mz, "charge": charge}

    return out_mz

# CalMzWindow function
def CalMzWindow(monoiso_mz, neutron_num, ppm, charge=1):
    # input
    # neutron_num = 5   # in the main function
    # ppm = 100   # in the main function
    # return: mz_windows

    # initialize output matrix
    mz_window = np.zeros(shape=(neutron_num + 1, 2), dtype='float')
    row_idxs = range(neutron_num + 1)

    # calc ppm factor
    ppm_low = 1 - ppm / 1e6
    ppm_high = 1 + ppm / 1e6

    # loop through rows
    for i in row_idxs:
        mz_tmp = np.array((monoiso_mz + i * C13_NEUTRON, monoiso_mz + i * H2_NEUTRON))
        mz_window[i, :] = np.min(mz_tmp * ppm_low), np.max(mz_tmp * ppm_high)

    return mz_window

# CalMid function
def CalMid(mz_window, ppm, rt_window, mzml_file):
    # parse mzml files
    from pyteomics import mzml

    # input
    # rt_window = (5, 5.5)  # from 5 to 5.5 min   # in the main function
    # mzml_file = "mca_1_l_d2o.mzML"   # in the main function
    # return: mean_mz, total_i, mid

    # import mzml file
    mz_dt = mzml.read(mzml_file)

    # identify rt window
    rt_min, rt_max = tuple(rt_window)
    # print("m/z window is from", rt_min, "to", rt_max, "min")

    # initialize output matrix of mean mz and intensities
    n_rows = mz_window.shape[0]
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
            index_mat = np.searchsorted(these_mzs, mz_window)
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

    return ({'mean_mz': mean_mz, 'total_i': total_i, 'mid': mid})

# main function
def main():
    # unpack arguments
    args = ParAug()

    # determine monoisotopic m/z and charge from structure
    structure = args.structure
    ion_info = CalMz(structure)
    monoiso_mz = ion_info["monoiso_mz"]
    charge = ion_info["charge"]

    # determine mz_windows from monoiso_mz
    neutron_num = args.max_neutrons
    ppm = args.ppm

    mz_window = CalMzWindow(monoiso_mz, neutron_num, ppm, charge=charge)

    # calculate MID raw data
    rt_window = (args.rt_start, args.rt_stop)
    mzml_file = args.mzml_file
    mid = CalMid(mz_window, ppm, rt_window, mzml_file)

    # prepare output
    today = datetime.datetime.today()
    m = np.arange(neutron_num + 1)
    C13_theo_mz = monoiso_mz + m * C13_NEUTRON
    H2_theo_mz = monoiso_mz + m * H2_NEUTRON

    out_df = pd.DataFrame({'m': m,
                           'mzml_file': mzml_file,
                           'structure': structure,
                           'analysis_date': today,
                           'C13_theo_mz': C13_theo_mz,
                           'H2_theo_mz': H2_theo_mz,
                           'mean_mz': mid['mean_mz'],
                           'raw_intensity': mid['total_i'],
                           'mid': mid['mid']
                           })
    # print("out_df: \n", out_df)
    out_df.to_csv("acutumine.csv")

def ParAug():
    try:
        parser = argparse.ArgumentParser(description='Gets an MID out of an mzML file given an ion structure.')
        parser.add_argument(
            '-f',
            "--mzml_file",
            action='store',
            help="mzML file from which to extract MID",
            required=True,
            type=str
        )
        parser.add_argument(
            '-s',
            "--structure",
            action='store',
            help=".sdf file of a single ion structure",
            required=True,
            type=str
        )

        parser.add_argument(
            '-o',
            "--output_file",
            action='store',
            help="output csv file",
            required=False,
            default='out.csv',
            type=str
        )

        parser.add_argument(
            '-r',
            "--rt_start",
            action='store',
            help="retention time in minutes at which to start MID extraction",
            required=False,
            default=0,
            type=float
        )

        parser.add_argument(
            '-t',
            "--rt_stop",
            action='store',
            help="retention time in minutes at which to end MID extraction",
            required=False,
            default=6,
            type=float
        )

        parser.add_argument(
            '-p',
            "--ppm",
            action='store',
            help="mass tolerance in parts per million (ppm) used for MID extraction",
            required=False,
            default=35,
            type=float
        )

        parser.add_argument(
            '-n',
            "--max_neutrons",
            action='store',
            help="maximum heavy neutrons to consider when extracting MIDs",
            required=False,
            default=5,
            type=int
        )

        parsed = parser.parse_args()
        return (parsed)
    except ValueError:
        print('Unable to parse arguments.')

# execute
if __name__ == '__main__':
    main()
