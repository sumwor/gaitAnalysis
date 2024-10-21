# general utility function needed in data analysis
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def bootstrap(data, dim, dim0, n_sample=1000):
    """
    input:
    data: data matrix for bootstrap
    dim: the dimension for bootstrap, should be data.shape[1]
    dim0: the dimension untouched, shoud be data.shape[0]
    n_sample: number of samples for bootstrap. default: 1000
    output:
    bootRes={'bootAve','bootHigh','bootLow'}
    """
    # Resample the rows of the matrix with replacement
    if len(data)>0:  # if input data is not empty
        bootstrap_indices = np.random.choice(data.shape[dim], size=(n_sample, data.shape[dim]), replace=True)

        # Bootstrap the matrix along the chosen dimension
        bootstrapped_matrix = np.take(data, bootstrap_indices, axis=dim)

        meanBoot = np.nanmean(bootstrapped_matrix,2)
        bootAve = np.nanmean(bootstrapped_matrix, axis=(1, 2))
        bootHigh = np.nanpercentile(meanBoot, 97.5, axis=1)
        bootLow = np.nanpercentile(meanBoot, 2.5, axis=1)

    else:  # return nans
        bootAve = np.full(dim0, np.nan)
        bootLow = np.full(dim0, np.nan)
        bootHigh = np.full(dim0, np.nan)
        # bootstrapped_matrix = np.array([np.nan])

    # bootstrapped_2d = bootstrapped_matrix.reshape(80,-1)
    # need to find a way to output raw bootstrap results
    tempData = {'bootAve': bootAve, 'bootHigh': bootHigh, 'bootLow': bootLow}
    index = np.arange(len(bootAve))
    bootRes = pd.DataFrame(tempData, index)

    return bootRes

def count_consecutive(listx):
    # count largest number of consecutive 1s in a given list
    count1 = 0
    maxConsec1 = 0
    for ii in range(len(listx)):
        if listx[ii] == 1:
            count1 = count1+1
            if count1 > maxConsec1:
                maxConsec1 = count1
        else:
            count1 = 0

    return maxConsec1


def butter_lowpass_filter(data, cutoff_freq, fs, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def fill_nans_and_split(vector, max_nan_chunk=5):
    # Helper function to linearly interpolate NaNs
    def linear_interpolation(vec, start, end):
        vec = np.array(vec)
        nans = np.isnan(vec)
        vec[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), vec[~nans])
        return vec

    # Finding NaN chunks
    n = len(vector)
    is_nan = np.isnan(vector)

    # Split indices based on large NaN chunks
    indices = np.arange(n)
    nan_chunks = np.split(indices, np.where(np.diff(is_nan) != 0)[0] + 1)

    result_vectors = []
    index_ranges = []

    i = 0
    # go over the vector to interpolate data first
    while i < len(nan_chunks):
        chunk = nan_chunks[i]
        if is_nan[chunk[0]]:  # Current chunk is NaN
            if len(chunk) <= max_nan_chunk:  # NaN chunk shorter than 5
                start = nan_chunks[i - 1][0]  # Start of previous non-NaN chunk
                end = nan_chunks[i + 1][-1]  # End of next non-NaN chunk
                vector[start:end + 1] = linear_interpolation(vector[start:end + 1], start, end)
                #result_vectors.append(vector[start:end + 1])
                #index_ranges.append((start, end))
            i += 2  # Skip next non-NaN chunk after a long NaN chunk
        else:
            i += 1  # Move to the next chunk

    # separate the chunks on new vector
    is_nan = np.isnan(vector)

    # Split indices based on large NaN chunks
    nan_chunks = np.split(indices, np.where(np.diff(is_nan) != 0)[0] + 1)
    i=0
    while i < len(nan_chunks):
        chunk = nan_chunks[i]
        if not is_nan[chunk[0]]: # if current chunk is not nan
            start = nan_chunks[i ][0]
            end = nan_chunks[i][-1]
            result_vectors.append(vector[start:end + 1])
            index_ranges.append((start, end))
        i+=1

    # Check for the last segment after processing all NaN chunks
    # if not is_nan[-1]:
    #     start = nan_chunks[-1][0]
    #     end = nan_chunks[-1][-1]
    #     result_vectors.append(vector[start:end + 1])
    #     index_ranges.append((start, end))

    return result_vectors, index_ranges

def read_rotarod_csv(filepath, animalID):
    """ read the csv file from the rotarod machine, take the animal we are looking for
    output a dataframe
    input:
    filepath: the path of the RR_results.csv file
    aniamlID: Id of the animal of interest
    output"
    """

    data = pd.read_csv(filepath, sep=';')
    data_needed = data[data['ID'] == animalID]
    keys = ['Trial', 'Date', 'Treatment', 'ID', 'Speed [RPM]', 'Latency [s]', 'Distance [mm]']
    output_dict = {}
    maxTrial = max(map(int, data_needed['Trial']))
    for key in keys:
        output_dict[key] = [[] for i in range(maxTrial)]

    for t in range(maxTrial):
        dat = data_needed[data_needed['Trial']==str(t+1)]
        for key in keys:
            if key in ['Latency [s]', 'Distance [mm]']:
                # find the PLATE-DN event
                if 'Event Description' in dat:
                    ind = dat[dat['Event Description']==('PLATE-DN')].index
                    if len(ind)>0:

                        output_dict[key][t] = float(dat[key][ind[0]])
                    else:
                        # find STOPPED event
                        #print(t)
                        ind = dat[dat['Event Description'] == ('STOPPED')].index
                        if len(ind) > 0:
                            if key == 'Latency [s]':
                                output_dict[key][t] = float(max(dat[key][ind]))
                            else:
                            # in this case calculate the distance manually
                                d = 30 # the rod diameter is 30 mm

                                # calculate the revolution first
                                omega_i = float(dat['Ramp Initial Speed [RPM]'].iloc[0]) / 60.0
                                omega_f = float(dat['Ramp Final Speed [RPM]'].iloc[0])  / 60.0

                                # Calculate angular acceleration in RPS^2
                                alpha = (omega_f - omega_i) / 300

                                # Calculate angular displacement in revolutions
                                theta = omega_i * float(max(dat['Latency [s]'][ind]))  + 0.5 * alpha * (float(max(dat['Latency [s]'][ind]))  ** 2)

                                output_dict['Distance [mm]'][t] = np.round(np.pi*d * theta)
                        else:
                            output_dict[key][t] = np.nan
                else:
                    print(f"Column Event Description does not exist in the dictionary.")
            else:
                output_dict[key][t] = dat[key].iloc[0]


    return output_dict

if __name__ == "__main__":
    x = [1, 1, 1, np.nan, 0.4, 0.6, np.nan, np.nan, 0, 1, 0,1, 1,1,1,1,0,1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1,0.9, 1.1,1.2]
    #result, index = fill_nans_and_split(x)
    filepath = r'/media/linda/WD_red_4TB/DeepLabCut_data/rotarod/Nlgn_rotarod/RR_Results.csv'
    animalID = 'ASD409'
    read_rotarod_csv(filepath, animalID)