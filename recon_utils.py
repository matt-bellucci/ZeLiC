import sys

import numpy as np
from fastdtw import fastdtw
from scipy import spatial, stats
import scipy.fftpack as spfft
from scipy.signal import savgol_filter

""" Utilities for signal reconstruction
This module contains functions that can downsample a signal, normalize signals
or datasets, and measure the distance between 2 signals.

- test


"""

def fixed_downsample(signal,samples,ret_ind=True,last_ind=True):
    N = len(signal)
    gap = N//samples
    r = N%samples
    ind = []
    for i in range(samples):
        indx = i*gap
        ind.append(indx)
    if last_ind:
        ind[-1] = N-1
    ind.sort()

    ds = signal[ind]
    if ret_ind:
        return (ind,ds)
    else:
        return ds
    
def fixed_downsample2(signal,samples,ret_ind=True,last_ind=True):
    N = len(signal)
    gap = N/float(samples-1)
    #r = N%samples
    ind = []
    for i in range(samples):
        indx = int(round(i*gap))
        ind.append(indx)
    if last_ind:
        ind[-1] = N-1
    ind.sort()

    ds = signal[ind]
    if ret_ind:
        return (ind,ds)
    else:
        return ds



def fixed_downsampl_dep(
        signal, nb_pts, random=True, ret_ind=True,first_ind=True,last_ind=True):

    """ Applies a fixed length downsampling

    This function takes a signal as input, and will return a new signal with
    the number of points required. It can do this in 2 ways:

    - Uniform: the function will take equidistant points of the input signal

    - Random: the function will take random points of the input signal

    Parameters
    ----------
    signal : list
        the signal to downsample
    nb_pts : integer
        the number of points the downsampled signal will have
    random : boolean
        if True, returns random downsampling,
        if False, returns uniform downsampling
    ret_ind : boolean
        if True, the function returns the list of indices
    first_ind : boolean
        if True, the downsampled signal will contain the first point of
        the original signal
    last_ind : boolean
        if True, the downsampled signal will contain the last point of
        the original signal

    Returns
    -------
    ind : list
        indices that correspond to the position of the downsampled points
    y : numpy.ndarray
        the downsampled signal

    """

    n = len(signal)
    ind = []

    if random:
    # uses a numpy function to randomly select indices of the original signal
        ind = np.random.choice(n, nb_pts, replace=False)
        ind.sort()
        ind = ind.tolist()
    else:
        # tries to find a uniform distance between the points
        # the euclidean division of n by nb_pts is the distance between the
        # points
        ratio = n // nb_pts
        # if remainder is not 0, we can not take equidistant pointw
        r = n % nb_pts
        sig = signal.copy()
        # this loop takes enough point on the extremities
        # to reduce the length of the signal and have a uniform distance
        for i in range(0, r):
            # if i is an even number
            if i%2 == 0:
                # take a point at the beginning of signal
                ind.append(i//2)
            # if i is odd
            else:
                # take a point at the end of signal
                ind.append(len(sig)-1 - (i//2))
        # once we remove enough points to have equidistant points, retain these
        # equidistant points
        for i in range((r+1)//2 + (r+1)%2 -1, nb_pts - (r+1)//2):
            # take points with ratio as a distance
            j = ratio*i
            ind.append(j)
        # sort the list to avoid any errors
        ind.sort()
    if first_ind:
        ind[0]=0
    if last_ind:
        ind[len(ind)-1] = len(signal)-1
    # take the values of the input signal at the indices chosen above
    y = np.array(signal[ind])
    if ret_ind:
        return (ind, y)
    else:
        return y

def var_downsample(signal, threshold=0.01, ret_ind=True, last_ind=True):
    # This function applies a variable length downsampling
    # Input :
    #       signal (list) : the signal to downsample
    #       threshold (float) : is the minimal variation needed to take a point
    #       ret_ind (boolean) : if True,
    #                          the function returns the list of indices
    #       last_ind (boolean) : if True, the downsampled signal will contain
    #                            the last point of the original signal
    # Output :
    #       if ret_ind=True : - ind (list): indices that correspond to the
    #                                      position of the downsampled points
    #                        - y (numpy array): the downsampled signal
    #       if ret_ind = False : -y (numpy array): the downsampled signal
    """ Applies a variable length downsampling

    This function takes a signal as input, and will return a new signal with
    a variable number of points. This number of points depend on the threshold
    chosen. It works with a "percentage rule". That means that we retain a
    point only if the variation between this point and the last point is took
    is greater than the threshold.


    For example, we have taken the first point, with a value of 100.
    The values of the points between 1 and 5 are contained in the interval
    [99,101] so we don't take them. 6th point has a value of 102, so that makes
    a variation greater than 1% (the threshold we chose for the example), so
    we retain the point. We then compare the next points with the 6th point.

    Parameters
    ----------
    signal : list
        the signal to downsample
    threshold : float
        the minimal variation needed to take a point. The condition is

        ``if np.abs(next_point-last_point) > threshold * np.abs(last_point)``

        ``retain next_point``

    ret_ind : boolean
        if True, the function returns the list of indices
    last_ind : boolean
        if True, the downsampled signal will contain the last point of
        the original signal

    Returns
    -------
    ind : list
        indices that correspond to the position of the downsampled points
    y : numpy.ndarray
        the downsampled signal

    """

    # we start with the first point
    ind = [0]
    # curr_val is the value of the last point we took
    curr_val = signal[0]
    # for all the remaining points in the signal
    for i in range(1, len(signal)-1):
        # check if the variation is greater than threshold
        #if np.abs((signal[i]-curr_val)) > threshold*np.abs(curr_val):
        if np.abs((signal[i]-curr_val)) > threshold:
            # if so, retain this point and take its value as curr_val
            ind.append(i)
            curr_val = signal[i]
    if last_ind:
        ind.append(len(signal)-1)
    # the output signal is the values of the input signal, at the indices we
    # chose above
        pass
    y = signal[ind]
    if ret_ind:
        return (ind, np.array(y))
    else:
        return np.array(y)

def adaptive_downsample(signal, threshold, distance, last_ind=True):

    """ Applies a combination of fixed and by percentage downsampling

    This downsampling method combines the by percentage rule, with a distance
    rule. We retain a point if one of these 2 conditions is met:

    - The variation is greater than the threshold
    - The distance between the last point taken and the point we compare is
        greater than the maximum distance chosen.

    This downsampling has the advantage of not missing important information
    such as sudden peaks, thanks to the percentage rule, and it avoids
    missing information if the signal is constant or varies very slowly, thanks
    to the distance rule downsampling.

    Parameters
    ----------
    signal : list
        the signal to downsample
    threshold : float
        the minimal variation needed to take a point. The condition is

        ``if np.abs(next_point-last_point) > threshold * np.abs(last_point)``
        ``retain next_point``
    distance : integer
        maximum distance between 2 points
    last_ind : boolean
        if True, the downsampled signal will contain the last point of
        the original signal

    Returns
    -------
    ind : list
        indices that correspond to the position of the downsampled points
    y : numpy.ndarray
        the downsampled signal
    labels : list
        list containing information about the type of downsampling for each
        point:

        - 0 if the point was chosen for its distance with the last point
        - 1 if the point was chosen because of its variation

    """
    labels = [0]
    ind = [0]
    last_ind = 0
    for i in range(1, len(signal)):

        ###################
        # PERCENTAGE RULE #
        ###################

        # if last point has a value of 0,the next point will be more than 1%
        # of 0 so we make sure to take it and avoid a division by 0
        if signal[ind[last_ind]] == 0:
            # to be sure to take it, we make the variation greater than
            # threshold
            variation = threshold + 1
        else:
            variation = np.abs(
            (signal[i]-signal[ind[last_ind]])/signal[ind[last_ind]])
        if variation >= threshold:
            ind.append(i)
            labels.append(1)
            last_ind += 1

        else:

            #################
            # DISTANCE RULE #
            #################

            if i - ind[last_ind] >= distance:
                ind.append(i)
                labels.append(0)
                last_ind += 1
            else:
                continue
    if last_ind:
        # we choose a label of 0 if the last point was not already retained
        if ind[last_ind] != len(signal)-1:
            ind.append(len(signal)-1)
            last_ind += 1
            labels.append(0)
    y = signal[ind]
    return (ind, y, labels)

def distance(signal1, signal2, metric='Euclidean'):

    """ Measures the distance between 2 signals

    This function can measure the distance between 2 signals
    using different metrics :

    - 'Dtw' : Dynamic Time Warping distance
    - 'Euclidean' : the Euclidean distance or norm 2
    - 'Braycurtis' : the Bray-Curtis distance
    - 'Manhattan' : the Manhattan distance or norm 1
    - 'Spearmanr' : the Spearman rank-order correlation
    - 'Cosine' : the cosine similarity measure
    - 'Minkowski' : the Minkowski distance
    - 'Correlation' : the correlation distance
    - 'Jaccard' : the Jaccard similarity coefficient
    - 'Canberra' : the Canberra distance
    - 'Chebyshev' : the Chebyshev distance

    Parameters
    ----------
    signal1 : list
        input signal
    signal2 : list
        input signal
    metric : string
        the metric to use, see above for a list of all metrics available

    Returns
    -------
    dist : float
        the distance between the 2 signals

    """
    if (np.array_equal(signal1-signal2, np.zeros(len(signal1)))):
        return 0
    else:
        if metric == 'Dtw':
            dist,path = fastdtw(signal1, signal2)
        elif metric == 'Euclidean':
            dist = spatial.distance.euclidean(signal1, signal2)
        elif metric == 'Braycurtis':
            dist = spatial.distance.braycurtis(signal1, signal2)
        elif metric == 'Manhattan':
            dist = spatial.distance.cityblock(signal1, signal2)
        elif metric == 'Spearmanr':
            prsn = stats.spearmanr(signal1, signal2)
            dist = prsn[0]
        elif metric == 'Cosine':
            dist = spatial.distance.cosine(signal1, signal2)
        elif metric == 'Minkowski':
            dist = spatial.distance.minkowski(signal1, signal2, p=2)
        elif metric == 'Correlation':
            dist = spatial.distance.correlation(signal1, signal2)
        elif metric == 'Jaccard':
            dist = spatial.distance.jaccard(signal1, signal2)
        #elif metric == 'Median':
        #    dist = np.median(signal1, signal2)
        #elif metric == 'Levenshtein':
        #    dist = levenshteinDistance(signal1, signal2)
        elif metric == 'Canberra':
            dist = spatial.distance.canberra(signal1, signal2)
        elif metric == 'Chebyshev':
            dist = spatial.distance.chebyshev(signal1, signal2)
        else:
            print('{0} is not implemented, returning euclidean distance'
                  .format(metric))
            dist = distance(signal1, signal2)
        return dist

def rel_accuracy(or_sig, recon_sig, random, metric='Euclidean'):
    """ Gives a percentage of error of a reconstruction

    This function gives an idea of how good a reconstruction is by comparing
    the distance between the original signal and the reconstructed signal to
    the distance between the original signal and a randomly generated one.
    Then it assumes that the distance between random and original signal is
    100% error, and with a rule of three, it deducts the percentage of error of
    the reconstruction.

    Sometimes, the percentage will be more than 100%, this means that the
    reconstruction was worse than a randomly generated signal.

    The randomly generated signal has to be given as input so it can be used
    to compare a full dataset and give consistent results.

    Parameters
    ----------
    or_sig : list
        the original signal
    recon_sig : list
        the reconstructed signal
    random : list
        the randomly generated signal.
    metric : string
        the metric to use to measure the distance. See ``distance``
        documentation for a list of all the metrics available

    Returns
    -------
    rel_acc : float
        the percentage of error of the reconstruction.

    """
    # distance between original and random, supposed to be high
    high = accuracy(or_sig,random,metric=metric)
    # distance between original and reconstructed, supposed to be low
    low = accuracy(or_sig,recon_sig,metric=metric)
    # high is 100% or 1 error, so the rule of three gives rel_acc = 1*low/high
    rel_acc = low/high
    return rel_acc

def normalize(signal):
    """ Normalization of a signal

    This function normalizes a signal X, with the following formula:

    `norm = X-mean(X)/std(x)`
    where std is the standard deviation.

    Parameters
    ----------
    signal : list
        the signal to normalize

    Returns
    -------
    y : numpy.ndarray
        the normalized signal

    """
    if np.std(signal) != 0:
        std = np.std(signal)
    else :
        std = 1
    y = np.array((signal-np.mean(signal))/std, dtype=np.float64)
    return y

def normalize_data(data, threshold=0, kind='whole'):

    """ Normalization of a dataset

    This function normalizes every signal of a dataset so they have values
    between -1 and 1.
    There are 3 ways of normalizing the dataset, see the parameter 'kind' for
    more information.

    Parameters
    ----------
    data : 2D array or equivalent
        the dataset to normalize
    threshold : float
        a value that will make the maximum value a bit larger to include new
        signals that may have a larger maximum value than the signals seen
        in the dataset
    kind : string
        - 'row':

        the row normalization normalizes every row by dividing all the
        values of the signal by the maximum, in absolute value, of the signal

        - 'whole_pos' :

        whole_pos stands for whole dataset with positive values.
        It makes sure the values are between 0 and 1. The normalization adds
        the smallest value to the whole dataset, if it is negative. Then it
        divides the dataset by the largest absolute value.

        - 'whole':

        normalizes the whole dataset, by dividing it by the largest
        absolute value of the dataset.

    Returns
    -------
    new_data : 2D array or equivalent
        the normalized dataset

    """
    data_cop = data.copy()
    data_cop = np.array(data_cop)
    new_data = []
    if kind=='row_pos':
        for signal in data_cop:
            mini = np.min(signal)
            if mini < 0:
                signal = signal + np.abs(mini)
            if mini > 0:
                signal = signal - np.abs(mini)
            maxi = np.max(signal)
            if maxi == 0:
                new_sig = signal
            else:
                new_sig = np.array(signal) / maxi
                new_data.append(new_sig)
        return new_data
    elif kind=='row':
        for signal in data_cop:
            maxi = np.max(np.abs(signal))
            maxi = maxi + threshold*maxi
            if maxi == 0:
                new_sig = signal
            else:
                new_sig = np.array(signal) / maxi
            new_data.append(new_sig)
        return new_data
    elif kind == 'whole_pos':
        mini = np.min(data_cop)
        if mini < 0:
            data_cop = data_cop + np.abs(mini)
        maxi = np.max(data_cop)
        maxi = maxi + threshold*maxi
        new_data = data_cop / maxi
        return new_data
    else:
        maxi = np.max(data_cop)
        maxi = maxi + threshold*maxi
        new_data = data_cop / maxi
        return new_data

def smooth(signal, intensity=11, deg=3):
    """ Smoothes a signal, with a Savgol filter

    Parameters
    ----------
    signal : list
        the signal to smooth
    intensity : integer
        see scipy.signal.savgol_filter for information
    deg : integer
        see scipy.signal.savgol_filter for information

    Returns
    -------
        smoothed_signal : list
            the smoothed signal
    """
    smoothed_signal = savgol_filter(signal, 11, 3)
    return smoothed_signal

def error_correction(y_original, ind, labels, threshold):

    """ Error correction algorithm for a reconstructed signal

    This function works only with signals that have been downsampled using
    the ``adaptive_downsample`` function and then reconstructed using any
    method.

    It uses the information given by the adaptive downsampling to make sure
    the reconstructed points are not larger than the threshold of the
    percentage rule. If it is the case, the point will be given the
    smallest (or largest) value acceptable.

    Parameters
    ----------
    y_original : list
        the reconstructed signal, after an adaptive downsampling
    ind : list
        indices that correspond to the position of the downsampled points
    labels : list
        the labels given by the ``adaptive_downsample`` function
    threshold : float
        the threshold value used for the downsampling

    Returns
    -------
    y : list
        the error corrected reconstructed signal

    """

# WARNING, this function has not been tested and may not improve the
# accuracy of the reconstruction, use with caution

    y = y_original.copy()
    ind = np.array(ind)
    # gives the position of the points that have a label == 1
    ind_var = [i for i in range(len(ind)) if labels[i]==1]
    for i in ind_var:
        # last known point before the point with label==1
        ref_point = y[ind[i-1]]
        # infimum of interval where the original point should be
        inf = ref_point - np.abs(ref_point)*threshold
        # supremum of interval where the original point should be
        sup = ref_point + np.abs(ref_point)*threshold
        # position of reconstructed points to correct,
        # between the 2 known points
        x = np.arange(ind[i-1]+1, ind[i],1)
        if x.size == 0:
            continue
        else:
            for pt in x:
                if y[pt]< inf:
                    y[pt] = ref_point*(1-threshold)
                elif y[pt]> sup:
                    y[pt]= ref_point*(1+threshold)
                else :
                    continue
    return y
