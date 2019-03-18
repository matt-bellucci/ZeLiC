import numpy as np
import scipy.fftpack as spfft
from scipy.signal import resample
from scipy import interpolate
from sklearn.linear_model import Lasso,OrthogonalMatchingPursuit
#import spams
import cvxpy as cvx

import recon_utils as utils


""" Reconstruction algorithms

This module contains functions that can reconstruct a signal from a
downsampled one. It contains implementations of compressed sensing,
interpolation and signal processing methods.

Compressed sensing
------------------
1. Orthogonal Matching Pursuit, with omp_recon and omp_batch_recon

2. l1 minimization solvers, with lasso_recon and cvx_recon

Spline interpolation
--------------------
1. zero_recon for a spline of 0th degree
2. linear_recon for a spline of 1st degree
3. quadratic_recon for a spline of 2nd degree
4. cubic_recon for a spline of 3rd degree
5. combined_recon for a combination of 0th and 1st degree splines
6. combined_con_recon, same as combined_recon but preserving
local convexity or concavity

Signal processing
-----------------
shannon_recon provides a reconstruction based on the shannon interpolation
and Fourier transform

"""

# ------- COMPRESSED SENSING FUNCTIONS -------

def omp_recon(
        signal, ind, transform='dct', target_pts=120, L=20, eps=0,
        numThreads=-1, retCoefs=False):
    """ Performs an Orthogonal Matching Pursuit technique, with spams library

    This algorithm is based on Compressed sensing theory and works as a
    greedy algorithm to find the sparsest coefficients in a given transform that
    fit the input signal.
    Then it returns the inverse transform of these coefficients

     Parameters
     ----------
    signal : list
        the downsampled signal to reconstruct
    ind : list
        the list of indices corresponding to the position of
        the downsampled points
    target_pts : integer
        the number of points the reconstructed signal should have
    L : integer
        the number of nonzeros that are supposed to be in
        the original signal's transform.
    eps : float
        see spams' OMP documentation.
    numThreads : integer
        number of threads to use (if -1, automatically chosen).
    transform : 'dct' or 'dst'
        the type of transform to use (discrete cosine or sine transform)
    retCeofs : boolean
        if True, will return the coefficients of the transform

     Returns
     -------
     Y : list
        the reconstructed signal
    sol : list
        the coefficients of the reconstructed signal's transform

    """
    X = np.asfortranarray(np.array([signal]).transpose(),dtype=np.float64)
    # X contains all the signals to solve
    if transform == 'dst':
        # generating the transform matrix
        phi = spfft.idst(np.identity(target_pts), axis=0)
    else:
        phi = spfft.idct(np.identity(target_pts), axis=0)
    # generating the matrix phi for the problem y=phi.x
    phi = phi[ind]
    D = np.asfortranarray(phi)
    # normalizing D
    D = np.asfortranarray(
        D / np.tile(np.sqrt((D*D).sum(axis=0)), (D.shape[0],1)),
        dtype= np.float64)


    alpha = spams.omp(X, D, L=L, eps=eps,
                      return_reg_path=False, numThreads=numThreads)
    sol = np.array(alpha.toarray()).transpose() * 2
    sol = sol[0]
    indz = np.nonzero(sol)

    if transform == 'dst':
        Y = spfft.idst(sol)
    else:
        Y = spfft.idct(sol)
    Y = utils.normalize(Y)
    if retCoefs:
        return (Y, sol)
    else:
        return Y
def omp_batch_recon(
        signal, ind, target_pts, n_nonzero_coefs=20,
        transform='dct', retCoefs=False):
    """ Performs an Orthogonal Matching Pursuit technique, with batch approach

    This algorithm is based on Compressed sensing theory and works as a
    greedy algorithm to find the sparsest coefficients in a given transform that
    fit the input signal.
    Then it returns the inverse transform of these coefficients

     Parameters
     ----------
    signal : list
        the downsampled signal to reconstruct
    ind : list
        the list of indices corresponding to the position of
        the downsampled points
    target_pts : integer
        the number of points the reconstructed signal should have
    n_nonzero_coefs : integer
        the number of nonzeros that are supposed to be in
        the original signal's transform.
    transform : 'dct' or 'dst'
        the type of transform to use (discrete cosine or sine transform)
    retCeofs : boolean
        if True, will return the coefficients of the transform

     Returns
     -------
     x : list
        the reconstructed signal
    coef : list
        the coefficients of the reconstructed signal's transform

    """
    if transform == 'dst':
        phi = spfft.idst(np.identity(target_pts), axis=0)
    else:
        phi = spfft.idct(np.identity(target_pts), axis=0)

    phi = phi[ind]
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    omp.fit(phi,signal)
    coef = omp.coef_

    if transform == 'dst':
        x = spfft.idst(coef,axis=0) + np.mean(signal)
    else:
        x = spfft.idct(coef,axis=0) + np.mean(signal)
    x = utils.normalize(x)
    if retCoefs:
        return(x,coef)
    else:
        return x

def lasso_recon(signal, ind, target_pts=120, alpha=0.001, retCoefs=False):
    """ Solves the l1 minimization problem with a lasso algorithm

    It transforms the input signal using discrete cosine transform, then solves
    a l1 minimization problem to find the sparsest coefficients that fit the
    given signal. It returns the inverse transform of these coefficients.

     Parameters
     ----------
    signal : list
        the downsampled signal to reconstruct
    ind : list
        the list of indices corresponding to the position of
        the downsampled points
    target_pts : integer
        the number of points the reconstructed signal should have
    alpha : float
        the parameter alpha of scikit-learn's lasso method (see documentation)
    retCeofs : boolean
        if True, will return the coefficients of the transform

     Returns
     -------
     y : list
        the reconstructed signal
    coefs : list
        the coefficients of the reconstructed signal's transform

    """
    D = spfft.dct(np.eye(target_pts))
    A = D[ind]
    lasso = Lasso(alpha=alpha)
    lasso.fit(A,signal)
    coefs = lasso.coef_
    y = spfft.idct(coefs)
    y = utils.normalize(y)
    if retCoefs:
        return (y,coefs)
    else:
        return y

def cvx_recon(signal, ind, target_pts=120, retCoefs=False):
    """ Solves the l1 minimization problem with CVXPY

    It transforms the input signal using discrete cosine transform, then solves
    a l1 minimization problem (with CVXPY, a convex minimization solver)
    to find the sparsest coefficients that fit the given signal.
    It returns the inverse transform of these coefficients.

     Parameters
     ----------
    signal : list
        the downsampled signal to reconstruct
    ind : list
        the list of indices corresponding to the position of
        the downsampled points
    target_pts : integer
        the number of points the reconstructed signal should have
    retCeofs : boolean
        if True, will return the coefficients of the transform

     Returns
     -------
     y : list
        the reconstructed signal
    x : list
        the coefficients of the reconstructed signal's transform

    """
    A = spfft.idct(np.identity(target_pts), norm='ortho', axis=0)
    A = A[ind]
    vx = cvx.Variable(target_pts)
    objective = cvx.Minimize(cvx.norm(vx, 1))
    constraints = [A*vx == signal]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=False)
    x = np.array(vx.value)
    x = np.squeeze(x)
    y = spfft.idct(x, norm='ortho', axis=0)
    y = utils.normalize(y)
    if retCoefs:
        return (y, x)
    else:
        return y
# ------- INTERPOLATION FUNCTIONS -------

def spline_recon(signal, ind, target_pts=120, kind='linear'):

    """ Use splines to reconstruct the signal
    It uses scipy.interpolate.interp1d to reconstruct a signal using splines.

     Parameters
     ----------
    signal : list
        the downsampled signal to reconstruct
    ind : list
        the list of indices corresponding to the position of
        the downsampled points
    target_pts : integer
        the number of points the reconstructed signal should have
    kind = 'zero','linear','quadratic' or 'cubic'
        the degree of the spline to use, see scipy.interpolate.interp1d
        documentation.

     Returns
     -------
     y : list
        the reconstructed signal

    """
    x = np.linspace(ind[0], ind[-1], target_pts)
    f = interpolate.interp1d(ind, signal, kind=kind)
    y = f(x)
    return y

def combined_recon(ds, ind, threshold, tolerance_ratio=0.1):

    """ A combination of zero and linear spline interpolation

    This function combines linear and zero spline interpolation, given a
    signal that has been downsampled using a by percentage downsampling. It
    uses the information given by this downsampling to choose between linear or
    zero interpolation.
    If the linear interpolation gives at least one point that has a variation
    superior to the threshold, it will use zero interpolation instead.

     Parameters
     ----------
    ds : list
        the downsampled signal to reconstruct
    ind : list
        the list of indices corresponding to the position of
        the downsampled points
    threshold : float
        the threshold used for the by percentage downsampling
    tolerance_ratio : float
        this ratio increases the interval of points where linear interpolation
        is used. It prevents the algorithm to use zero if linear is closer to
        the actual signal. It calculates a larger threshold, as such :

        ``new_threshold = threshold*(1+tolerance_ratio)``

     Returns
     -------
     y : list
        the reconstructed signal

    """

    # Generate the linear interpolation
    x = np.arange(ind[-1]+1)
    f_linear = interpolate.interp1d(ind, ds, kind='linear')
    f_zero = interpolate.interp1d(ind, ds, kind='zero')
    y_linear = f_linear(x)
    y_zero = f_zero(x)
    y = y_linear.copy()
    # the new threshold, with some tolerance so we do not use
    # zero interpolation for a small difference with the threshold
    tolerance = threshold * (1+tolerance_ratio) #

    # Check if the points are in the correct interval
    last_ind = ind[0]
    for i in range(1, len(ind)):
        # j is the index of the points between 2 known points
        j = last_ind + 1
        out = False
        var = np.abs(y_linear[j]-ds[i-1])
        while (not out) and (j < ind[i]):
            var = np.abs(y_linear[j]-ds[i-1])
            if var > tolerance*np.abs(ds[i-1]):
                out=True
            j +=1
        # if one point is outside the interval, use zero interpolation
        # for the segment
        if out:
            for j in range(last_ind+1, ind[i]):
                y[j] = y_zero[j]
        last_ind = ind[i]
    return y
def combined_fixed_recon(ds, ind, threshold, tolerance_ratio=0.1,plot=False):

    """ A combination of zero and linear spline interpolation

    This function combines linear and zero spline interpolation, given a
    signal that has been downsampled using a by percentage downsampling. It
    uses the information given by this downsampling to choose between linear or
    zero interpolation.
    If the linear interpolation gives at least one point that has a variation
    superior to the threshold, it will use zero interpolation instead.

     Parameters
     ----------
    ds : list
        the downsampled signal to reconstruct
    ind : list
        the list of indices corresponding to the position of
        the downsampled points
    threshold : float
        the threshold used for the by percentage downsampling
    tolerance_ratio : float
        this ratio increases the interval of points where linear interpolation
        is used. It prevents the algorithm to use zero if linear is closer to
        the actual signal. It calculates a larger threshold, as such :

        ``new_threshold = threshold*(1+tolerance_ratio)``

     Returns
     -------
     y : list
        the reconstructed signal

    """

    x = np.arange(ind[-1]+1)
    tolerance = threshold * (1+tolerance_ratio)
    f_linear = interpolate.interp1d(ind, ds, kind='linear')
    f_pchip = interpolate.PchipInterpolator(ind,ds)
    y = [ds[0]]
    for i in range(1, len(ind)):
        indx = np.arange(ind[i-1]+1, ind[i]+1, 1)
        indx_conditions = (i-2 >= 0) and (i+1 < len(ind))
        out = False
        j = ind[i]-1
        while (not out) and (j > ind[i-1]):
            var = np.abs(f_linear(x[j]) - ds[i-1])
            if var > tolerance*np.abs(ds[i-1]):
                out = True
            j -= 1

        y_ = f_linear(indx)
        if out:
            #f_nearest = interpolate.interp1d([ind[i-1],ind[i]],[ds[i-1],ds[i]],kind='nearest')
            #y_ = f_nearest(indx).tolist()
            ind = np.array(ind,dtype=int)
            ds = np.array(ds,dtype=float)
            y_ = f_pchip(indx)
            #y_.append(ds[i])
        y = np.concatenate((y, y_))
    return y

def combined_con_recon(
        ds, ind, threshold, tolerance_ratio=0.1, min_distance=3,
        max_distance=-0.5, previous_distance=3):

    """ A shape preserving combination of zero and linear spline interpolation

    This function is similar to `combined_recon` but adds a shape-preserving
    aspect. If an interval of the input signal is considered convex or concave,
    the function will generate a point in the middle of this interval, with a
    value that is the average of 2 values :

    - the minimum (or maximum) possible, which means that it stays within a 1%
        variation for a convex (or concave) signal

    - the value that would have been given by the `combined_recon` function

    This ensures a reconstruction that preserves convexity or concavity.


     Parameters
     ----------
    ds : list
        the downsampled signal to reconstruct
    ind : list
        the list of indices corresponding to the position of
        the downsampled points
    threshold : float
        the threshold used for the by percentage downsampling
    tolerance_ratio : float
        this ratio increases the interval of points where linear interpolation
        is used. It prevents the algorithm to use zero if linear is closer to
        the actual signal. It calculates a larger threshold, as such :

        ``new_threshold = threshold*(1+tolerance_ratio)``
    min_distance : int
        the minimal distance between the 2 points where we want to interpolate
        when we can assume convexity or concavity
    max_distance : int
        the maximal distance between 2 points where we can assume convexity
        or concavityself.
        If set at a negative number, it will calculate :

        ``new_max_distance = abs(max_distance) * ind[-1]``
    previous_distance : int
        the minimal distance between the last 2 points where we can assume
        convexity or concavity. If the points are too close the signal is less
        likely to be convex or concave

     Returns
     -------
     y : list
        the reconstructed signal

    """
    if max_distance < 0:
         # if negative, it will take a proportion
         # of the length of the signal.
         # If signal is of length 120 and max_distance = 0.5, then
         # max_distance = 0.5*120 = 60
        max_distance = np.abs(max_distance) * ind[-1]

    x = np.arange(ind[-1]+1)
    tolerance = threshold * (1+tolerance_ratio)
    f_linear = interpolate.interp1d(ind, ds, kind='linear')
    y = [ds[0]]
    for i in range(1, len(ind)):
        indx = np.arange(ind[i-1]+1, ind[i]+1, 1)
        # conditions on the existence of the points we use
        indx_conditions = (i-2 >= 0) and (i+1 < len(ind))
        # conditions on the distance between the 2 points we want to
        # interpolate, as well as the 2 points before.
        # If the 2 points to interpolate are too close, linear works well
        # if they are too far, the convexity assumption is less likely
        # if the 2 points before were too close, the convexity assumption
        # is less likely to be true
        dist_conditions = ((ind[i]-ind[i-1] < max_distance) and
            (ind[i]-ind[i-1] > min_distance) and
            (ind[i-1]-ind[i-2] > previous_distance))
        # if the function is convex or concave on the chosen interval
        change_conditions = np.sign(ds[i-1]-ds[i-2]) != np.sign(ds[i]-ds[i-1])
        out = False
        # we go from last index to first because it is usually in
        # the last points that the points are out the interval,
        # so we minimize the complexity
        j = ind[i]-1
        while (not out) and (j > ind[i-1]):
            var = np.abs(f_linear(x[j]) - ds[i-1])
            if var > tolerance*np.abs(ds[i-1]):
                out = True
            j -= 1

        if indx_conditions and dist_conditions and change_conditions :
            sign_before = np.sign(ds[i-1] - ds[i-2])
            ind_mid = (ind[i]+ind[i-1]) // 2
            # if convex
            if sign_before < 0:
                bound = ds[i-1] - np.abs(ds[i-1])*tolerance
            # if concave
            else:
                bound = ds[i-1] + np.abs(ds[i-1])*tolerance

            # check if smart_linear_interpolation would use zero or linear
            # interpolation
            if out:
            # if zero interpolation, we choose a point in the
            # middle of zero and the percentage rule limit
                pt_mid = (ds[i-1] + bound)/2
                interp_indx = [ind[i-1], ind_mid, ind[i]-1, ind[i]]
                interp_points = [ds[i-1], pt_mid, ds[i-1], ds[i]]
            else:
            # if linear interpolation, we choose a point in the middle
            # of linear and the percentage rule limit
                pt_mid = (f_linear(ind_mid)+bound) / 2
                interp_indx = [ind[i-1], ind_mid, ind[i]]
                interp_points = [ds[i-1], pt_mid, ds[i]]
            # we connect the points using a linear interpolation
            f_ = interpolate.interp1d(interp_indx, interp_points,
                                      kind='linear')
            y_ = f_(indx)
        else:
            # use the smart_linear_interpolation algorithm
            y_ = f_linear(indx)
            if out:
                y_ = [ds[i-1] for k in indx[1:]]
                y_.append(ds[i])
        y = np.concatenate((y, y_))
    return y

def combined_con_fixed_recon(ds, ind, threshold, tolerance_ratio=0.1):
    ind = np.array(ind,dtype=int)
    ds = np.array(ds,dtype=float)
    x = np.arange(ind[-1]+1)
    tolerance = threshold * (1+tolerance_ratio)
    f_linear = interpolate.interp1d(ind, ds, kind='linear')
    y = [ds[0]]
    for i in range(1, len(ind)):
        indx = np.arange(ind[i-1]+1, ind[i]+1, 1)
        # conditions on the existence of the points we use
        indx_conditions = (i-2 >= 0) and (i+1 < len(ind))
        # conditions on the distance between the 2 points we want to
        # interpolate, as well as the 2 points before.
        # If the 2 points to interpolate are too close, linear works well
        # if they are too far, the convexity assumption is less likely
        # if the 2 points before were too close, the convexity assumption
        # is less likely to be true
        # if the function is convex or concave on the chosen interval
        change_conditions = np.sign(ds[i-1]-ds[i-2]) != np.sign(ds[i]-ds[i-1])
        out = False
        # we go from last index to first because it is usually in
        # the last points that the points are out the interval,
        # so we minimize the complexity
        j = ind[i]-1
        while (not out) and (j > ind[i-1]):
            var = np.abs(f_linear(x[j]) - ds[i-1])
            if var > tolerance*np.abs(ds[i-1]):
                out = True
            j -= 1

        if indx_conditions and change_conditions :
            sign_before = np.sign(ds[i-1] - ds[i-2])
            ind_mid = (ind[i]+ind[i-1]) // 2
            # if convex
            if sign_before < 0:
                bound = ds[i-1] - np.abs(ds[i-1])*tolerance
            # if concave
            else:
                bound = ds[i-1] + np.abs(ds[i-1])*tolerance

            # check if smart_linear_interpolation would use zero or linear
            # interpolation
            if out:
            # if zero interpolation, we choose a point in the
            # middle of zero and the percentage rule limit
                pt_mid = (ds[i-1] + bound)/2
                interp_indx = [ind[i-2],ind[i-1],ind[i]-1, ind[i],ind[i+1]]
                interp_points = [ds[i-2],ds[i-1],ds[i-1], ds[i],ds[i+1]]
                y_ = interpolate.pchip_interpolate(interp_indx,interp_points,indx)
            else:
            # if linear interpolation, we choose a point in the middle
            # of linear and the percentage rule limit
                pt_mid = (f_linear(ind_mid)+bound) / 2
                interp_indx = np.array([ind[i-2],ind[i-1],ind_mid, ind[i]],dtype=int)
                interp_points = np.array([ds[i-2],ds[i-1],pt_mid, ds[i]],dtype=float)
                indx = np.array(indx,dtype=int)
                f_ = interpolate.interp1d(interp_indx,interp_points,kind='linear')
                y_ = f_(indx)
            # we connect the points using a linear interpolation



        else:
            # use the smart_linear_interpolation algorithm
            y_ = f_linear(indx)
            if out:
                y_ = interpolate.pchip_interpolate(ind,ds,indx)
        y = np.concatenate((y, y_))
    return y

# ------- SIGNAL PROCESSING FUNCTIONS -------

def shannon_recon(signal, target_pts=120):

    """ Use Fourier transform and Shannon reconstruction
    It uses scipy.signal.resample to add points to the input signal, using
    the Fourier transform and the Shannon reconstruction

     Parameters
     ----------
    signal : list
        the downsampled signal to reconstruct

    target_pts : integer
        the number of points the reconstructed signal should have
    alpha : float
        the parameter alpha of scikit-learn's lasso method (see documentation)
    retCeofs : boolean
        if True, will return the coefficients of the transform

     Returns
     -------
     y : list
        the reconstructed signal

    """
    return resample(signal, target_pts)
