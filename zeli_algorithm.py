import numpy as np
from scipy import interpolate

import recon_utils as utils

def ZeLiC(
        ds, ind, threshold, tolerance_ratio=0.1, min_distance=3,
        max_distance=-0.5, previous_distance=3, convexity_cond = True):

    tolerance = tolerance_ratio*threshold
    
    if max_distance < 0:
         # if negative, it will take a proportion
         # of the length of the signal.
         # If signal is of length 120 and max_distance = 0.5, then
         # max_distance = 0.5*120 = 60
        max_distance = np.abs(max_distance) * ind[-1]

    x = np.arange(ind[-1]+1)
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
            if var > tolerance:
                out = True
            j -= 1

        if indx_conditions and dist_conditions and change_conditions and convexity_cond:
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

def ZeChipC(
        ds, ind, threshold, tolerance_ratio=0.1, min_distance=3,
        max_distance=-0.5, previous_distance=3, convexity_cond = True):

    tolerance = tolerance_ratio*threshold
    
    if max_distance < 0:
         # if negative, it will take a proportion
         # of the length of the signal.
         # If signal is of length 120 and max_distance = 0.5, then
         # max_distance = 0.5*120 = 60
        max_distance = np.abs(max_distance) * ind[-1]

    x = np.arange(ind[-1]+1)
    #f_linear = interpolate.interp1d(ind, ds, kind='linear')
    f_pchip = interpolate.PchipInterpolator(ind,ds)
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
            var = np.abs(f_pchip(x[j]) - ds[i-1])
            if var > tolerance:
                out = True
            j -= 1

        if indx_conditions and dist_conditions and change_conditions and convexity_cond:
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
                pt_mid = (f_pchip(ind_mid)+bound) / 2
                interp_indx = [ind[i-1], ind_mid, ind[i]]
                interp_points = [ds[i-1], pt_mid, ds[i]]
            # we connect the points using a linear interpolation
            f_ = interpolate.PchipInterpolator(interp_indx, interp_points)
            y_ = f_(indx)
        else:
            # use the smart_linear_interpolation algorithm
            y_ = f_pchip(indx)
            if out:
                y_ = [ds[i-1] for k in indx[1:]]
                y_.append(ds[i])
        y = np.concatenate((y, y_))
    return y

