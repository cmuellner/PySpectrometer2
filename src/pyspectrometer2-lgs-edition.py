#!/usr/bin/env python3

'''
PySpectrometer2 Les Wright 2022
https://www.youtube.com/leslaboratory
https://github.com/leswright1977

Little Garden Spectrometer Edition cmuellner 2024
https://github.com/cmuellner

This project is a follow on from: https://github.com/leswright1977/PySpectrometer
'''

import argparse
import base64
import datetime
import time
from math import factorial

import numpy as np
import cv2
from scipy.interpolate import UnivariateSpline

# Colors are in CV2 have to be set in BGR
BLACK = (0, 0, 0)
ONE = (1, 1, 1)
WHITE = (255, 255, 255)
GRAY100 = (100, 100, 100)
GRAY200 = (200, 200, 200)
YELLOW = (0, 255, 255)

# Window properties
# The window has a string ID and is a composition of
# - header with textual state information
# - ROI preview (cropped camera data)
# - graph with intensities
# - waterfall graph
# Window name (also string IDs)
WINDOW_NAME = "PySpectrometer 2"
WINDOW_WIDTH = 1500
WINDOW_HEADER_START = 0
WINDOW_HEADER_HEIGHT = 80
WINDOW_ROI_START = WINDOW_HEADER_START + WINDOW_HEADER_HEIGHT
WINDOW_ROI_HEIGHT = 80
WINDOW_GRAPH_START = WINDOW_ROI_START + WINDOW_ROI_HEIGHT
WINDOW_GRAPH_HEIGHT = 400
WINDOW_WATERFALL_START = WINDOW_GRAPH_START + WINDOW_GRAPH_HEIGHT
WINDOW_WATERFALL_HEIGHT = 300
WINDOW_HEIGHT = WINDOW_WATERFALL_START + WINDOW_WATERFALL_HEIGHT

# Frame properties
# The little garden spectrometer has a 1920x1080 sensor.
# However, there is nothing going on outside of a 800x600 crop.
# Therefore, increasing will only lower the frame rate.
FRAME_WIDTH=800
FRAME_HEIGHT=600
FPS=21
DEVICE_DEFAULT = 0 # /dev/videoN

# Region of interest (RIO) properties
# There might be some variation from device to device, but most likely
# the settings below are at least good to start with.
# The settings below assume a frame size of 800x600:
#   80 px in height from (middle + ROI_HEIGHT_OFFSET).
#   We read the whole X-axis.
#   Intensity data is expected in the middle of the ROI.
ROI_WIDTH = 500
ROI_X_START = 0
ROI_HEIGHT = 80
ROI_Y_START = int((FRAME_HEIGHT - ROI_HEIGHT) / 2) + 27

# Reported intensity of the sensor in case we get no light into the spectrometer
DARK_INTENSITY = 14

# Smoothening parameters
SAVITZKY_GOLAY_WINDOW_SIZE_DEFAULT = 17
# Polynomial degree (max: 15)
SAVITZKY_GOLAY_POLYNOMIAL_DEFAULT = 15
# Minumum distance between peaks (max: 100)
PEAK_MINDIST_DEFAULT = 50
# Threshold (max: 100)
PEAK_THRES_DEFAULT = 20
# Noise reduction history size
NOISE_REDUCTION_HISTSIZE_DEFAULT = 8

# Convert a given spectral color (specified by its wavelength) into RGB values
# in the interval [0.0..1.0].  A scale factor allows to move the interval.
# E.g. if scale=255 then the interval of the RGB values will be [0.0..255.0].
# If toint is set to True, then the result will be rounded to the next integer.
# In case of no color (0, 0, 0) is returned unless the parameter gray provides
# a non-zero value that should be used for each RGB component instead.
#
# There are several sources in the internet for this conversion.
# Note, that this is simplified (i.e. performance optimized).
# To make this precise you would need to use the CIE31 color matching functions
# to calculate XYZ and then convert XYZ to RGB (a multiplication by 255 will
# put the numbers in the target interval).
def wavelength_to_rgb(nm, scale=255, toint=True, gray=155):
        gamma = 0.8
        factor = 0.0
        r = 0.0
        g = 0.0
        b = 0.0

        # Violett -> blue
        if 380 <= nm < 440:
            r = -(nm - 440) / (440 - 380)
            g = 0.0
            b = 1.0
        # Blue -> bluegreen
        elif 440 <= nm < 490:
            r = 0.0
            g = (nm - 440) / (490 - 440)
            b = 1.0
        # Bluegreen -> Green
        elif 490 <= nm < 510:
            r = 0.0
            g = 1.0
            b = -(nm - 510) / (510 - 490)
        # Green -> orange
        elif 510 <= nm < 580:
            r = (nm - 510) / (580 - 510)
            g = 1.0
            b = 0.0
        # Orange -> Red
        elif 580 <= nm < 645:
            r = 1.0
            g = -(nm - 645) / (645 - 580)
            b = 0.0
        # Red
        elif 645 <= nm < 781:
            r = 1.0
            g = 0.0
            b = 0.0

        # Linear falloff near the vision limits
        if 380 <= nm < 420:
            factor = 0.3 + 0.7 * (nm - 380) / (420 - 380)
        elif 420 <= nm < 701:
            factor = 1.0
        elif 701 <= nm < 781:
            factor = 0.3 + 0.7 * (780 - nm) / (780 - 700)

        # Calculate the scaled RGB values
        if r > 0.0 and factor > 0.0:
            r = ((r * factor) ** gamma) * scale
        if g > 0.0 and factor > 0.0:
            g = ((g * factor) ** gamma) * scale
        if b > 0.0 and factor > 0.0:
            b = ((b * factor) ** gamma) * scale

        # Round if necessary
        if toint:
            r = int(round(r))
            g = int(round(g))
            b = int(round(b))

        # No-color fixup
        if r == 0.0 and g == 0.0 and b == 0.0:
            r = gray
            g = gray
            b = gray

        return (r, g, b)

# The Savitzky Golay filter is a particular type of low-pass filter, well
# adapted for data smoothing. For further information see:
#   http://www.wire.tu-bs.de/OLDWEB/mameyer/cmr/savgol.pdf
# or:
#   http://www.dalkescientific.com/writings/NBN/data/savitzky_golay.py
# for a pre-numpy implementation.
# Source:
#   https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = np.abs(np.int32(window_size))
        order = np.abs(np.int32(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

# Dummy class to imitate a CV2 VideoCapture that constantly returns a given frame
class DummyVideoCapture:
    def __init__(self, imgpath, frame_width, frame_height, fps):
        # Read the given image
        gray = cv2.imread(imgpath, 0)

        # Check the dimensions
        height, width = gray.shape
        if frame_width != width or frame_height != height:
            return

        # We have a grayscale image and want to convert to YUYV.
        # Believe it or not, there is no way to get CV2 convert to YUYV.
        # What needs to be done is GRAY -> BGR -> YUV, then manual conversion.

        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        yuv= cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        # Create YUYV from YUV
        y0 = np.expand_dims(yuv[...,0][::,::2], axis=2)
        u = np.expand_dims(yuv[...,1][::,::2], axis=2)
        y1 = np.expand_dims(yuv[...,0][::,1::2], axis=2)
        v = np.expand_dims(yuv[...,2][::,::2], axis=2)
        yuyv = np.concatenate((y0, u, y1, v), axis=2)
        yuyv_cvt = yuyv.reshape(yuyv.shape[0], yuyv.shape[1] * 2,
                                int(yuyv.shape[2] / 2))

        self.frame = yuyv_cvt
        self.fps = fps

    def isOpened(self):
        return True

    def read(self):
        time.sleep(1 / self.fps)
        return (True, self.frame)

    def release(self):
        self.frame = None

# Get the peaks in y
# Source:
#   https://bitbucket.org/lucashnegri/peakutils/raw/f48d65a9b55f61fb65f368b75a2c53cbce132a0c/peakutils/peak.py
def get_peak_indices(y, thres=0.3, min_dist=1, thres_abs=False):
    """Peak detection routine.

    Finds the numeric index of the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.
    thres_abs: boolean
        If True, the thres value will be interpreted as an absolute value, instead of
        a normalized threshold.

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected.
        When using with Pandas DataFrames, iloc should be used to access the values at the returned positions.
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    if not thres_abs:
        thres = thres * (np.max(y) - np.min(y)) + np.min(y)

    min_dist = int(min_dist)

    # compute first order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    zeros, = np.where(dy == 0)

    # check if the signal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])

    if len(zeros):
        # compute first order difference of zero indexes
        zeros_diff = np.diff(zeros)
        # check when zeros are not chained together
        zeros_diff_not_one, = np.add(np.where(zeros_diff != 1), 1)
        # make an array of the chained zero indexes
        zero_plateaus = np.split(zeros, zeros_diff_not_one)

        # fix if leftmost value in dy is zero
        if zero_plateaus[0][0] == 0:
            dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)

        # fix if rightmost value of dy is zero
        if len(zero_plateaus) and zero_plateaus[-1][-1] == len(dy) - 1:
            dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)

        # for each chain of zero indexes
        for plateau in zero_plateaus:
            median = np.median(plateau)
            # set leftmost values to leftmost non zero values
            dy[plateau[plateau < median]] = dy[plateau[0] - 1]
            # set rightmost and middle values to rightmost non zero values
            dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]

    # find the peaks by using the first order difference
    peaks = np.where(
        (np.hstack([dy, 0.0]) < 0.0)
        & (np.hstack([0.0, dy]) > 0.0)
        & (np.greater(y, thres))
    )[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks

# Query the user for wavelength of the clicked points.
# Return two arrays with the pixel positions and the wavelengths.
def get_cal_data(click_events):
    calcomplete = False
    pxdata = []
    wldata = []
    print("Enter known wavelengths for observed pixels!")

    for pos in click_events:
        px = pos[0]
        wavelength = ""
        while wavelength == "":
            wavelength = input("Enter wavelength for " + str(px) + " px: ")
            try:
                wavelength = float(wavelength)
            except:
                print("Invalid input (need numeric data like '577.7')")
                wavelength = ""
        pxdata.append(px)
        wldata.append(wavelength)

    if len(pxdata) < 3:
        print("Not enough points (need at least three)!")
        print("Calibration aborted!")
        return ([], [])

    return (pxdata, wldata)

# Write the calibration data into a file.
def write_cal_data(pxdata, wldata):
    pxdata = ','.join(map(str, pxdata))
    wldata = ','.join(map(str, wldata))

    f = open('caldata.txt', 'w')
    f.write(pxdata + '\n')
    f.write(wldata + '\n')
    f.close()

    print("Calibration data written.")

# Read the calibration data from the calibration file.
# This will always provide data for further processing.
# The first element in the return tuple will be 0 if the data
# from the calibration file could be used. If it is 1, then
# default calibration data is returned.
def read_cal_data():
    err = 0
    pxdata = []
    wldata = []

    try:
        print("Loading calibration data...")
        f = open('caldata.txt', 'r')
        lines = f.readlines()
        f.close()

        for line in lines:
            line = line.strip()

            # Let's ignore comments
            if line[0] == ';' or line[0] == '#':
                continue

            vals = line.split(',')

            if len(pxdata) == 0:
                pxdata = [int(v) for v in vals]
            elif len(wldata) == 0:
                wldata = [float(v) for v in vals]
            else:
                break

        if (len(pxdata) != len(wldata) or
            len(pxdata) < 3):
            err = 1
    except:
        err = 1

    if err != 0:
        print("Loading calibration data failed (missing caldata.txt or corrupted data!")
        print("Loading default calibration.")
        print("Calibration is highly recommended!\n")

        # Values below are calibration data of a real device
        pxdata = [100, 199, 356, 541, 742, 802, 862, 900, 977, 1040]
        wldata = [405.4, 436.6, 487.7, 546.5, 611.6, 631.1, 650.8, 662.6, 687.7, 707.0]

    return (err, pxdata, wldata)

def process_cal_data(frame_width, pxdata, wldata):
    max_degree = len(pxdata) - 1;
    best_corr = 0
    best_wl = []
    best_degree = 0

    def is_monotonic(npdata):
        return np.all(np.diff(npdata) >= 0)

    def update_best(f):
        nonlocal best_corr
        nonlocal best_wl
        nonlocal best_degree

        # Calculate the wavelength for each pixel position
        wl = f(range(frame_width))

        # Check for monotony
        if not is_monotonic(wl):
            return

        # Calculate the Pearson correlation coefficient
        act = f(np.asarray(pxdata))
        corr_matrix = np.corrcoef(wldata, act)
        corr = corr_matrix[0, 1]

        # Pick the degree with the best correlation
        if corr > best_corr:
            best_corr = corr
            best_wl = wl
            best_degree = degree

    # Try polynomials of different degrees
    # Note, that the result MUST be monotonic and must have a correlation
    # coefficient higher than zero.  We will achieve that for reasonable
    # calibration data.  If we fail, then this is a sign that the calibration
    # data is not plausible.
    for degree in range(1, max_degree):
        p = np.polynomial.Polynomial.fit(pxdata, wldata, degree,
                                         domain=[0, frame_width])
        update_best(p)

    best_wl = [round(v, 6) for v in best_wl]

    print("Calculated calibration using %d degree function (from %d points)" %
          (best_degree, len(pxdata)))

    calmsg1 = "Calibrated!"
    calmsg2 = "Cal deg: %d" % (best_degree)

    return [best_wl, calmsg1, calmsg2]

def get_irradiance_calibration_table(wl_data):
    # Data here is from Lao Kang (measured on his spectrometer)
    FACTOR_350_379 = [2.568] * 30
    FACTOR_380_779 = [
        2.568, 2.568, 2.568, 2.568, 2.568, 2.568, 2.562, 2.532, 2.504, 2.364,
        2.282, 2.258, 2.19,  2.072, 1.94,  1.836, 1.753, 1.642, 1.555, 1.49 ,
        1.423, 1.37,  1.3,   1.243, 1.214, 1.18,  1.155, 1.155, 1.16,  1.165,
        1.175, 1.188, 1.204, 1.221, 1.25,  1.263, 1.258, 1.277, 1.291, 1.32 ,
        1.348, 1.376, 1.381, 1.376, 1.359, 1.344, 1.326, 1.313, 1.292, 1.273,
        1.254, 1.24,  1.218, 1.204, 1.186, 1.173, 1.147, 1.117, 1.089, 1.064,
        1.049, 1.04,  1.038, 1.036, 1.036, 1.036, 1.036, 1.038, 1.041, 1.046,
        1.049, 1.051, 1.052, 1.055, 1.057, 1.057, 1.056, 1.054, 1.053, 1.049,
        1.043, 1.035, 1.028, 1.02,  1.01,  0.996, 0.979, 0.963, 0.945, 0.927,
        0.908, 0.893, 0.88,  0.868, 0.857, 0.846, 0.838, 0.828, 0.82,  0.813,
        0.81,  0.808, 0.808, 0.809, 0.812, 0.814, 0.816, 0.819, 0.822, 0.827,
        0.832, 0.84,  0.85,  0.859, 0.868, 0.874, 0.88,  0.886, 0.891, 0.896,
        0.9,   0.904, 0.908, 0.91,  0.909, 0.908, 0.906, 0.903, 0.899, 0.895,
        0.891, 0.887, 0.883, 0.878, 0.873, 0.867, 0.86,  0.852, 0.845, 0.839,
        0.833, 0.827, 0.822, 0.817, 0.813, 0.808, 0.805, 0.803, 0.802, 0.8  ,
        0.798, 0.796, 0.795, 0.794, 0.794, 0.794, 0.793, 0.793, 0.794, 0.789,
        0.786, 0.784, 0.788, 0.793, 0.797, 0.8,   0.804, 0.808, 0.812, 0.815,
        0.818, 0.821, 0.824, 0.825, 0.827, 0.827, 0.827, 0.826, 0.824, 0.823,
        0.822, 0.822, 0.821, 0.82,  0.818, 0.817, 0.815, 0.813, 0.809, 0.806,
        0.803, 0.802, 0.802, 0.801, 0.8,   0.798, 0.796, 0.794, 0.792, 0.791,
        0.791, 0.792, 0.795, 0.798, 0.802, 0.806, 0.81,  0.814, 0.82,  0.825,
        0.832, 0.839, 0.845, 0.852, 0.858, 0.864, 0.87,  0.877, 0.885, 0.895,
        0.905, 0.915, 0.924, 0.935, 0.943, 0.952, 0.958, 0.967, 0.975, 0.984,
        0.993, 1.001, 1.012, 1.02,  1.028, 1.031, 1.036, 1.039, 1.042, 1.046,
        1.05,  1.055, 1.061, 1.065, 1.069, 1.071, 1.071, 1.069, 1.066, 1.064,
        1.062, 1.062, 1.063, 1.064, 1.064, 1.062, 1.058, 1.053, 1.048, 1.044,
        1.043, 1.044, 1.045, 1.044, 1.041, 1.036, 1.029, 1.023, 1.018, 1.015,
        1.01,  1.007, 1.005, 1.003, 1,     0.995, 0.992, 0.991, 0.99,  0.987,
        0.985, 0.984, 0.983, 0.98,  0.978, 0.979, 0.982, 0.985, 0.991, 0.996,
        1.002, 1.003, 1.003, 1.001, 1.003, 1.005, 1.008, 1.011, 1.016, 1.018,
        1.017, 1.016, 1.018, 1.026, 1.032, 1.039, 1.045, 1.053, 1.057, 1.057,
        1.06,  1.064, 1.07,  1.073, 1.076, 1.081, 1.086, 1.087, 1.089, 1.091,
        1.093, 1.095, 1.098, 1.102, 1.104, 1.108, 1.111, 1.115, 1.117, 1.118,
        1.122, 1.126, 1.128, 1.131, 1.132, 1.134, 1.138, 1.14,  1.142, 1.144,
        1.145, 1.146, 1.148, 1.148, 1.145, 1.147, 1.141, 1.138, 1.14,  1.14 ,
        1.14,  1.143, 1.145, 1.149, 1.15,  1.152, 1.156, 1.156, 1.157, 1.161,
        1.165, 1.17,  1.172, 1.174, 1.176, 1.179, 1.183, 1.186, 1.189, 1.192,
        1.204, 1.2,   1.205, 1.21,  1.215, 1.219, 1.225, 1.23,  1.244, 1.241,
        1.246, 1.252, 1.257, 1.264, 1.271, 1.278, 1.283, 1.299, 1.299, 1.309,
        1.314, 1.322, 1.332, 1.34,  1.347, 1.367, 1.368, 1.376, 1.387, 1.398]
    FACTOR_780_800 = [1.41] * 21
    FACTOR = FACTOR_350_379 + FACTOR_380_779 + FACTOR_780_800

    wl = range(350, 801)
    factor_interp = np.interp(wl_data, wl, FACTOR, 0, 0)
    return factor_interp

def generate_graticule(wl_data):
    # Get lowest and highest wavelength rounded to next integer
    # We also create a margin of 10 nm for better measure???
    low = int(round(wl_data[0])) - 10
    high = int(round(wl_data[-1])) + 10

    tens = []
    fifties = []
    for i in range(low, high):
        if (i % 10 == 0):
            # Position contains pixelnumber and wavelength
            position = min(enumerate(wl_data), key=lambda x: abs(i - x[1]))

            # We found the right position, if the error is less than 1
            if abs(i - position[1]) < 1:
                tens.append(position[0])

                if (i % 50 == 0):
                    labelpos = position[0]
                    labeltxt = int(round(position[1]))
                    labeldata = [labelpos, labeltxt]
                    fifties.append(labeldata)

    return (tens, fifties)

def main():
    print("PySpectrometer 2 - Little Garden Spectrometer Edition")
    print("Les Wright 2022 [Little Garden Spectrometer Edition by cmuellner 2024]")

    # Create parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",
                        help="Video Device number e.g. 0"
                             ", use `v4l2-ctl --list-devices`")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fullscreen", action="store_true",
                       help="Enable fullscreen")
    group.add_argument("--waterfall", action="store_true",
                       help="Enable waterfall (Windowed only)")

    # Parse command line arguments
    args = parser.parse_args()

    # Log the displaying mode
    dispFullscreen = False
    if args.fullscreen:
        print("Fullscreen Spectrometer enabled")
        dispFullscreen = True

    # Get the camera properties
    if args.device:
        dev = args.device
    else:
        dev = DEVICE_DEFAULT

    if dev.isnumeric():
        # Connect to the camera and set it up
        cap = cv2.VideoCapture('/dev/video'+dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("Error: Could not open camera device: /dev/video"+str(dev))
            return

        frame_width = FRAME_WIDTH
        frame_height = FRAME_HEIGHT
        print("Requested camera caps: %d px x %d px"
              % (frame_width, frame_height))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Adjust capturing settings.
        # The following command helps to validate the values:
        #   v4l2-ctl --list-ctrls -d /dev/videoN
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
        cap.set(cv2.CAP_PROP_CONTRAST, 32)
        cap.set(cv2.CAP_PROP_SATURATION, 50)
        cap.set(cv2.CAP_PROP_HUE, 0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_TEMPERATURE, 5600)
        # To set manual exposure, we need to set 'auto' first
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, 1000)
        cap.set(cv2.CAP_PROP_BACKLIGHT, 0)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print("New camera caps: %d px x %d px @ %d fps"
              % (frame_width, frame_height, fps))

        # Make sure the ROI is within the frame size
        if frame_width < (ROI_X_START + ROI_WIDTH) or frame_height < (ROI_Y_START + ROI_HEIGHT):
            print("Error: ROI too big for frame size!")
            return
        # By default CV2 converts all frames into BGR.
        # We don't want this, because we want unmodified values.
        # The sensor data is sent in 'YUYV' (YUYV 4:2:2), which can be check with
        #   v4l2-ctl --list-formats /dev/videoN
        # Note: The sensor does not have a Bayer matrix anymore, so YUYV is
        #       calculated based on wrong assumptions. We'll fix that later.
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    else:
        cap = DummyVideoCapture(dev, FRAME_WIDTH, FRAME_HEIGHT, FPS)
        fps = FPS

    # Create the window
    # - automatic size calculation
    # - no resizing
    # - no menu bar
    # Note, that moveWIndow() does not work with Wayland
    if dispFullscreen == True:
        cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        window_flags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL
        cv2.namedWindow(WINDOW_NAME, window_flags)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    # Settings for smoothening
    savpoly = SAVITZKY_GOLAY_POLYNOMIAL_DEFAULT

    # Settings for peak detection
    mindist = PEAK_MINDIST_DEFAULT
    thresh = PEAK_THRES_DEFAULT

    # Font that is used in the application
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize intensities array with zeros
    intensity = np.zeros(WINDOW_WIDTH, dtype=np.uint8)

    # Special states for measuring and pixel recording
    measure = False
    unit_is_px = False
    holdpeaks = False
    smoothen = False
    calibrate_irradiance = False

    # Last time saved message
    saveMsg = "No data saved"

    # Create waterfall data (all black at this time)
    waterfall = np.zeros([WINDOW_WATERFALL_HEIGHT, WINDOW_WIDTH, 3], dtype=np.uint8)

    # Process the calibration data
    (cal_err, pxdata, wldata) = read_cal_data()
    caldata = process_cal_data(WINDOW_WIDTH, pxdata, wldata)
    wl_data = caldata[0]
    if cal_err == 0:
        calmsg1 = caldata[1]
        calmsg2 = caldata[2]
    else:
        calmsg1 = "UNCALIBRATED!"
        calmsg2 = "Perform Calibration!"
    factor_interp = get_irradiance_calibration_table(wl_data)

    # Generate graticule
    (tens, fifties) = generate_graticule(wl_data)

    # Generate header
    banner_txt = np.zeros((WINDOW_HEADER_HEIGHT, WINDOW_WIDTH, 3), np.uint8)
    cv2.putText(banner_txt, "PySpectrometer 2",
                 (20, WINDOW_HEADER_HEIGHT - 36),
                 cv2.FONT_HERSHEY_DUPLEX, 1.5, ONE, 2)

    grad_img = np.zeros((WINDOW_HEADER_HEIGHT, 400, 3), np.uint8)
    for idx, wl in enumerate(range(380, 780)):
        (r, g, b) = wavelength_to_rgb(wl)
        col = np.full((WINDOW_HEADER_HEIGHT, 3), (b, g, r))
        grad_img[:, idx] = col
    # Scale the gradient to match the font area
    grad_scaled = cv2.resize(grad_img, (500, WINDOW_HEADER_HEIGHT))
    g_h, g_w, _ = grad_scaled.shape
    # Extend as needed
    grad_mask = np.pad(grad_scaled, ((0, WINDOW_HEADER_HEIGHT - g_h),
                                  (20, WINDOW_WIDTH - g_w - 20),
                                  (0, 0)))
    header = np.multiply(banner_txt, grad_mask)

    cv2.putText(header, "Les Wright 2022 [Little Garden Spectrometer Edition by cmuellner 2024]",
                (23, WINDOW_HEADER_HEIGHT - 10),
                font, 0.4, WHITE, 1)

    # Function to save the contents of the current window into a PNG
    def snapshot(wl_data, intensity, window_stack, graph, frame):
        t = datetime.datetime.now()
        tstamp = t.strftime("%Y%m%d_%H%M%S")

        cv2.imwrite("window-" + tstamp + ".png", window_stack)
        cv2.imwrite("spectrum-" + tstamp + ".png", graph)
        bwimage = cv2.cvtColor(frame, cv2.COLOR_YUV2GRAY_YUYV)
        cv2.imwrite("raw-" + tstamp + ".png", bwimage)

        # Generate irradiance data by interpolation it at interger wavelengths
        wl = range(350, 801)
        irr = np.interp(wl, wl_data, intensity, 0, 0)
        max_irr = np.max(irr)
        norm_irr = np.divide(irr, max_irr)

        f = open("irradiance-" + tstamp + '.csv', 'w')
        f.write("wl,irr\n")
        for x in zip(wl, norm_irr):
            f.write(str(x[0]) + "," + str(x[1]) + "\n")
        f.close()

        return "Last Save: " + t.strftime("%H:%M:%S")

    # Setup mouse support
    cursor_x = 0
    cursor_y = 0
    click_events = []
    def handle_mouse(event, x, y, flags, param):
        nonlocal cursor_x
        nonlocal cursor_y
        nonlocal click_events

        if measure == True:
            if event == cv2.EVENT_MOUSEMOVE:
                # Store the position so we can print a cursor
                cursor_x = x
                cursor_y = y
            if event == cv2.EVENT_LBUTTONDOWN:
                # Store the click for calibration
                click_events.append([x, y])
    # Listen for mouse events in the window
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse)

    while(cap.isOpened()):
        # Capture a new frame, exit on failure
        ret, frame = cap.read()
        if ret == False:
            break

        # We get data in 'YUYV' format.
        # Intensities are calculated from a conversion (back) to grayscale.
        # For the color output (the graphs), we further create a BGR image.
        # We need to do this first, because CV2 can't handle YUYV data and
        # will mess up the result (crop, resize, flip, ...).
        bwimage = cv2.cvtColor(frame, cv2.COLOR_YUV2GRAY_YUYV)

        # Users expect the graph to show lower wavelengths (blue-ish parts)
        # on the left and higher wavelengths (red-ish parts) on the right.
        # The sensor captures in opposite direction. Therefore, we flip
        # the image.
        bwimage = cv2.flip(bwimage, 1)

        # Crop the (spectrometer-specific) RIO:
        #   80 px in height from (middle + ROI_HEIGHT_OFFSET).
        #   We read the whole X-axis.
        #   Intensity data is expected in the middle of the ROI.
        x = ROI_X_START
        w = ROI_WIDTH
        y = ROI_Y_START
        h = ROI_HEIGHT
        h_data = int(h / 2)
        bwimage = bwimage[y:y+h, x:x+w]

        # Scale the image to match the window width
        bwimage = cv2.resize(bwimage, (WINDOW_WIDTH, ROI_HEIGHT))

        ###
        # Data analysis and postprocessing
        ###

        # Now process the intensity data and display it
        for i in range(WINDOW_WIDTH):
            # Get the three vertical pixels in the middle of the ROI
            # and calculate the average.
            # XXX: is average the best idea? probably maximum would be better
            px0 = bwimage[h_data - 1, i]
            px1 = bwimage[h_data, i]
            px2 = bwimage[h_data + 1,i]
            data = (int(px0) + px1 + px2) / 3

            # Uncomment the line below for measuring DARK_INTENSITY
            # Don't forget to not let any light into the spectrometer
            #print("intensity[400]: %d" % (data))

            # Subtract the dark frame value
            if data > DARK_INTENSITY:
                data = data - DARK_INTENSITY
            else:
                data = 0

            # Set the new intensity in the graph data
            if holdpeaks != True or data > intensity[i]:
                intensity[i] = data

        # Smoothen the curve
        if smoothen:
            intensity = savitzky_golay(intensity,
                                       SAVITZKY_GOLAY_WINDOW_SIZE_DEFAULT,
                                       savpoly)
            intensity = np.rint(intensity).astype(np.uint8)

        if calibrate_irradiance:
            factor_interp = get_irradiance_calibration_table(wl_data)
            intensity = np.multiply(intensity, factor_interp).astype(np.uint8)

        # Find peaks
        max_intensity = np.max(intensity)
        if max_intensity > 0:
            scaled_thresh = thresh / max_intensity
        else:
            scaled_thresh = 1

        peak_indices = get_peak_indices(intensity.astype(int), scaled_thresh, min_dist=mindist)

        ###
        # Prepare cropped region
        ###

        # Create lines around a region with a height of 3px around the center.
        cropped = cv2.cvtColor(bwimage, cv2.COLOR_GRAY2BGR)
        cv2.line(cropped, (0, h_data - 2), (WINDOW_WIDTH, h_data - 2), WHITE, 1)
        cv2.line(cropped, (0, h_data + 2), (WINDOW_WIDTH, h_data + 2), WHITE, 1)

        ###
        # Prepare graph
        ###

        # Create a new white graph
        graph = np.zeros([WINDOW_GRAPH_HEIGHT, WINDOW_WIDTH, 3], dtype=np.uint8)
        graph.fill(255)

        # Display a graticule calibrated with cal data
        textoffset = 12
        # Vertial lines every 10 nm
        for position in tens:
            cv2.line(graph, (position, 15), (position, WINDOW_GRAPH_HEIGHT),
                     GRAY200, 1)

        # Vertical lines with labels every 50 nm
        for positiondata in fifties:
            cv2.line(graph, (positiondata[0], 15),
                     (positiondata[0], WINDOW_GRAPH_HEIGHT), BLACK, 1)
            if unit_is_px:
                txt = str(positiondata[0]) + "px"
            else:
                txt = str(positiondata[1]) + "nm"
            cv2.putText(graph, txt, (positiondata[0] - textoffset, 12),
                        font, 0.4, BLACK, 1, cv2.LINE_AA)

        # Draw horizontal lines
        step = WINDOW_GRAPH_HEIGHT
        while step > 64:
            step = int(step / 2)
        for i in range (step, WINDOW_GRAPH_HEIGHT, step):
            y_pos = WINDOW_GRAPH_HEIGHT - i
            cv2.line(graph, (0, y_pos), (WINDOW_WIDTH, y_pos), GRAY200, 1)

        # Draw the intensity data
        for index, i in enumerate(intensity):
            # Derive the color from the wvalenthData array
            (r, g, b) = wavelength_to_rgb(round(wl_data[index]))

            # Draw a BGR line for each intensity (and a black line above)
            height = WINDOW_GRAPH_HEIGHT - int(round(WINDOW_GRAPH_HEIGHT * i / 255))
            cv2.line(graph, (index, WINDOW_GRAPH_HEIGHT), (index, height),
                     (b, g, r), 1)
            cv2.line(graph, (index, height - 1), (index, height), BLACK, 1)

        # Label the peaks
        textoffset = 12
        for index in peak_indices:
            i = intensity[index]
            height = WINDOW_GRAPH_HEIGHT - int(round(WINDOW_GRAPH_HEIGHT * i / 255))
            cv2.rectangle(graph, ((index - textoffset) - 2, height),
                          ((index - textoffset) + 60, height - 15), YELLOW, -1)
            cv2.rectangle(graph, ((index - textoffset) - 2, height),
                          ((index - textoffset) + 60, height - 15), BLACK, 1)
            if unit_is_px:
                txt = str(index) + "px"
            else:
                txt = str(round(wl_data[index], 1)) + "nm"

            cv2.putText(graph, txt,
                        (index - textoffset, height - 3), font, 0.4, BLACK, 1,
                        cv2.LINE_AA)
            # Flag pole
            cv2.line(graph, (index, height), (index, height + 10), BLACK, 1)

        if measure == True:
            # Show the cursor
            y_offset = WINDOW_GRAPH_START
            y_top = cursor_y - y_offset - 20
            y_mid = cursor_y - y_offset
            y_bot = cursor_y - y_offset + 20
            x_lef = cursor_x - 20
            x_mid = cursor_x
            x_rig = cursor_x + 20

            cv2.line(graph, (x_mid, y_top), (x_mid, y_bot), BLACK, 1)
            cv2.line(graph, (x_lef, y_mid), (x_rig, y_mid), BLACK, 1)

            txt = None
            if unit_is_px:
                txt = str(cursor_x) + 'px'
            else:
                txt = str(round(wl_data[cursor_x], 2)) + 'nm'

            cv2.putText(graph, txt, (x_mid + 5, y_mid - 5), font, 0.4, BLACK, 1,
                        cv2.LINE_AA)
        else:
            #also make sure the click array stays empty
            click_events = []

        # Draw the mouse clicks as dots
        for pos in click_events:
            y_offset = WINDOW_GRAPH_START
            x = pos[0]
            y = pos[1] - y_offset
            cv2.circle(graph, (x, y), 5, BLACK, -1)
            cv2.putText(graph, str(x), (x + 5, y), font, 0.4, BLACK, 1,
                        cv2.LINE_AA)

        ###
        # Prepare waterfall
        ###

        # Update the waterfall data
        wdata = np.zeros([1, WINDOW_WIDTH, 3], dtype=np.uint8)
        for index, i in enumerate(intensity):
            # Get the spectral color of the wavelength in RGB
            (r, g, b) = wavelength_to_rgb(round(wl_data[index]),
                                          scale=intensity[index],
                                          gray=intensity[index])

            # Set as BGR (as CV2 wants everything in BGR)
            wdata[0, index] = (b, g, r)
        # Append new data on top
        waterfall = np.insert(waterfall, 0, wdata, axis=0)
        # Drop last line
        waterfall = waterfall[:-1]

        ###
        # Compose the window stack
        ###

        window_stack = np.vstack((header, cropped, graph, waterfall))

        ###
        # Adjustments in the final window contents
        # This is required because we need to keep a copy of the untained
        # contents of the header and the waterfall for the next loop iteration.
        ###

        # Add dividing lines between the window elements
        cv2.line(window_stack,(0, WINDOW_ROI_START),
                 (WINDOW_WIDTH, WINDOW_ROI_START), WHITE, 1)
        cv2.line(window_stack,(0, WINDOW_GRAPH_START),
                 (WINDOW_WIDTH, WINDOW_GRAPH_START), WHITE, 1)
        cv2.line(window_stack,(0, WINDOW_WATERFALL_START),
                 (WINDOW_WIDTH, WINDOW_WATERFALL_START), WHITE, 1)

        # Header: print 1st header column
        x_pos = WINDOW_WIDTH - 310

        cv2.putText(window_stack, calmsg1, (x_pos, 15), font, 0.4, YELLOW, 1,
                    cv2.LINE_AA)

        cv2.putText(window_stack, calmsg2, (x_pos, 33), font, 0.4, YELLOW, 1,
                    cv2.LINE_AA)

        if calibrate_irradiance:
            txt = "Irr. cal.: ON"
        else:
            txt = "Irr. cal.: OFF"
        cv2.putText(window_stack, txt, (x_pos, 51), font, 0.4, YELLOW, 1,
                    cv2.LINE_AA)

        cv2.putText(window_stack, saveMsg, (x_pos, 69), font, 0.4, YELLOW, 1,
                    cv2.LINE_AA)

        # Header: print 2nd column
        x_pos = WINDOW_WIDTH - 160

        if holdpeaks:
            txt = "Holdpeaks ON"
        else:
            txt = "Holdpeaks OFF"
        cv2.putText(window_stack, txt, (x_pos, 15), font, 0.4, YELLOW, 1,
                    cv2.LINE_AA)

        txt = "Savgol filter: "
        if smoothen:
            txt += str(savpoly)
        else:
            txt += "OFF"
        cv2.putText(window_stack, txt, (x_pos, 33), font, 0.4, YELLOW, 1,
                    cv2.LINE_AA)

        txt = "Label Peak Width: " + str(mindist)
        cv2.putText(window_stack, txt, (x_pos, 51), font, 0.4, YELLOW, 1,
                    cv2.LINE_AA)

        txt = "Label Threshold: " + str(thresh)
        cv2.putText(window_stack, txt, (x_pos, 69), font, 0.4, YELLOW, 1,
                    cv2.LINE_AA)

        # Waterfall: draw vertical lines every whole 50 nm
        textoffset = 12
        y_start = WINDOW_WATERFALL_START
        y_end = WINDOW_WATERFALL_START + WINDOW_WATERFALL_HEIGHT
        for positiondata in fifties:
            for i in range(y_start + 20, y_end, 20):
                cv2.line(window_stack, (positiondata[0], i),
                         (positiondata[0], i), BLACK, 2)
                cv2.line(window_stack, (positiondata[0], i),
                         (positiondata[0], i), GRAY200, 1)

            if unit_is_px:
                txt = str(positiondata[0]) + "px"
            else:
                txt = str(positiondata[1]) + "nm"

            cv2.putText(window_stack, txt,
                        (positiondata[0] - textoffset, y_end - 5),
                        font, 0.4, BLACK, 2, cv2.LINE_AA)
            cv2.putText(window_stack, txt,
                        (positiondata[0] - textoffset, y_end - 5),
                        font, 0.4, WHITE, 1, cv2.LINE_AA)

        ###
        # Put the window stack into the window
        ###

        cv2.imshow(WINDOW_NAME, window_stack)

        ###
        # Check for keyboard inputs
        ###

        key = cv2.waitKey(1)

        # Quit loop
        if key == ord('q'):
            break

        # Save data to file system
        elif key == ord("s"):
            saveMsg = snapshot(wl_data, intensity, window_stack, graph, frame)

        # Hold peaks
        elif key == ord('h'):
            holdpeaks = not holdpeaks

        # Smoothen the curve
        elif key == ord('-'):
            smoothen = not smoothen

        elif key == ord('i'):
            calibrate_irradiance = not calibrate_irradiance

        # Enable the mouse cursor
        elif key == ord("m"):
            measure = not measure

        # Display unit is px
        elif key == ord("p"):
            unit_is_px = not unit_is_px

        # Clear marked points
        elif key == ord("x"):
            click_events = []

        # Calibrate marked points
        elif key == ord("c"):
            (pxdata, wldata) = get_cal_data(click_events)
            if len(pxdata) > 0:
                # Process the calibration data
                write_cal_info(pxdata, wldata)
                caldata = process_cal_data(WINDOW_WIDTH, pxdata, wldata)
                wl_data = caldata[0]
                calmsg1 = caldata[1]
                calmsg2 = caldata[2]
                factor_interp = get_irradiance_calibration_table(wl_data)

                # Re-generate graticule
                (tens, fifties) = generate_graticule(wl_data)

        # Increment the SavGol poly
        elif key == ord("o"):
                if savpoly < 15:
                    savpoly += 1
        # Decrement the SavGol poly
        elif key == ord("l"):
                if savpoly > 0:
                    savpoly -= 1
        # Increase the peak width
        elif key == ord("i"):
                if mindist < 100:
                    mindist += 1
        # Decrease the peak width
        elif key == ord("k"):
                if mindist > 0:
                    mindist -= 1
        # Increment the label threshold
        elif key == ord("u"):
                if thresh < 100:
                    thresh += 1
        # Decrement the label threshold
        elif key == ord("j"):
                if thresh > 0:
                    thresh -= 1

    # Everything done, release the device
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
