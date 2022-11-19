"""
Phase retrieval image reconstruction for single particle imaging.

Requirements
------------
`numpy` `scipy` `cupy` `pytorch` `h5py` `pandas` `pillow` `matplotlib` `tqdm`

Introduction
------------
1. Inheriting the following algorithm: ER, HIO, ER-HIO, ASR, HPR, RAAR, DM.
2. Different phasing algorithms and parameters can be freely combined.
3. The Oversampling smoothness (OSS) filtering framework is available.
4. Threshold and Area support update algorithms with Gaussian blur.
5. Using CUDA for reconstruction. Using non-blocking multithreading for output.

Author: Jiyang (Julian) OU Julian-OU@outlook.com\\
College of Physics, Sichaun University, CHINA.\\
Copyright (c) 2020-2022. All rights reserved.
"""

import os
import warnings
from multiprocessing import Pool, cpu_count

import cupy as cp
import numpy as np
import pandas as pd
import torch.nn.functional as nn
import torch.utils.dlpack as dl
from cupy.fft import fft2 as fft
from cupy.fft import ifft2 as ifft
from cupy.fft import irfft2 as irfft
from cupy.fft import rfft2 as rfft
from cupyx.scipy.special import gammaln
from tqdm import tqdm

import fileioput as fio

PI = np.pi
INF = np.inf
NAN = np.nan


def shift(data):
    return cp.fft.fftshift(data, (-2, -1))


def ishift(data):
    return cp.fft.ifftshift(data, (-2, -1))


def convolve(data1, data2):
    return irfft(rfft(data1) * rfft(data2), s=cp.shape(data1))


def correlate(data1, data2):
    return irfft(cp.conj(rfft(data1)) * rfft(data2), s=cp.shape(data1))


def errCallBack(err):
    print(f"Error: {str(err)}")


def SetFourierGrid(size):
    x = cp.arange(-0.5, 0.5, 1 / size[0])
    y = cp.arange(-0.5, 0.5, 1 / size[1])
    xx, yy = cp.meshgrid(x, y, indexing="ij")
    return xx, yy


def SetRealGrid(size):
    x = cp.arange(-size[0] / 2, size[0] / 2)
    y = cp.arange(-size[1] / 2, size[1] / 2)
    xx, yy = cp.meshgrid(x, y, indexing="ij")
    return xx, yy


def GaussianKernel(size, sigma):
    """
    Return a Gaussian convolution kernel.

    G(m,n)=exp[-(m^2+n^2)/(2*sigma^2)]/(2*pi*sigma^2)
    """
    xx, yy = SetRealGrid(size)
    sigma2 = sigma * sigma
    grid2 = shift(xx**2 + yy**2) / sigma2
    return cp.exp(-grid2 / 2) / (2 * PI * sigma2)


def GaussianWindow(size, sigma):
    """
    Return a sequence of Gaussian Windows for filtering.

    w(l,m,n)=exp[-(m^2+n^2)/(2*sigma[l]^2)]
    """
    xx, yy = SetFourierGrid(size)
    sigma2 = sigma * sigma
    grid2 = shift(xx**2 + yy**2)
    try:
        grid2 = grid2 / sigma2
    except:
        grid2 = grid2[cp.newaxis, :, :] / sigma2[:, cp.newaxis, cp.newaxis]
    return cp.exp(-grid2 / 2)


def PoissonDeviate(lam: float, seed=None):
    """
    Add Poisson noise to a image (diffraction pattern).

    Parameters
    ----------
     - intensity (cupy.ndarray): Image intensity.
     - noiseRatio (float): Poisson noise ratio.
     - seed (default: None): The seed for the random number generator.

    Returns
    -------
     - poissonDev (cupy.ndarray): Poisson deviates corresponds to lambda one to one.

    References
    ----------
    1. W. H. Press et al., Numerical recipes : the art of scientific computing (Cambridge\
        University Press, 2007) p.373
    2. W. H. Press et al., Numerical recipes in C : the art of scientific computing (Cambridge\
        University Press, 1992) p.294
    """

    cp.random.seed(seed)
    poissonDev = cp.empty_like(lam)
    loc1 = lam < 5
    loc2 = lam > 13.5
    loc3 = cp.logical_and(lam >= 5, lam <= 13.5)
    if cp.sum(loc1):  # Using product of uniforms method (waiting time)
        lamexp = cp.exp(-lam[loc1])
        deviate = -cp.ones_like(lamexp)
        time = cp.ones_like(deviate)
        idx1 = cp.arange(cp.sum(loc1))
        while cp.size(idx1):
            deviate[idx1] = deviate[idx1] + 1
            time[idx1] = time[idx1] * cp.random.rand(cp.size(idx1))
            idx1 = idx1[time[idx1] > lamexp[idx1]]
        poissonDev[loc1] = deviate
    if cp.sum(loc2):  # Using fast acceptance-rejection method
        lamloc = lam[loc2]
        sqrtlam = cp.sqrt(lamloc)
        deviate = -cp.ones_like(lamloc)
        idx2 = cp.arange(cp.sum(loc2))
        while cp.size(idx2):
            u = 0.64 * cp.random.rand(cp.size(idx2))
            v = -0.68 + 1.28 * cp.random.rand(cp.size(idx2))
            v2 = cp.square(v)
            rejection = cp.logical_or(  # Outer squeeze for fast rejection
                cp.logical_and(v >= 0, v2 > 6.5 * u * (0.64 - u) * (u + 0.2)),
                cp.logical_and(v < 0, v2 > 9.6 * u * (0.66 - u) * (u + 0.07)),
            )
            other = cp.logical_not(rejection)
            k = cp.floor(
                sqrtlam[idx2][other] *
                (v[other] / u[other]) + lamloc[idx2][other] + 0.5
            )
            try:
                other[other] = k >= 0
                k = k[k >= 0][other[other]]
            except:
                None
            if not (cp.any(other)):
                continue
            u = u[other]
            v = v[other]
            u2 = cp.square(u)
            v2 = v2[other]
            acceptance = cp.logical_or(  # Inner squeeze for fast acceptance
                cp.logical_and(v >= 0, v2 < 15.2 * u2 * \
                               (0.61 - u) * (0.8 - u)),
                cp.logical_and(v < 0, v2 < 6.76 * u2 * (0.62 - u) * (1.4 - u)),
            )
            other[other] = acceptance
            deviate[idx2[other]] = k[acceptance]
            idx2 = idx2[cp.logical_not(other)]
        poissonDev[loc2] = deviate
    if cp.sum(loc3):  # Using ratio-of-uniforms method
        lamloc = lam[loc3]
        sqrtlam = cp.sqrt(lamloc)
        loglam = cp.log(lamloc)
        deviate = -cp.ones_like(lamloc)
        idx3 = cp.arange(cp.sum(loc3))
        while cp.size(idx3):
            u = 0.64 * cp.random.rand(cp.size(idx3))
            v = -0.68 + 1.28 * cp.random.rand(cp.size(idx3))
            k = cp.floor(sqrtlam[idx3] * (v / u) + lamloc[idx3] + 0.5)
            other = k >= 0  # rejection = k < 0
            u = u[other]
            k = k[other]
            u2 = cp.square(u)
            p = sqrtlam[idx3][other] * cp.exp(
                -lamloc[idx3][other] + k * loglam[idx3][other] - gammaln(k + 1)
            )
            acceptance = p > u2
            other[other] = acceptance
            deviate[idx3[other]] = k[acceptance]
            idx3 = idx3[cp.logical_not(other)]
        poissonDev[loc3] = deviate
    return poissonDev


def Simulation(
    objectInput,
    oversamplingRatio: float,
    emitPhotonsNumber: int,
    patternCoverRatio: float = 0,
    patternCoverShape="c",
    patternCoverUpper: float = INF,
    patternCoverLower: float = 0,
    seed=None,
):
    """
    Generating simulated diffraction from the oversampled object with Poisson noise.

    Parameters
    ----------
     - objectInput (cupy.ndarray): Parameter of the poisson distribution.
     - oversamplingRatio (float): The oversampling ratio druing the diffraction pattern simulation.
     - emitPhotonsNumber (int): The number of photons emitted in the diffraction pattern simulation.
     - patternCoverRatio (float): The cover ratio of the low frequency part of the diffraction pattern.
     - patternCoverUpper (float): The upper limit of the number of photons effectively detected by each pixel.
     - patternCoverLower (float): The lower limit of the number of photons effectively detected by each pixel
     - seed (default: None): The seed for the random number generator of Poisson noise.

    Returns
    -------
     - patternNoisedInt (cupy.ndarray): Simulated diffraction pattern with Possion noise.
     - patternNoiseFree (cupy.ndarray): The expectation of an ideal diffraction pattern.
     - patternMask (cupy.ndarray): The mask of pattern for covered pixels.
     - perfectSupport (cupy.ndarray): Perfect tight support of object.
     - SignalNoiseRatio (float): Signal to noise ratio of simulated diffraction pattern.
    """

    realSpace = objectInput / cp.max(objectInput)  # Normalizing
    padding = (
        np.array([cp.shape(realSpace)] * 2) * (oversamplingRatio - 1) / 2
    ).astype("int64")
    # Generating zero-density region
    realSpace = shift(cp.pad(realSpace, padding, "constant"))
    size = cp.shape(realSpace)
    perfectSupport = realSpace > 0
    fourierSpace = fft(realSpace)
    # Shifted pattern intensity
    patternNoiseFree = cp.abs(fourierSpace) ** 2
    patternNoiseFree[patternNoiseFree > patternCoverUpper] = 0
    patternNoiseFree[patternNoiseFree < patternCoverLower] = 0
    if patternCoverRatio != 0:  # Generating missing pixels
        xx, yy = SetFourierGrid(size)
        xx = shift(xx) * 2
        yy = shift(yy) * 2
        if patternCoverShape == "r":
            patternNoiseFree[
                cp.logical_and(
                    cp.abs(xx) < patternCoverRatio, cp.abs(
                        yy) < patternCoverRatio
                ),
            ] = 0
        elif patternCoverShape == "c":
            patternNoiseFree[xx**2 + yy**2 < (patternCoverRatio) ** 2] = 0
        elif patternCoverShape == "d":
            patternNoiseFree[cp.abs(xx) + cp.abs(yy) < patternCoverRatio] = 0
        elif patternCoverShape == "upper":
            num = round(size[0] * size[1] * patternCoverRatio**2)
            mask = cp.ones(size, dtype="bool")
            mask[cp.argpartition(cp.ravel(realSpace), -num)[-num:]] = False
            patternNoiseFree[cp.reshape(mask, size)] = 0
        else:
            raise RuntimeError("Unsupported shape.")
    if emitPhotonsNumber > 0:
        totalIntensity = cp.sum(patternNoiseFree)
        patternNoiseFree = (
            patternNoiseFree * emitPhotonsNumber / totalIntensity
        )  # Setting the number of photons
        patternNoisedInt = PoissonDeviate(
            patternNoiseFree, seed
        )  # Generating poisson noise
    else:
        patternNoisedInt = patternNoiseFree
    patternNoiseFree = cp.round(patternNoiseFree)
    patternMask = patternNoisedInt != 0  # Generating missing pixels mask
    SignalNoiseRatio = 10 * cp.log10(
        cp.mean(patternNoiseFree)
        / cp.mean((cp.abs(patternNoisedInt - patternNoiseFree)))
    )  # Calculating noise ratio
    return (
        patternNoisedInt,
        patternNoiseFree,
        patternMask,
        perfectSupport,
        SignalNoiseRatio,
        realSpace,
        fourierSpace,
    )


class OSS(object):
    """
    General phasing class.

    Using the following functions to set and control the reconstruction process.

    Initialization
    --------------
    - `SetPatternMask(method, **kwargs)`
    - `SetInitRealSpace(method, **kwargs)`
    - `SetInitSupport(method, **kwargs)`

    Phasing Algorithm
    -----------------
    Successive calling can concatenate different algorithms.
    - `SetPhasingMethod(method, **kwargs)`

    Image Output
    ------------
    The following functions must be called to set output format.
    - `SetOutput(path, period, processes)`
    - `SetOutputFourierSpace(h5, mat)`
    - `SetOutputRealSpace(h5, mat, png, tif)`
    - `SetOutputSupport(h5, mat, png, tif)`

    OSS Framework (optional)
    ------------------------
    Call only when needed. If not, OSS Framework will not be used.
    - `SetOSSFramework(period, alpha)`

    Support Update (optional)
    -------------------------
    If the following functions are not call, it will be phased with fixed support.
    - `SetSupportUpdateMethod(self, period, offset)`
    - `SetSupportUpdateBlur(radius, *args)`
    - `SetSupportUpdateThreshold(threshold, *args)`
    - `SetSupportUpdateArea(area, *args)`

    Process Control
    ---------------
    - `Run()`: Once the setup is complete, call `Run()` to start phasing.
    - `Reset()`: Reset all settings (for future GUIs).
    """

    def __init__(
        self,
        patternIntensity,
        patternIntyError=None,
        gpuDevice: int = 0,
        processes: int = cpu_count(),
    ) -> None:
        """
        Import diffraction pattern intensity as the input of the reconstruction.

        Parameters
        ----------
        - patternIntensity (array_like): Diffraction pattern intensity distribution.
        """
        self.Reset()
        cp.cuda.Device(gpuDevice).use()
        self.processes = processes
        patternIntensity = cp.copy(patternIntensity)
        self.patternIntensity = patternIntensity
        self.pattern = cp.sqrt(patternIntensity)
        self.size = cp.shape(patternIntensity)
        if patternIntyError is not None:
            self.FourierModulusProjector = self.FourierModulusProjectorLimit
            self.patternIntyError = cp.copy(patternIntyError)
            self.patternUpper = cp.sqrt(
                self.patternIntensity + self.patternIntyError)
            self.patternLower = cp.sqrt(
                self.patternIntensity - self.patternIntyError)

    def __getstate__(self):
        # Do not pickle pool
        self_dict = self.__dict__.copy()
        del self_dict["pool"]
        return self_dict

    def __setstate__(self, state):
        # Increment instance variable
        self.__dict__.update(state)

    def Reset(self):
        """
        Reset all settings (for future GUIs).
        """
        self.phasingMethod = []
        self.OSSFilterUpdateTemp = NAN
        self.supportUpdatePeriod = INF
        self.supportUpdateMod = INF
        self.supportUpdateBlur = False
        self.supportUpdateThreshold = False
        self.supportUpdateArea = False
        self.outputPeriod = INF
        self.outputMod = INF
        self.outputFourierSpace = False
        self.outputRealSpace = False
        self.outputSupport = False

    #### Initialization ####

    def SetPatternMask(self, method: str, value=0.0, patternMask=None):
        """
        Settings for the mask of the diffraction pattern.

        Parameters
        ----------
        - method (str): The available method include `"match"` and `"custom"`.
            - `"match"`: Generating the mask by matching the values in the diffraction pattern.
            - `"custom"`: Setting the mask manually.
        - value (float): Only available for `mothod="match"`. Pixels with the passed in `value` of
            the diffraction pattern will be set to unavailable pixels.
        - patternMask (cupy.ndarray): Only available for `mothod="custom"`. Pass in a bool array
            of the same shape as the pattern. `True` indicates that the pixel at the corresponding
            position is available, and `False` indicates that it is unavailable.
        """
        if method == "match":
            self.mask = self.patternIntensity != value
        elif method == "custom":
            if patternMask is None:
                raise ValueError('Please input parameter "patternMask".')
            else:
                self.mask = cp.copy(patternMask)
        else:
            raise ValueError("Unsupported method.")

    def SetInitRealSpace(
        self, method: str, zeroLimit=1e-16, seed=None, initRealSpace=None
    ):
        """
        Settings for the initial guess of real space.

        Parameters
        ----------
        - method (str): The available method include `"random"`, `"zero"`, `"inverse"`, and `"custom"`.
            - `"random"`: Generating the real space with random phase and approximately zero amplitude.
            - `"zero"`: Generating the real space with zero phase and amplitude.
            - `"inverse"`: Generating the real space from the inverse Fourier transform of the diffraction
                pattern.
            - `"custom"`: Setting the real space manually.
        - zeroLimit (float): Only available for `mothod="random"`. The approximate limit of zero.
        - seed (default: None): Only available for `mothod="random"`. The seed for the random number
            generator of initial phase.
        - initRealSpace (cupy.ndarray): Only available for `mothod="custom"`. Pass in a array of the same
            shape as the pattern. The type of the array can be either float or complex.
        """
        if method == "random":
            cp.random.seed(seed)
            ang = cp.random.rand(
                self.size[0] * self.size[1]).reshape(self.size)
            self.initRealSpace = zeroLimit * cp.exp(1j * 2 * PI * ang)
        elif method == "zero":
            self.initRealSpace = cp.zeros(self.size)
        elif method == "inverse":
            self.initRealSpace = ifft(self.pattern)
        elif method == "custom":
            if initRealSpace is None:
                raise ValueError('Please input parameter "initRealSpace".')
            else:
                self.initRealSpace = initRealSpace
        else:
            raise ValueError("Unsupported method.")

    def SetInitSupport(
        self, method: str, threshold=0.02, blurRadius=50, initSupport=None
    ):
        """
        Settings for the initial guess of support.

        Parameters
        ----------
        - method (str): The available method include `"auto"` and `"custom"`.
            - `"auto"`: Generating the support by autocorrelation.
            - `"custom"`: Setting the support manually.
        - threshold (float): Only available for `mothod="auto"`. Threshold level of
            autocorrelation maximum to Generate the support.
        - blurRadius (float or int): Only available for `mothod="auto"`. Gaussian blur radius. The
            support generated through autocorrelation is often separated. The continuous support
            can be obtained by using Gaussian blur on the discrete support. The blured support
            update threshold is 0.5.
        - initRealSpace (cupy.ndarray): Only available for `mothod="custom"`. Pass in a bool array
            of the same shape as the pattern. `True` indicates that the pixels at the corresponding
            region is in the support, and `False` indicates that it is outside the support.
        """
        if method == "auto":
            correlation = cp.abs(ifft(self.patternIntensity))
            self.initSupport = correlation > cp.max(correlation) * threshold
            self.initSupport = convolve(
                GaussianKernel(self.size, blurRadius), self.initSupport
            )
            self.initSupport = self.initSupport > cp.max(
                self.initSupport) * 0.5
        elif method == "custom":
            if initSupport is None:
                raise ValueError('Please input parameter "initSupport".')
            else:
                self.initSupport = cp.copy(initSupport)
        else:
            raise ValueError("Unsupported method.")

    #### Images Output ####

    def SetOutputMethod(self, path="./out/", period: int = 1000):
        """
        General settings for output of reconstructed results.

        Parameters
        ----------
        - path (str): Directory to output the reconstruction results. Both relative and absolute
            paths are available. The directory will be created if it does not exist.
        - period (int): Cyclic output period. If set to infinity, only the initial guess and the
            final reconstruction result will be output.
        - processes(int, default is the number of local CPU threads): The number of processes used
            for output. In order to reduce the impact on reconstruction efficiency, non-blocking
            parallel output is carried out by `multiprocessing.Pool.apply_async`.

        Notes
        -----
        This function must be called with the following functions to complete the output settings.
        - `SetOutputFourierSpace(h5, mat)`
        - `SetOutputRealSpace(h5, mat, png, tif)`
        - `SetOutputSupport(h5, mat, png, tif)`
        """
        if path[-1] != "/":
            path = path + "/"
        if not os.path.exists(path):
            warnings.warn(
                "The directory does not exist and will be created.",
                RuntimeWarning,
            )
            os.makedirs(path)
        if len(os.listdir(path)) != 0:
            warnings.warn(
                "The directory is not empty, and the last output will be overwritten.",
                RuntimeWarning,
            )
        self.outputPath = path
        self.outputPeriod = period
        self.outputMod = period - 1

    def SetOutputFourierSpace(self, h5=True, mat=False):
        """
        Settings for output of reconstructed Fourier space.

        Parameters
        ----------
        - h5 (bool): Whether to output shifted .h5 files. The shifted complex Fourier space will be
            stored in the database with the node named `"data"`.
        - mat (bool): Whether to output shifted .mat files. The shifted complex Fourier space will
            be stored in the variable named `"data"`.
        """
        self.outputFourierSpace = True
        self.outputFourierSpaceH5 = h5
        self.outputFourierSpaceMAT = mat

    def SetOutputRealSpace(self, h5=True, mat=False, png=True, tif=False):
        """
        Settings for output of reconstructed real space.

        Parameters
        ----------
        - h5 (bool): Whether to output shifted .h5 files. The shifted complex real space will be
            stored in the HDF5 database with the node named `"data"`.
        - mat (bool): Whether to output shifted .mat files. The shifted complex real space will be
            stored in MATLAB® file with the variable named `"data"`.
        - png (bool): Whether to output centered .png images. The centered amplitude of real space
            will be generated as an 8bit image.
        - tif (bool): Whether to output centered .tif images. The centered amplitude of real space
            will be generated as an 16bit image.
        """
        self.outputRealSpace = True
        self.outputRealSpaceH5 = h5
        self.outputRealSpaceMAT = mat
        self.outputRealSpacePNG = png
        self.outputRealSpaceTIF = tif

    def SetOutputSupport(self, h5=True, mat=False, png=True, tif=False):
        """
        Settings for output of reconstructed support.

        Parameters
        ----------
        - h5 (bool): Whether to output shifted .h5 files. The shifted support with bool type will
            be stored in the HDF5 database with the node named `"data"`.
        - mat (bool): Whether to output shifted .mat files. The shifted support with bool type will
            be stored in MATLAB® file with the variable named `"data"`.
        - png (bool): Whether to output centered .png images. The centered support will be
            generated as an image. White represents the region of support and black represents
            outside of support.
        - tif (bool): Whether to output centered .tif images. The centered support will be
            generated as an image. White represents the region of support and black represents
            outside of support.
        """
        self.outputSupport = True
        self.outputSupportH5 = h5
        self.outputSupportMAT = mat
        self.outputSupportPNG = png
        self.outputSupportTIF = tif

    def Output(self, realSpace, i: int):
        if self.outputFourierSpace:
            fourierSpace = cp.asnumpy(self.fourierSpace)
            fileName = self.outputPath + "{:0>8d}".format(i) + "-FourierSpace"
            if self.outputFourierSpaceH5:
                self.pool.apply_async(
                    fio.WriteH5,
                    (
                        fourierSpace,
                        fileName + ".h5",
                    ),
                    error_callback=errCallBack,
                )
            if self.outputFourierSpaceMAT:
                self.pool.apply_async(
                    fio.WriteMAT,
                    (
                        fourierSpace,
                        fileName + ".mat",
                    ),
                    error_callback=errCallBack,
                )
        if self.outputRealSpace:
            realSpace = cp.asnumpy(realSpace)
            fileName = self.outputPath + "{:0>8d}".format(i) + "-RealSpace"
            if self.outputRealSpaceH5:
                self.pool.apply_async(
                    fio.WriteH5,
                    (
                        realSpace,
                        fileName + ".h5",
                    ),
                    error_callback=errCallBack,
                )
            if self.outputRealSpaceMAT:
                self.pool.apply_async(
                    fio.WriteMAT,
                    (
                        realSpace,
                        fileName + ".mat",
                    ),
                    error_callback=errCallBack,
                )
            if self.outputRealSpacePNG:
                self.pool.apply_async(
                    fio.WritePNG,
                    (
                        realSpace,
                        fileName + ".png",
                    ),
                    error_callback=errCallBack,
                )
            if self.outputRealSpaceTIF:
                self.pool.apply_async(
                    fio.WriteTIF,
                    (
                        realSpace,
                        fileName + ".tif",
                    ),
                    error_callback=errCallBack,
                )
        if self.outputSupport:
            support = cp.asnumpy(self.support)
            fileName = self.outputPath + "{:0>8d}".format(i) + "-Support"
            if self.outputSupportH5:
                self.pool.apply_async(
                    fio.WriteH5,
                    (
                        support,
                        fileName + ".h5",
                    ),
                    error_callback=errCallBack,
                )
            if self.outputSupportMAT:
                self.pool.apply_async(
                    fio.WriteMAT,
                    (
                        support,
                        fileName + ".mat",
                    ),
                    error_callback=errCallBack,
                )
            if self.outputSupportPNG:
                self.pool.apply_async(
                    fio.WritePNG,
                    (
                        support,
                        fileName + ".png",
                    ),
                    error_callback=errCallBack,
                )
            if self.outputSupportTIF:
                self.pool.apply_async(
                    fio.WriteTIF,
                    (
                        support,
                        fileName + ".tif",
                    ),
                    error_callback=errCallBack,
                )

    #### OSS Framework ####

    def SetOSSFramework(self, alpha=cp.arange(5, 0, -0.5), period: int = INF):
        """
        Settings for OSS Framework. OSS Framework applying spatial frequency filters to the pixels
        outside the support at different stages of the iterative process, reducing the oscillations
        in the reconstruction.

        Parameters
        ----------
        - period (int): The period of changing fillter, i.e. period of changing alpha.
        - alpha (array_like): The standard deviation of Gaussian window in different stages.
        """
        self.OSSFilterUpdateTemp = period
        self.filterLen = len(alpha)
        self.filter = GaussianWindow(self.size, cp.asarray(alpha))

    def OSSActivation(self):
        self.OSSFilter = self.OSSFramework
        self.filterIdx = 0
        self.OSSFilterUpdatePeriod = self.OSSFilterUpdateTemp
        self.OSSFilterUpdateMod = self.OSSFilterUpdateTemp - 1

    def OSSFramework(self, realSpace):
        """
        References
        ----------
        J. A. Rodriguez et al., J. Appl. Crystallogr. 46, 312 (2013).\
            https://doi.org/10.1107/S0021889813002471
        """
        realSpace[cp.logical_not(self.support)] = ifft(
            fft(realSpace) * self.filter[self.filterIdx]
        )[cp.logical_not(self.support)]
        return realSpace

    def OSSFilterUpdate(self):
        if self.filterIdx < self.filterLen:
            self.filterIdx = self.filterIdx + 1

    #### Support Update ####

    def SetSupportUpdateMethod(self, period=50, offset=0):
        """
        Settings for support update. If this function is not called, the support is fixed
        throughout the phasing process. If call, Support is updated with a specific iteration
        period.

        Parameters
        ----------
        - period (int): The period of support update.
        - offset (int): If the period of support update is `t` and the offset is `s`, the update
            will be carried out when `n=k*t+s`, where `k` is a natural number.

        Notes
        -----
        This function must be called with any of the following functions to complete settings.
        - `SetSupportUpdateThreshold(threshold, *args)`
        - `SetSupportUpdateArea(area, *args)`

        Another function `SetSupportUpdateBlur(radius, *args)` can optionally be called.
        """
        self.supportUpdatePeriod = period
        self.supportUpdateMod = offset

    def SetSupportUpdateBlur(self, radius: float, radiusMin=NAN, dropRate=0.0):
        """
        Settings for Gaussian blur on support update. Gaussian blur can smooth the object and avoid
        the lack of support caused by setting the area too small or the threshold too agressive.
        However, a large blur radius can also lead to looser support. The solution is to make the
        blur radius drop by the same proportion with each support update.

        Parameters
        ----------
        - radius (float): Gaussian blur radius, in pixels.
        - radiusMin (float, optional): The minimum Gaussian blur radius can be dropped to.
        - dropRate (float, optional): The drop rate proportion of Gaussian blur radius. If not set,
            the Gaussian blur radius will not change with each support update.
        """
        self.supportUpdateBlur = True
        self.radiusInit = radius
        self.radiusDrop = 1 - dropRate
        self.radiusMin = radiusMin

    def SetSupportUpdateThreshold(
        self, threshold: float, thresholdMin=NAN, dropRate=0.0
    ):
        """
        Settings for Threshold algorithm on support update. Threshold algorithm compares the value
        of each pixel with the highest value. If the ratio of a value of pixel to the highest
        value is higher than the threshold passed in, the pixel will be included in the support.
        As phasing proceeds, support tightens and the edges of the object become clearer, droping
        the threshold by the same proportion could avoid lack of support.

        Parameters
        ----------
        - threshold (float): The minimum ratio of the pixel value within the support to the highest
            value. Must be set between 0 and 1.
        - thresholdMin (float, optional): The minimum threshold can be dropped to.
        - dropRate (float, optional): The proportion of drop rate for threshold. If not set, the
            threshold will not change with each support update.

        Notes
        -----
        When used together with the area algorithm, i.e., when `SetSupportUpdateArea()` is called,
        the support will be the union of support produced by two algorithms.
        """
        self.supportUpdateThreshold = True
        self.thresholdInit = threshold
        self.thresholdDrop = 1 - dropRate
        self.thresholdMin = thresholdMin

    def SetSupportUpdateArea(self, area: float, areaMin=NAN, dropRate=0.0):
        """
        Settings for Area algorithm on support update. Area algorithm choose the most intense part
        of pixels as support. The size of the most intense part of pixels called area. The passed
        in area is given by units of the whole field. It prevents excessive shrinkage of support.
        A large value of area is often set at the beginning of phasing to search losser support,
        and then dropped to tighten support.

        Parameters
        ----------
        - area (float): The proportion of the number of pixels in support within the whole field.
            Must be set between 0 and 1.
        - areaMin (float, optional): The minimum area can be dropped to.
        - dropRate (float, optional): The proportion of drop rate for area. If not set, the area
            will not change with each support update.

        Notes
        -----
        When used together with the area algorithm, i.e., when `SetSupportUpdateThreshold()` is
        called, the support will be the union of support produced by two algorithms.
        """
        self.supportUpdateArea = True
        self.areaInit = area
        self.areaDrop = 1 - dropRate
        self.areaMin = areaMin

    def SupportUpdate(self, realSpace):
        realSpace = cp.abs(realSpace)
        if self.supportUpdateBlur:
            realSpace = self.SupportUpdateBlur(realSpace)
        if self.supportUpdateThreshold & self.supportUpdateArea:
            self.support = cp.logical_or(
                self.SupportUpdateThreshold(realSpace),
                self.SupportUpdateArea(realSpace),
            )
        elif self.supportUpdateThreshold:
            self.support = self.SupportUpdateThreshold(realSpace)
        elif self.supportUpdateArea:
            self.support = self.SupportUpdateArea(realSpace)
        else:
            raise RuntimeError("The support update method is not set")

    def SupportUpdateBlur(self, realSpace):
        """
        References
        ----------
        S. Marchesini et al., Phys. Rev. B 68, 140101(R) (2003).\
            https://doi.org/10.1103/physrevb.68.140101
        """
        if self.radius > self.radiusMin:
            self.radius = self.radiusDrop * self.radius
        return convolve(GaussianKernel(self.size, self.radius), realSpace)

    def SupportUpdateThreshold(self, realSpace):
        """
        References
        ----------
        S. Marchesini et al., Phys. Rev. B 68, 140101(R) (2003).\
            https://doi.org/10.1103/physrevb.68.140101
        """
        if self.threshold > self.thresholdMin:
            self.threshold = self.thresholdDrop * self.threshold
        return realSpace > cp.max(realSpace) * self.threshold

    def SupportUpdateArea(self, realSpace):
        """
        References
        ----------
        F. R. N. C. Maia et al., J. Appl. Crystallogr. 43, 1535 (2010).\
            https://doi.org/10.1107/s0021889810036083
        """
        if self.area > self.areaMin:
            self.area = self.areaDrop * self.area
        size = self.size[0] * self.size[1]
        num = round(size * self.area)
        support = cp.zeros(size, dtype="bool")
        support[cp.argpartition(cp.ravel(realSpace), -num)[-num:]] = True
        return cp.reshape(support, self.size)

    #### Phasing Algorithm ####

    def SetPhasingMethod(self, method: str, iterations: int, **kwargs):
        """
        Add reconstruction algorithm to phasing process. Each call to this function will add a
        stage of reconstruction algorithm, which concatenates different algorithms and
        parameters.

        Parameters
        ----------
        - method (str): Reconstruction algorithm. The available reconstruction algorithm include\
        "ER"`, `"SF"`, `"HIO"`, `"ERHIO"`, `"ASR"`, `"HPR"`, `"RAAR"`, and `"DM"`.
            - `"ER"`: P_s P_m.
            - `"SF"`: R_s P_m.
            - `"HIO"`: if in support: P_m else: (I - beta P_m). Need to pass in the parameter `beta`.
            - `"ERHIO"`: Alternate the ER and HIO algorithms periodically. Need to pass in parameter
                `beta` (same as HIO), `HIO` (the number of HIO iterations per period), and `ER` (the
                number of ER iterations per period).
            - `"ASR"`: (R_s R_m + I) / 2.
            - `"HPR"`: (R_s (R_m + (beta - 1) P_m) + I + (1 - beta) P_m) / 2. Need to pass in the
                parameter `beta`.
            - `"RAAR"`: beta (R_s R_m + I) / 2 + (1 - beta) P_m. Need to pass in the parameter `beta`.
            - `"DM"`:  I + beta (P_s ((1 + gammaS) P_m - gammaS I) - P_m ((1 + gammaM) P_s - gammaM I)).
                Need to pass in the parameter `beta`, `gammaS`, and `gammaM`. If I don't pass in
                `gammaS`, `gammaS` will be set to `-1 / beta`. If I don't pass in `gammaM`, `gammaM`
                will be set to `1 / beta`.
        """
        phasingMethod = {"method": method, "iterations": iterations}
        try:
            beta = kwargs["beta"]
        except:
            beta = 0.9
        if method in ["HIO", "HPR", "RAAR"]:
            phasingMethod["beta"] = beta
        elif method == "ERHIO":
            phasingMethod["beta"] = beta
            phasingMethod["ERiterations"] = kwargs["ER"]
            phasingMethod["HIOiterations"] = kwargs["HIO"]
            phasingMethod["ERHIOperiod"] = kwargs["ER"] + kwargs["HIO"]
        elif method == "DM":
            phasingMethod["beta"] = beta
            try:
                phasingMethod["gammaS"] = kwargs["gammaS"]
                phasingMethod["gammaM"] = kwargs["gammaM"]
            except:
                phasingMethod["gammaS"] = -1 / beta
                phasingMethod["gammaM"] = 1 / beta
        elif not (method in ["ER", "SF", "ASR"]):
            raise ValueError("Unsupported phasing method.")
        self.phasingMethod.append(phasingMethod)

    def FourierModulusProjectorLimit(self, realSpace):
        fourierSpace = fft(realSpace)
        self.fourierSpace = cp.copy(fourierSpace)
        fourierSpaceMask = cp.abs(fourierSpace[self.mask])
        upper = fourierSpaceMask > self.patternUpperMask
        lower = fourierSpaceMask < self.patternLowerMask
        fourierSpaceMask[upper] = self.patternUpperMask[upper]
        fourierSpaceMask[lower] = self.patternLowerMask[lower]
        fourierSpace[self.mask] = fourierSpaceMask * cp.exp(
            1j * cp.angle(fourierSpace[self.mask])
        )
        return ifft(fourierSpace)

    def FourierModulusProjector(self, realSpace):
        fourierSpace = fft(realSpace)
        self.fourierSpace = cp.copy(fourierSpace)
        fourierSpace[self.mask] = self.patternMask * cp.exp(
            1j * cp.angle(fourierSpace[self.mask])
        )
        return ifft(fourierSpace)

    def RealSupportProjector(self, realSpace):
        realSpace[cp.logical_not(self.support)] = 0
        return realSpace

    def FourierModulusReflector(self, realSpace):
        return 2 * self.FourierModulusProjector(realSpace) - realSpace

    def RealSupportReflector(self, realSpace):
        return 2 * self.RealSupportProjector(realSpace) - realSpace

    def HIOFunctionProjector(self, afterFMPj, realSpace):
        outSupport = cp.logical_not(self.support)
        afterFMPj[outSupport] = (
            realSpace[outSupport] - self.beta * afterFMPj[outSupport]
        )
        return afterFMPj

    def ERSovler(self, realSpace):
        """
        References
        ----------
        A. Levi and H. Stark, J. Opt. Soc. Am. A 1, 932 (1984).\
            https://doi.org/10.1364/josaa.1.000932
        """
        realSpace = self.FourierModulusProjector(realSpace)
        realSpace = self.RealSupportProjector(realSpace)
        return realSpace

    def SFSovler(self, realSpace):
        """
        References
        ----------
        J. P. Abrahams and A. G. W. Leslie, Acta Crystallogr. D 52, 30 (1996).\
            https://doi.org/10.1107/S0907444995008754
        """
        realSpace = self.FourierModulusProjector(realSpace)
        realSpace = self.RealSupportReflector(realSpace)
        return realSpace

    def HIOSovler(self, realSpace):
        """
        References
        ----------
        J. R. Fienup, Opt. Lett. 3, 27 (1978).\
            https://doi.org/10.1364/ol.3.000027
        """
        afterFMPj = self.FourierModulusProjector(realSpace)
        realSpace = self.HIOFunctionProjector(afterFMPj, realSpace)
        return realSpace

    def ASRSovler(self, realSpace):
        """
        References
        ----------
        H. H. Bauschke, P. L. Combettes, and D. R. Luke, J. Opt. Soc. Am. A 19, 1334 (2002).\
            https://doi.org/10.1364/josaa.19.001334
        """
        reflector = self.RealSupportReflector(
            self.FourierModulusReflector(realSpace))
        return (reflector + realSpace) / 2

    def HPRSovler(self, realSpace):
        """
        References
        ----------
        H. H. Bauschke, P. L. Combettes, and D. R. Luke, J. Opt. Soc. Am. A 20, 1025 (2003).\
            https://doi.org/10.1364/josaa.20.001025
        """
        projector = (1 - self.beta) * self.FourierModulusProjector(realSpace)
        reflector = self.RealSupportReflector(
            self.FourierModulusReflector(realSpace) - projector
        )
        return (reflector + realSpace + projector) / 2

    def RAARSovler(self, realSpace):
        """
        References
        ----------
        D. R. Luke, Inverse Prob. 21, 37 (2004).\
            https://doi.org/10.1088/0266-5611/21/1/004
        """
        projector = (1 - self.beta) * self.FourierModulusProjector(realSpace)
        reflector = self.RealSupportReflector(
            self.FourierModulusReflector(realSpace))
        return self.beta * (reflector + realSpace) / 2 + projector

    def DMSovler(self, realSpace):
        """
        References
        ----------
        V. Elser, J. Opt. Soc. Am. A 20, 40 (2003).\
            https://10.1364/josaa.20.000040
        """
        RSestimate = self.RealSupportProjector(
            (1 + self.gammaS) * self.FourierModulusProjector(realSpace)
            - self.gammaS * realSpace
        )
        FMestimate = self.FourierModulusProjector(
            (1 + self.gammaM) * self.RealSupportProjector(realSpace)
            - self.gammaM * realSpace
        )
        return self.beta * (RSestimate - FMestimate) / 2 + realSpace

    def Iterator(self, realSpace):
        for i in tqdm(range(self.iterations)):
            realSpace = self.Sovler(realSpace)
            realSpace = self.OSSFilter(realSpace)
            self.UpdateAndOutput(realSpace, i)
        return realSpace

    def ERIterator(self, realSpace):
        for i in tqdm(range(self.iterations)):
            realSpace = self.ERSovler(realSpace)
            if i % self.outputPeriod == self.outputMod:
                self.Output(realSpace, self.cumsum + i)
        return realSpace

    def ERHIOIterator(self, realSpace):
        for i in tqdm(range(self.iterations)):
            if i % self.ERHIOperiod < self.HIOiterations:
                realSpace = self.HIOSovler(realSpace)
            else:
                realSpace = self.ERSovler(realSpace)
            realSpace = self.OSSFilter(realSpace)
            self.UpdateAndOutput(realSpace, i)
        return realSpace

    def UpdateAndOutput(self, realSpace, i):
        if (1 + i) % self.OSSFilterUpdatePeriod == 0:
            self.OSSFilterUpdate()
        if (self.cumsum + i) % self.supportUpdatePeriod == self.supportUpdateMod:
            self.SupportUpdate(realSpace)
        if (self.cumsum + i) % self.outputPeriod == 0:
            self.Output(realSpace, self.cumsum + i)

    def PhasingIterator(self, realSpace):
        self.patternMask = self.pattern[self.mask]
        if self.FourierModulusProjector == self.FourierModulusProjectorLimit:
            self.patternUpperMask = self.patternUpper[self.mask]
            self.patternLowerMask = self.patternLower[self.mask]
        self.fourierSpace = ifft(realSpace)
        for method in self.phasingMethod:
            for item in method.items():
                exec("self." + item[0] + "=item[1]")
            if self.method == "ER":
                realSpace = self.ERIterator(realSpace)
            elif self.method == "ERHIO":
                realSpace = self.ERHIOIterator(realSpace)
            else:
                self.Sovler = eval("self." + self.method + "Sovler")
                realSpace = self.Iterator(realSpace)
            self.cumsum = self.cumsum + self.iterations
        self.Output(realSpace, self.cumsum - 1)
        return realSpace

    def Ready(self):
        self.cumsum = 1
        self.support = cp.copy(self.initSupport)
        self.OSSFilterUpdatePeriod = INF
        self.OSSFilterUpdateMod = INF
        self.OSSFilter = lambda realSpace: realSpace
        if self.supportUpdateBlur:
            self.radius = self.radiusInit
        if self.supportUpdateThreshold:
            self.threshold = self.thresholdInit
        if self.supportUpdateArea:
            self.area = self.areaInit

    def Run(self):
        """
        Once the setup is complete, call `Run()` to start phasing.
        """
        self.pool = Pool(processes=self.processes)
        self.Ready()
        if not np.isnan(self.OSSFilterUpdateTemp):
            self.OSSActivation()
        self.PhasingIterator(self.initRealSpace)
        self.pool.close()
        self.pool.join()


class HMG(OSS):
    """
    General phasing class.

    Using the following functions to set and control the reconstruction process.

    Initialization
    --------------
    - `SetPatternMask(method, **kwargs)`
    - `SetInitRealSpace(method, **kwargs)`
    - `SetInitSupport(method, **kwargs)`

    Multigrid
    ---------
    Set the number of levels for multiple grids.
    - `SetMultigridMethod(num)`

    Phasing Algorithm
    -----------------
    Successive calling can concatenate different algorithms.
    - `SetPhasingMethod(method, girdLevel, **kwargs)`

    Image Output
    ------------
    The following functions must be called to set output format.
    - `SetOutput(path, period, processes)`
    - `SetOutputFourierSpace(h5, mat)`
    - `SetOutputRealSpace(h5, mat, png, tif)`
    - `SetOutputSupport(h5, mat, png, tif)`

    OSS Framework (optional)
    ------------------------
    Call only when needed. If not, OSS Framework will not be used.
    - `SetOSSFramework(period, alpha)`

    Support Update (optional)
    -------------------------
    If the following functions are not call, it will be phased with fixed support.
    - `SetSupportUpdateMethod(self, period, offset)`
    - `SetSupportUpdateBlur(radius, *args)`
    - `SetSupportUpdateThreshold(threshold, *args)`
    - `SetSupportUpdateArea(area, *args)`

    Process Control
    ---------------
    - `Run()`: Once the setup is complete, call `Run()` to start phasing.
    - `Reset()`: Reset all settings (for future GUIs).
    """

    def __init__(
        self,
        patternIntensity,
        patternIntyError=None,
        gpuDevice: int = 0,
        processes: int = cpu_count(),
    ) -> None:
        super().__init__(patternIntensity, patternIntyError, gpuDevice, processes)
        self.gridLevels = 1
        self.multigridMethod = [[]]

    def SetMultigridMethod(self, gridLevels: int, interpMode="bilinear"):
        self.gridLevels = gridLevels
        self.multigridMethod = [[] for i in range(gridLevels)]
        if interpMode in ["nearest", "bilinear", "bicubic", "area"]:
            self.interpMode = interpMode
        else:
            raise ValueError("Unsupported interpolation method.")

    def SetPhasingMethod(
        self, method: str, iterations: int, gridLevel: int = None, **kwargs
    ):
        """
        Add reconstruction algorithm to phasing process for different grid levels. Each call to this
        function will add a stage of reconstruction algorithm, which concatenates different algorithms
        and parameters.

        Parameters
        ----------
        - method (str): Reconstruction algorithm. The available reconstruction algorithm include\
        "ER"`, `"SF"`, `"HIO"`, `"ERHIO"`, `"ASR"`, `"HPR"`, `"RAAR"`, and `"DM"`.
            - `"ER"`: P_s P_m.
            - `"SF"`: R_s P_m.
            - `"HIO"`: if in support: P_m else: (I - beta P_m). Need to pass in the parameter `beta`.
            - `"ERHIO"`: Alternate the ER and HIO algorithms periodically. Need to pass in parameter
                `beta` (same as HIO), `HIO` (the number of HIO iterations per period), and `ER` (the
                number of ER iterations per period).
            - `"ASR"`: (R_s R_m + I) / 2.
            - `"HPR"`: (R_s (R_m + (beta - 1) P_m) + I + (1 - beta) P_m) / 2. Need to pass in the
                parameter `beta`.
            - `"RAAR"`: beta (R_s R_m + I) / 2 + (1 - beta) P_m. Need to pass in the parameter `beta`.
            - `"DM"`:  I + beta (P_s ((1 + gammaS) P_m - gammaS I) - P_m ((1 + gammaM) P_s - gammaM I)).
                Need to pass in the parameter `beta`, `gammaS`, and `gammaM`. If I don't pass in
                `gammaS`, `gammaS` will be set to `-1 / beta`. If I don't pass in `gammaM`, `gammaM`
                will be set to `1 / beta`.
        - gridLevel (int or tuple[int]): If `None`, the algorithm is added to all levels of the grid.\
            The coarsest grid is defined as level 0. Use tuples to add the same algorithm to multiple
            levels of grid simultaneously.
        """
        if gridLevel is None:
            self.phasingMethod = []
            super().SetPhasingMethod(method, iterations, **kwargs)
            for i in range(self.gridLevels):
                self.multigridMethod[i].append(self.phasingMethod[0])
        elif isinstance(gridLevel, (list, tuple)):
            for i in gridLevel:
                self.phasingMethod = self.multigridMethod[i]
                super().SetPhasingMethod(method, iterations, **kwargs)
        elif isinstance(gridLevel, int) and (0 <= gridLevel < self.gridLevels):
            self.phasingMethod = self.multigridMethod[gridLevel]
            super().SetPhasingMethod(method, iterations, **kwargs)
        else:
            raise ValueError(
                "GridLevel must be an integer and between 0 and the number of multigrid levels (inclusives)"
            )

    def Restriction(self, data, mode):
        data = dl.from_dlpack(data.toDlpack())
        data = nn.interpolate(
            data[cp.newaxis, cp.newaxis, :],
            scale_factor=0.5,
            mode=mode,
            recompute_scale_factor=False,
        )[0, 0]
        return cp.fromDlpack(dl.to_dlpack(data))

    def Prolongation(self, data, mode):
        data = dl.from_dlpack(data.toDlpack())
        data = nn.interpolate(
            data[cp.newaxis, cp.newaxis, :],
            scale_factor=2,
            mode=mode,
            align_corners=False if mode in ["bilinear", "bicubic"] else None,
            recompute_scale_factor=False,
        )[0, 0]
        return cp.fromDlpack(dl.to_dlpack(data))

    def FourierRestriction(self, data):
        size = cp.shape(data)
        data = ishift(data)
        data = data[size[0] // 4: -size[0] // 4, size[1] // 4: -size[1] // 4]
        data = shift(data)
        if data.dtype == "bool":
            return data
        else:
            return data / 4

    def MultiGridRestriction(self, data):
        if data.dtype == "bool":
            support = self.Restriction(data.astype("float16"), "area")
            return support >= 0.5
        else:
            data = ishift(data)
            mag = cp.abs(data)
            ang = cp.angle(data)
            mag = self.Restriction(mag, "area")
            ang = self.Restriction(ang, "area")
            return shift(mag * cp.exp(1j * ang))

    def MultiGridProlongation(self, data):
        if data.dtype == "bool":
            support = self.Prolongation(data.astype("float16"), "nearest")
            return support.astype("bool")
        else:
            data = ishift(data)
            mag = cp.abs(data)
            ang = cp.angle(data)
            mag = self.Prolongation(mag, self.interpMode)
            ang = self.Prolongation(ang, self.interpMode)
            return shift(mag * cp.exp(1j * ang))

    def HalfMultiGrid(self, gridLevel, realSpace):
        """
        References
        ----------
        U. Trottenberg et al., Multigrid (Elsevier Science \& Techn., 2000) p.47
        """
        if not os.path.exists(self.outputPath + str(gridLevel) + "/"):
            os.makedirs(self.outputPath + str(gridLevel) + "/")
        # If it is not in the coarsest grid level, continue restricting.
        if gridLevel < self.gridLevels - 1:
            # Restricion
            realSpace = self.MultiGridRestriction(realSpace)
            if self.supportUpdatePeriod == INF:
                support = cp.copy(self.support)
            else:
                self.radius = self.radius / 2
                self.radiusMin = self.radiusMin / 2
            self.support = self.MultiGridRestriction(self.support)
            mask = cp.copy(self.mask)
            self.mask = self.FourierRestriction(self.mask)
            pattern = cp.copy(self.pattern)
            self.pattern = self.FourierRestriction(self.pattern)
            if self.FourierModulusProjector == self.FourierModulusProjectorLimit:
                patternUpper = cp.copy(self.patternUpper)
                patternLower = cp.copy(self.patternLower)
                self.patternUpper = self.FourierRestriction(self.patternUpper)
                self.patternLower = self.FourierRestriction(self.patternLower)
            # Performing next coarse-grid
            realSpace = self.HalfMultiGrid(gridLevel + 1, realSpace)
            # Prolongation
            realSpace = self.MultiGridProlongation(realSpace)
            if self.supportUpdatePeriod == INF:
                self.support = cp.copy(support)
            else:
                self.support = self.MultiGridProlongation(self.support)
                self.radius = self.radius * 2
                self.radiusMin = self.radiusMin * 2
            self.mask = cp.copy(mask)
            self.pattern = cp.copy(pattern)
            if self.FourierModulusProjector == self.FourierModulusProjectorLimit:
                self.patternUpper = cp.copy(patternUpper)
                self.patternLower = cp.copy(patternLower)
        # Phasing
        if gridLevel == 0 and not np.isnan(self.OSSFilterUpdateTemp):
            self.OSSActivation()
        self.size = cp.shape(self.pattern)
        self.outputPath = self.outputPath + str(gridLevel) + "/"
        self.phasingMethod = self.multigridMethod[gridLevel]
        realSpace = self.PhasingIterator(realSpace)
        self.outputPath = self.outputPath[:-2]
        return realSpace

    def Run(self):
        self.pool = Pool(processes=self.processes)
        self.Ready()
        self.HalfMultiGrid(0, self.initRealSpace)
        self.pool.close()
        self.pool.join()


class Phasing(HMG):
    """
    General phasing class.

    Using the following functions to set and control the reconstruction process.

    Initialization
    --------------
    - `SetPatternMask(method, **kwargs)`
    - `SetInitRealSpace(method, **kwargs)`
    - `SetInitSupport(method, **kwargs)`

    Multigrid
    ---------
    Set the number of levels for multiple grids.
    - `SetMultigridMethod(num)`

    Phasing Algorithm
    -----------------
    Successive calling can concatenate different algorithms.
    - `SetPhasingMethod(method, girdLevel, **kwargs)`

    Image Output
    ------------
    The following functions must be called to set output format.
    - `SetOutput(path, period, processes)`
    - `SetOutputFourierSpace(h5, mat)`
    - `SetOutputRealSpace(h5, mat, png, tif)`
    - `SetOutputSupport(h5, mat, png, tif)`

    OSS Framework (optional)
    ------------------------
    Call only when needed. If not, OSS Framework will not be used.
    - `SetOSSFramework(period, alpha)`

    Support Update (optional)
    -------------------------
    If the following functions are not call, it will be phased with fixed support.
    - `SetSupportUpdateMethod(self, period, offset)`
    - `SetSupportUpdateBlur(radius, *args)`
    - `SetSupportUpdateThreshold(threshold, *args)`
    - `SetSupportUpdateArea(area, *args)`

    Process Control
    ---------------
    - `Run([start, ]stop, **kwarg)`: Once the setup is complete, call `Run()` to start phasing.
    - `Reset()`: Reset all settings (for future GUIs).
    """

    def __init__(
        self,
        patternIntensity,
        patternIntyError=None,
        gpuDevice: int = 0,
        processes: int = cpu_count(),
    ) -> None:
        super().__init__(patternIntensity, patternIntyError, gpuDevice, processes)

    def SetInitRealSpace(
        self, method: str, zeroLimit=1e-16, seed=None, initRealSpace=None
    ):
        if method == "random" and type(seed) != int:
            self.randomList = True
            self.zeroLimit = zeroLimit
            self.seedList = seed
        else:
            self.randomList = False
            super().SetInitRealSpace(method, zeroLimit, seed, initRealSpace)

    def ImportResult(
        self, path: str = None, i: int = None, realSpaceList=None, supportList=None
    ):
        if path is None:
            self.realSpaceList = cp.asarray(realSpaceList)
            self.supportList = cp.asarray(supportList)
            dirList = np.arange(cp.shape(self.realSpaceList)[0])
        else:
            if path[-1] != "/":
                path = path + "/"
            self.outputPath = path
            dirList = sorted(os.listdir(path))
            for dir in dirList[:]:
                if not os.path.isdir(path + dir):
                    dirList.remove(dir)
            self.realSpaceList = []
            self.supportList = []
            for dir in dirList[:]:
                try:
                    realSpace = fio.ReadFile(
                        path + dir + "/0/" +
                        "{:0>8d}".format(i) + "-RealSpace.h5",
                        "data",
                    )
                    self.realSpaceList.append(realSpace)
                    support = fio.ReadFile(
                        path + dir + "/0/" +
                        "{:0>8d}".format(i) + "-Support.h5", "data"
                    )
                    self.supportList.append(support)
                except:
                    dirList.remove(dir)
            self.realSpaceList = cp.asarray(self.realSpaceList)
            self.supportList = cp.asarray(self.supportList)
        self.list = pd.DataFrame({"dir": dirList})

    def SynchronizeMag(self):
        """
        Synchronize the magnitude of multiple reconstruction results.
        """
        try:
            model = cp.abs(self.realSpaceMod)
            s = 0
        except:
            cor = correlate(
                cp.abs(self.realSpaceList[0]),
                GaussianWindow(self.size, min(self.size) / 2),
            )
            shift = np.unravel_index(int(cp.argmax(cor)), self.size)
            self.supportList[0] = cp.roll(
                self.supportList[0], shift, axis=(0, 1))
            self.realSpaceList[0] = cp.roll(
                self.realSpaceList[0], shift, axis=(0, 1))
            model = cp.abs(self.realSpaceList[0])
            if cp.sum(model[: self.size[0] // 2]) > cp.sum(model[self.size[0] // 2:]):
                self.supportList[0] = cp.rot90(self.supportList[0], 2)
                self.realSpaceList[0] = cp.rot90(
                    cp.conj(self.realSpaceList[0]), 2)
                model = cp.rot90(model, 2)
            s = 1
        for i in range(s, self.len):
            mag = cp.abs(self.realSpaceList[i])
            cor = correlate(mag, model)
            cort = correlate(cp.rot90(mag, 2), model)
            if cp.max(cor) < cp.max(cort):
                cor = cort
                self.supportList[i] = cp.rot90(self.supportList[i], 2)
                self.realSpaceList[i] = cp.rot90(
                    cp.conj(self.realSpaceList[i]), 2)
            shift = np.unravel_index(int(cp.argmax(cor)), self.size)
            self.supportList[i] = cp.roll(
                self.supportList[i], shift, axis=(0, 1))
            self.realSpaceList[i] = cp.roll(
                self.realSpaceList[i], shift, axis=(0, 1))

    def SynchronizeArg(self):
        """
        Synchronize the argument (phase) of multiple reconstruction results.
        """
        try:
            model = cp.conj(self.realSpaceMod)
            s = 0
        except:
            phase = -cp.angle(cp.nansum(self.realSpaceList[0] ** 2)) / 2
            self.realSpaceList[0] = self.realSpaceList[0] * cp.exp(1j * phase)
            if cp.sum(cp.real(self.realSpaceList[0])) < 0:
                self.realSpaceList[0] = -self.realSpaceList[0]
            model = cp.conj(self.realSpaceList[0])
            s = 1
        for i in range(s, self.len):
            phase = -cp.angle(cp.nansum(model * self.realSpaceList[i]))
            self.realSpaceList[i] = self.realSpaceList[i] * cp.exp(1j * phase)

    def OutputAve(self):
        if self.outputRealSpace:
            list = ["realSpaceAve", "realSpaceStd"]
            for i in range(2):
                realSpace = cp.asnumpy(eval("self." + list[i]))
                fileName = self.outputPath + "########-R" + list[i][1:]
                if self.outputRealSpaceH5:
                    fio.WriteH5(realSpace, fileName + ".h5")
                if self.outputRealSpaceMAT:
                    fio.WriteMAT(realSpace, fileName + ".mat")
                if self.outputRealSpacePNG:
                    fio.WritePNG(realSpace, fileName + ".png")
                if self.outputRealSpaceTIF:
                    fio.WriteTIF(realSpace, fileName + ".tif")
        if self.outputSupport:
            list = ["supportAve", "supportLim"]
            for i in range(2):
                support = cp.asnumpy(eval("self." + list[i]))
                fileName = self.outputPath + "########-S" + list[i][1:]
                if self.outputSupportH5:
                    fio.WriteH5(support, fileName + ".h5")
                if self.outputSupportMAT:
                    fio.WriteMAT(support, fileName + ".mat")
                if self.outputSupportPNG:
                    fio.WritePNG(support, fileName + ".png")
                if self.outputSupportTIF:
                    fio.WriteTIF(support, fileName + ".tif")

    def Run(self, *times, ave=False, RFupper=0.14, SPlower=0.8):
        """
        Once the setup is complete, call `Run([start, ]stop, **kwarg)` to start phasing.

        Parameters
        ----------
        - start (int): Start number. The default value is 0.
        - stop (int): End number.
        - ave (Boolean): Whether to average multiple results.
        - RFupper (float): The upper Fourier error limit for successful reconstruction.
        - SPlower (float): The lower limit to generate support from averaged support.
        """
        self.pool = Pool(processes=self.processes)
        outputPath = self.outputPath
        try:
            len = times[1] - times[0]
        except:
            len = times[0]
        if ave:
            self.realSpaceList = np.empty((len,) + self.size, dtype="complex")
            self.supportList = np.empty((len,) + self.size, dtype="bool")
        dirList = []
        for i in range(*times):
            print("i={:0>4d}".format(i))
            dirList.append("i={:0>4d}".format(i))
            self.outputPath = outputPath + "{:0>4d}".format(i) + "/"
            if not os.path.exists(self.outputPath):
                os.makedirs(self.outputPath)
            if self.randomList:
                if self.seedList is not None:
                    cp.random.seed(self.seedList[i])
                ang = cp.random.rand(
                    self.size[0] * self.size[1]).reshape(self.size)
                self.initRealSpace = self.zeroLimit * cp.exp(1j * 2 * PI * ang)
            self.Ready()
            realSpace = self.HalfMultiGrid(0, self.initRealSpace)
            if ave:
                self.realSpaceList[i - times[0]] = cp.asnumpy(realSpace)
                self.supportList[i - times[0]] = cp.asnumpy(self.support)
        self.pool.close()
        self.pool.join()
        self.outputPath = outputPath
        if ave:
            self.list = pd.DataFrame({"dir": dirList})
            self.realSpaceList = cp.asarray(self.realSpaceList)
            self.supportList = cp.asarray(self.supportList)
            self.Ave(RFupper, SPlower)

    def Ave(self, RFupper, SPlower):
        """
        Synchronize and average multiple results.
        
        Parameters
        ----------
        - start (int): Start number. The default value is 0.
        - stop (int): End number.
        - ave (Boolean): Whether to average multiple results.
        - RFupper (float): The upper Fourier error limit for successful reconstruction.
        - SPlower (float): The lower limit to generate support from averaged support.
        """
        rf = self.ErrorFourier()
        self.list.insert(1, "RF", rf)
        success = rf < RFupper
        self.list.insert(2, "success", success)
        isminRF = rf == cp.min(rf)
        self.list.insert(3, "isminRF", isminRF)
        self.list.to_csv(self.outputPath + "########-Output.csv", index=False)
        self.realSpaceList = self.realSpaceList[np.argwhere(success)[:, 0]]
        self.supportList = self.supportList[np.argwhere(success)[:, 0]]
        self.len = cp.shape(self.realSpaceList)[0]
        self.SynchronizeMag()
        self.SynchronizeArg()
        self.supportAve = cp.mean(self.supportList, axis=0)
        self.supportLim = self.supportAve > SPlower
        self.realSpaceAve = cp.mean(self.realSpaceList, axis=0)
        self.realSpaceStd = cp.std(self.realSpaceList, axis=0)
        # Relative Standard Deviation
        cor = correlate(
            self.supportAve,
            GaussianWindow(self.size, min(self.size) / 2),
        )
        shift = np.unravel_index(int(cp.argmax(cor)), self.size)
        self.supportAve = cp.roll(self.supportAve, shift, axis=(0, 1))
        self.supportLim = cp.roll(self.supportLim, shift, axis=(0, 1))
        self.realSpaceAve = cp.roll(self.realSpaceAve, shift, axis=(0, 1))
        self.realSpaceStd = cp.roll(self.realSpaceStd, shift, axis=(0, 1))
        self.OutputAve()
        consistency = ICC(self.realSpaceList, self.supportList)
        print(
            "Error for Fourier magnitude:",
            np.min(rf),
            "index:",
            np.argwhere(isminRF)[0, 0],
        )
        print("Success rate of reconstruction:",
              np.sum(success) / np.size(success))
        print("Average error in Fourier space:", np.mean(rf[success]))
        print(
            "Relative Standard Deviation:", cp.mean(
                self.realSpaceStd[self.supportLim])
        )
        print("Consistency for magnitude:", consistency[0])
        print("Consistency for phase:", consistency[1])

    def ErrorFourier(self):
        """
        Error for Fourier magnitudes.
        """
        mask = self.mask[np.newaxis, :, :]
        pattern = self.pattern * mask
        fourierSpace = cp.abs(fft(self.realSpaceList) * mask)
        scaling = cp.sum(pattern) / cp.sum(fourierSpace, axis=(1, 2))
        fourierSpace = fourierSpace * scaling[:, cp.newaxis, cp.newaxis]
        if self.FourierModulusProjector == self.FourierModulusProjectorLimit:
            patternUpper = self.patternUpper * mask
            patternLower = self.patternLower * mask
            upper = fourierSpace > patternUpper
            lower = fourierSpace < patternLower
            diff = cp.zeros_like(fourierSpace)
            diff[upper] = (fourierSpace - patternUpper)[upper]
            diff[lower] = (patternLower - fourierSpace)[lower]
        else:
            diff = cp.abs(fourierSpace - pattern)
        ef = cp.sum(diff, axis=(1, 2)) / cp.sum(pattern)
        return cp.asnumpy(ef)

    def ErrorReal(self):
        """
        Error for real space.
        """
        scaling = cp.sum(cp.abs(self.realSpaceMod)) / cp.sum(
            cp.abs(self.realSpaceList), axis=(1, 2)
        )
        self.realSpaceList = self.realSpaceList * \
            scaling[:, cp.newaxis, cp.newaxis]
        diff = cp.abs(self.realSpaceList - self.realSpaceMod[cp.newaxis, :, :])
        er = cp.sum(diff, axis=(1, 2)) / cp.sum(cp.abs(self.realSpaceMod))
        return cp.asnumpy(er)

    def Sim(self, RFupper, realSpaceMod):
        self.realSpaceMod = realSpaceMod
        ef = self.ErrorFourier()
        self.list.insert(1, "EF", ef)
        success = ef < RFupper
        self.list.insert(2, "success", success)
        isminRF = ef == cp.min(ef)
        self.list.insert(3, "isminRF", isminRF)
        self.len = cp.shape(self.realSpaceList)[0]
        self.SynchronizeMag()
        self.SynchronizeArg()
        er = self.ErrorReal()
        self.list.insert(2, "ER", er)
        try:
            self.list.to_csv(self.outputPath +
                             "########-Output.csv", index=False)
            for i in range(self.len):
                fio.WritePNG(
                    cp.asnumpy(cp.abs(self.realSpaceList[i])),
                    self.outputPath + "{:0>3d}".format(i) + ".png",
                )
        except:
            warnings.warn("The output path is not set.")
        print("Success rate of reconstruction:",
              np.sum(success) / np.size(success))
        print("Average error in Fourier space:", np.mean(ef))
        print("Average error in real space:", np.mean(er))


def PRTF(data0, data):
    """
    Phase retrieval transfer function.
    """
    data0 = ishift(cp.abs(data0))
    data = ishift(cp.abs(data))
    data = data * cp.sum(data0[data0 != 0] ** 2) / \
        cp.sum(data[data0 != 0] ** 2)
    xx, yy = SetFourierGrid(data0.shape)
    frequencyGrid = cp.sqrt(xx**2 + yy**2)
    k = np.linspace(0, 0.5, 100, endpoint=False)
    prtf = cp.zeros(100)
    for i in range(100):
        position = frequencyGrid < (i + 1) / 200
        prtf[i] = cp.mean(data[position] / data0[position])
        frequencyGrid[position] = NAN
    return k, cp.asnumpy(prtf)


def ICC(realSpaceList, supportList):
    """
    Interclass Correlation Coefficient, 1-way random effects, the mean of k raters/measurements,\
        absolute agreement
    """

    def ICC1k(data):
        size = cp.shape(data)
        len = size[0]
        num = size[1] * size[2]
        data = cp.reshape(data, (len, num))
        mean = cp.mean(data)
        sst = cp.sum((data - mean) ** 2)
        ssr = cp.sum((cp.mean(data, 0) - mean) ** 2) * len
        msr = ssr / (num - 1)
        msw = (sst - ssr) / (num * len - num)
        return (msr - msw) / msr

    mag = cp.abs(realSpaceList)
    ang = cp.angle(realSpaceList)
    supportList = cp.logical_not(supportList)
    mag[supportList] = 0
    ang[supportList] = 0
    return cp.asnumpy(ICC1k(mag)), cp.asnumpy(ICC1k(ang))


def correlation(data1, data2):
    return cp.real(cp.sum(data1 * cp.conj(data2))) / cp.sqrt(
        cp.sum(cp.square(cp.abs(data1))) * cp.sum(cp.square(cp.abs(data2)))
    )


def FRC(data0, data):
    data1 = ishift(fft(cp.rot90(data0, 2)))
    data0 = ishift(fft(data0))
    data = ishift(fft(data))
    xx, yy = SetFourierGrid(data0.shape)
    frequencyGrid = cp.sqrt(xx**2 + yy**2)
    k = np.linspace(0, 0.5, 100, endpoint=False)
    frc0 = cp.zeros(100)
    frc1 = cp.zeros(100)
    for i in range(100):
        position = frequencyGrid < (i + 1) / 200
        frc0[i] = correlation(data0[position], data[position])
        frc1[i] = correlation(data1[position], data[position])
        frequencyGrid[position] = NAN
    return k, cp.asnumpy(frc0), cp.asnumpy(frc1)


def twin(data0, data):
    data0 = data0*cp.sum(data)/cp.sum(data0)
    data1 = fft(cp.rot90(data0, 2))
    data0 = fft(data0)
    data = fft(data)
    error0 = (cp.abs(data - data0)) / cp.abs(data0)
    error1 = (cp.abs(data - data1)) / cp.abs(data1)
    error0 = dl.from_dlpack(error0.toDlpack())
    error0 = nn.interpolate(
        error0[cp.newaxis, cp.newaxis, :],
        scale_factor=0.125,
        mode="area",
        recompute_scale_factor=False,
    )[0]
    error0 = cp.fromDlpack(dl.to_dlpack(error0))
    error1 = dl.from_dlpack(error1.toDlpack())
    error1 = nn.interpolate(
        error1[cp.newaxis, cp.newaxis, :],
        scale_factor=0.125,
        mode="area",
        recompute_scale_factor=False,
    )[0]
    error1 = cp.fromDlpack(dl.to_dlpack(error1))
    error = cp.min(cp.vstack((error0, error1)), axis=0)
    error0 = error0[0]
    error1 = error1[0]
    twin = cp.empty_like(error, dtype="float")
    twin[cp.logical_or(error >= 0.6, error0 == error1)] = 0
    twin[cp.logical_and(error < 0.6, error0 < error1)] = 1
    twin[cp.logical_and(error < 0.6, error0 > error1)] = -1
    return ishift(twin)
