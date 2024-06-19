import numpy as np
from .base import FittableModel
from astropy.convolution import convolve_fft
from scipy.ndimage import rotate

class Fittable1D(FittableModel):  # ciao=33, hello="world"
    def __init__(self, **kwargs):
        super().__init__()

    N_DIMENSIONS = 1  # non prendo x e y da evaluate
    IS_COMPOSITE = False
    N_INPUTS = 1  # x e y
    N_OUTPUTS = 1  # z


class Fittable2D(FittableModel):  # ciao=33, hello="world"
    def __init__(self, **kwargs):
        super().__init__()

    N_DIMENSIONS = 2  # non prendo x e y da evaluate
    IS_COMPOSITE = False
    N_INPUTS = 2  # x e y
    N_OUTPUTS = 1  # z


class Kernel2D(FittableModel):
    def __init__(self, **kwargs):
        super().__init__()

    N_DIMENSIONS = 2  # non prendo x e y da evaluate
    IS_COMPOSITE = False
    N_INPUTS = 1  # x e y
    N_OUTPUTS = 1  # z


class Kernel1D(FittableModel):
    def __init__(self, **kwargs):
        super().__init__()

    N_DIMENSIONS = 1  # non prendo x e y da evaluate
    IS_COMPOSITE = False
    N_INPUTS = 1  # x e y
    N_OUTPUTS = 1  # z


class GaussianModel(Fittable2D):
    
    def evaluate(self, x, y, amp, x0, y0, sigma_x, sigma_y, theta):
        """Two dimensional Gaussian function."""
        cost2 = np.cos(theta) ** 2
        sint2 = np.sin(theta) ** 2
        sin2t = np.sin(2.0 * theta)
        xstd2 = sigma_x**2
        ystd2 = sigma_y**2
        xdiff = x - x0
        ydiff = y - y0
        a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
        b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
        c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
        return amp * np.exp(-((a * xdiff**2) + (b * xdiff * ydiff) + (c * ydiff**2)))


class SersicModel(Fittable2D):
    def evaluate(self, x, y, amp, re, n, x0, y0, ellip, theta):
        """Two dimensional Sersic profile function."""
        from scipy.special import gammaincinv

        bn = gammaincinv(2.0 * n, 0.5)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_maj = np.abs((x - x0) * cos_theta + (y - y0) * sin_theta)
        x_min = np.abs(-(x - x0) * sin_theta + (y - y0) * cos_theta)

        b = (1 - ellip) * re
        expon = 2.0
        inv_expon = 1.0 / expon
        z = ((x_maj / re) ** expon + (x_min / b) ** expon) ** inv_expon
        return amp * np.exp(-bn * (z ** (1 / n) - 1.0))


class PSF(Kernel2D):
    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        self.data = data

    def evaluate(self, data):
        return convolve_fft(data, self.data)


class Rotation2D(Kernel2D):
    def __init__(self, angle, **kwargs):
        super().__init__(**kwargs)
        self.angle = angle

    def evaluate(self, data):
        return rotate(data, self.angle, reshape=False)
    

class Line(Fittable1D):
    def evaluate(self, x, x0, a=2, b=2):
        return a * (x - x0) + b


class PowerLaw(Fittable1D):
    def evaluate(self, x, alpha):
        return x**alpha


class Gaussian1D(Fittable1D):
    def evaluate(self, x, amp, x0, stddev):
        """
        Gaussian1D model function.
        """
        return amp * np.exp(-0.5 * (x - x0) ** 2 / stddev**2)