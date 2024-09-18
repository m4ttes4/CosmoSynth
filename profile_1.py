import cProfile
import pstats

from models.zoo import Fittable1D
import numpy as np
from models.zoo import Fittable1D
from models.optimizers import NelderMead
import numpy as np
import matplotlib.pyplot as plt
# defining our simple models

class AGaussian1D(Fittable1D):
    @staticmethod
    def evaluate(x, amp, x0, stddev):
        return amp * np.exp(-0.5 * (x - x0) ** 2 / stddev**2)
    
    # create our simple model
class Gaussian1D(Fittable1D):
    @staticmethod
    def evaluate(x, mu, sigma, A):
        return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


class Line1D(Fittable1D):
    @staticmethod
    def evaluate(x, a, b):
        return a * x + b
    
X = np.linspace(0, 10, 100)

def generate_fake_data(mu, sigma, A, num_points=100, noise_level=0.2):
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, num_points)
    y_true = Gaussian1D.evaluate(x, mu, sigma, A)
    noise = noise_level * np.random.randn(num_points)
    y_noisy = y_true + noise
    return x, y_noisy


if __name__ == '__main__':
    gaus = Gaussian1D()

    # True params
    true_mu = 10.77
    true_sigma = 6.44
    true_A = 1.5

    # G
    x_data, y_data = generate_fake_data(true_mu, true_sigma, true_A)

    # Initial guess
    initial_guess = [3.6, 1.5, 0.6]  # [mu, sigma, A]

    # or equivalently... a dictionary is supported
    initial_guess = {"mu": 4, "A": 4, "sigma": 2}

    optimizer = NelderMead(model=gaus, treshold=1e-6, delta=0.1)

    with cProfile.Profile() as profile:

        '''gaussian = Gaussian1D(name="MyGaussian", amp=10, x0=5)

        line = Line1D(name="MyLine", a=0.5)

        gaussian_values = gaussian([X])
        #cProfile.run("gaussian_values = gaussian([X])")

        '''

    results = optimizer.minimize(
        x0=initial_guess, grid=[x_data], data=[y_data], progress=True, maxiter=1000
    )

    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats()
    results.dump_stats('profiler.prof')


