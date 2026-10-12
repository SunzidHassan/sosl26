"""Olfaction: Gaussian plume model and the Bayesian source map P(src | C).

For every candidate source cell the plume model predicts the concentration
the robot should measure; each step multiplies the map by how well that
prediction matches the measured (rescaled) concentration.
"""

import math

import numpy as np


def gaussian_plume(x, z, source, q_s=2000, D=10, U=0, tau=1000, psi_deg=0):
    """Concentration at (x, z) for a source at `source` = (x_s, z_s).

        c = q_s / (4 pi D r) * exp(-del_z U / (2 D) - r / lambda)
        lambda = sqrt(D tau / (1 + tau U^2 / (4 D)))

    q_s: emission rate, D: diffusion coefficient, tau: decay time, U: wind
    speed and psi_deg its direction; r is the distance to the source and
    del_z the downwind offset. With U = 0 the plume is radially symmetric.
    x, z and the source coordinates can be scalars or numpy arrays.
    """
    psi = math.radians(psi_deg)
    lambd = math.sqrt((D * tau) / (1 + (tau * U**2) / (4 * D)))   # decay length
    delta_x = np.asarray(x, dtype=float) - source[0]
    delta_z = np.asarray(z, dtype=float) - source[1]
    r = np.hypot(delta_x, delta_z)
    r = np.where(r == 0, 1e-3, r)                                    # avoid division by 0 at the source
    del_z = -(delta_x * math.cos(psi) + delta_z * math.sin(psi))     # offset along the wind
    return q_s / (4 * math.pi * D * r) * np.exp((-del_z * U) / (2 * D) - r / lambd)


def map_entropy(raw):
    """Shannon entropy (bits) of a non-negative map, after normalising it to sum to 1."""
    p = np.asarray(raw, dtype=float).ravel()
    total = p.sum()
    if total < 1e-12:
        return 0.0
    p = p / total
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log2(p)))


class BayesianAgent:
    """Bayesian belief map over source locations on the (x_points, z_points) grid.

    Cells outside `inside` (bool (H, W), e.g. the search boundary) have
    probability 0 and stay 0 after every update.
    """

    def __init__(self, x_points, z_points, sigma_noise, inside=None,
                 q_s=2000, D=10, tau=1000, U=0, psi_deg=0):
        self.x_points = np.asarray(x_points, dtype=float)
        self.z_points = np.asarray(z_points, dtype=float)
        self.sigma_noise = sigma_noise
        self.plume_params = dict(q_s=q_s, D=D, U=U, tau=tau, psi_deg=psi_deg)
        shape = (len(self.z_points), len(self.x_points))
        self.inside = np.ones(shape, dtype=bool) if inside is None else np.asarray(inside, dtype=bool)
        self.prob_map = self.uniform()
        # Candidate source position of every cell
        self.Xs, self.Zs = np.meshgrid(self.x_points, self.z_points)

    def uniform(self):
        return self.inside / self.inside.sum()

    def likelihood(self, measured, expected):
        """Gaussian sensor noise: p(measured | expected)."""
        s = self.sigma_noise
        return np.exp(-((measured - expected) ** 2) / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s + 1e-9)

    def posterior(self, concentration, robot_x, robot_z):
        """Bayes update with one (rescaled) reading taken at (robot_x, robot_z)."""
        # Expected reading at the robot for a source in every cell, in one vectorised call.
        expected = gaussian_plume(robot_x, robot_z, (self.Xs, self.Zs), **self.plume_params)
        self.prob_map = self.prob_map * self.likelihood(concentration, expected)
        total = self.prob_map.sum()
        if total > 1e-9:
            self.prob_map /= total
        else:
            print("Warning: Probability map sum is close to zero after update; resetting to uniform.")
            self.prob_map = self.uniform()
