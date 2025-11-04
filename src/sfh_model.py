import cogsworth
import astropy.units as u
import numpy as np
from scipy.integrate import cumulative_trapezoid
import logging


class ClusteredRecentNearSunWagg2022(cogsworth.sfh.Wagg2022):
    def __init__(self, components=["low_alpha_disc"], component_masses=[1],
                 near_thresh=3 * u.kpc,
                 cluster_radius=1 * u.pc,
                 n_per_cluster=10000, **kwargs):
        self.near_thresh = near_thresh
        self.cluster_radius = cluster_radius
        self.n_per_cluster = n_per_cluster
        super().__init__(components=components, component_masses=component_masses, **kwargs)

    def sun_pos(self, t):
        sun_R = 8.122 * u.kpc
        sun_v = 229.40403 * u.km / u.s
        T = ((2 * np.pi * sun_R) / sun_v).to(u.Myr)
        theta = ((2 * np.pi * t / T) % (2 * np.pi)).decompose() * u.rad
        x = sun_R * np.cos(theta)
        y = sun_R * np.sin(theta)
        return x, y

    def draw_radii(self, size=None, component="low_alpha_disc"):
        if component != "low_alpha_disc":
            raise NotImplementedError()

        return np.random.uniform(8.122 - self.near_thresh.to(u.kpc).value,
                                 8.122 + self.near_thresh.to(u.kpc).value, size) * u.kpc

    def draw_lookback_times(self, size=None, component="low_alpha_disc"):
        if component != "low_alpha_disc":
            raise NotImplementedError()

        U = np.random.rand(size)
        norm = 1 / (self.tsfr * np.exp(-self.galaxy_age / self.tsfr) * (np.exp(200 * u.Myr / self.tsfr) - 1))
        tau = self.tsfr * np.log((U * np.exp(self.galaxy_age / self.tsfr)) / (norm * self.tsfr) + 1)

        return tau

    def sample(self):
        """Sample from the distributions for each component, combine and save in class attributes"""
        n_clusters = int(np.ceil(self._size / self.n_per_cluster))
        # create an array of which component each point belongs to
        self._which_comp = np.repeat("low_alpha_disc", self._size)

        self._tau = self.draw_lookback_times(n_clusters)
        sun_x, sun_y = self.sun_pos(-self._tau)

        angle_offset = np.random.uniform(0, 2 * np.pi, n_clusters) * u.rad
        r_offset = np.random.rand(n_clusters)**(0.5) * self.near_thresh
        x_offset, y_offset = r_offset * np.cos(angle_offset), r_offset * np.sin(angle_offset)
        x, y = sun_x + x_offset, sun_y + y_offset

        rho = ((x**2 + y**2)**(0.5)).to(u.kpc)
        z = self.draw_heights(n_clusters)

        # shuffle the samples so components are well mixed (mostly for plotting)
        random_order = np.random.permutation(n_clusters)
        self._tau = self._tau[random_order]
        rho = rho[random_order]
        z = z[random_order]
        x = x[random_order]
        y = y[random_order]

        self._tau = np.repeat(self._tau, self.n_per_cluster)[:self._size]
        rho = np.repeat(rho, self.n_per_cluster)[:self._size]
        x = np.repeat(x, self.n_per_cluster)[:self._size]
        y = np.repeat(y, self.n_per_cluster)[:self._size]
        z = np.repeat(z, self.n_per_cluster)[:self._size]

        # spread out each cluster
        x, y, z = np.random.normal([x.to(u.kpc).value, y.to(u.kpc).value, z.to(u.kpc).value],
                                   self.cluster_radius.to(u.kpc).value / np.sqrt(3),
                                   size=(3, self._size)) * u.kpc

        self._x = x
        self._y = y
        self._z = z

        # compute the metallicity given the other values
        self._Z = self.get_metallicity()


class RecentSB15Annulus(cogsworth.sfh.SandersBinney2015):
    """A modification of the SB15 SFH to only include recent star formation within a given age cutoff and
    only in a specific range of angular momenta (to mimic an annulus around the Sun)."""
    def __init__(self, age_cutoff=200 * u.Myr, **kwargs):
        self.age_cutoff = age_cutoff
        super().__init__(**kwargs)

    def draw_lookback_times(self):
        tau_range = np.linspace(0, self.tau_m * (1 - 1e-10), 100000)
        tau_pdf = np.exp(tau_range / self.tau_F - self.tau_S / (self.tau_m - tau_range))
        tau_cdf = cumulative_trapezoid(tau_pdf, tau_range, initial=0)

        lim = (tau_cdf / tau_cdf[-1])[tau_range >= self.age_cutoff][0]
        U = np.random.uniform(0, lim, self._size)
        self._tau = self._inv_cdf(U) * u.Gyr
        return self._tau

    def _generate_df(self, J, component, tau):
        df_val = super()._generate_df(J=J, component=component, tau=tau)
        J_r, J_z, J_phi = J.T
        df_val[(J_phi < 1.17) | (J_phi > 2.8)] = 0.0
        return df_val


class ClusteredNearSun(cogsworth.sfh.StarFormationHistory):
    """A star formation history that samples clusters of stars formed near the Sun's position using the
    Sanders and Binney (2015) SFH model."""
    def __init__(self, size, sfh_model, sfh_params,
                 near_thresh=3 * u.kpc,
                 cluster_radius=1 * u.pc,
                 n_per_cluster=10000,
                 velocity_dispersion=2 * u.km / u.s,
                 immediately_sample=True):
        self._size = size
        self.near_thresh = near_thresh
        self.cluster_radius = cluster_radius
        self.n_per_cluster = n_per_cluster
        self.velocity_dispersion = velocity_dispersion
        self.sfh_model = sfh_model
        self.sfh_params = sfh_params
        self.__citations__ = []

        if immediately_sample:
            self.sample()

    def sun_pos(self, t):
        """Approximate the Sun's position at lookback time t assuming a circular orbit."""
        sun_R = 8.122 * u.kpc
        sun_v = 229.40403 * u.km / u.s
        T = ((2 * np.pi * sun_R) / sun_v).to(u.Myr)
        theta = ((2 * np.pi * t / T) % (2 * np.pi)).decompose() * u.rad
        x = sun_R * np.cos(theta)
        y = sun_R * np.sin(theta)
        return x, y

    def sample(self):
        # determine how many clusters we need to sample
        n_clusters = int(np.ceil(self.size / self.n_per_cluster))

        # sample clusters until we have enough stars near the Sun
        sampled_sfhs = []
        while sum(len(s) for s in sampled_sfhs) < n_clusters:
            sfh = self.sfh_model(size=n_clusters * 10, **self.sfh_params)
            sun_x, sun_y = self.sun_pos(-sfh.tau)
            dist_to_sun = np.sqrt((sfh.x - sun_x)**2 + (sfh.y - sun_y)**2)
            masked_sfh = sfh[dist_to_sun <= self.near_thresh]
            print(f"  Selected {len(masked_sfh)} stars within {self.near_thresh} of the Sun")
            sampled_sfhs.append(masked_sfh)
        self._sampled_sfh = cogsworth.sfh.concat(*sampled_sfhs)[:n_clusters]

        # repeat each sampled star to create clusters
        data_attributes = ["_tau", "_Z", "_x", "_y", "_z", "_which_comp", "v_R", "v_T", "v_z", "v_x", "v_y"]
        for attr in data_attributes:
            if hasattr(self._sampled_sfh, attr):
                data = np.repeat(getattr(self._sampled_sfh, attr), self.n_per_cluster)[:self.size]
                setattr(self, attr, data)

        # apply cluster spread to positions
        self._x, self._y, self._z = np.random.normal(
            [self._x.to(u.kpc).value,
             self._y.to(u.kpc).value,
             self._z.to(u.kpc).value],
            self.cluster_radius.to(u.kpc).value / np.sqrt(3),
            size=(3, self._size)
        ) * u.kpc

        # if the sfh has velocities, apply velocity dispersion
        if hasattr(self._sampled_sfh, "v_R"):
            self.v_R, self.v_T, self.v_z = np.random.normal(
                [self.v_R.to(u.km/u.s).value,
                 self.v_T.to(u.km/u.s).value,
                 self.v_z.to(u.km/u.s).value],
                self.velocity_dispersion.to(u.km/u.s).value / np.sqrt(3),
                size=(3, self._size)
            ) * u.km / u.s

            if hasattr(self._sampled_sfh, "v_x"):
                v_phi = (self.v_T / self.rho)
                self.v_x = (self.v_R * np.cos(self.phi) - self.rho * np.sin(self.phi) * v_phi)
                self.v_y = (self.v_R * np.sin(self.phi) + self.rho * np.cos(self.phi) * v_phi)
        else:
            # warn the user that no velocities were applied
            logging.getLogger("cogsworth").warning("The provided SFH model does not have velocities; no velocity dispersion applied.")

        # delete the sampled SFH otherwise the saving function explodes
        del self._sampled_sfh
