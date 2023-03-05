import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import scipy
from scipy.interpolate import pchip_interpolate


class Planet():
    """
    |
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(self, atmos_func='exponential',
                 atmos_filename=os.sep.join((os.path.dirname(__file__), '..',
                                             'resources',
                                             'AltitudeDensityTable.csv')),
                 Cd=1., Ch=0.1, Q=1e7, Cl=1e-3, alpha=0.3,
                 Rp=6371e3, g=9.81, H=8000., rho0=1.2):
        """
        Set up the initial parameters and constants for the target planet

        Parameters
        ----------
        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function rho = rho0 exp(-z/H).
            Options are 'exponential', 'tabular' and 'constant'
        atmos_filename : string, optional
            Name of the filename to use with the tabular atmos_func option
        Cd : float, optional
            The drag coefficient
        Ch : float, optional
            The heat transfer coefficient
        Q : float, optional
            The heat of ablation (J/kg)
        Cl : float, optional
            Lift coefficient
        alpha : float, optional
            Dispersion coefficient
        Rp : float, optional
            Planet radius (m)
        rho0 : float, optional
            Air density at zero altitude (kg/m^3)
        g : float, optional
            Surface gravity (m/s^2)
        H : float, optional
            Atmospheric scale height (m)
        """

        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0
        self.atmos_filename = atmos_filename

        try:
            # set function to define atmoshperic density
            if atmos_func == 'exponential':
                self.rhoa = lambda z: self.rho0 * np.exp(-z/self.H)
            elif atmos_func == 'tabular':
                data = pd.read_csv(atmos_filename, comment='#', delimiter=' ',
                                   skipinitialspace=True,
                                   names=['Altitude', 'Density'])
                altitude = data.Altitude.values
                density = data.Density.values

                # Interpolate data using table
                self.rhoa = lambda z: pchip_interpolate(altitude, density, z)

            elif atmos_func == 'constant':
                self.rhoa = lambda z: rho0
            else:
                raise NotImplementedError(
                    "atmos_func must be 'exponential', 'tabular' or 'constant'"
                    )
        except NotImplementedError:
            print("atmos_func {} not implemented yet.".format(atmos_func))
            print("Falling back to constant density atmosphere for now")
            self.rhoa = lambda z: rho0

    def F(self, u, strength, density):
        """
        Define a coupled set of ordinary differential equations for
        du/dt = f(t, u)

        Parameters
        ----------
        u : DataFrame
            A pandas dataframe with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius for the last timestep
        strength : float
            The strength of the asteroid (i.e. the maximum pressure it can
            take before fragmenting) in N/m^2
        density : float
            The density of the asteroid in kg/m^3

        Returns
        -------
        result : DataFrame
            A pandas dataframe with columns for the change of
            velocity, mass, angle, altitude, horizontal distance,
            radius and time
        """
        # u = [v, m, theta, z, x, r]
        v, m, theta, z, _, r = u
        area = np.pi * r**2
        rhoa = self.rhoa(z)

        if rhoa * v**2 < strength:
            drdt = 0
        else:
            drdt = np.sqrt(7/2*self.alpha*rhoa/density)*v

        dvdt = -self.Cd*rhoa*area*(v**2)/(2*m) + self.g*np.sin(theta)
        dmdt = -self.Ch*rhoa*area*(v**3)/(2*self.Q)
        dthetadt = (self.g*np.cos(theta)/v
                    - self.Cl*rhoa*area*v/(2*m)
                    - v*np.cos(theta)/(self.Rp+z))
        dzdt = -v*np.sin(theta)
        dxdt = v*np.cos(theta)/(1+z/self.Rp)

        return np.array([dvdt, dmdt, dthetadt, dzdt, dxdt, drdt])

    def solve_atmospheric_entry(
            self, radius, velocity, density, strength, angle,
            init_altitude=100e3, dt=0.05, tmax=300., radians=False):
        """
        Solve the system of differential equations for a given impact scenario

        Parameters
        ----------
        radius : float
            The radius of the asteroid in meters
        velocity : float
            The entery speed of the asteroid in meters/second
        density : float
            The density of the asteroid in kg/m^3
        strength : float
            The strength of the asteroid (i.e. the maximum pressure it can
            take before fragmenting) in N/m^2
        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians
        init_altitude : float, optional
            Initial altitude in m
        dt : float, optional
            The output timestep, in s
        dt : float, optional
            The maximal timestep, in s
        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the dataframe will have the same units as the
            input
        Returns
        -------
        Result : DataFrame
            A pandas dataframe containing the solution to the system.
            Includes the following columns:
            'velocity', 'mass', 'angle', 'altitude',
            'distance', 'radius', 'time'
        """
        # initial condition
        v0 = velocity
        m0 = density * (4/3) * np.pi * radius**3
        z0 = init_altitude
        x0 = 0
        theta0 = math.radians(angle) if not radians else angle
        r0 = radius
        f = np.array([v0, m0, theta0, z0, x0, r0])
        fs = np.zeros((100000000, 6))
        fs[0, :] = f
        i = 1

        t = 0
        ts = np.array(0)

        # RK4 schematic
        while f[3] >= 0:
            # End on the ground
            k1 = dt*self.F(f, strength, density)
            k2 = dt*self.F(f+0.5*k1, strength, density)
            k3 = dt*self.F(f+0.5*k2, strength, density)
            k4 = dt*self.F(f+k3, strength, density)
            f = f + (k1 + 2*k2 + 2*k3 + k4) / 6
            t += dt

            fs[i, :] = f
            i += 1
            ts = np.append(ts, t)

            # End on the ground
            if (f[1] <= 0.01) | (f[3] > z0):
                break

        fs = fs[0:i, :]

        return pd.DataFrame(
            {
                'velocity': fs[:, 0],
                'mass': fs[:, 1],
                'angle': fs[:, 2]*(180/np.pi) if not radians else fs[:, 2],
                'altitude': fs[:, 3],
                'distance': fs[:, 4],
                'radius': fs[:, 5],
                'time': ts
            }, index=range(len(ts)))

    def calculate_energy(self, result):
        """
        Function to calculate the kinetic energy lost per unit altitude in
        kilotons TNT per km, for a given solution.

        Parameters
        ----------
        result : DataFrame
            A pandas dataframe with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time
        Returns
        -------
        Result : DataFrame
            Returns the dataframe with additional column ``dedz`` which is the
            kinetic energy lost per unit altitude
        """

        # Replace these lines with your code to add the dedz column to
        # the result DataFrame
        result = result.copy()
        result.insert(len(result.columns),
                      'dedz', np.array(np.nan))

        dz = ((np.array(result.altitude)[0:-1]
              - np.array(result.altitude)[1:])/1000)  # km
        E = np.array(0.5 * result.mass * result.velocity**2)/4.184e12
        result['dedz'] = np.append(0, (E[0:-1] - E[1:]) / dz)

        return result

    def analyse_outcome(self, result):
        """
        Inspect a pre-found solution to calculate the impact and airburst stats

        Parameters
        ----------
        result : DataFrame
            pandas dataframe with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time
        Returns
        -------
        outcome : Dict
            dictionary with details of the impact event, which should contain
            the key:
                ``outcome`` (which should contain one of the
                following strings: ``Airburst`` or ``Cratering``),
            as well as the following 4 keys:
                ``burst_peak_dedz``, ``burst_altitude``,
                ``burst_distance``, ``burst_energy``

        Examples
        -------
        >>> import numpy
        >>> planet = Planet(atmos_func='tabular')
        >>> result = planet.solve_atmospheric_entry(radius=35, angle=45,
        ... init_altitude=10e3,tmax=120,
        ... strength=1e7, density=3000, velocity=19e3)
        >>> energy = planet.calculate_energy(result)
        >>> res = planet.analyse_outcome(energy)
        >>> res['outcome']
        'Airburst'
        >>> numpy.isclose(res['burst_peak_dedz'], 5186.86921609922)
        True
        >>> numpy.isclose(res['burst_altitude'], 4241.464397644265)
        True
        >>> numpy.isclose(res['burst_distance'], 5756.907451545796)
        True
        >>> numpy.isclose(res['burst_energy'], 14516.354931930628)
        True

        """

        location = np.argmax(result.dedz)
        burst_peak_dedz = result._get_value(location, 'dedz')
        burst_altitude = result._get_value(location, 'altitude')
        burst_distance = result._get_value(location, 'distance')

        burst_mass = result._get_value(location, 'mass')
        burst_velocity = result._get_value(location, 'velocity')
        initial_mass = result._get_value(0, 'mass')
        initial_velocity = result._get_value(0, 'velocity')

        burst_energy = (((0.5 * initial_mass * initial_velocity**2)
                        - (0.5 * burst_mass * burst_velocity**2)) / 4.184e12)

        cratering_mass = result._get_value(len(result)-1, 'mass')
        cratering_velocity = result._get_value(len(result)-1, 'velocity')
        remain_energy = ((0.5 * cratering_mass * cratering_velocity**2)
                         / 4.184e12)

        # Use the index of maximal dedz as the condition that determine outcome
        if (result.radius[location] > result.radius[0] and
                location != len(result)-1):
            burst_outcome = 'Airburst'

        else:
            burst_outcome = 'Cratering'
            burst_altitude = 0
            burst_energy = max(remain_energy, burst_energy)

        outcome = {'outcome': burst_outcome,
                   'burst_peak_dedz': burst_peak_dedz,
                   'burst_altitude': burst_altitude,
                   'burst_distance': burst_distance,
                   'burst_energy': burst_energy}
        return outcome

    def plotting(self, result):
        """
        Plot simple graphical output of the evolution
        of the asteroid in the atmosphere
        Figures produced are:
            - time histories of velocity, mass, angle,
            altitude, distance and radius
            - altitude histories of velocity, mass, angle,
            time, distance and radius

        Parameters
        ----------
        result : DataFrame
            A pandas dataframe with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time

        Returns
        ----------
        result: Matplotlib Figure

        """
        # Plot the evolution of the asteroid vs time
        _, axes1 = plt.subplots(2, 3, figsize=(20, 10))
        titles = ['velocity (m/s) vs time (s)',
                  'mass (kg) vs time (s)',
                  'angle (rad) vs time (s)',
                  'altitude (m) vs time (s)',
                  'distance (m) vs time (s)',
                  'radius (m) vs time (s)']
        x_lists = [result.velocity,
                   result.mass,
                   result.angle,
                   result.altitude,
                   result.distance,
                   result.radius]
        x_labels = ['velocity (m/s)',
                    'mass (kg)',
                    'angle (rad)',
                    'altitude (m)',
                    'distance (m)',
                    'radius (m)']
        for ax, x_val, x_label, title in zip(axes1.flatten(), x_lists,
                                             x_labels, titles):
            ax.plot(x_val, result.time)
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel('time (s)')
            plt.tight_layout(pad=5)

        # Plot the evolution of the asteroid vs altitude
        _, axes2 = plt.subplots(2, 3, figsize=(20, 10))
        titles_alt = ['velocity (m/s) vs altitude (m)',
                      'mass (kg) vs altitude (m)',
                      'angle (rad) vs altitude (m)',
                      'time (s) vs altitude (m)',
                      'distance (m) vs altitude (m)',
                      'radius (m) vs altitude (m)']
        x_lists[3] = result.time
        x_labels[3] = 'time (s)'
        for ax, x_val, x_label, title in zip(axes2.flatten(), x_lists,
                                             x_labels, titles_alt):
            ax.plot(x_val, result.altitude)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel('altitude (m)', fontsize=10)
            plt.tight_layout(pad=5)
        plt.show()

    def plot_energy_curve(self, result, outcome):
        """
        Plot energy deposition curve
        |
        Parameters
        ----------
        result : DataFrame
            A pandas dataframe with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time
        outcome : Dict
            dictionary with details of the impact event, which should contain
            the key:
                ``outcome`` (which should contain one of the
                following strings: ``Airburst`` or ``Cratering``),
            as well as the following 4 keys:
                ``burst_peak_dedz``, ``burst_altitude``,
                ``burst_distance``, ``burst_energy``

        Returns
        ----------
        result: Matplotlib Figure
        """
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(result.dedz, result.altitude/1e3, 'k')
        ax.scatter(0, max(result.altitude)/1e3, c='k',
                   marker='*', label='Break-up altitude')
        ax.scatter(outcome['burst_peak_dedz'], outcome['burst_altitude']/1e3,
                   c='b', marker='o', label='Burst altitude')
        ax.set_xlabel('Energy per unit height [Kt/km]')
        ax.set_ylabel('Altitude [km]')
        ax.set_title('Energy deposition curve')
        ax.legend(loc='best', fontsize=12)
        plt.show()
