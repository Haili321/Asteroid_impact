import os
import math
import numpy as np
import pandas as pd
import math


class Planet():
    """
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
                data = pd.read_csv(atmos_filename, comment='#', delimiter=' ', \
                                   skipinitialspace=True,names=['Altitude', 'Density'])
                
                z_max = data.Altitude.max()
                z_min = data.Altitude.min()
                altitude = data.Altitude.values
                density = data.Density.values

                # Interpolate data using table. If z exceed the range, use exponential function
                self.rhoa = lambda z: np.select([(z >= z_min) & (z <= z_max), (z < z_min) | (z > z_max)],
                                                            [np.interp(z, altitude, density), self.rho0 * np.exp(-z/self.H)])
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
        Define a coupled set of ordinary differential equations for du/dt = f(t, u)
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
        dthetadt = self.g*np.cos(theta)/v - self.Cl*rhoa*area*v/(2*m) - v*np.cos(theta)/(self.Rp+z)
        dzdt = -v*np.sin(theta)
        dxdt = v*np.cos(theta)/(1+z/self.Rp)
        
        return np.array([dvdt, dmdt, dthetadt, dzdt, dxdt, drdt])

    def solve_atmospheric_entry(
            self, radius, velocity, density, strength, angle,
            init_altitude=100e3, dt=0.005, tmax = 1000, radians=False):
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

        #initial condition
        v0 = velocity
        m0 = density * (4/3) * np.pi * radius**3
        z0 = init_altitude
        x0 = 0
        theta0 = math.radians(angle) if radians == False else angle
        r0 = radius
        f = np.array([v0, m0, theta0, z0, x0, r0])
        fs = f.copy()

        t = 0
        ts = np.array(0)

        # RK4 schematic
        while t < tmax:
            k1 = dt*self.F(f, strength, density)
            k2 = dt*self.F(f+0.5*k1, strength, density)
            k3 = dt*self.F(f+0.5*k2, strength, density)
            k4 = dt*self.F(f+k3, strength, density)
            f = f + (k1 + 2*k2 + 2*k3 + k4) / 6
            t += dt
            if f[3] <= 0.1:
                break
            fs = np.vstack((fs, f))
            ts = np.append(ts, t)
        
        df = pd.DataFrame(fs, columns = ['velocity', 'mass', 'angle', 'altitude', 'distance', 'radius'])


        return pd.DataFrame({'velocity': df['velocity'],
                             'mass': df['mass'],
                             'angle': df['angle'],
                             'altitude': df['altitude'],
                             'distance': df['distance'],
                             'radius': df['radius'],
                             'time': ts}, index=range(len(ts)))
        

    def calculate_energy(self, result):
        """
        Function to calculate the kinetic energy lost per unit altitude in
        kilotons TNT per km, for a given solution.
        Parameters 
        ----------
        result : DataFrame
            A pandas dataframe with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time
        Returns : DataFrame
            Returns the dataframe with additional column ``dedz`` which is the
            kinetic energy lost per unit altitude
        """

        # Replace these lines with your code to add the dedz column to
        # the result DataFrame
        result = result.copy()
        result.insert(len(result.columns),
                      'dedz', np.array(np.nan))

        E = np.array(0.5 * result.mass * result.velocity**2)/4.184e12
        result['dedz'] = np.append(0, (E[0:-1] - E[1:]) / (np.array(result['altitude'])[0:-1] - np.array(result['altitude'])[1:]))

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
        """
        # return the first max radius
        # location = np.argmax(result['radius'])


        test1 = result.dedz.iloc[-1]
        test2 = result.dedz.iloc[-2]
        
        if test1 - test2 < 0:
            burst_outcome = 'Airburst'
            location = np.argmax(result.dedz)
            burst_altitude = result._get_value(location, 'altitude')
            #burst_peak_dedz = result._get_value(location, 'dedz')


        else:
            burst_outcome = 'Cratering'
            location = result.shape[0]-1
            burst_altitude = 0
            #burst_peak_dedz = 0

        burst_peak_dedz = result._get_value(location, 'dedz')
        #burst_altitude = result._get_value(location, 'altitude')
        burst_distance = result._get_value(location, 'distance')
        burst_mass = result._get_value(location, 'mass')
        burst_velocity = result._get_value(location, 'velocity')


        initial_mass = result._get_value(0, 'mass')
        initial_velocity = result._get_value(0, 'velocity')
        burst_energy = -(((0.5 * burst_mass * burst_velocity**2) / (4.184e12)) - (0.5 * initial_mass * initial_velocity ** 2 / (4.184e12)))
        cratering_mass =  result._get_value(len(result)-1, 'mass')
        cratering_velocity = result._get_value(len(result)-1, 'velocity')
        remain_energy = 0.5 * cratering_mass * cratering_velocity ** 2 / (4.184e12)

        if burst_outcome == 'Airburst':
            pass
        elif (burst_outcome == 'Cratering') & (remain_energy > burst_energy):
            burst_energy = remain_energy
        
        outcome = {'outcome': burst_outcome,
                   'burst_peak_dedz': burst_peak_dedz,
                   'burst_altitude': burst_altitude,
                   'burst_distance': burst_distance,
                   'burst_energy': burst_energy}
        return outcome


       


planet = Planet(atmos_func='tabular')


result = planet.solve_atmospheric_entry(radius=35, angle=45,
                                       strength=1e7, density=3000,
                                       velocity=19e3)

energy = planet.calculate_energy(result)
print(energy)
res = planet.analyse_outcome(energy)
print(res)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20,10))
loss_energy = energy['dedz']
x = energy['altitude']
ax.plot(loss_energy,x)
ax.set_xlabel('Energy per unit height [kt km-1]')
ax.set_ylabel('Attitude')
ax.set_title('loss energy with altitude')
plt.show()