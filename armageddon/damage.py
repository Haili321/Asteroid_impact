import pandas as pd
import numpy as np
from armageddon import locator


def p(r, Energy, burst_altitude, p_threshold):
    """
    Calculate the pressure(p) in this wave.

    Parameters
    ----------

    r: arraylike, float
        horizontal range (meter)
    Energy: float
        burst energy kiloton of TNT
    burst_altitude: float
        burst altitude, in meter
    p_threshold: float
        the target pressure that wish to find r for damage zone

    Returns
    -------
    pressure: float
        the pressure unit is Pa

    Examples
    --------
    >>> p(1e-6, 6000, 9000, 27e3)
    20145.2535399684

    """
    value = (r**2 + burst_altitude**2) / Energy**(2 / 3)
    return 3.14e11 * value**-1.3 + 1.8e7 * value**-0.565 - p_threshold


def level_dicision(min,
                   max,
                   Energy,
                   burst_altitude,
                   p_threshold,
                   stopping_tolerance=1e-6,
                   max_iter=100):
    """
    return root to help damage level decision

    Parameters
    ----------

    min, max: float
        the initial interval
    Energy: float
        burst energy kiloton of TNT
    burst_altitude: float
        burst altitude, in meter
    p_threshold: float
        the target pressure that wish to find r for damage zone
    stopping_tolerance: float
        a value for stopping loop
    max_iter: Integer
        the maximum number of iterations allowed

    Returns
    -------
    mid: float
        final horizontal range (meter) root in the interval

    Examples
    --------

    >>> print(level_dicision(min=1e-6, \
              max=2**15+1e-6, \
              Energy=7e3, \
              burst_altitude=8e3, \
              p_threshold=1e3, \
              stopping_tolerance=1e-6, \
              max_iter=100))
    115971.3167305456
    """

    interval = np.abs(max - min)
    while np.sign(p(min, Energy, burst_altitude, p_threshold)) == np.sign(
            p(max, Energy, burst_altitude, p_threshold)):
        min += interval
        max += interval

    # use max_iter to avoid infinite iteration problem
    n = 0
    while n <= max_iter:
        mid = (min + max) / 2.
        p_cal = p(mid, Energy, burst_altitude, p_threshold)

        if p_cal == 0. or (max - min) / 2. < stopping_tolerance:
            return mid
        n += 1

        if np.sign(p_cal) == np.sign(
                p(min, Energy, burst_altitude, p_threshold)):
            min = mid
        else:
            max = mid

    raise RuntimeError('Hit maximum number of iterations with no root found')


def damage_zones(outcome, lat, lon, bearing, pressures):
    """
    Calculate the latitude and longitude of the surface zero location and the
    list of airblast damage radii (m) for a given impact scenario.

    Parameters
    ----------

    outcome: Dict
        the outcome dictionary from an impact scenario
    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    bearing: float
        Bearing (azimuth) relative to north of meteoroid trajectory (degrees)
    pressures: float, arraylike
        List of threshold pressures to define airblast damage levels

    Returns
    -------

    blat: float
        latitude of the surface zero point (degrees)
    blon: float
        longitude of the surface zero point (degrees)
    damrad: arraylike, float
        List of distances specifying the blast radii
        for the input damage levels

    Examples
    --------

    >>> outcome = {'burst_altitude': 8e3, 'burst_energy': 7e3,\
    'burst_distance': 90e3, 'burst_peak_dedz': 1e3,\
    'outcome': 'Airburst'}
    >>> result = damage_zones(outcome, 52.79, -2.95, 135, \
    pressures=[1e3, 3.5e3, 27e3, 43e3])
    >>> np.allclose([result[0], result[1]],\
        [52.21396905216966, -2.015908861677074])
    True
    >>> np.allclose(result[2], [115971.3167305456, 42628.366516159615,\
        9575.214233444764, 5835.983451889588])
    True
    """

    # Transfer the elat, elon, bearing into radian form
    lat, lon, bearing = np.radians([lat, lon, bearing])
    # calculate r/Rp value
    r_Rp = outcome['burst_distance'] / 6.371e6
    blat = np.arcsin(
        np.sin(lat) * np.cos(r_Rp) +
        np.cos(lat) * np.sin(r_Rp) * np.cos(bearing))
    blon = np.arctan(
        np.sin(bearing) * np.sin(r_Rp) * np.cos(lat) /
        (np.cos(r_Rp) - np.sin(lat) * np.sin(blat))) + lon
    blat, blon = np.degrees([blat, blon])  # Transform the result into degree
    blat = blat.tolist()
    blon = blon.tolist()  # Transform the result into float instead of float64

    # calculate damrad
    # parse and check input parameters
    try:
        Energy = float(outcome['burst_energy'])
    except ValueError:
        print('Energy should be a valid number')
    assert Energy >= 0, 'burst energy must be positive'

    try:
        burst_altitude = float(outcome['burst_altitude'])
    except ValueError:
        print('burst_altitude should be a valid number')
    assert burst_altitude >= 0, 'burst altitude must be positive'

    # if energy is zero, all impact radius is zero
    if Energy == 0:
        damrad = [0 for _ in pressures]
        return blat, blon, damrad

    # define relevant constant
    min_r = 1e-6
    dr = 2**15
    max_p = p(min_r, Energy, burst_altitude, p_threshold=0)
    damrad = [
        level_dicision(min_r, min_r + dr, Energy, burst_altitude, p_threshold)
        if max_p >= p_threshold else 0. for p_threshold in pressures
    ]

    return blat, blon, damrad


fiducial_means = {
    'radius': 35,
    'angle': 45,
    'strength': 1e7,
    'density': 3000,
    'velocity': 19e3,
    'lat': 53.0,
    'lon': -2.5,
    'bearing': 115.
}
fiducial_stdevs = {
    'radius': 1,
    'angle': 1,
    'strength': 5e6,
    'density': 500,
    'velocity': 1e3,
    'lat': 0.025,
    'lon': 0.025,
    'bearing': 0.5
}


def impact_risk(planet,
                means=fiducial_means,
                stdevs=fiducial_stdevs,
                pressure=27.e3,
                nsamples=100,
                sector=True):
    """
    Perform an uncertainty analysis to calculate the risk for each affected
    UK postcode or postcode sector

    Parameters
    ----------
    planet: armageddon.Planet instance
        The Planet instance from which to solve the atmospheric entry

    means: dict
        A dictionary of mean input values for the uncertainty analysis. This
        should include values for ``radius``, ``angle``, ``strength``,
        ``density``, ``velocity``, ``lat``, ``lon`` and ``bearing``

    stdevs: dict
        A dictionary of standard deviations for each input value. This
        should include values for ``radius``, ``angle``, ``strength``,
        ``density``, ``velocity``, ``lat``, ``lon`` and ``bearing``

    pressure: float
        A single pressure at which to calculate the damage zone for each impact

    nsamples: int
        The number of iterations to perform in the uncertainty analysis

    sector: logical, optional
        If True (default) calculate the risk for postcode sectors, otherwise
        calculate the risk for postcodes

    Returns
    -------
    risk: DataFrame
        A pandas DataFrame with columns for postcode (or postcode sector) and
        the associated risk. These should be called ``postcode`` or ``sector``,
        and ``risk``.
    """

    # use Gaussian Distribution to store data
    gaussian_distribution = {}
    for i in means:
        gaussian_distribution[i] = np.random.normal(means.get(i),
                                                    stdevs.get(i), nsamples)

    postcode_affected = []
    Postcodelocator_Instance = None
    for i in range(nsamples):
        # calculate the result of the math formula given in solve.py
        result = planet.solve_atmospheric_entry(
            radius=gaussian_distribution['radius'][i],
            angle=gaussian_distribution['angle'][i],
            strength=gaussian_distribution['strength'][i],
            density=gaussian_distribution['density'][i],
            velocity=gaussian_distribution['velocity'][i])
        # store the energy data
        result = planet.calculate_energy(result)
        # get the event type, airburst or cratering
        outcome = planet.analyse_outcome(result)
        # return surface zero location lat, lon and the damage radii list
        blat, blon, damrad = damage_zones(
            outcome,
            lat=gaussian_distribution['lat'][i],
            lon=gaussian_distribution['lon'][i],
            bearing=gaussian_distribution['bearing'][i],
            pressures=[pressure])
        # print(f"{i+1}th level3 damage radii: {damrad}")
        # create new object of postcode locator
        Postcodelocator_Instance = locator.PostcodeLocator()
        # postcode in the given area
        current_postcode = Postcodelocator_Instance.get_postcodes_by_radius(
            [blat, blon], radii=damrad, sector=sector)
        # store current postcode in the list
        postcode_affected += current_postcode
    postcode_affected_list = []
    # change list format
    for i in postcode_affected:
        postcode_affected_list += i

    postcode_probability = {}
    postcode_affected_list_set = set(postcode_affected_list)
    # get unique postcode through set data type
    for i in postcode_affected_list_set:
        # The probability that a postcode is within a specified damage level
        postcode_probability[i] = postcode_affected_list.count(i) / nsamples

    postcode_probability = {
        # key is postcode, value is probability
        postcode: probability
        # select the valid pairs
        for postcode, probability in postcode_probability.items()
        if probability != 0
    }
    risk = []
    sector_list = []
    unit = []
    for postcode in postcode_probability:
        if sector:
            # sector is true
            sector_list += [postcode]
        else:
            unit += [postcode]

        # calculate risk
        risk += [
            postcode_probability[postcode] *
            Postcodelocator_Instance.get_population_of_postcode(
                [[postcode]], sector=sector)[-1][-1]
        ]

    if sector:
        # sector is true
        return pd.DataFrame({'sector': sector_list, 'risk': risk})
    else:
        return pd.DataFrame({'postcode': unit, 'risk': risk})
