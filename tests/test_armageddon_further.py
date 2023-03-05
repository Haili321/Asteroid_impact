import pandas as pd
import os
import armageddon
from unittest import TestCase


data_base_path = os.sep.join((os.path.dirname(__file__),
                              'test_data'))

earth = armageddon.Planet(atmos_func='tabular')


def test_solve_atmosphere():

    test01_output = pd.read_csv(os.sep.join((data_base_path,
                                             'solve_atmospheric_data.csv')))

    pd.testing.assert_frame_equal(
        earth.solve_atmospheric_entry(radius=35,
                                      angle=45,
                                      init_altitude=10e3,
                                      strength=1e3,
                                      density=3000,
                                      velocity=19e3), test01_output)


def test_calculate_energy():

    test02_output = pd.read_csv(os.sep.join((data_base_path,
                                             'calculate_energy_data.csv')))

    input02 = earth.solve_atmospheric_entry(radius=35, angle=45,
                                            init_altitude=10e3,
                                            strength=1e3, density=3000,
                                            velocity=19e3)

    pd.testing.assert_frame_equal(earth.calculate_energy(input02),
                                  test02_output)


def test_analyse_outcome():

    inp = earth.solve_atmospheric_entry(radius=35, angle=45,
                                        init_altitude=10e3,
                                        strength=1e3, density=3000,
                                        velocity=19e3)
    input03 = earth.calculate_energy(inp)

    res = earth.analyse_outcome(input03)

    expected_result = {
        'outcome': 'Airburst',
        'burst_peak_dedz': 5186.8692160992105,
        'burst_altitude': 4241.464397644264,
        'burst_distance': 5756.907451545797,
        'burst_energy': 14516.354931930628
    }

    for key in res.keys():
        TestCase().assertAlmostEqual(res[key], expected_result[key])
