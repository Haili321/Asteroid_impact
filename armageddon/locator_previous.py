"""Module dealing with postcode information."""

from datetime import date
import os
import sys

import numpy as np
import pandas as pd


def great_circle_distance(latlon1, latlon2):
    """
    Calculate the great circle distance (in metres) between pairs of
    points specified as latitude and longitude on a spherical Earth
    (with radius 6371 km).

    Parameters
    ----------

    latlon1: arraylike
        latitudes and longitudes of first point (as [n, 2] array for n points) unit:degree
    latlon2: arraylike
        latitudes and longitudes of second point (as [m, 2] array for m points) unit:degree

    Returns
    -------

    numpy.ndarray
        Distance in metres between each pair of points (as an n x m array)

    Examples
    --------

    >>> import numpy
    >>> fmt = lambda x: numpy.format_float_scientific(x, precision=3)
    >>> with numpy.printoptions(formatter={'all': fmt}): print(great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0]))
    [[1.286e+05]
     [6.378e+04]]
    """
    # deal with naked list
    if not isinstance(latlon1[0], list):
        latlon = []
        latlon.append(latlon1)
        latlon1 = latlon
    if not isinstance(latlon2[0], list):
        latlon = []
        latlon.append(latlon2)
        latlon2 = latlon

    EARTH_RADIUS = 6371000  # unit is meter
    # convert to array
    latlon1 = np.array(latlon1)
    latlon2 = np.array(latlon2)
    # convert degree to radians
    latlon1 = latlon1 * np.pi / 180
    latlon2 = latlon2 * np.pi / 180
    distance = np.empty((len(latlon1), len(latlon2)), float)
    for i, item_1 in enumerate(latlon1):
        for j, item_2 in enumerate(latlon2):
            if item_1[0] == item_2[0]:
                # same latitude
                distance[i][j] = np.abs(item_1[1] - item_2[1]) * EARTH_RADIUS * np.cos(item_1[0])
            elif item_1[1] == item_2[1]:
                # same longitude
                distance[i][j] = np.abs(item_1[0] - item_2[0]) * EARTH_RADIUS
            else:
                # law of spherical cosines(through %timeit to check, this method is fastest)
                distance[i][j] = np.arccos(
                    np.sin(item_1[0]) * np.sin(item_2[0]) + np.cos(item_1[0]) * np.cos(item_2[0]) *
                    np.cos(np.abs(item_1[1] - item_2[1]))) * EARTH_RADIUS
    return distance


class PostcodeLocator(object):
    """Class to interact with a postcode database file."""

    def __init__(self, postcode_file='resources/full_postcodes.csv',
                 census_file='resources/population_by_postcode_sector.csv',
                 norm=great_circle_distance):
        """
        Parameters
        ----------

        postcode_file : str, optional
            Filename of a .csv file containing geographic
            location data for postcodes.

        census_file :  str, optional
            Filename of a .csv file containing census data by postcode sector.

        norm : function
            Python function defining the distance between points in latitude-longitude space.

        """
        self.postcode_file = postcode_file
        self.census_file = census_file
        self.norm = norm

    def get_postcodes_by_radius(self, X, radii, sector=False):
        """
        Return (unit or sector) postcodes within specific distances of
        input location.

        Parameters
        ----------
        X : arraylike
            Latitude-longitude pair of centre location
        radii : arraylike
            array of radial distances from X
        sector : bool, optional
            if true return postcode sectors, otherwise postcode units

        Returns
        -------
        list of lists
            Contains the lists of postcodes closer than the elements of radii to the location X.


        Examples
        --------

        >>> locator = PostcodeLocator(postcode_file='./resources/full_postcodes.csv', census_file='./resources/population_by_postcode_sector.csv')
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773), [0.13e3])
        [['SW7 2AZ', 'SW7 2BT', 'SW7 2BU', 'SW7 2DD', 'SW7 5HF', 'SW7 5HG', 'SW7 5HQ']]
        >>> locator.get_postcodes_by_radius((51.4981, -0.1773), [0.4e3, 0.2e3], True)
        [['SW7 1', 'SW7 2', 'SW7 3', 'SW7 4', 'SW7 5', 'SW7 9'], ['SW7 1', 'SW7 2', 'SW7 3', 'SW7 4', 'SW7 5', 'SW7 9']]
        """

        # read the file, data type conversion and prepare data.
        df = pd.read_csv(self.postcode_file)

        X = list(X)

        res = []
        if not sector:
            # for unit
            # units array(1D)
            postcodes_array = df[['Postcode']].values.flatten()
            # unit coordinates array list(2D)
            coordinates_array = df[['Latitude', 'Longitude']].values.tolist()
            # calculate the distance for all postcodes.
            distances = self.norm(coordinates_array, X)
            # distance with X array(1D: [1, ])
            distances_array = distances.flatten()
            for ra in radii:
                # for each radius, find out units in zone
                list_ra = postcodes_array[distances_array < ra].tolist()
                # current radius list added to all_radii_list
                res.append(list_ra)
        else:
            # for sector

            # add a new sector column in df and group by the value of this column.
            df['sector'] = df['Postcode'].str[0:5]

            units_coordinates = df[['Latitude', 'Longitude']].values.tolist()
            distances = self.norm(units_coordinates, X)
            distances = distances.flatten()
            df['distance'] = distances

            # group by sector
            group = df.groupby('sector')
            # for each group(each sector), calculate the min unit distance
            data = group['distance'].min()
            min_distances = data.values

            # array of sectors
            sector_array = np.array(data.index)

            for ra in radii:
                # for each radius, find sectors in this zone
                # If at least one unit in one sector is within this zone, this sector is within this zone.
                list_ra = sector_array[min_distances < ra].tolist()
                # current radius list added to all_radii_list
                res.append(list_ra)
        return res

    def get_population_of_postcode(self, postcodes, sector=False):
        """
        Return populations of a list of postcode units or sectors.

        Parameters
        ----------
        postcodes : list of lists
            list of postcode units or postcode sectors
        sector : bool, optional
            if true return populations for postcode sectors, otherwise postcode units

        Returns
        -------
        list of lists
            Contains the populations of input postcode units or sectors


        Examples
        --------

        >>> locator = PostcodeLocator(postcode_file='./resources/full_postcodes.csv', census_file='./resources/population_by_postcode_sector.csv')
        >>> locator.get_population_of_postcode([['SW7 2AZ', 'SW7 2BT', 'SW7 2BU', 'SW7 2DD'], ['SA8 3AB', 'SA8 3AD', 'SA8 3AE']])
        [[18, 18, 18, 18], [44, 44, 44]]
        >>> locator.get_population_of_postcode([['SW7  2']], True)
        [[2283]]
        >>> locator.get_population_of_postcode([[]], True)
        [[]]
        >>> locator.get_population_of_postcode([], True)
        []
        """
        census_df = pd.read_csv(self.census_file)
        postcodes_df = pd.read_csv(self.postcode_file)

        res = []
        if sector:
            # for sector
            # if sector input each does not have one extra space, such as 'SA8 3', eliminate extra space in census_df to unify the form.
            if len(postcodes) > 0 and len(postcodes[0]) > 0 and len(postcodes[0][0]) == 5:
                census_df['geography'] = census_df['geography'].str[:4] + census_df['geography'].str[5:]

            for level in postcodes:
                level_list = []
                for postcode in level:
                    # for each sector, look up the population in census_df, if not exist, population = 0
                    row_select = census_df[census_df['geography'] == postcode]
                    population = row_select.iloc[0]['Variable: All usual residents; measures: Value'] if row_select.shape[0] > 0 else 0
                    level_list.append(population)
                res.append(level_list)
        else:
            # for unit
            # unit input each does not has one extra space, which is different with the form in censtus
            # eliminate extra space in census_df to unify the form
            census_df['geography'] = census_df['geography'].str[:4] + census_df['geography'].str[5:]
            for level in postcodes:
                level_list = []
                for postcode in level:
                    # obtain the sector which this unit belongs to
                    sector_postcode = postcode[:-2]
                    # obtain the corresponding row in census_df for the sector
                    row_select = census_df[census_df['geography'] == sector_postcode]
                    if row_select.shape[0] > 0:
                        # obetain the population for the sector
                        population_in_sector = row_select.iloc[0]['Variable: All usual residents; measures: Value']
                        # use sector to look up units in postcodes_df
                        units_in_sector = postcodes_df[postcodes_df['Postcode'].str.contains(sector_postcode)]
                        # calculate the number of units in this sector
                        num_units = units_in_sector.shape[0]
                        # unit population = sector population / units number (use the average as unit population)
                        population = int(population_in_sector / num_units)
                    else:
                        # if sector which unit belongs to does not exist, unit population = 0
                        population = 0
                    level_list.append(population)
                res.append(level_list)
        return res

locator = PostcodeLocator()
postcodes = locator.get_postcodes_by_radius((51.499840, -0.124663), [0.13e4, 0.13e3], sector=False)
print(postcodes)