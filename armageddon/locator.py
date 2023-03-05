"""Module dealing with postcode information."""

import os
import copy
import numpy as np
import pandas as pd

__all__ = ['PostcodeLocator', 'great_circle_distance']


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the haversine distance between two poins (lat, lon)
    on earth with radius 6371 km.
    Parameters
    ----------
    lat1: float
        latitude value from first point.
    lon1: float
        longitude value from first point.
    lat2: float
        latitude value from second point.
    lon2: float
        longitude value from second point.
    Returns
    -------
    numpy.ndarray
        Distance in two points on earth in meters.
    Examples
    --------
    >>> res = haversine(10, 12, 31.321, 12.987)
    >>> np.isclose(res, 2372973.2601616434)
    True
    """

    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)

    lond = np.abs(lon2 - lon1)
    latd = np.abs(lat2 - lat1)

    tmp = np.sin(latd/2.)**2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(lond/2.)**2

    return np.abs(2 * np.arcsin(np.sqrt(tmp)) * 6_371_000)


def great_circle_distance(latlon1, latlon2):
    """
    Calculate the great circle distance (in metres) between pairs of
    points specified as latitude and longitude on a spherical Earth
    (with radius 6371 km).
    Parameters
    ----------
    latlon1: arraylike
        latitudes and longitudes of first point (as [n, 2] array for n points)
    latlon2: arraylike
        latitudes and longitudes of second point (as [m, 2] array for m points)
    Returns
    -------
    numpy.ndarray
        Distance in metres between each pair of points (as an n x m array)
    Examples
    --------
    >>> import numpy
    >>> res = great_circle_distance([[54.0, 0.0], [55, 0.0]], [55, 1.0])
    >>> np.allclose(res, [[128580.53670808], [63778.24657475]])
    True
    """

    # Copying
    latlon1 = copy.deepcopy(latlon1)
    latlon2 = copy.deepcopy(latlon2)

    # check data type (latlon1)
    if isinstance(latlon1, list) or isinstance(latlon1, tuple):
        latlon1 = np.array(latlon1)

    elif isinstance(latlon1, np.ndarray):
        pass

    else:
        raise ValueError("'latlon1' is not from type list "
                         "of lists or numpy array!")

    # check data type (latlon2)
    if isinstance(latlon2, list) or isinstance(latlon2, tuple):
        latlon2 = np.array(latlon2)

    elif isinstance(latlon2, np.ndarray):
        pass

    else:
        raise ValueError("'latlon2' is not from type list "
                         "of lists or numpy array!")

    # Check sizes
    if latlon1.size == 0:
        raise ValueError("'latlon1' is empty!")

    if latlon2.size == 0:
        raise ValueError("'latlon2' is empty!")

    # reshapeing arrays [1, 2] -> [[1, 2]]
    if latlon1.ndim == 1:
        latlon1 = latlon1.reshape(1, 2)

    if latlon2.ndim == 1:
        latlon2 = latlon2.reshape(1, 2)

    n = latlon1.shape[0]
    m = latlon2.shape[0]

    # vectorized
    df = pd.DataFrame(columns=['lat1', 'lon1', 'lat2', 'lon2'])

    # create dataframe with lat1, lon1, lat2, lon2 as columns
    df['lat1'] = np.tile(latlon1[:, 0], m)
    df['lon1'] = np.tile(latlon1[:, 1], m)
    df['lat2'] = np.repeat(latlon2[:, 0], n)
    df['lon2'] = np.repeat(latlon2[:, 1], n)

    # create new column that represents the distance
    # for the pair (lat1, lon1) - (lat2, lon2)
    df['distance'] = haversine(df['lat1'], df['lon1'], df['lat2'], df['lon2'])

    return df['distance'].values.reshape(m, n).T


class PostcodeLocator(object):
    """Class to interact with a postcode database file."""

    def __init__(self,
                 postcode_file=os.sep.join(
                     (os.path.dirname(__file__), '..',
                      'resources',
                      'full_postcodes.csv')),
                 census_file=os.sep.join(
                     (os.path.dirname(__file__), '..',
                      'resources',
                      'population_by_postcode_sector.csv')),
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
            Python function defining the distance between points in
            latitude-longitude space.
        """

        self.postcode_data = pd.read_csv(postcode_file, usecols=['Postcode',
                                                                 'Latitude',
                                                                 'Longitude'])
        self.census_data = pd.read_csv(census_file,
                                       usecols=['geography code',
                                                'Variable: All usual ' +
                                                'residents; measures: Value'])

        # remove wehitespaces
        self.census_data['geography code'] = \
            self.census_data['geography code'] \
                .apply(lambda x: x.replace(' ', ''))

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
            Contains the lists of postcodes closer than the elements
            of radii to the location X.
        Examples
        --------
        >>> locator = PostcodeLocator()
        >>> res = locator.get_postcodes_by_radius((51.4981, -0.1773), [0.13e3])
        >>> res == [['SW7 2AZ', 'SW7 2BT', 'SW7 2BU',
        ... 'SW7 2DD', 'SW7 5HF',
        ... 'SW7 5HG', 'SW7 5HQ']]
        True
        """

        # Copying
        X = copy.deepcopy(X)
        radii = copy.deepcopy(radii)

        # check data type (X)
        if isinstance(X, list) or isinstance(X, tuple):
            X = np.array(X)

        elif isinstance(X, np.ndarray):
            pass

        else:
            raise ValueError("'X' is not from type list "
                             "of lists or numpy array!")

        # check data type (radii)
        if isinstance(radii, list) or isinstance(radii, tuple):
            radii = np.array(radii)

        elif isinstance(radii, np.ndarray):
            pass

        else:
            raise ValueError("'radii' is not from type list "
                             "of lists or numpy array!")

        # check data type (sector)
        if isinstance(sector, bool):
            pass

        else:
            raise ValueError("'sector' is not from type bool!")

        # Check sizes
        if X.size == 0:
            raise ValueError("'X' is empty!")

        if radii.size == 0:
            raise ValueError("'radii' is empty!")

        # calculate the distance from every postcode to X
        self.postcode_data['distances'] = \
            self.norm(self.postcode_data[['Latitude', 'Longitude']].values, X)

        result = []

        # check if distance is smaller or equal to radius
        for r in radii:

            res = \
                self.postcode_data[
                    self.postcode_data['distances'] <= r]['Postcode'].values

            # if only sectors should be returned -> only keep first
            # 5 character und make the list unique
            if sector:
                result.append(list(dict.fromkeys(res.astype('U5'))))

            else:
                result.append(list(res))

        return result

    def get_population_of_postcode(self, postcodes, sector=False):
        """
        Return populations of a list of postcode units or sectors.
        Parameters
        ----------
        postcodes : list of lists
            list of postcode units or postcode sectors
        sector : bool, optional
            if true return populations for postcode sectors,
            otherwise returns populations for postcode units
        Returns
        -------
        list of lists
            Contains the populations of input postcode units or sectors
        Examples
        --------
        >>> locator = PostcodeLocator()
        >>> res = locator.get_population_of_postcode([['SW7 2AZ', 'SW7 2BT',\
        'SW7 2BU', 'SW7 2DD']])
        >>> np.allclose(res, [[18.71311475409836, 18.71311475409836,\
        18.71311475409836, 18.71311475409836]])
        True
        >>> res = locator.get_population_of_postcode([['SW7  2']], True)
        >>> np.allclose(res, [[2283]])
        True
        """

        # Copying
        postcodes = copy.deepcopy(postcodes)

        # check data type (postcodes)
        # not using numpy array becauase 'rows' can have
        # different length
        if isinstance(postcodes, list):
            pass

        else:
            raise ValueError("'postcodes' is not from type list "
                             "of lists!")

        # check data type (sector)
        if isinstance(sector, bool):
            pass

        else:
            raise ValueError("'sector' is not from type bool!")

        # Check sizes
        if not any(postcodes):
            if len(postcodes) == 0:
                raise ValueError("'postcodes' is empty!")

            else:
                return [[0]] * len(postcodes)

        result = []
        COL_NAME_POPULATION = 'Variable: All usual residents; measures: Value'

        # If sectors should be returned
        if sector:
            for row in postcodes:

                # if no there are no postcodes for
                # one damage level -> 0 peope are in
                # danger
                if len(row) == 0:
                    result.append([0])
                    continue

                # remove whitespaces
                postcodes_df = pd.DataFrame(row, columns=['postcodes'])
                postcodes_df['postcodes'] = postcodes_df['postcodes'].apply(
                    lambda x: x.replace(' ', ''))

                # merging postcodes with census data to get population size
                merged_df = postcodes_df.merge(
                    self.census_data,
                    how='left',
                    left_on='postcodes',
                    right_on='geography code'
                ).fillna(0.0)

                result.append(merged_df[COL_NAME_POPULATION].values.tolist())

        # If units should be returned
        else:
            for row in postcodes:
                row = np.array(row).astype('U5')

                # if no there are no postcodes for
                # one damage level -> 0 peope are in
                # danger
                if len(row) == 0:
                    result.append([0])
                    continue

                # remove whitespaces
                postcodes_df = pd.DataFrame(row, columns=['postcodes'])
                postcodes_df['postcodes'] = postcodes_df['postcodes'].apply(
                    lambda x: x.replace(' ', ''))

                # getting the number of units for each sector
                number_units_series = pd.Series(
                    self.postcode_data['Postcode'].values.astype('U5'),
                    name='postcode_number'
                ).apply(lambda x: x.replace(' ', '')).value_counts()

                # merging postcodes with the number of unitzs for
                # each sector
                number_postcodes_df = postcodes_df.merge(
                    number_units_series,
                    how='left',
                    left_on='postcodes',
                    right_index=True)

                # merging postcodes with census data to get population size
                merged_df = number_postcodes_df.merge(
                    self.census_data,
                    how='left',
                    left_on='postcodes',
                    right_on='geography code'
                ).fillna(0.0)

                merged_df['population'] = merged_df[COL_NAME_POPULATION] / \
                    merged_df['postcode_number']

                result.append(merged_df['population'].values.tolist())

        return result
