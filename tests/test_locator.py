import numpy as np
import pytest
import copy
import armageddon


# Tesing distance function
class TestDistance(object):

    # test valid input datatypes
    @pytest.mark.parametrize('latlon1, latlon2', [
        ([[20.0, 10.0], [15.0, 0.0]], [33.0, 2.0]),
        ([[20.0, 10], [15, 0.0]], [33.0, 2.0]),
        ([[20, 0], [55, 0]], [55, 1]),
        ([20, 0], [[55, 1], [55, 0]]),
        ([(20, 0), [55, 0]], [55, 1]),
        ([(20, 0), (55, 0)], [55, 1]),
        (((20, 0), (55, 0)), [55, 1]),
        (([20, 0], [55, 0]), [55, 1]),
        ([[20, 0], [55, 0]], (55, 1)),
        ([[20, 0], [55, 0]], [[55, 1]]),
        ([[20, 0], [55, 0]], ((55, 1))),
        ([[20, 0], [55, 0]], ([55, 1])),
        ([[20, 0], [55, 0]], [(55, 1)]),
        (np.array([[20, 0], [15, 0]]), [33, 2]),
        ([[20, 0], [15, 0]], np.array([33, 2])),
        (np.array([[20, 0], [15, 0]]), np.array([33, 2])),
        (np.array([[20, 0], [15, 0]]), np.array([[33, 2]])),
        (np.array([15, 0]), np.array([[33, 2], [20, 0]])),
    ])
    def testValidDataTypes(self, latlon1, latlon2):
        try:
            armageddon.great_circle_distance(latlon1, latlon2)

        except ValueError:
            assert False

    # test not valid input datatypes (latlon1)
    @pytest.mark.parametrize('latlon1, latlon2', [
        ({'key': 'value'}, [15.0, 0.0]),
        (True, [33.0, 2.0]),
        ('testing', [[20, 0], [55, 0]])
    ])
    def testNotValidDataTypesLatLon1(self, latlon1, latlon2):
        with pytest.raises(ValueError,
                           match="'latlon1' is not from type list "
                           "of lists or numpy array!"):
            armageddon.great_circle_distance(latlon1, latlon2)

    # test not valid input datatypes (latlon2)
    @pytest.mark.parametrize('latlon1, latlon2', [
        ([15.0, 0.0], {'key': 'value'}),
        ([33.0, 2.0], True),
        ([[20, 0], [55, 0]], 'testing')
    ])
    def testNotValidDataTypesLatLon2(self, latlon1, latlon2):
        with pytest.raises(ValueError,
                           match="'latlon2' is not from type list "
                           "of lists or numpy array!"):
            armageddon.great_circle_distance(latlon1, latlon2)

    # test if function changed input data
    @pytest.mark.parametrize('latlon1, latlon2', [
        ([[20, 0], [55, 0]], [55, 1]),
        (((20, 0), (55, 0)), [55, 1]),
        ([[20, 0], [55, 0]], (55, 1)),
        ([[20, 0], [55, 0]], ((55, 1))),
        ([[20, 0], [55, 0]], ([55, 1])),
        (np.array([[20, 0], [15, 0]]), [33, 2]),
        ([[20, 0], [15, 0]], np.array([33, 2])),
    ])
    def testIfInputChanged(self, latlon1, latlon2):
        input1 = copy.deepcopy(latlon1)
        input2 = copy.deepcopy(latlon2)

        armageddon.great_circle_distance(latlon1, latlon2)

        assert type(input1) == type(latlon1)
        assert type(input2) == type(latlon2)

        if isinstance(input1, np.ndarray):
            assert np.array_equal(input1, latlon1)

        else:
            assert input1 == latlon1

        if isinstance(input2, np.ndarray):
            assert np.array_equal(input2, latlon2)

        else:
            assert input2 == latlon2

    # test empty input (latlon1)
    @pytest.mark.parametrize('latlon1, latlon2', [
        ([], [15.0, 0.0]),
        ((), [33.0, 2.0]),
        (np.array([]), [[20, 0], [55, 0]])
    ])
    def testEmptyInputLatLon1(self, latlon1, latlon2):
        with pytest.raises(ValueError,
                           match="'latlon1' is empty!"):
            armageddon.great_circle_distance(latlon1, latlon2)

    # test empty input (latlon2)
    @pytest.mark.parametrize('latlon1, latlon2', [
        ([15.0, 0.0], []),
        ([33.0, 2.0], ()),
        ([[20, 0], [55, 0]], np.array([]))
    ])
    def testEmptyInputLatLon2(self, latlon1, latlon2):
        with pytest.raises(ValueError,
                           match="'latlon2' is empty!"):
            armageddon.great_circle_distance(latlon1, latlon2)

    # test output
    @pytest.mark.parametrize('latlon1, latlon2, result', [
        ([38.8976, -77.0366], [38.8976, -77.0366], [[0]]),
        ([38.8976, -77.0366], [39.9496, -75.1503], [[199830.22873474]]),
        ([38.8976, -77.0366],
            [[39.9496, -75.1503], [38.8976, -77.0366]],
            [[199830.22873474, 0]]),
        ([[38.8976, -77.0366], [39.9496, -75.1503]],
            [[39.9496, -75.1503]],
            [[199830.22873474], [0]])
    ])
    def testOutput(self, latlon1, latlon2, result):
        output = armageddon.great_circle_distance(latlon1, latlon2)
        print(output)
        assert (np.isclose(output, result)).all()


@pytest.fixture(scope='module')
def locator():
    return armageddon.PostcodeLocator()

# Tesing postcodes function


@pytest.mark.usefixtures('locator')
class TestPostcodes(object):

    # test valid input datatypes
    @pytest.mark.parametrize('X, radii, sector', [
        ([15.0, 0.0], [33.0, 2.0, 21, 34], False),
        ([15.0, 0.0], [33.0, 2.0, 21, 34], True),
        ((15.0, 0.0), [33.0, 2.0, 21, 34], True),
        ([15.0, 0.0], (33.0, 2.0, 21, 34), False),
        ((15.0, 0.0), (33.0, 2.0, 21, 34), True),
    ])
    def testValidDataTypes(self, locator, X, radii, sector):
        try:
            locator.get_postcodes_by_radius(X, radii, sector)

        except ValueError:
            assert False

    # test not valid input datatypes (X)
    @pytest.mark.parametrize('X, radii, sector', [
        (True, [33.0, 2.0, 21, 34], True),
        ('test', [33.0, 2.0, 21, 34], True),
        (123, [33.0, 2.0, 21, 34], True),
        ({'key': 'value'}, [33.0, 2.0, 21, 34], True)
    ])
    def testNotValidDataTypesX(self, locator, X, radii, sector):
        with pytest.raises(ValueError,
                           match="'X' is not from type list "
                           "of lists or numpy array!"):
            locator.get_postcodes_by_radius(X, radii, sector)

    # test not valid input datatypes (radii)
    @pytest.mark.parametrize('X, radii, sector', [
        ([15.0, 0.0], True, True),
        ([15.0, 0.0], 'test', True),
        ([15.0, 0.0], 123, True),
        ([15.0, 0.0], {'key': 'value'}, True)
    ])
    def testNotValidDataTypesRadii(self, locator, X, radii, sector):
        with pytest.raises(ValueError,
                           match="'radii' is not from type list "
                           "of lists or numpy array!"):
            locator.get_postcodes_by_radius(X, radii, sector)

    # test not valid input datatypes (sector)
    @pytest.mark.parametrize('X, radii, sector', [
        ([15.0, 0.0], [33.0, 2.0, 21, 34], 123),
        ([15.0, 0.0], [33.0, 2.0, 21, 34], 'test'),
        ([15.0, 0.0], [33.0, 2.0, 21, 34], [1, 2]),
        ([15.0, 0.0], [33.0, 2.0, 21, 34], np.array([1, 2])),
        ([15.0, 0.0], [33.0, 2.0, 21, 34], {'key': 'value'})
    ])
    def testNotValidDataTypesSector(self, locator, X, radii, sector):
        with pytest.raises(ValueError,
                           match="'sector' is not from type bool!"):
            locator.get_postcodes_by_radius(X, radii, sector)

    # test if function changed input data
    @pytest.mark.parametrize('X, radii, sector', [
        ([15.0, 0.0], [33.0, 2.0, 21, 34], True),
        ((15.0, 0.0), [33.0, 2.0, 21, 34], True),
        ([15.0, 0.0], (33.0, 2.0, 21, 34), True),
    ])
    def testIfInputChanged(self, locator, X, radii, sector):
        input1 = copy.deepcopy(X)
        input2 = copy.deepcopy(radii)
        input3 = copy.deepcopy(sector)

        locator.get_postcodes_by_radius(X, radii, sector)

        assert type(input3) == type(sector)

        if isinstance(input1, np.ndarray):
            assert np.array_equal(input1, X)

        else:
            assert input1 == X

        if isinstance(input2, np.ndarray):
            assert np.array_equal(input2, radii)

        else:
            assert input2 == radii

    # test empty input (X)
    @pytest.mark.parametrize('X, radii, sector', [
        ([], [15.0, 0.0], True),
        ((), [33.0, 2.0], True),
        (np.array([]), [20, 0, 55, 0], True)
    ])
    def testEmptyInputX(self, locator, X, radii, sector):
        with pytest.raises(ValueError,
                           match="'X' is empty!"):
            locator.get_postcodes_by_radius(X, radii, sector)

    # test empty input (radii)
    @pytest.mark.parametrize('X, radii, sector', [
        ([15.0, 0.0], [], True),
        ([15.0, 0.0], (), True),
        ([15.0, 0.0], np.array([]), True)
    ])
    def testEmptyRadii(self, locator, X, radii, sector):
        with pytest.raises(ValueError,
                           match="'radii' is empty!"):
            locator.get_postcodes_by_radius(X, radii, sector)

    # test output
    @pytest.mark.parametrize('X, radii, sector, result', [
        ([0, 0], [0, 0, 0, 0], True, [[], [], [], []]),
        ([-33.865143, 151.209900],
         [50000, 4000, 3000, 20],
         True, [[], [], [], []])
    ])
    def testOutput(self, locator, X, radii, sector, result):
        output = locator.get_postcodes_by_radius(X, radii, sector)
        assert np.allclose(output, result)


# Tesing postcodes function
@pytest.mark.usefixtures('locator')
class TestPopulation(object):

    # test valid input datatypes
    @pytest.mark.parametrize('postcodes, sector', [
        ([[], [], []], True),
        ([['SE1 8ST', 'SE2 2SA'], ['SE8 2ST'], []], True),
        ([['SE1 8ST', 'SE2 2SA'], ['SE8 2ST'], []], False),
    ])
    def testValidDataTypes(self, locator, postcodes, sector):
        try:
            locator.get_population_of_postcode(postcodes, sector)

        except ValueError:
            assert False

    # test not valid input datatypes (postcodes)
    @pytest.mark.parametrize('postcodes, sector', [
        (True, True),
        (123, True),
        ('Test', True),
        ({'key': 'value'}, True),
        ((), True),
    ])
    def testNotValidDataTypesPostcodes(self, locator, postcodes, sector):
        with pytest.raises(ValueError,
                           match="'postcodes' is not from type list "
                           "of lists!"):
            locator.get_population_of_postcode(postcodes, sector)

    # test not valid input datatypes (sector)
    @pytest.mark.parametrize('postcodes, sector', [
        ([[], [], []], 123),
        ([[], [], []], 'test'),
        ([[], [], []], {'key': 'value'}),
        ([[], [], []], [123, 234]),
        ([[], [], []], (12, 34)),
        ([[], [], []], np.array([1, 2]))
    ])
    def testNotValidDataTypesSector(self, locator, postcodes, sector):
        with pytest.raises(ValueError,
                           match="'sector' is not from type bool!"):
            locator.get_population_of_postcode(postcodes, sector)

    # test if function changed input data
    @pytest.mark.parametrize('postcodes, sector', [
        ([[], [], []], True),
        ([['SE1 8ST', 'SE2 2SA'], ['SE8 2ST'], []], True),
        ([['SE1 8ST', 'SE2 2SA']], True),
        ([['SE1 8ST', 'SE2 2SA'], ['SE8 2ST']], True)
    ])
    def testIfInputChanged(self, locator, postcodes, sector):
        input1 = copy.deepcopy(postcodes)
        input2 = copy.deepcopy(sector)

        locator.get_population_of_postcode(postcodes, sector)

        assert type(input2) == type(sector)

        if isinstance(input1, np.ndarray):
            assert np.array_equal(input1, postcodes)

        else:
            assert input1 == postcodes

    # test empty input (postcodes)
    @pytest.mark.parametrize('postcodes, sector', [
        ([], True),
        ([], False),
    ])
    def testEmptyInputPostcodes(self, locator, postcodes, sector):
        with pytest.raises(ValueError,
                           match="'postcodes' is empty!"):
            locator.get_population_of_postcode(postcodes, sector)

    # test output
    @pytest.mark.parametrize('postcodes, sector, result', [
        ([[], [], []], True, [[0], [0], [0]]),
        ([['AL11'], [], []], True, [[5453], [0], [0]]),
        ([['AL11', 'SA733'], [], ['SA92']], True, [[5453, 5246], [0], [7281]]),
        ([['AL11', 'SA733'], ['SA91', 'SA92'], ['SA84', 'SA92']], True,
            [[5453, 5246],
             [7898, 7281],
             [7787, 7281]]),
        ([[], [], []], False, [[0], [0], [0]]),
        ([['AL1 1AG'], [], []], False, [[28.54973821989529], [0], [0]]),
        ([['AL1 1AG'], ['AL1 1AG'], ['AL1 1AG']], False,
            [[28.54973821989529],
             [28.54973821989529],
             [28.54973821989529]]),
        ([['AL1 1AG', 'SE1 8YU'], [], ['SE1 8YU']], False,
            [[28.54973821989529, 22.119565217391305],
             [0],
             [22.119565217391305]]),
    ])
    def testOutput(self, locator, postcodes, sector, result):
        output = locator.get_population_of_postcode(postcodes, sector)
        print(output)
        assert output == result
