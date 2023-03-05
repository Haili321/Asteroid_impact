from importlib import import_module
import doctest
import pytest


class TestDocStrings(object):

    @pytest.mark.parametrize('module_name', [
        ('armageddon.solver'),
        ('armageddon.damage'),
        ('armageddon.locator'),
        ('armageddon.mapping'),
    ])
    def testFunctions(self, module_name):
        mod = import_module(module_name)
        failures, _ = doctest.testmod(mod, report=True)

        assert failures == 0
