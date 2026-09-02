import pytest
from ffem.mapping.radial_resolution import RadialResolutionPolicy

def test_required_resolution_bands():
    p=RadialResolutionPolicy(); assert p.cell_size(1,1)==pytest.approx(.05); assert p.cell_size(20,0)==pytest.approx(.10); assert p.cell_size(40,0)==pytest.approx(.25); assert p.cell_size(80,0)==pytest.approx(.50)

def test_policy_validation():
    assert RadialResolutionPolicy().validate()
