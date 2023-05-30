import pytest
import numpy as np

from ..src.polute_process import *

# from polute_process import PoluteProcessInspector, get_drain_characteristic_function

def get_test_inspector():
    area_size = np.array([9, 9])
    partion = 10, 10
    k_1 = 10
    k_2 = 10
    u_cr = 12
    drain_characteristic = get_drain_characteristic_function(u_critical=u_cr)
    sourse = 5

    John = PoluteProcessInspector(area_size=area_size, partion=partion, k_1=k_1, k_2=k_2,
                                  drain_characteristic=drain_characteristic, sourse=sourse)

    return John

def test_index_of():
    John = get_test_inspector()
    
    assert John._index_of(c=3, l=4) == 43
