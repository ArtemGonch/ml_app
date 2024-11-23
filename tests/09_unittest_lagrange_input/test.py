import numpy as np
from solution import *
from common import assert_ndarray_equal


def test_lagrange():
    GramXy = np.array([[0.23981859, 0.0585316, 0.2107551], 
                  [0.0585316, 0.20847433, 0.21285568],
                  [0.2107551, 0.21285568, 0.31939051]])
    
    a = np.array([1, 1, 1])
    
    actual = lagrange(GramXy, a)

    correct = np.array(2.134016)

    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 1 Failed")


def test_lagrange_derive():
    GramXy = np.array([[0.23981859, 0.0585316, 0.2107551], 
                  [0.0585316, 0.20847433, 0.21285568],
                  [0.2107551, 0.21285568, 0.31939051]])
    
    a = np.array([1, 1, 1])
    
    actual = lagrange_derive(GramXy, a)

    correct = np.array([0.490895, 0.520138, 0.256999])

    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 2 Failed")
