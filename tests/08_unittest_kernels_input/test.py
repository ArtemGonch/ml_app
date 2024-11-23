import numpy as np
from solution import *
from common import assert_ndarray_equal


def test_linear_kernel_1():
    x = (np.array([-0.06, -0.48591111]),  np.array([0.9090147, -0.448]))

    actual = kernel_linear(x[0], x[1])

    correct = np.array(0.163147)
    
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 1 Failed")


def test_linear_kernel_2():
    x = (np.array([-0.06, -0.48591111]),  np.array([0.9090147, -0.448]))

    actual = kernel_linear(x[1], x[0])

    correct = np.array(0.163147)
    
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 2 Failed")


def test_rbf_kernel_1():
    x = (np.array([-0.06, -0.48591111]),  np.array([0.9090147, -0.448]))

    actual = kernel_rbf(x[0], x[1])

    correct = np.array(0.624869)
    
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 3 Failed")


def test_rbf_kernel_2():
    x = (np.array([-0.06, -0.48591111]),  np.array([0.9090147, -0.448]))

    actual = kernel_rbf(x[0], x[1], l=2)

    correct = np.array(0.889093)
    
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 4 Failed")


def test_rbf_kernel_3():
    x = (np.array([-0.06, -0.48591111]),  np.array([0.9090147, -0.448]))

    actual = kernel_rbf(x[0], x[1], l=0.5)

    correct = np.array(0.15246)
    
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 5 Failed")


def test_rbf_kernel_4():
    x = (np.array([-0.06, -0.48591111]),  np.array([0.9090147, -0.448]))

    actual = kernel_rbf(x[1], x[0], l=0.5)

    correct = np.array(0.15246)
    
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 6 Failed")



def test_poly_kernel_1():
    x = (np.array([-0.06, -0.48591111]),  np.array([0.9090147, -0.448]))

    actual = kernel_poly(x[0], x[1])

    correct = np.array(1.352912)
    
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 7 Failed")


def test_poly_kernel_2():
    x = (np.array([-0.06, -0.48591111]),  np.array([0.9090147, -0.448]))

    actual = kernel_poly(x[0], x[1], d=2)

    correct = np.array(1.352912)
    
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 8 Failed")


def test_poly_kernel_3():
    x = (np.array([-0.06, -0.48591111]),  np.array([0.9090147, -0.448]))

    actual = kernel_poly(x[0], x[1], d=3)

    correct = np.array(1.573636)
    
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 9 Failed")


def test_poly_kernel_4():
    x = (np.array([-0.06, -0.48591111]),  np.array([0.9090147, -0.448]))

    actual = kernel_poly(x[1], x[0], d=3)

    correct = np.array(1.573636)
    
    assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 10 Failed")