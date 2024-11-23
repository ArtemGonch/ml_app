import numpy as np
from solution import *
from common import assert_ndarray_equal

def test_step_1():
    model = BinaryEstimatorSVM(C=1.0,
                            fit_intercept=True)
    
    model.coef_ = np.array([[-1.31980307], [ 0.33426974]])
    model.intercept_ = 2.3

    X = np.array([[-1.27599669, 0.83473676]
                , [0.834732, 0.40096667]
                , [0.71524899, 0.94828181]
                , [-1.36385697, 1.08768775]
                , [-1.44876697, 0.79743398]
                , [0.5275475, 0.50570868]
                , [-1.15731421, -0.67281411]
                , [-1.41285223, 0.72616832]
                , [-1.26116455, 1.0807142]
                , [-0.00422831, -2.0688571]
                , [0.59346072, 0.58952085]
                , [-0.01037196, -1.78490712]
                , [-1.27959017, -0.64675803]
                , [-1.20396896, 0.5540786]
                , [0.99893463, 0.19640479]
                , [1.2213383, 0.18877463]
                , [-1.23188415, -0.68323095]
                , [0.1109394, -1.69829392]
                , [-1.41501468, 1.28928821]])
    
    y = np.array([-1., -1., -1., -1., -1., -1., 1., -1., -1., -1., -1., -1., 1., -1., -1., -1., 1., -1., -1.])
    
    _ = model.loss(X, y)

    actual_grad, actual_grad_intercept = model.loss_grad(X, y)

    model.step(actual_grad, actual_grad_intercept)

    correct_coef, correct_intercept = np.array([[-1.262665], [ 0.29445 ]]), np.array(2.14)
    
    assert_ndarray_equal(actual=model.coef_, correct=correct_coef,  err_msg="Test 1 Failed")
    assert_ndarray_equal(actual=model.intercept_, correct=correct_intercept,  err_msg="Test 1 Failed")


def test_step_2():
    model = BinaryEstimatorSVM(C=0.007,
                            fit_intercept=True)
    
    model.coef_ = np.array([[-1.31980307], [ 0.33426974]])
    model.intercept_ = 2.3

    X = np.array([[-1.27599669, 0.83473676]
                , [0.834732, 0.40096667]
                , [0.71524899, 0.94828181]
                , [-1.36385697, 1.08768775]
                , [-1.44876697, 0.79743398]
                , [0.5275475, 0.50570868]
                , [-1.15731421, -0.67281411]
                , [-1.41285223, 0.72616832]
                , [-1.26116455, 1.0807142]
                , [-0.00422831, -2.0688571]
                , [0.59346072, 0.58952085]
                , [-0.01037196, -1.78490712]
                , [-1.27959017, -0.64675803]
                , [-1.20396896, 0.5540786]
                , [0.99893463, 0.19640479]
                , [1.2213383, 0.18877463]
                , [-1.23188415, -0.68323095]
                , [0.1109394, -1.69829392]
                , [-1.41501468, 1.28928821]])
    
    y = np.array([-1., -1., -1., -1., -1., -1., 1., -1., -1., -1., -1., -1., 1., -1., -1., -1., 1., -1., -1.])
    
    _ = model.loss(X, y)

    actual_grad, actual_grad_intercept = model.loss_grad(X, y)

    model.step(actual_grad, actual_grad_intercept)

    correct_coef, correct_intercept = np.array([[-1.27577 ], [ 0.297769]]), np.array(2.14)

    assert_ndarray_equal(actual=model.coef_, correct=correct_coef,  err_msg="Test 2 Failed")
    assert_ndarray_equal(actual=model.intercept_, correct=correct_intercept,  err_msg="Test 2 Failed")


def test_step_3():
    model = BinaryEstimatorSVM(C=0.0,
                            fit_intercept=True)
    
    model.coef_ = np.array([[-1.31980307], [ 0.33426974]])
    model.intercept_ = 2.3

    X = np.array([[-1.27599669, 0.83473676]
                , [0.834732, 0.40096667]
                , [0.71524899, 0.94828181]
                , [-1.36385697, 1.08768775]
                , [-1.44876697, 0.79743398]
                , [0.5275475, 0.50570868]
                , [-1.15731421, -0.67281411]
                , [-1.41285223, 0.72616832]
                , [-1.26116455, 1.0807142]
                , [-0.00422831, -2.0688571]
                , [0.59346072, 0.58952085]
                , [-0.01037196, -1.78490712]
                , [-1.27959017, -0.64675803]
                , [-1.20396896, 0.5540786]
                , [0.99893463, 0.19640479]
                , [1.2213383, 0.18877463]
                , [-1.23188415, -0.68323095]
                , [0.1109394, -1.69829392]
                , [-1.41501468, 1.28928821]])
    
    y = np.array([-1., -1., -1., -1., -1., -1., 1., -1., -1., -1., -1., -1., 1., -1., -1., -1., 1., -1., -1.])
    
    _ = model.loss(X, y)

    actual_grad, actual_grad_intercept = model.loss_grad(X, y)

    model.step(actual_grad, actual_grad_intercept)

    correct_coef, correct_intercept = np.array([[-1.275863], [ 0.297793]]), np.array(2.14)

    assert_ndarray_equal(actual=model.coef_, correct=correct_coef,  err_msg="Test 3 Failed")
    assert_ndarray_equal(actual=model.intercept_, correct=correct_intercept,  err_msg="Test 3 Failed")


def test_step_4():
    model = BinaryEstimatorSVM(C=0.007,
                            fit_intercept=False)
    
    model.coef_ = np.array([[-1.31980307], [ 0.33426974]])

    X = np.array([[-1.27599669, 0.83473676]
                , [0.834732, 0.40096667]
                , [0.71524899, 0.94828181]
                , [-1.36385697, 1.08768775]
                , [-1.44876697, 0.79743398]
                , [0.5275475, 0.50570868]
                , [-1.15731421, -0.67281411]
                , [-1.41285223, 0.72616832]
                , [-1.26116455, 1.0807142]
                , [-0.00422831, -2.0688571]
                , [0.59346072, 0.58952085]
                , [-0.01037196, -1.78490712]
                , [-1.27959017, -0.64675803]
                , [-1.20396896, 0.5540786]
                , [0.99893463, 0.19640479]
                , [1.2213383, 0.18877463]
                , [-1.23188415, -0.68323095]
                , [0.1109394, -1.69829392]
                , [-1.41501468, 1.28928821]])
    
    y = np.array([-1., -1., -1., -1., -1., -1., 1., -1., -1., -1., -1., -1., 1., -1., -1., -1., 1., -1., -1.])
    
    _ = model.loss(X, y)

    actual_grad, actual_grad_intercept = model.loss_grad(X, y)

    model.step(actual_grad, actual_grad_intercept)

    correct_coef = np.array([[-1.253568], [ 0.301621]])

    assert_ndarray_equal(actual=model.coef_, correct=correct_coef,  err_msg="Test 4 Failed")


def test_step_5():
    model = BinaryEstimatorSVM(C=0.0,
                            fit_intercept=False)
    
    model.coef_ = np.array([[-1.31980307], [ 0.33426974]])

    X = np.array([[-1.27599669, 0.83473676]
                , [0.834732, 0.40096667]
                , [0.71524899, 0.94828181]
                , [-1.36385697, 1.08768775]
                , [-1.44876697, 0.79743398]
                , [0.5275475, 0.50570868]
                , [-1.15731421, -0.67281411]
                , [-1.41285223, 0.72616832]
                , [-1.26116455, 1.0807142]
                , [-0.00422831, -2.0688571]
                , [0.59346072, 0.58952085]
                , [-0.01037196, -1.78490712]
                , [-1.27959017, -0.64675803]
                , [-1.20396896, 0.5540786]
                , [0.99893463, 0.19640479]
                , [1.2213383, 0.18877463]
                , [-1.23188415, -0.68323095]
                , [0.1109394, -1.69829392]
                , [-1.41501468, 1.28928821]])
    
    y = np.array([-1., -1., -1., -1., -1., -1., 1., -1., -1., -1., -1., -1., 1., -1., -1., -1., 1., -1., -1.])
    
    _ = model.loss(X, y)

    actual_grad, actual_grad_intercept = model.loss_grad(X, y)

    model.step(actual_grad, actual_grad_intercept)

    correct_coef = np.array([[-1.25366 ], [ 0.301644]])

    assert_ndarray_equal(actual=model.coef_, correct=correct_coef,  err_msg="Test 5 Failed")


def test_step_6():
    model = BinaryEstimatorSVM(C=0.1,
                            fit_intercept=True)
    
    model.coef_ = np.array([[-1.07], [ 0.6974]])
    model.intercept_ = -106.3

    X = np.array([[-1.27599669, 0.83473676]
                , [0.834732, 0.40096667]
                , [0.7152, 0.94828181]
                , [-1.36385697, 2.768775]
                , [-1.44897, 0.79743398]
                , [0.5275475, 0.50570868]
                , [-1.15731421, -0.67281411]
                , [-1.41285223, 0.72616832]
                , [-1.2455, 1.0807142]
                , [-0.081, -2.0688571]
                , [0.59346072, 0.58952085]
                , [-0.0196, -1.712]
                , [1.27959017, -0.64675803]
                , [-1., 0.5540786]
                , [0.99893463, 0.19640479]
                , [1.2213383, 0.18877463]
                , [1.23188415, -0.63095]
                , [0.1109394, -1.69829392]
                , [-1.41501468, 1.28928821]])
    
    y = np.array([-1., 1., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1., 1., 1., -1., -1., 1., -1., -1.])
    
    _ = model.loss(X, y)

    actual_grad, actual_grad_intercept = model.loss_grad(X, y)

    model.step(actual_grad, actual_grad_intercept)

    correct_coef, correct_intercept = np.array([[-1.059958], [ 0.70145 ]]), np.array(-106.25)

    assert_ndarray_equal(actual=model.coef_, correct=correct_coef,  err_msg="Test 6 Failed")
    assert_ndarray_equal(actual=model.intercept_, correct=correct_intercept,  err_msg="Test 6 Failed")