import numpy as np
from common import assert_ndarray_equal, load_test_data, draw_cotour
from os.path import abspath, dirname, join
from solution import *
import pickle


def test_classifier_1():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  trainer = NonLinearDualSVM(C=20.0, kernel='linear')

  with open("./tests/13_unittest_nonlinear_predict_svm_input/test_estimators_1.pickle", 'rb') as f:
    trainer.estimators = pickle.load(f)

  actual =  trainer.predict(X_test)

  correct = np.array([0., 1., 1., 3., 2., 0., 0., 0., 0., 3., 0., 1., 0., 0., 0., 1., 0., 2., 0., 1., 2., 0., 0., 2.
            , 0., 3., 0., 1., 2., 0., 1., 2., 2., 0., 2., 0., 1., 1., 3., 3., 2., 3., 0., 0., 2., 0., 2., 2.
            , 0., 1., 0., 0., 1., 1., 2., 0., 1., 1., 1., 2., 1., 3., 0., 2., 1., 1., 1., 0., 1., 3., 0., 3.
            , 2., 2., 2., 3., 2., 0., 1., 0., 1., 3., 0., 1., 0., 3., 0., 2., 3., 2., 0., 0., 2., 3., 0., 3.
            , 2., 1., 0., 3., 3., 2., 0., 3., 2., 1., 3., 2., 0., 0., 3., 0., 2., 1., 2., 2., 1., 2., 0., 2.
            , 0., 1., 3., 2., 0., 0., 1., 0., 1., 0., 2., 0., 2., 1., 1., 0., 2., 1., 3., 3., 2., 3., 3., 1.
            , 1., 0., 3., 3., 3., 2., 0., 1., 3., 2., 2., 3., 0., 1., 0., 1.])

  assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 1 Failed")

      
def test_classifier_2():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  trainer = NonLinearDualSVM(C=0.0001, kernel='linear')

  with open("./tests/13_unittest_nonlinear_predict_svm_input/test_estimators_2.pickle", 'rb') as f:
    trainer.estimators = pickle.load(f)

  actual =  trainer.predict(X_test)

  correct = np.array([0., 3., 1., 3., 2., 1., 2., 2., 2., 3., 0., 1., 0., 2., 0., 1., 0., 2., 0., 3., 2., 0., 0., 2.
            , 1., 3., 0., 1., 2., 2., 1., 2., 2., 2., 2., 0., 1., 3., 3., 3., 3., 3., 0., 0., 2., 1., 2., 2.
            , 1., 1., 0., 1., 1., 1., 2., 2., 3., 1., 3., 3., 1., 3., 0., 2., 1., 3., 1., 0., 1., 3., 2., 3.
            , 2., 2., 2., 3., 2., 0., 1., 2., 1., 3., 2., 1., 0., 3., 0., 3., 3., 2., 2., 2., 2., 3., 0., 3.
            , 3., 1., 0., 3., 3., 2., 2., 3., 2., 1., 3., 2., 0., 0., 3., 1., 2., 1., 2., 2., 1., 2., 2., 2.
            , 0., 1., 3., 2., 1., 0., 1., 2., 1., 1., 3., 0., 3., 1., 3., 2., 2., 1., 3., 3., 2., 3., 3., 1.
            , 1., 0., 3., 3., 3., 3., 0., 1., 3., 2., 2., 3., 2., 3., 2., 1.])

  assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 2 Failed")
