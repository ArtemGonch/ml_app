import numpy as np
from common import assert_ndarray_equal, load_test_data, draw_cotour
from os.path import abspath, dirname, join
from solution import *
import pickle


def test_classifier_1():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'linear_test_1'))

  trainer = LinearPrimalSVM(lr=0.01, C=0.01, n_epochs=1000, batch_size=128, fit_intercept=True, drop_last=True)
  with open("./tests/06_unittest_linear_predict_svm_input/test_estimators_1.pickle", 'rb') as f:
    trainer.list_of_models = pickle.load(f)

  trainer.n_classes_ = len(trainer.list_of_models)
  actual =  trainer.predict(X_test)

  correct = np.array([2, 0, 1, 2, 1, 1, 0, 2, 0, 1, 0, 3, 1, 3, 2, 1, 1, 1, 0, 2, 3, 1, 1, 3, 3, 3, 3, 1, 1, 3, 3, 1, 0, 0, 0, 2, 2
    , 0, 1, 2, 0, 3, 0, 1, 2, 3, 0, 2, 3, 1, 0, 2, 1, 3, 2, 3, 3, 0, 0, 2, 1, 0, 2, 2, 3, 0, 3, 1, 3, 1, 3, 1, 1, 3
    , 0, 3, 0, 3, 3, 3, 1, 0, 2, 2, 3, 3, 3, 0, 1, 2, 1, 0, 2, 2, 2, 1, 1, 0, 0, 0, 2, 0, 2, 3, 3, 2, 0, 2, 2, 0, 0
    , 2, 3, 1, 1, 1, 1, 2, 1, 2, 3, 3, 2, 3, 0, 0, 2, 0, 2, 3, 0, 3, 1, 0, 1, 1, 1, 0, 1, 2, 1, 2, 3, 3, 1, 1, 3, 0
    , 2, 0, 2, 3, 0, 2, 0, 2, 2, 2, 3, 0])

  assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 1 Failed")

      
def test_classifier_2():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'linear_test_2'))

  trainer = LinearPrimalSVM(lr=0.01, C=0.01, n_epochs=1000, batch_size=128, fit_intercept=True, drop_last=True)
  with open("./tests/06_unittest_linear_predict_svm_input/test_estimators_2.pickle", 'rb') as f:
    trainer.list_of_models = pickle.load(f)

  
  trainer.n_classes_ = len(trainer.list_of_models)

  actual =  trainer.predict(X_test)

  correct = np.array([2, 0, 1, 2, 1, 0, 0, 2, 0, 1, 0, 3, 0, 3, 2, 2, 1, 1, 0, 2, 3, 1,
       1, 3, 3, 3, 3, 1, 1, 3, 3, 1, 0, 0, 0, 2, 2, 0, 1, 2, 0, 3, 0, 1,
       2, 3, 0, 2, 3, 1, 0, 2, 1, 3, 2, 3, 3, 0, 0, 2, 1, 0, 2, 2, 3, 0,
       3, 1, 3, 1, 3, 1, 1, 3, 0, 3, 0, 3, 3, 3, 1, 0, 2, 2, 3, 3, 3, 0,
       1, 2, 1, 0, 2, 2, 2, 1, 1, 0, 0, 0, 2, 0, 2, 3, 3, 3, 0, 2, 2, 0,
       0, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 2, 3, 0, 0, 2, 0, 2, 3, 0, 3,
       1, 0, 1, 1, 1, 0, 0, 2, 1, 2, 3, 3, 0, 1, 2, 0, 2, 0, 2, 3, 0, 2,
       0, 2, 2, 2, 3, 0])

  assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 2 Failed")


def test_classifier_3():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'linear_test_3'))

  trainer = LinearPrimalSVM(lr=0.01, C=0.01, n_epochs=1000, batch_size=128, fit_intercept=True, drop_last=True)
  with open("./tests/06_unittest_linear_predict_svm_input/test_estimators_3.pickle", 'rb') as f:
    trainer.list_of_models = pickle.load(f)

  trainer.n_classes_ = len(trainer.list_of_models)

  actual =  trainer.predict(X_test)

  correct = np.array([1, 4, 3, 5, 2, 2, 2, 2, 5, 2, 3, 5, 0, 2, 4, 5, 4, 5, 4, 0, 1, 0,
       2, 5, 2, 5, 0, 5, 0, 1, 1, 0, 5, 5, 5, 1, 0, 1, 5, 4, 2, 4, 4, 0,
       5, 0, 2, 4, 5, 3, 4, 0, 2, 2, 3, 1, 3, 3, 1, 2, 5, 1, 5, 1, 0, 2,
       2, 0, 0, 0, 2, 0, 3, 3, 0, 5, 0, 2, 5, 2, 2, 1, 5, 3, 3, 0, 1, 1,
       4, 4, 4, 1, 4, 2, 4, 1, 5, 2, 0, 1, 4, 2, 3, 5, 2, 3, 2, 4, 0, 3,
       2, 0, 4, 3, 4, 4, 4, 1, 5, 4, 1, 3, 0, 1, 4, 3, 4, 3, 5, 4, 5, 3,
       3, 3, 0, 5, 4, 2, 1, 0, 2, 1, 4, 1, 5, 4, 2, 0, 5, 1, 4, 1, 4, 1,
       0, 1, 3, 2, 1, 3])

  assert_ndarray_equal(actual=actual, correct=correct,  err_msg="Test 3 Failed")