import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_test_data(test_data):
    return pickle.load(open(f'{test_data}.pickle', 'rb'))

def assert_value_is_ndarray(value):
    __tracebackhide__ = True
    assert isinstance(value, (np.ndarray, np.generic)), (
        f"Value should be an instance of np.ndarray, but it is {type(value)}."
    )


def assert_dtypes_compatible(actual_dtype, correct_dtype):
    __tracebackhide__ = True
    assert (
            np.can_cast(actual_dtype, correct_dtype, casting='same_kind') and
            np.can_cast(correct_dtype, actual_dtype, casting='same_kind')
    ), (
        "The dtypes of actual value and correct value are not the same "
        "and can't be safely converted.\n"
        f"actual.dtype={actual_dtype}, correct.dtype={correct_dtype}"
    )


def assert_shapes_match(actual_shape, correct_shape):
    __tracebackhide__ = True
    assert (
            len(actual_shape) == len(correct_shape) and
            actual_shape == correct_shape
    ), (
        "The shapes of actual value and correct value are not the same.\n"
        f"actual.shape={actual_shape}, correct.shape={correct_shape}"
    )


def assert_ndarray_equal(*, actual, correct, rtol=0, atol=1e-6, err_msg=''):
    __tracebackhide__ = True
    assert_value_is_ndarray(actual)
    assert_dtypes_compatible(actual.dtype, correct.dtype)
    assert_shapes_match(actual.shape, correct.shape)
    np.testing.assert_allclose(actual, correct, atol=atol, rtol=rtol,
                               verbose=True, err_msg=err_msg)
    




def draw_cotour(model, X_test, y_test, filename, num=501):

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.tight_layout()
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.title('Тестовая выборка и области классов', fontsize=16)
    plt.grid()
    classes = set(y_test)
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    for i in classes:
        ax.plot(*X_test[np.squeeze(y_test) == i].T, 'o', color=colors[i]) 

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(*xlim, num)
    y = np.linspace(*ylim, num)
    X = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    Y = model.predict(X)
    con = ax.contourf(
        x, y, np.squeeze(Y).reshape(num, num).T,
        levels=np.arange(len(classes) + 1)-0.5,
        colors=colors,
        alpha=0.25
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.savefig(filename, format='jpg', dpi=300)
