from joblib import Parallel, delayed, parallel_config


def divide(x, y):
    try:
        result = x / y
    except ZeroDivisionError:
        result = float("nan")
    return result


data = [(1, 2), (3, 0), (5, 2)]

with parallel_config(backend="loky", n_jobs=-1, verbose=50):
    results = Parallel()(delayed(divide)(x, y) for x, y in data)

print(results)
