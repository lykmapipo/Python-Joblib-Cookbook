import joblib


def square(x):
    return x**2


results = joblib.Parallel(n_jobs=-1)(joblib.delayed(square)(i) for i in range(10))

print(results)
