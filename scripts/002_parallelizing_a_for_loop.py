import joblib


def process_item(item):
    return item**2


items = list(range(10))

results = joblib.Parallel(n_jobs=-1)(joblib.delayed(process_item)(item) for item in items)

print(results)
