from joblib import Parallel, delayed


def process_item(item):
    return item**2


items = list(range(10))

results = Parallel(n_jobs=-1)(delayed(process_item)(item) for item in items)

print(results)
