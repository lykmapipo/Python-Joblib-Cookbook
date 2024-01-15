from joblib import Parallel, delayed, parallel_config
from ray.util.joblib import register_ray


def square(x):
    return x**2


# Register Ray Backend to be called with parallel_config(backend="ray")
register_ray()

# See: https://docs.ray.io/en/latest/ray-core/walkthrough.html
if __name__ == "__main__":
    with parallel_config(backend="ray"):
        results = Parallel(verbose=10)(delayed(square)(i) for i in range(10))

    print(results)
