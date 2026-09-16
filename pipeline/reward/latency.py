import time


def measure_latency(action_fn, *args, **kwargs):
    """
    Execute an action and measure its execution time.

    Returns:
        result, latency_seconds
    """

    start_time = time.perf_counter()

    result = action_fn(*args, **kwargs)

    end_time = time.perf_counter()

    latency_seconds = end_time - start_time

    return result, latency_seconds


def latency_score(
    latency_seconds: float,
    target_latency: float = 2.0,
    max_latency: float = 15.0,
) -> float:
    """
    Convert latency into a normalized score between 0 and 1.

    target_latency:
        Latency considered very good.

    max_latency:
        Latency considered very poor.

    Returns:
        1.0 = very fast
        0.0 = very slow
    """

    if latency_seconds <= target_latency:
        return 1.0

    if latency_seconds >= max_latency:
        return 0.0

    score = 1.0 - (
        (latency_seconds - target_latency)
        / (max_latency - target_latency)
    )

    return round(max(0.0, min(1.0, score)), 4)