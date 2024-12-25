from benchllama import benchmark

def run_benchllama(model_name):
    results = benchmark(
        models=[model_name],
        dataset="medqa"
    )
    return results