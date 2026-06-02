import os
import random
from bipia_benchmark.sampling import save_samples

_cache = {}


def sample_random_question(
    dataset_name: str = "email",
    seed: int = 2026,
    output_dir: str | None = "bipia_benchmark/experiment_results",
) -> dict | tuple[dict, str]:
    """
    Sample one random question/attack pair from BIPIA dataset.

    Args:
        dataset_name: Dataset name (email, code, table, etc.)
        seed: Random seed for reproducibility
        output_dir: Base directory for saving samples. Defaults to "bipia_benchmark/experiment_results".
                   Set to None to disable file saving.
                   Files saved to: {output_dir}/{dataset_name}_YYYYMMDD_HHMMSS/samples.json

    Returns:
        If output_dir is None: dict with keys question, attack_str, context
        If output_dir provided: tuple of (sample_dict, output_dir_path)
    """
    # Load data once and cache it
    if dataset_name not in _cache:
        from bipia.data import AutoPIABuilder

        module_dir = os.path.dirname(os.path.abspath(__file__))
        benchmark_path = os.path.join(module_dir, "BIPIA", "benchmark", dataset_name, "test.jsonl")
        attack_path = os.path.join(module_dir, "BIPIA", "benchmark", "text_attack_test.json")

        builder = AutoPIABuilder.from_name(dataset_name)()
        df = builder(benchmark_path, attack_path, enable_stealth=True)
        _cache[dataset_name] = df.to_dict(orient="records")

    samples = _cache[dataset_name]

    # Seed right before choice to ensure reproducible selection
    random.seed(seed)
    selected = random.choice(samples)

    sample_dict = {
        "question": selected.get("question", ""),
        "attack_str": selected.get("attack_str", ""),
        "context": selected.get("context", ""),
    }

    if output_dir is None:
        return sample_dict

    _, out_path = save_samples([sample_dict], dataset_name, output_dir)
    return sample_dict, out_path
