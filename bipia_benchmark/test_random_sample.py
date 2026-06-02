import json
import os
import tempfile

import pytest
from bipia_benchmark.random_sample import sample_random_question


class TestRandomSampleQuestion:
    """Test random sampling from BIPIA email dataset"""

    def test_returns_dict_with_required_fields(self):
        """Sample has question, attack_str, and context fields"""
        sample = sample_random_question(seed=42, output_dir=None)

        assert isinstance(sample, dict)
        assert "question" in sample
        assert "attack_str" in sample
        assert "context" in sample

    def test_fields_are_strings(self):
        """All fields are non-empty strings"""
        sample = sample_random_question(seed=42, output_dir=None)

        assert isinstance(sample["question"], str)
        assert isinstance(sample["attack_str"], str)
        assert isinstance(sample["context"], str)
        assert len(sample["question"]) > 0
        assert len(sample["attack_str"]) > 0
        assert len(sample["context"]) > 0

    def test_same_seed_returns_same_sample(self):
        """Reproducibility: same seed always returns identical sample"""
        sample1 = sample_random_question(seed=42, output_dir=None)
        sample2 = sample_random_question(seed=42, output_dir=None)

        assert sample1["question"] == sample2["question"]
        assert sample1["attack_str"] == sample2["attack_str"]
        assert sample1["context"] == sample2["context"]

    def test_different_seeds_return_different_samples(self):
        """Randomness: different seeds return different samples"""
        sample_42 = sample_random_question(seed=42, output_dir=None)
        sample_99 = sample_random_question(seed=99, output_dir=None)

        # At least one field should differ
        differs = (
            sample_42["question"] != sample_99["question"] or
            sample_42["attack_str"] != sample_99["attack_str"] or
            sample_42["context"] != sample_99["context"]
        )
        assert differs, "Different seeds should produce different samples"

    def test_multiple_runs_same_seed_identical(self):
        """Run same seed 3 times, all results identical"""
        samples = [sample_random_question(seed=123, output_dir=None) for _ in range(3)]

        for i in range(1, 3):
            assert samples[0]["question"] == samples[i]["question"]
            assert samples[0]["attack_str"] == samples[i]["attack_str"]
            assert samples[0]["context"] == samples[i]["context"]

    def test_question_starts_with_q(self):
        """Question field is valid prompt"""
        sample = sample_random_question(seed=42, output_dir=None)
        assert sample["question"].startswith("Q:")

    def test_attack_str_is_base64_or_text(self):
        """Attack string is non-empty"""
        sample = sample_random_question(seed=42, output_dir=None)
        assert sample["attack_str"].strip() != ""

    def test_no_output_dir_returns_dict(self):
        """Without output_dir, return value is a dict"""
        result = sample_random_question(output_dir=None)
        assert isinstance(result, dict)
        assert not isinstance(result, tuple)

    def test_with_output_dir_returns_tuple(self):
        """With output_dir, return value is (dict, path) tuple"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = sample_random_question(output_dir=tmpdir, seed=42)
            assert isinstance(result, tuple)
            assert len(result) == 2
            sample_dict, out_path = result
            assert isinstance(sample_dict, dict)
            assert isinstance(out_path, str)

    def test_output_file_created_with_correct_content(self):
        """Output directory created with samples.json containing sample data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dict, out_path = sample_random_question(seed=7777, output_dir=tmpdir)

            assert os.path.exists(out_path)
            samples_file = os.path.join(out_path, "samples.json")
            assert os.path.exists(samples_file)

            with open(samples_file) as fh:
                saved_data = json.load(fh)
            assert len(saved_data) == 1
            assert saved_data[0] == sample_dict

    def test_output_dir_has_correct_format(self):
        """Output directory name follows dataset_YYYYMMDD_HHMMSS format"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _, out_path = sample_random_question(dataset_name="email", output_dir=tmpdir)
            dirname = os.path.basename(out_path)
            # Should be email_YYYYMMDD_HHMMSS
            assert dirname.startswith("email_")
            assert len(dirname) == len("email_20260527_143000")

    def test_same_seed_with_output_produces_different_dirs(self, monkeypatch):
        """Same seed produces identical data but different output directories"""
        from datetime import datetime
        from bipia_benchmark import sampling

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock timestamps to be different
            timestamps = iter([
                datetime(2026, 5, 27, 14, 41, 0),
                datetime(2026, 5, 27, 14, 41, 1),
            ])
            monkeypatch.setattr(sampling, "_now", lambda: next(timestamps))

            _, path1 = sample_random_question(seed=5555, output_dir=tmpdir)
            _, path2 = sample_random_question(seed=5555, output_dir=tmpdir)

            # Different directories (different timestamps)
            assert path1 != path2

            # But identical file content
            with open(os.path.join(path1, "samples.json")) as f1:
                data1 = json.load(f1)
            with open(os.path.join(path2, "samples.json")) as f2:
                data2 = json.load(f2)
            assert data1 == data2

    def test_multiple_seeds_produce_different_questions(self):
        """Different seeds produce different questions (not all identical)"""
        # Email dataset only has 34 unique questions out of 11,250 samples
        # So sampling with different seeds should hit different questions
        samples = [sample_random_question(seed=s, output_dir=None) for s in range(1000, 1150)]
        questions = [s["question"] for s in samples]

        # Should have some diversity - at least a few different questions
        unique_questions = len(set(questions))
        assert unique_questions > 1, "Expected at least 2 different questions"
        # With 150 samples and 34 unique questions in the dataset,
        # expect to hit at least 10 different questions by chance
        assert unique_questions >= 5, f"Insufficient diversity: {unique_questions} unique questions in 150 samples"

    def test_default_output_dir_is_bipia_benchmark_experiment_results(self):
        """Default output_dir uses bipia_benchmark/experiment_results by default"""
        import inspect
        from bipia_benchmark.random_sample import sample_random_question as func

        # Check the function signature to verify the default
        sig = inspect.signature(func)
        default_output_dir = sig.parameters["output_dir"].default

        # Verify default contains the expected path
        assert "bipia_benchmark/experiment_results" in default_output_dir or "bipia_benchmark\\experiment_results" in default_output_dir
