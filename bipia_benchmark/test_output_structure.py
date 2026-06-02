"""Tests that benchmark output follows the correct structure.

Reference: bipia_benchmark/experiment_results/email_20260527_140511/
Expected files:
  - asr_results.json     (list of vulnerable sample dicts)
  - comparison_summary.json (summary with dataset, defenses, stats)
  - defense_results.jsonl   (one JSON object per line, per sample)
"""

import json
import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reference_dir():
    """Path to known-good reference output."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "experiment_results", "email_20260527_140511")


@pytest.fixture
def reference_asr(reference_dir):
    with open(os.path.join(reference_dir, "asr_results.json")) as f:
        return json.load(f)


@pytest.fixture
def reference_summary(reference_dir):
    with open(os.path.join(reference_dir, "comparison_summary.json")) as f:
        return json.load(f)


@pytest.fixture
def reference_defense_lines(reference_dir):
    lines = []
    with open(os.path.join(reference_dir, "defense_results.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


# ---------------------------------------------------------------------------
# Reference output sanity (proves our fixture data is valid)
# ---------------------------------------------------------------------------

class TestReferenceOutputExists:
    def test_reference_dir_exists(self, reference_dir):
        assert os.path.isdir(reference_dir)

    def test_has_three_files(self, reference_dir):
        expected = {"asr_results.json", "comparison_summary.json", "defense_results.jsonl"}
        actual = set(os.listdir(reference_dir))
        assert expected == actual, f"Expected exactly {expected}, got {actual}"


# ---------------------------------------------------------------------------
# asr_results.json structure
# ---------------------------------------------------------------------------

class TestAsrResultsStructure:
    def test_is_list(self, reference_asr):
        assert isinstance(reference_asr, list)

    def test_each_entry_has_required_fields(self, reference_asr):
        required = {"context", "attack_name", "attack_str", "task_name",
                     "ideal", "question", "position", "id", "baseline_response"}
        for entry in reference_asr:
            missing = required - set(entry.keys())
            assert not missing, f"Sample id={entry.get('id')} missing fields: {missing}"

    def test_id_is_integer(self, reference_asr):
        for entry in reference_asr:
            assert isinstance(entry["id"], int), f"id should be int, got {type(entry['id'])}"

    def test_task_name_matches_dataset(self, reference_asr):
        for entry in reference_asr:
            assert entry["task_name"] == "email"

    def test_position_is_valid(self, reference_asr):
        valid_positions = {"start", "middle", "end"}
        for entry in reference_asr:
            assert entry["position"] in valid_positions, (
                f"Invalid position '{entry['position']}' for sample id={entry['id']}"
            )

    def test_question_starts_with_q(self, reference_asr):
        for entry in reference_asr:
            assert entry["question"].startswith("Q:"), (
                f"Question should start with 'Q:', got: {entry['question'][:30]}"
            )

    def test_baseline_response_is_nonempty_string(self, reference_asr):
        for entry in reference_asr:
            assert isinstance(entry["baseline_response"], str)
            assert len(entry["baseline_response"]) > 0

    def test_attack_str_is_nonempty(self, reference_asr):
        for entry in reference_asr:
            assert len(entry["attack_str"].strip()) > 0

    def test_context_is_nonempty(self, reference_asr):
        for entry in reference_asr:
            assert len(entry["context"].strip()) > 0


# ---------------------------------------------------------------------------
# comparison_summary.json structure
# ---------------------------------------------------------------------------

class TestComparisonSummaryStructure:
    REQUIRED_TOP_KEYS = {
        "dataset", "total_samples_tested", "target_model", "timestamp",
        "mock_mode_active", "baseline_scan_completed", "defenses_test_completed",
        "sampling", "prompt_statistics", "defenses",
    }

    def test_has_all_top_level_keys(self, reference_summary):
        missing = self.REQUIRED_TOP_KEYS - set(reference_summary.keys())
        assert not missing, f"Missing top-level keys: {missing}"

    def test_dataset_is_string(self, reference_summary):
        assert isinstance(reference_summary["dataset"], str)
        assert len(reference_summary["dataset"]) > 0

    def test_total_samples_tested_is_positive_int(self, reference_summary):
        n = reference_summary["total_samples_tested"]
        assert isinstance(n, int)
        assert n >= 0

    def test_timestamp_format(self, reference_summary):
        ts = reference_summary["timestamp"]
        assert isinstance(ts, str)
        assert len(ts) == len("20260527_140511"), f"Unexpected timestamp format: {ts}"

    def test_mock_mode_is_bool(self, reference_summary):
        assert isinstance(reference_summary["mock_mode_active"], bool)

    def test_baseline_and_defenses_completed_are_bool(self, reference_summary):
        assert isinstance(reference_summary["baseline_scan_completed"], bool)
        assert isinstance(reference_summary["defenses_test_completed"], bool)

    def test_sampling_has_seed_and_budget(self, reference_summary):
        sampling = reference_summary["sampling"]
        assert "seed" in sampling
        assert "sample_budget" in sampling
        assert isinstance(sampling["seed"], int)
        assert isinstance(sampling["sample_budget"], int)

    def test_prompt_statistics_structure(self, reference_summary):
        stats = reference_summary["prompt_statistics"]
        assert "target_prompts" in stats
        assert "judge_prompts" in stats
        assert "total_prompts" in stats
        assert stats["total_prompts"] == stats["target_prompts"] + stats["judge_prompts"]

    def test_defenses_is_dict(self, reference_summary):
        assert isinstance(reference_summary["defenses"], dict)
        assert len(reference_summary["defenses"]) > 0

    def test_each_defense_has_rate_fields(self, reference_summary):
        for name, data in reference_summary["defenses"].items():
            assert "blocks" in data, f"Defense '{name}' missing 'blocks'"
            assert "success_rate" in data, f"Defense '{name}' missing 'success_rate'"
            assert "attack_success_rate" in data, f"Defense '{name}' missing 'attack_success_rate'"

    def test_defense_blocks_is_int(self, reference_summary):
        for name, data in reference_summary["defenses"].items():
            assert isinstance(data["blocks"], int), (
                f"Defense '{name}' blocks should be int, got {type(data['blocks'])}"
            )

    def test_defense_rates_are_percentage_strings(self, reference_summary):
        for name, data in reference_summary["defenses"].items():
            for field in ("success_rate", "attack_success_rate"):
                val = data[field]
                assert isinstance(val, str), f"Defense '{name}' {field} should be string"
                assert val.endswith("%"), f"Defense '{name}' {field} should end with '%', got '{val}'"

    def test_total_samples_matches_asr_count(self, reference_summary, reference_asr):
        assert reference_summary["total_samples_tested"] == len(reference_asr)


# ---------------------------------------------------------------------------
# defense_results.jsonl structure
# ---------------------------------------------------------------------------

class TestDefenseResultsStructure:
    def test_is_nonempty(self, reference_defense_lines):
        assert len(reference_defense_lines) > 0

    def test_each_line_is_valid_json_object(self, reference_defense_lines):
        for i, obj in enumerate(reference_defense_lines):
            assert isinstance(obj, dict), f"Line {i} is not a JSON object"

    def test_each_line_has_required_fields(self, reference_defense_lines):
        required = {"sample_id", "original_question", "defenses"}
        for i, obj in enumerate(reference_defense_lines):
            missing = required - set(obj.keys())
            assert not missing, f"Line {i} missing fields: {missing}"

    def test_defenses_field_is_dict(self, reference_defense_lines):
        for i, obj in enumerate(reference_defense_lines):
            assert isinstance(obj["defenses"], dict), f"Line {i} defenses not a dict"

    def test_each_defense_result_has_required_fields(self, reference_defense_lines):
        required = {"prompt", "response", "attack_thwarted", "attack_successful"}
        for i, obj in enumerate(reference_defense_lines):
            for def_name, def_data in obj["defenses"].items():
                missing = required - set(def_data.keys())
                assert not missing, (
                    f"Line {i}, defense '{def_name}' missing: {missing}"
                )

    def test_attack_thwarted_is_bool(self, reference_defense_lines):
        for obj in reference_defense_lines:
            for def_name, def_data in obj["defenses"].items():
                assert isinstance(def_data["attack_thwarted"], bool)

    def test_attack_successful_is_bool(self, reference_defense_lines):
        for obj in reference_defense_lines:
            for def_name, def_data in obj["defenses"].items():
                assert isinstance(def_data["attack_successful"], bool)

    def test_thwarted_and_successful_are_opposites(self, reference_defense_lines):
        for obj in reference_defense_lines:
            for def_name, def_data in obj["defenses"].items():
                if "error" not in def_data:
                    assert def_data["attack_thwarted"] != def_data["attack_successful"], (
                        f"sample {obj['sample_id']}, defense '{def_name}': "
                        f"thwarted={def_data['attack_thwarted']} and successful={def_data['attack_successful']} "
                        "should be opposites"
                    )

    def test_prompt_is_nonempty_string(self, reference_defense_lines):
        for obj in reference_defense_lines:
            for def_name, def_data in obj["defenses"].items():
                assert isinstance(def_data["prompt"], str)
                assert len(def_data["prompt"]) > 0

    def test_line_count_matches_summary(self, reference_defense_lines, reference_summary):
        assert len(reference_defense_lines) == reference_summary["total_samples_tested"]


# ---------------------------------------------------------------------------
# _save_results output location
# ---------------------------------------------------------------------------

class TestSaveResultsOutputLocation:
    """Verify _save_results writes to bipia_benchmark/experiment_results by default."""

    def test_default_base_dir_is_bipia_experiment_results(self):
        """_save_results default base_dir should be bipia_benchmark/experiment_results, not experiment_output."""
        from bipia_benchmark.benchmark_defense import BipiaDefenseBenchmark
        import inspect

        sig = inspect.signature(BipiaDefenseBenchmark._save_results)
        base_dir_param = sig.parameters.get("base_dir")
        assert base_dir_param is not None, "_save_results should have a base_dir parameter"

        default = base_dir_param.default
        assert "bipia_benchmark" in str(default) and "experiment_results" in str(default), (
            f"Default base_dir should contain 'bipia_benchmark/experiment_results', got '{default}'"
        )

    def test_output_folder_name_format(self):
        """Output folder should be {dataset}_{YYYYMMDD_HHMMSS}."""
        from bipia_benchmark.benchmark_defense import BipiaDefenseBenchmark

        b = BipiaDefenseBenchmark(dataset_name="email", mock=True)
        folder_name = f"{b._dataset_name}_{b._timestamp}"
        assert folder_name.startswith("email_")
        assert len(b._timestamp) == len("20260527_140511")

    def test_save_creates_three_files(self):
        """_save_results should produce asr_results.json, comparison_summary.json, defense_results.jsonl."""
        from bipia_benchmark.benchmark_defense import BipiaDefenseBenchmark

        with tempfile.TemporaryDirectory() as tmpdir:
            b = BipiaDefenseBenchmark(dataset_name="email", mock=True)
            vuln = [{"id": 0, "question": "Q: test?", "context": "ctx", "attack_str": "atk"}]
            defense_data = [{
                "sample_id": 0,
                "original_question": "Q: test?",
                "defenses": {
                    "none": {
                        "prompt": "ctx Q: test?",
                        "response": "mock",
                        "attack_thwarted": False,
                        "attack_successful": True,
                    }
                },
            }]
            summary = {
                "dataset": "email",
                "total_samples_tested": 1,
                "target_model": None,
                "timestamp": b._timestamp,
                "mock_mode_active": True,
                "baseline_scan_completed": True,
                "defenses_test_completed": True,
                "sampling": {"seed": 2026, "sample_budget": 50},
                "prompt_statistics": {"target_prompts": 1, "judge_prompts": 1, "total_prompts": 2},
                "defenses": {"none": {"blocks": 0, "success_rate": "0.00%", "attack_success_rate": "100.00%"}},
            }

            b._save_results(vuln, defense_data, summary, base_dir=tmpdir, force=True)

            run_folder = b._run_folder
            expected_files = {"asr_results.json", "comparison_summary.json", "defense_results.jsonl"}
            actual_files = set(os.listdir(run_folder))
            assert expected_files == actual_files, (
                f"Expected {expected_files}, got {actual_files}"
            )

    def test_save_asr_results_valid_json(self):
        """asr_results.json should be valid JSON list."""
        from bipia_benchmark.benchmark_defense import BipiaDefenseBenchmark

        with tempfile.TemporaryDirectory() as tmpdir:
            b = BipiaDefenseBenchmark(dataset_name="email", mock=True)
            vuln = [{"id": 0, "question": "Q: test?"}]
            b._save_results(vuln, [], {"baseline_scan_completed": True}, base_dir=tmpdir, force=True)

            with open(os.path.join(b._run_folder, "asr_results.json")) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert data == vuln

    def test_save_defense_results_is_jsonl(self):
        """defense_results.jsonl should have one JSON object per line."""
        from bipia_benchmark.benchmark_defense import BipiaDefenseBenchmark

        with tempfile.TemporaryDirectory() as tmpdir:
            b = BipiaDefenseBenchmark(dataset_name="email", mock=True)
            defense_data = [
                {"sample_id": 0, "original_question": "Q1", "defenses": {}},
                {"sample_id": 1, "original_question": "Q2", "defenses": {}},
            ]
            b._save_results([], defense_data, {"baseline_scan_completed": True}, base_dir=tmpdir, force=True)

            with open(os.path.join(b._run_folder, "defense_results.jsonl")) as f:
                lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 2
            for line in lines:
                obj = json.loads(line)
                assert isinstance(obj, dict)

    def test_save_comparison_summary_valid_json(self):
        """comparison_summary.json should be valid JSON dict."""
        from bipia_benchmark.benchmark_defense import BipiaDefenseBenchmark

        with tempfile.TemporaryDirectory() as tmpdir:
            b = BipiaDefenseBenchmark(dataset_name="email", mock=True)
            summary = {"dataset": "email", "baseline_scan_completed": True}
            b._save_results([], [], summary, base_dir=tmpdir, force=True)

            with open(os.path.join(b._run_folder, "comparison_summary.json")) as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert data["dataset"] == "email"


# ---------------------------------------------------------------------------
# Cross-file consistency
# ---------------------------------------------------------------------------

class TestCrossFileConsistency:
    def test_asr_sample_ids_appear_in_defense_results(self, reference_asr, reference_defense_lines):
        asr_ids = {entry["id"] for entry in reference_asr}
        defense_ids = {obj["sample_id"] for obj in reference_defense_lines}
        assert defense_ids.issubset(asr_ids), (
            f"Defense results reference sample_ids not in asr_results: {defense_ids - asr_ids}"
        )

    def test_defense_names_consistent_across_files(self, reference_summary, reference_defense_lines):
        summary_defenses = set(reference_summary["defenses"].keys())
        for obj in reference_defense_lines:
            line_defenses = set(obj["defenses"].keys())
            assert line_defenses == summary_defenses, (
                f"Sample {obj['sample_id']} has defenses {line_defenses}, "
                f"summary has {summary_defenses}"
            )

    def test_no_duplicate_sample_ids_in_defense_results(self, reference_defense_lines):
        ids = [obj["sample_id"] for obj in reference_defense_lines]
        assert len(ids) == len(set(ids)), "Duplicate sample_ids in defense_results.jsonl"

    def test_no_duplicate_ids_in_asr_results(self, reference_asr):
        ids = [entry["id"] for entry in reference_asr]
        assert len(ids) == len(set(ids)), "Duplicate ids in asr_results.json"


# ---------------------------------------------------------------------------
# Docker container integration
# ---------------------------------------------------------------------------

class TestDockerContainerIntegration:
    """Tests that verify benchmark runs inside Docker and produces correct output."""

    def test_dockerfile_exists(self):
        here = os.path.dirname(os.path.abspath(__file__))
        assert os.path.isfile(os.path.join(here, "Dockerfile")), (
            "Dockerfile must exist in bipia_benchmark/"
        )

    def test_dockerfile_installs_pytest(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "Dockerfile")) as f:
            content = f.read()
        assert "pytest" in content, "Dockerfile should install pytest for in-container testing"

    def test_dockerfile_workdir_is_app(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "Dockerfile")) as f:
            content = f.read()
        assert "WORKDIR /app" in content

    def test_dockerfile_copies_bipia_benchmark(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "Dockerfile")) as f:
            content = f.read()
        assert "bipia_benchmark" in content, "Dockerfile must copy bipia_benchmark/ into container"

    def test_dockerfile_copies_defenses(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "Dockerfile")) as f:
            content = f.read()
        assert "defenses" in content, "Dockerfile must copy defenses/ into container"

    def test_cmd_is_single_python_command(self):
        """CMD should be a simple single python invocation, not a shell script chain."""
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "Dockerfile")) as f:
            lines = f.readlines()

        active_cmd_lines = [
            l.strip() for l in lines
            if l.strip().startswith("CMD") and not l.strip().startswith("#")
        ]
        assert len(active_cmd_lines) == 1, (
            f"Expected exactly 1 active CMD line, got {len(active_cmd_lines)}"
        )
        cmd = active_cmd_lines[0]
        assert "python" in cmd, f"CMD should invoke python, got: {cmd}"
        assert "&&" not in cmd, f"CMD should be single command, not chained: {cmd}"
        assert "|" not in cmd, f"CMD should be single command, not piped: {cmd}"

    def test_benchmark_output_goes_to_bipia_benchmark_experiment_results(self):
        """Output must land in bipia_benchmark/experiment_results/, not project root."""
        here = os.path.dirname(os.path.abspath(__file__))
        expected_output_dir = os.path.join(here, "experiment_results")
        assert os.path.isdir(expected_output_dir), (
            "bipia_benchmark/experiment_results/ directory must exist"
        )

    def test_mock_run_produces_correct_output_structure(self):
        """A mock benchmark run should produce all 3 expected files in bipia_benchmark/experiment_results/."""
        from bipia_benchmark.benchmark_defense import BipiaDefenseBenchmark

        with tempfile.TemporaryDirectory() as tmpdir:
            b = BipiaDefenseBenchmark(dataset_name="email", mock=True, search_limit=5)
            b.vulnerable_samples = [
                {
                    "id": i,
                    "question": f"Q: test {i}?",
                    "context": f"ctx {i}",
                    "attack_str": f"atk {i}",
                    "attack_name": "test-0",
                    "task_name": "email",
                    "ideal": "unknown",
                    "position": "middle",
                    "baseline_response": "mock response",
                }
                for i in range(3)
            ]
            b.detailed_results = [
                {
                    "sample_id": i,
                    "original_question": f"Q: test {i}?",
                    "defenses": {
                        "none": {
                            "prompt": f"ctx {i} Q: test {i}?",
                            "response": "mock",
                            "attack_thwarted": False,
                            "attack_successful": True,
                        }
                    },
                }
                for i in range(3)
            ]
            b.baseline_scan_completed = True
            b.defenses_test_completed = True
            b.per_defense_blocks = {"none": 0}
            b._defense_names = ["none"]

            b._save_results(
                b.vulnerable_samples, [], {},
                base_dir=tmpdir, force=True,
            )
            b._finalize_save()

            run_folder = b._run_folder
            files = set(os.listdir(run_folder))

            assert "asr_results.json" in files
            assert "comparison_summary.json" in files
            assert "defense_results.jsonl" in files

            with open(os.path.join(run_folder, "comparison_summary.json")) as f:
                summary = json.load(f)
            assert summary["dataset"] == "email"
            assert summary["baseline_scan_completed"] is True
            assert summary["defenses_test_completed"] is True

    def test_docker_compose_exists_at_project_root(self):
        """docker compose up --build should work - compose file must exist."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        compose_candidates = [
            "docker-compose.yml",
            "docker-compose.yaml",
            "compose.yml",
            "compose.yaml",
        ]
        found = any(
            os.path.isfile(os.path.join(project_root, name))
            for name in compose_candidates
        )
        assert found, (
            f"No docker compose file found at project root ({project_root}). "
            "Expected one of: " + ", ".join(compose_candidates)
        )
