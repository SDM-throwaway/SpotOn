"""
Tests for sampling behavior in BipiaDefenseBenchmark.

TDD - RED phase: tests import select_samples from benchmark_defense.
They will fail until that function is extracted and assigns sample 'id' fields.
"""

import json
import os
import re
import stat
import tempfile

import pytest

# We import only the pure helper, not the whole module, so we don't need
# the BIPIA library or API keys to run these tests.
from bipia_benchmark.sampling import save_samples, select_samples


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_samples(n: int) -> list[dict]:
    """Return n synthetic sample dicts that mimic BIPIA row structure."""
    return [
        {
            "question": f"question_{i}",
            "context": f"context_{i}",
            "attack_str": f"attack_{i}",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestSamplingDeterminism:
    def test_same_seed_produces_same_questions(self):
        """seed=42 must select the same questions on every run."""
        samples = make_samples(100)
        run1 = select_samples(samples, seed=42, limit=20)
        run2 = select_samples(samples, seed=42, limit=20)
        assert [s["question"] for s in run1] == [s["question"] for s in run2]

    def test_different_seeds_produce_different_subsets(self):
        """seed=42 and seed=43 must not select identical subsets."""
        samples = make_samples(100)
        q42 = [s["question"] for s in select_samples(samples, seed=42, limit=20)]
        q43 = [s["question"] for s in select_samples(samples, seed=43, limit=20)]
        assert q42 != q43

    def test_full_pool_returned_when_limit_is_zero(self):
        """limit=0 means no cap - return all samples."""
        samples = make_samples(30)
        assert len(select_samples(samples, seed=1, limit=0)) == 30

    def test_limit_is_respected(self):
        samples = make_samples(100)
        assert len(select_samples(samples, seed=7, limit=15)) == 15


# ---------------------------------------------------------------------------
# sample_id assignment
# ---------------------------------------------------------------------------

class TestSampleIdAssignment:
    def test_every_sample_has_id(self):
        """select_samples must set 'id' on every returned sample."""
        samples = make_samples(50)
        selected = select_samples(samples, seed=42, limit=10)
        for s in selected:
            assert "id" in s, f"'id' key missing from sample: {s}"

    def test_sample_id_is_not_unknown(self):
        """'id' must never be the sentinel string 'unknown'."""
        samples = make_samples(50)
        selected = select_samples(samples, seed=42, limit=10)
        for s in selected:
            assert s["id"] != "unknown", f"sample id is 'unknown': {s}"

    def test_sample_ids_are_unique_within_run(self):
        """No two samples in the same run may share an 'id'."""
        samples = make_samples(100)
        selected = select_samples(samples, seed=42, limit=30)
        ids = [s["id"] for s in selected]
        assert len(ids) == len(set(ids)), "Duplicate sample ids found"

    def test_sample_ids_are_deterministic_across_runs(self):
        """Same seed must produce the same id sequence on repeated calls."""
        samples = make_samples(100)
        ids1 = [s["id"] for s in select_samples(samples, seed=99, limit=25)]
        ids2 = [s["id"] for s in select_samples(samples, seed=99, limit=25)]
        assert ids1 == ids2

    def test_sample_ids_differ_across_seeds(self):
        """Different seeds must yield different question orderings."""
        samples = make_samples(100)
        q1 = [s["question"] for s in select_samples(samples, seed=1, limit=20)]
        q2 = [s["question"] for s in select_samples(samples, seed=2, limit=20)]
        assert q1 != q2

    def test_id_is_not_none(self):
        """'id' must be set to a non-None value."""
        samples = make_samples(50)
        selected = select_samples(samples, seed=42, limit=10)
        for s in selected:
            assert s["id"] is not None


# ---------------------------------------------------------------------------
# save_samples
# ---------------------------------------------------------------------------

TIMESTAMP_RE = re.compile(r"email_\d{8}_\d{6}$")


class TestSaveSamplesDirectory:
    def test_output_dir_created(self, tmp_path):
        """save_samples must create a directory inside base_dir."""
        samples = make_samples(5)
        selected = select_samples(samples, seed=1, limit=5)
        _, out_dir = save_samples(selected, dataset_name="email", base_dir=str(tmp_path))
        assert os.path.isdir(out_dir)

    def test_output_dir_name_format(self, tmp_path):
        """Directory name must match email_YYYYMMDD_HHMMSS."""
        samples = make_samples(5)
        selected = select_samples(samples, seed=1, limit=5)
        _, out_dir = save_samples(selected, dataset_name="email", base_dir=str(tmp_path))
        assert TIMESTAMP_RE.search(os.path.basename(out_dir))

    def test_output_dir_inside_base_dir(self, tmp_path):
        """Output directory must be a child of base_dir."""
        samples = make_samples(5)
        selected = select_samples(samples, seed=1, limit=5)
        _, out_dir = save_samples(selected, dataset_name="email", base_dir=str(tmp_path))
        assert os.path.dirname(out_dir) == str(tmp_path)

    def test_two_runs_different_directories(self, tmp_path, monkeypatch):
        """Same seed must produce different output dirs (different timestamps)."""
        import bipia_benchmark.sampling as smod
        from datetime import datetime

        timestamps = ["20260101_000001", "20260101_000002"]
        call_count = {"n": 0}

        def fake_now():
            ts = timestamps[call_count["n"] % 2]
            call_count["n"] += 1
            return datetime.strptime(ts, "%Y%m%d_%H%M%S")

        monkeypatch.setattr(smod, "_now", fake_now)

        samples = make_samples(5)
        selected = select_samples(samples, seed=1, limit=5)
        _, dir1 = save_samples(selected, dataset_name="email", base_dir=str(tmp_path))
        _, dir2 = save_samples(selected, dataset_name="email", base_dir=str(tmp_path))
        assert dir1 != dir2


class TestSaveSamplesFiles:
    def test_samples_file_created(self, tmp_path):
        """save_samples must write a samples.json file."""
        samples = make_samples(5)
        selected = select_samples(samples, seed=1, limit=5)
        _, out_dir = save_samples(selected, dataset_name="email", base_dir=str(tmp_path))
        assert os.path.isfile(os.path.join(out_dir, "samples.json"))

    def test_file_contents_match_returned_samples(self, tmp_path):
        """File content must be identical to the returned sample list."""
        samples = make_samples(10)
        selected = select_samples(samples, seed=7, limit=5)
        returned, out_dir = save_samples(selected, dataset_name="email", base_dir=str(tmp_path))
        with open(os.path.join(out_dir, "samples.json")) as fh:
            on_disk = json.load(fh)
        assert on_disk == returned

    def test_same_seed_same_file_contents(self, tmp_path):
        """Same seed must produce identical file contents across runs."""
        samples = make_samples(20)

        sel1 = select_samples(samples, seed=99, limit=5)
        _, dir1 = save_samples(sel1, dataset_name="email", base_dir=str(tmp_path))

        sel2 = select_samples(samples, seed=99, limit=5)
        _, dir2 = save_samples(sel2, dataset_name="email", base_dir=str(tmp_path))

        with open(os.path.join(dir1, "samples.json")) as f1:
            data1 = json.load(f1)
        with open(os.path.join(dir2, "samples.json")) as f2:
            data2 = json.load(f2)

        assert data1 == data2


class TestSaveSamplesReturnValue:
    def test_returns_tuple_of_two(self, tmp_path):
        samples = make_samples(5)
        selected = select_samples(samples, seed=1, limit=5)
        result = save_samples(selected, dataset_name="email", base_dir=str(tmp_path))
        assert isinstance(result, tuple) and len(result) == 2

    def test_first_element_is_the_sample_list(self, tmp_path):
        samples = make_samples(5)
        selected = select_samples(samples, seed=1, limit=5)
        returned, _ = save_samples(selected, dataset_name="email", base_dir=str(tmp_path))
        assert returned == selected

    def test_second_element_is_string_path(self, tmp_path):
        samples = make_samples(5)
        selected = select_samples(samples, seed=1, limit=5)
        _, out_dir = save_samples(selected, dataset_name="email", base_dir=str(tmp_path))
        assert isinstance(out_dir, str)


class TestSaveSamplesEdgeCases:
    def test_empty_sample_list(self, tmp_path):
        """save_samples must handle an empty list without raising."""
        returned, out_dir = save_samples([], dataset_name="email", base_dir=str(tmp_path))
        assert returned == []
        assert os.path.isdir(out_dir)

    def test_base_dir_created_if_missing(self, tmp_path):
        """base_dir need not exist beforehand - it should be created."""
        new_base = os.path.join(str(tmp_path), "nonexistent", "nested")
        _, out_dir = save_samples([], dataset_name="email", base_dir=new_base)
        assert os.path.isdir(out_dir)

    def test_existing_dir_not_overwritten(self, tmp_path, monkeypatch):
        """If two calls land on the same timestamp, neither must destroy the other."""
        import bipia_benchmark.sampling as smod
        from datetime import datetime

        fixed_ts = "20260101_120000"
        monkeypatch.setattr(smod, "_now", lambda: datetime.strptime(fixed_ts, "%Y%m%d_%H%M%S"))

        samples = make_samples(3)
        selected = select_samples(samples, seed=1, limit=3)
        save_samples(selected, dataset_name="email", base_dir=str(tmp_path))
        # Second call with same timestamp must not raise
        save_samples(selected, dataset_name="email", base_dir=str(tmp_path))

    @pytest.mark.skipif(os.getuid() == 0, reason="root ignores permission bits")
    def test_permission_error_propagates(self, tmp_path):
        """An unwritable base_dir must raise PermissionError."""
        locked = os.path.join(str(tmp_path), "locked")
        os.makedirs(locked)
        os.chmod(locked, stat.S_IRUSR | stat.S_IXUSR)  # r-x, no write
        try:
            with pytest.raises(PermissionError):
                save_samples([], dataset_name="email", base_dir=locked)
        finally:
            os.chmod(locked, stat.S_IRWXU)
