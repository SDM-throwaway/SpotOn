"""
Pytest configuration and fixtures for BIPIA tests.
Provides mocks for bipia module when it's not installed.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock


class MockDataFrame:
    """Simple DataFrame mock that mimics pandas DataFrame behavior needed by tests."""

    def __init__(self, data):
        self.data = data

    def to_dict(self, orient="records"):
        """Return data as list of dicts (records orientation)."""
        return self.data


class MockAutoDataBuilder:
    """Mock for any AutoPIABuilder dataset builder instance."""

    def __call__(self, benchmark_path, attack_path, enable_stealth=True):
        """Load data from local BIPIA files."""
        samples = []

        # Load benchmark data
        if Path(benchmark_path).exists():
            with open(benchmark_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sample = json.loads(line)
                        samples.append(sample)

        # Load attack data
        attack_data = {}
        if Path(attack_path).exists():
            with open(attack_path) as f:
                attack_data = json.load(f)

        # Merge attack data into samples
        attack_list = []
        if isinstance(attack_data, dict):
            # Flatten attack_data if it's a dict of lists (like text_attack_test.json)
            for key, value in attack_data.items():
                if isinstance(value, list):
                    attack_list.extend(value)
                elif isinstance(value, str):
                    attack_list.append(value)

        for i, sample in enumerate(samples):
            # Add attack_str if not present
            if "attack_str" not in sample or not sample.get("attack_str"):
                if attack_list:
                    # Cycle through attack list if needed
                    sample["attack_str"] = attack_list[i % len(attack_list)]
                else:
                    sample["attack_str"] = f"injection_test_{i}"

        return MockDataFrame(samples)


def create_mock_bipia():
    """Create mock bipia module."""
    bipia_module = MagicMock()
    bipia_module.data.AutoPIABuilder.from_name = MagicMock(
        side_effect=lambda name: MockAutoDataBuilder
    )
    return bipia_module


# Register the mock before any imports of bipia
if "bipia" not in sys.modules:
    sys.modules["bipia"] = create_mock_bipia()
    sys.modules["bipia.data"] = sys.modules["bipia"].data
