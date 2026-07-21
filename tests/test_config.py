# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
import sys
from unittest.mock import patch

from pydantic import ValidationError

from heretic.config import ScorerConfig, SeededARATrial, Settings


def make_settings(**overrides) -> Settings:
    data = {
        "model": "dummy/model",
        "good_prompts": {
            "dataset": "good-dataset",
            "split": "train[:1]",
            "column": "text",
        },
        "bad_prompts": {
            "dataset": "bad-dataset",
            "split": "train[:1]",
            "column": "text",
        },
        "scorers": [
            {
                "plugin": "heretic.scorers.keyword_rate.KeywordRate",
                "optimization": "minimize",
            },
            {
                "plugin": "heretic.scorers.kl_divergence.KLDivergence",
                "optimization": "minimize",
            },
        ],
    }
    data.update(overrides)
    with patch.object(sys, "argv", [sys.argv[0]]):
        return Settings.model_validate(data)


class ScorerConfigTests(unittest.TestCase):
    def test_accepts_slug_like_instance_name(self) -> None:
        config = ScorerConfig(
            plugin="heretic.scorers.keyword_rate.KeywordRate",
            optimization="minimize",
            instance_name="small-1",
        )

        self.assertEqual(config.instance_name, "small-1")

    def test_rejects_empty_instance_name(self) -> None:
        with self.assertRaises(ValidationError):
            ScorerConfig(
                plugin="heretic.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
                instance_name=" \t",
            )

    def test_rejects_whitespace_in_instance_name(self) -> None:
        for instance_name in ["small name", "small\tname", "small\nname"]:
            with self.subTest(instance_name=instance_name):
                with self.assertRaisesRegex(
                    ValidationError, "whitespace is not allowed"
                ):
                    ScorerConfig(
                        plugin="heretic.scorers.keyword_rate.KeywordRate",
                        optimization="minimize",
                        instance_name=instance_name,
                    )

    def test_rejects_dot_in_instance_name(self) -> None:
        with self.assertRaisesRegex(ValidationError, "'\\.' is not allowed"):
            ScorerConfig(
                plugin="heretic.scorers.keyword_rate.KeywordRate",
                optimization="minimize",
                instance_name="small.name",
            )


class SettingsCompatibilityTests(unittest.TestCase):
    def test_quantized_ara_auto_enables_lora(self) -> None:
        settings = make_settings(
            quantization="bnb_4bit",
            use_ara=True,
            use_ara_lora=False,
        )

        self.assertTrue(settings.use_ara_lora)
        self.assertTrue(settings._auto_enabled_use_ara_lora)

    def test_quantized_directional_run_does_not_enable_lora(self) -> None:
        settings = make_settings(
            quantization="bnb_4bit",
            use_ara=False,
            use_ara_lora=False,
        )

        self.assertFalse(settings.use_ara_lora)
        self.assertFalse(settings._auto_enabled_use_ara_lora)

    def test_seed_ara_trials_are_parsed(self) -> None:
        settings = make_settings(
            seed_ara_trials=[
                {
                    "start_layer_index": 9,
                    "end_layer_index": 36,
                    "preserve_good_behavior_weight": 0.4826,
                    "steer_bad_behavior_weight": 0.0002,
                    "overcorrect_relative_weight": 1.0491,
                    "neighbor_count": 15,
                }
            ],
        )

        self.assertEqual(len(settings.seed_ara_trials), 1)
        self.assertIsInstance(settings.seed_ara_trials[0], SeededARATrial)
        self.assertEqual(settings.seed_ara_trials[0].start_layer_index, 9)
        self.assertEqual(settings.seed_ara_trials[0].end_layer_index, 36)

    def test_seed_ara_trials_validate_layer_range(self) -> None:
        with self.assertRaisesRegex(
            ValidationError, "end_layer_index must be greater than start_layer_index"
        ):
            make_settings(
                seed_ara_trials=[
                    {
                        "start_layer_index": 12,
                        "end_layer_index": 12,
                        "preserve_good_behavior_weight": 0.5,
                        "steer_bad_behavior_weight": 0.0002,
                        "overcorrect_relative_weight": 1.0,
                        "neighbor_count": 15,
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
