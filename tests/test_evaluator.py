# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import sys
import unittest
from unittest.mock import patch

from heretic.config import Settings
from heretic.evaluator import BUILTIN_KL_PLUGIN, BUILTIN_PIQA_PLUGIN, Evaluator


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
                "plugin": BUILTIN_KL_PLUGIN,
                "optimization": "minimize",
            },
        ],
    }
    data.update(overrides)
    with patch.object(sys, "argv", [sys.argv[0]]):
        return Settings.model_validate(data)


class EvaluatorCompatibilityTests(unittest.TestCase):
    def make_evaluator(self, settings: Settings) -> Evaluator:
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.settings = settings
        return evaluator

    def test_use_piqa_replaces_builtin_kl_divergence(self) -> None:
        evaluator = self.make_evaluator(make_settings(use_piqa=True))

        configs = evaluator._get_effective_scorer_configs()

        self.assertEqual(
            [
                "heretic.scorers.keyword_rate.KeywordRate",
                BUILTIN_PIQA_PLUGIN,
            ],
            [config.plugin for config in configs],
        )
        self.assertEqual("maximize", configs[1].optimization)

    def test_use_piqa_does_not_duplicate_explicit_piqa(self) -> None:
        evaluator = self.make_evaluator(
            make_settings(
                use_piqa=True,
                scorers=[
                    {
                        "plugin": "heretic.scorers.keyword_rate.KeywordRate",
                        "optimization": "minimize",
                    },
                    {
                        "plugin": BUILTIN_KL_PLUGIN,
                        "optimization": "minimize",
                    },
                    {
                        "plugin": BUILTIN_PIQA_PLUGIN,
                        "optimization": "none",
                    },
                ],
            )
        )

        configs = evaluator._get_effective_scorer_configs()

        self.assertEqual(
            1,
            sum(config.plugin == BUILTIN_PIQA_PLUGIN for config in configs),
        )
        piqa_config = next(
            config for config in configs if config.plugin == BUILTIN_PIQA_PLUGIN
        )
        self.assertEqual("maximize", piqa_config.optimization)

    def test_use_piqa_appends_piqa_when_missing(self) -> None:
        evaluator = self.make_evaluator(
            make_settings(
                use_piqa=True,
                scorers=[
                    {
                        "plugin": "heretic.scorers.keyword_rate.KeywordRate",
                        "optimization": "minimize",
                    }
                ],
            )
        )

        configs = evaluator._get_effective_scorer_configs()

        self.assertEqual(
            [
                "heretic.scorers.keyword_rate.KeywordRate",
                BUILTIN_PIQA_PLUGIN,
            ],
            [config.plugin for config in configs],
        )
        self.assertEqual("maximize", configs[-1].optimization)


if __name__ == "__main__":
    unittest.main()
