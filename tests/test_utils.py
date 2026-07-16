# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import unittest

from optuna.study import StudyDirection
from optuna.trial import create_trial

from heretic.config import Settings
from heretic.utils import (
    generate_reproduce_json,
    get_settings_from_reproduction,
    get_trial_parameters,
    get_trial_sort_key,
    get_trial_user_attrs_from_reproduction,
)


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
    return Settings.model_validate(data)


def make_score_records() -> list[dict[str, object]]:
    return [
        {
            "name": "Keywords",
            "score": {
                "value": 0.2,
                "rich_display": "2/10",
                "md_display": "2/10",
            },
            "baseline": {
                "value": 0.5,
                "rich_display": "5/10",
                "md_display": "5/10",
            },
        },
        {
            "name": "PIQA acc_norm",
            "score": {
                "value": 0.7,
                "rich_display": "0.7000",
                "md_display": "0.7000",
            },
            "baseline": {
                "value": 0.6,
                "rich_display": "0.6000",
                "md_display": "0.6000",
            },
        },
    ]


class TrialFormattingTests(unittest.TestCase):
    def test_get_trial_parameters_formats_directional_trials(self) -> None:
        trial = create_trial(
            values=[0.0],
            user_attrs={
                "direction_index": 12.3456,
                "parameters": {
                    "attn.o_proj": {
                        "max_weight": 1.2345,
                        "max_weight_position": 18.7654,
                        "min_weight": 0.3333,
                        "min_weight_distance": 7.6543,
                    }
                },
            },
        )

        parameters = get_trial_parameters(make_settings(use_ara=False), trial)

        self.assertEqual("12.35", parameters["direction_index"])
        self.assertEqual("1.23", parameters["attn.o_proj.max_weight"])
        self.assertEqual("18.77", parameters["attn.o_proj.max_weight_position"])
        self.assertEqual("0.33", parameters["attn.o_proj.min_weight"])
        self.assertEqual("7.65", parameters["attn.o_proj.min_weight_distance"])

    def test_get_trial_parameters_formats_ara_trials(self) -> None:
        trial = create_trial(
            values=[0.0],
            user_attrs={
                "ara_parameters": {
                    "start_layer_index": 3,
                    "end_layer_index": 9,
                    "preserve_good_behavior_weight": 0.125,
                    "steer_bad_behavior_weight": 0.0625,
                    "overcorrect_relative_weight": 1.25,
                    "neighbor_count": 7,
                }
            },
        )

        parameters = get_trial_parameters(make_settings(use_ara=True), trial)

        self.assertEqual(
            {
                "start_layer_index": "3",
                "end_layer_index": "9",
                "preserve_good_behavior_weight": "0.1250",
                "steer_bad_behavior_weight": "0.0625",
                "overcorrect_relative_weight": "1.2500",
                "neighbor_count": "7",
            },
            parameters,
        )


class ReproductionTests(unittest.TestCase):
    def test_missing_ara_flags_in_legacy_reproduction_default_to_directional(self) -> None:
        reproduction_information = {
            "settings": {
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
            }
        }

        settings_data = get_settings_from_reproduction(reproduction_information)

        self.assertFalse(settings_data["use_ara"])
        self.assertFalse(settings_data["use_ara_lora"])

    def test_generate_reproduce_json_and_restore_directional_parameters(self) -> None:
        scores = make_score_records()
        trial = create_trial(
            values=[0.0],
            user_attrs={
                "index": 4,
                "direction_index": 14.25,
                "parameters": {
                    "attn.o_proj": {
                        "max_weight": 1.1,
                        "max_weight_position": 20.0,
                        "min_weight": 0.2,
                        "min_weight_distance": 5.0,
                    }
                },
                "scores": scores,
            },
        )
        settings = make_settings(use_ara=False)

        reproduction_information = json.loads(
            generate_reproduce_json(
                settings,
                trial,
                timestamp="2026-07-16T00:00:00",
                uploaded_model_hashes={"model.safetensors": "abc123"},
                include_system_information=False,
            )
        )

        self.assertEqual("3", reproduction_information["version"])
        self.assertEqual(
            {
                "direction_index": 14.25,
                "abliteration_parameters": trial.user_attrs["parameters"],
            },
            reproduction_information["parameters"],
        )
        self.assertEqual(
            {
                "direction_index": 14.25,
                "parameters": trial.user_attrs["parameters"],
                "scores": scores,
            },
            get_trial_user_attrs_from_reproduction(
                settings,
                reproduction_information,
            ),
        )

    def test_generate_reproduce_json_and_restore_ara_parameters(self) -> None:
        scores = make_score_records()
        trial = create_trial(
            values=[0.0],
            user_attrs={
                "index": 7,
                "ara_parameters": {
                    "start_layer_index": 2,
                    "end_layer_index": 11,
                    "preserve_good_behavior_weight": 0.5,
                    "steer_bad_behavior_weight": 0.125,
                    "overcorrect_relative_weight": 0.75,
                    "neighbor_count": 5,
                },
                "scores": scores,
            },
        )
        settings = make_settings(use_ara=True)

        reproduction_information = json.loads(
            generate_reproduce_json(
                settings,
                trial,
                timestamp="2026-07-16T00:00:00",
                uploaded_model_hashes={"model.safetensors": "def456"},
                include_system_information=False,
            )
        )

        self.assertEqual(
            {
                "ara_parameters": trial.user_attrs["ara_parameters"],
            },
            reproduction_information["parameters"],
        )
        self.assertEqual(
            {
                "ara_parameters": trial.user_attrs["ara_parameters"],
                "scores": scores,
            },
            get_trial_user_attrs_from_reproduction(
                settings,
                reproduction_information,
            ),
        )


class TrialSortingTests(unittest.TestCase):
    def test_get_trial_sort_key_normalizes_maximize_objectives(self) -> None:
        lower_piqa = create_trial(
            values=[0.0, 0.0],
            user_attrs={
                "scores": [
                    {
                        "name": "Keywords",
                        "score": {"value": 0.2},
                    },
                    {
                        "name": "PIQA acc_norm",
                        "score": {"value": 0.6},
                    },
                ]
            },
        )
        higher_piqa = create_trial(
            values=[0.0, 0.0],
            user_attrs={
                "scores": [
                    {
                        "name": "Keywords",
                        "score": {"value": 0.2},
                    },
                    {
                        "name": "PIQA acc_norm",
                        "score": {"value": 0.8},
                    },
                ]
            },
        )

        sorted_trials = sorted(
            [lower_piqa, higher_piqa],
            key=lambda trial: get_trial_sort_key(
                trial,
                ["Keywords", "PIQA acc_norm"],
                [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE],
            ),
        )

        self.assertIs(higher_piqa, sorted_trials[0])
        self.assertIs(lower_piqa, sorted_trials[1])


if __name__ == "__main__":
    unittest.main()
