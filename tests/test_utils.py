# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import sys
import unittest
from unittest.mock import patch

from huggingface_hub import ModelCardData
from optuna.study import StudyDirection
from optuna.trial import create_trial

from heretic.config import Settings
from heretic.utils import (
    generate_evaluation_section,
    generate_reproduce_json,
    get_settings_from_reproduction,
    get_readme_intro,
    get_trial_parameters,
    get_trial_sort_key,
    get_trial_user_attrs_from_reproduction,
    update_model_card_data,
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
    with patch.object(sys, "argv", [sys.argv[0]]):
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


def make_benchmark_runs() -> list[dict[str, object]]:
    return [
        {
            "benchmark": "WinoGrande",
            "results": {
                "name": "winogrande",
                "sample_len": 1267,
                "acc,none": 0.4949,
                "acc_stderr,none": 0.0141,
            },
            "original_results": {
                "name": "winogrande",
                "sample_len": 1267,
                "acc,none": 0.5051,
                "acc_stderr,none": 0.0141,
            },
        },
        {
            "benchmark": "EQ-Bench",
            "results": {
                "name": "eq_bench",
                "sample_len": 171,
                "eqbench,none": 8.5117,
                "eqbench_stderr,none": 2.5467,
                "percent_parseable,none": 65.4971,
                "percent_parseable_stderr,none": 3.6460,
            },
            "original_results": {
                "name": "eq_bench",
                "sample_len": 171,
                "eqbench,none": 6.0554,
                "eqbench_stderr,none": 1.9705,
                "percent_parseable,none": 55.5556,
                "percent_parseable_stderr,none": 3.8111,
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
    def test_missing_ara_flags_in_legacy_reproduction_default_to_directional(
        self,
    ) -> None:
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


class ModelCardTests(unittest.TestCase):
    def test_update_model_card_data_sets_derivative_metadata(self) -> None:
        card_data = ModelCardData(
            tags=["gemma4_unified"],
            pipeline_tag="any-to-any",
        )

        update_model_card_data(
            card_data,
            make_settings(
                model="google/gemma-4-12B-it-qat-q4_0-unquantized",
                use_ara=True,
            ),
            contains_reproducibility_information=True,
        )

        self.assertEqual(
            "google/gemma-4-12B-it-qat-q4_0-unquantized",
            card_data.base_model,
        )
        self.assertEqual("transformers", card_data.library_name)
        self.assertEqual("any-to-any", card_data.pipeline_tag)
        self.assertIn("gemma4_unified", card_data.tags)
        self.assertIn("heretic", card_data.tags)
        self.assertIn("ara", card_data.tags)
        self.assertIn("reproducible", card_data.tags)
        self.assertIn("not-for-all-audiences", card_data.tags)

    def test_generate_evaluation_section_formats_benchmark_comparisons(self) -> None:
        section = generate_evaluation_section(make_benchmark_runs())

        self.assertIn("## Evaluation", section)
        self.assertIn(
            "| WinoGrande | 1267 | Accuracy | 49.49% | 50.51% | -1.02 pp | 1.41 pp | 1.41 pp |",
            section,
        )
        self.assertIn(
            "| EQ-Bench | 171 | EQ-Bench score | 8.51 | 6.06 | +2.46 | 2.55 | 1.97 |",
            section,
        )
        self.assertIn(
            "| EQ-Bench | 171 | Parseable outputs | 65.50% | 55.56% | +9.94 pp | 3.65 pp | 3.81 pp |",
            section,
        )
        self.assertIn("The largest decline is `-1.02 pp`", section)
        self.assertIn("The largest improvement is `+9.94 pp`", section)

    def test_get_readme_intro_includes_evaluation_and_upstream_note(self) -> None:
        trial = create_trial(
            values=[0.0],
            user_attrs={
                "ara_parameters": {
                    "start_layer_index": 19,
                    "end_layer_index": 42,
                    "preserve_good_behavior_weight": 0.9174,
                    "steer_bad_behavior_weight": 0.0004,
                    "overcorrect_relative_weight": 0.9004,
                    "neighbor_count": 13,
                },
                "scores": make_score_records(),
            },
        )

        with patch("heretic.utils.version", return_value="1.4.0"):
            readme_intro = get_readme_intro(
                make_settings(
                    model="google/gemma-4-12B-it-qat-q4_0-unquantized",
                    use_ara=True,
                ),
                trial,
                contains_reproducibility_information=True,
                evaluation_section=generate_evaluation_section(make_benchmark_runs()),
                includes_upstream_model_card=True,
            )

        self.assertIn("## Evaluation", readme_intro)
        self.assertIn(
            "The remainder of this README reproduces the upstream model card",
            readme_intro,
        )
        self.assertIn("This model is reproducible!", readme_intro)


if __name__ == "__main__":
    unittest.main()
