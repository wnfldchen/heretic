# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import lm_eval

from heretic.plugin import Context
from heretic.scorer import Score, Scorer


class PIQA(Scorer):
    @property
    def score_name(self) -> str:
        return "PIQA acc_norm"

    def get_score(self, ctx: Context) -> Score:
        results = lm_eval.simple_evaluate(
            model=ctx.get_lm_eval_model(),
            tasks=["piqa"],
        )
        piqa_acc_norm = float(results["results"]["piqa"]["acc_norm,none"])
        return Score(
            value=piqa_acc_norm,
            rich_display=f"{piqa_acc_norm:.4f}",
            md_display=f"{piqa_acc_norm:.4f}",
        )
