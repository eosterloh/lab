"""Public function-calling data and synthetic traces for the ensemble roles.

SFT rows are TRL ``messages`` (``user`` = the harness prompt from
``thinker_prompt`` / ``tooler_prompt`` / ``coder_prompt``, ``assistant`` = the
raw completion that role's parser accepts). Public function-calling rows keep
their own tool list inside that same tooler template and are tagged
``source=foreign_tools`` (``origin`` is the dataset id) so a trainer can mix
them with native harness rows. DPO rows are TRL
``prompt`` / ``chosen`` / ``rejected`` strings. The harness calls
``backend.generate`` with that raw prompt and no chat template; point the
trainer at these strings the same way, or the model trains on a different
prefix than it sees at inference.

Build::

    python -m lab.posttrain.data build --role tooler --out <dir> [--max N]
"""

from lab.posttrain.data.build import build_role

__all__ = ["build_role"]
