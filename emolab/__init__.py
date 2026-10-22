"""emolab: multi-model facial-expression benchmarking and integration.

This package provides the emolab.inference/1.0 compatibility interface: a
framework-independent adapter contract so facial-expression models built by
different projects (OpenCode mini-Xception/ViT, Claude Code Emotion Lab,
Codex body-language) can be evaluated under identical conditions without
merging their repositories.

Nothing here is model-specific: adapters bring their own weights, preprocessing
and class mappings; this package only defines the contract they must honour.
"""

__version__ = "0.9.0"
