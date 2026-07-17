"""Curator tier — a human-in-the-loop labeling substrate for the framing corpus.

Produces the durable labeled pair set that later feeds the bi-encoder frame
detector (sub-project B) and the larger-set Cartographer recalibration
(sub-project C). Consumes gin.cartographer; imported by nothing.
See docs/superpowers/specs/2026-07-17-curator-ui-label-store-design.md.
"""
