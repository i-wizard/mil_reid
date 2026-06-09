"""
Machine-learning core for the animal re-identification demo.

This package is the first of three parts of the demo app (the others being a
FastAPI wrapper and a browser client). It is deliberately self-contained: the
``ml.inference`` subpackage is the only seam the API layer should import, so
that no model code is duplicated outside of here.

The approach is patch-bag Multiple Instance Learning (MIL): each image is a bag
of patch instances, a frozen backbone embeds the patches, and a small trainable
gated-attention head pools them into a single identity embedding used for
open-set retrieval.
"""
