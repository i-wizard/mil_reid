"""
HTTP layer (Part 2) for the animal re-identification demo.

A thin FastAPI wrapper over the ML core's inference seam (``ml.inference``). It
only marshals HTTP ⇆ the existing inference functions — no model logic lives
here — so that the Part 3 browser client (and any other consumer) can drive
enrollment, identification, and explanation over the network.
"""
