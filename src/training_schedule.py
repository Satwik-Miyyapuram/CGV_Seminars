"""Centralized training schedule constants shared across the codebase.

Place iteration numbers and step lists here so other modules can import
them instead of hard-coding the same literals in multiple places.
"""

# Phase boundaries and key iteration numbers
SURFEL_DENSIFICATION_STOP = 10000
SURFEL_PHASE_END = 20000
VISIBILITY_PRUNE_STEP = 15000
CLAMP_18K_STEP = 18000
CLAMP_19K_STEP = 19000
GAUSSIAN_SPAWN_STEP = SURFEL_PHASE_END

# Saving / visualization milestones
LOSS_GRAPH_STEP = 49999
FIXED_VIEW_STEPS = [
    1000,
    5000,
    9999,
    10001,
    14999,
    15001,
    17999,
    18001,
    18999,
    19001,
    19999,
    20001,
    25000,
    30000,
    35000,
    40000,
    45000,
    LOSS_GRAPH_STEP,
]
MILESTONE_STEPS = [9999, 10001, 14999, 15001, 17500, 19999, 20001]

# Derived helper step used for assembling composites in the model
COMPOSITE_ASSEMBLY_STEP = LOSS_GRAPH_STEP - SURFEL_PHASE_END
