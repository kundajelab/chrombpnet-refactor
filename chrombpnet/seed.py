"""The default of ``-s/--seed``, the one random seed of every command.

Each command that draws anything at random takes ``-s/--seed`` and draws from
it; this is its default, and the seed of library callers that pass none. The
README's "Randomized seeds" section lists what each command draws with it; outside it,
TF-MoDISco seeds itself (modisco-lite takes no seed). chrombpnet 1.x drew the
training negatives with a fixed seed of 1; ``--seed 1`` makes that draw (the
README says when that gives 1.x's negatives).
Imports nothing, so any module can use it.
"""

DEFAULT_SEED = 1234
