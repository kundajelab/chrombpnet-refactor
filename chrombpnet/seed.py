"""The default of ``-s/--seed``, the one random seed of every command.

Each command that draws anything at random takes ``-s/--seed`` and draws from
it; this is its default, and the seed of library callers that pass none.
Imports nothing, so any module can use it.
"""

DEFAULT_SEED = 1234
