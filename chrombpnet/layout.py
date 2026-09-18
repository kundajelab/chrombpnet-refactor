"""The output directory contract for a pipeline run.

Every pipeline writes into a fixed four-directory tree under ``--output-dir``
and prefixes filenames with ``--file-prefix``. That contract used to live in two
places at once: ``CHROMBPNET.py`` created the directories, and ``pipelines.py``
built roughly seventy ``os.path.join(args.output_dir, "auxiliary/{}...".format(
fpx))`` strings to read and write inside them. Keeping the two in step was
manual, and a typo in any one of those strings produced a missing-file error
somewhere much later in the run.
"""

from __future__ import annotations

import os
import shutil


class RunLayout:
    """Resolves paths inside one pipeline run's output tree.

    ``file_prefix`` is the user's ``--file-prefix``; the trailing underscore
    that every filename needs is added here rather than at each of the four
    places that previously recomputed ``fpx``.
    """

    #: Subdirectories created for a training run, in creation order.
    TRAINING_DIRS = ("logs", "auxiliary", "models", "evaluation")
    #: Subdirectories a qc run needs. It creates fewer, and tolerates their
    #: existing: qc consumes user-provided peak files and must not demand a
    #: pristine directory (defect 1 in chrombpnet_bugs.md).
    QC_DIRS = ("auxiliary", "evaluation")

    def __init__(self, output_dir, file_prefix=None):
        self.output_dir = output_dir
        self.prefix = f"{file_prefix}_" if file_prefix else ""

    @classmethod
    def from_args(cls, args):
        return cls(args.output_dir, getattr(args, "file_prefix", None))

    # -- directories -------------------------------------------------------
    def _in(self, subdir, name=""):
        return os.path.join(self.output_dir, subdir, f"{self.prefix}{name}")

    def aux(self, name=""):
        return self._in("auxiliary", name)

    def models(self, name=""):
        return self._in("models", name)

    def logs(self, name=""):
        return self._in("logs", name)

    def evaluation(self, name=""):
        return self._in("evaluation", name)

    def interpret_subsample(self, name=""):
        return os.path.join(self.output_dir, "auxiliary", "interpret_subsample",
                            f"{self.prefix}{name}")

    def modisco_dir(self, kind):
        """``evaluation/modisco_profile/`` or ``evaluation/modisco_counts/``.

        Not prefixed: these directory names never carried the file prefix.
        """
        return os.path.join(self.output_dir, "evaluation", f"modisco_{kind}") + os.sep

    def unprefixed(self, subdir, name):
        """A path in ``subdir`` that deliberately omits the file prefix."""
        return os.path.join(self.output_dir, subdir, name)

    # -- creation ----------------------------------------------------------
    def create_training_dirs(self):
        """Create the training tree, refusing to reuse an existing one.

        ``exist_ok=False`` is deliberate: a training run writes checkpoints and
        logs whose names it does not vary, so reusing a directory would
        silently mix two runs' outputs.
        """
        for subdir in self.TRAINING_DIRS:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=False)

    def create_qc_dirs(self):
        """Create the qc tree, reusing it if present (see :data:`QC_DIRS`)."""
        for subdir in self.QC_DIRS:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

    def create_interpret_subsample_dir(self):
        os.makedirs(os.path.join(self.output_dir, "auxiliary",
                                 "interpret_subsample"), exist_ok=False)

    # -- moving files between subdirectories -------------------------------
    def move(self, from_subdir, to_subdir, name):
        """Move ``name`` between two subdirectories of this run.

        shutil.move rather than os.rename, which fails outright when the two
        directories are on different filesystems.
        """
        shutil.move(self._in(from_subdir, name), self._in(to_subdir, name))
