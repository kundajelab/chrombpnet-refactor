"""The CLI invocations the regression harness replays.

Scenarios run in order inside one workspace and may consume the outputs of
earlier ones, which is how the real pipelines are used: ``bias train`` produces
the bias checkpoint that ``train`` consumes, and ``train`` produces the
checkpoints that ``qc`` and ``pred_bw`` consume.

Geometry comes from :mod:`tests.regression.fixture`; the sizes are chosen so the
whole sweep runs in minutes rather than hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import fixture

# Shared model/training geometry. Tiny but architecturally valid.
GEOMETRY = [
    "-il", str(fixture.INPUTLEN),
    "-ol", str(fixture.OUTPUTLEN),
    "-fil", "4",
    "-dil", "1",
    "-bs", "8",
    "-e", "2",
    "-es", "2",
    "-s", "1234",
]


@dataclass(frozen=True)
class Scenario:
    """One ``chrombpnet`` invocation.

    ``argv`` placeholders are formatted with the fixture paths and the workspace
    root, so a scenario can refer to another scenario's output directory.
    """

    name: str
    argv: list[str]
    #: Directory the command writes into, relative to the workspace root.
    #: The training commands require their subdirectories not to exist, so the
    #: harness clears this before each run.
    output: str
    #: modiscolite is only an optional [reports] extra, and the qc pipelines
    #: run motif discovery, so those scenarios are skipped where it is absent.
    needs_modiscolite: bool = False
    #: Set to the reason when a scenario's output is not reproducible. Such
    #: scenarios still run -- that catches crashes and import errors -- but
    #: their artifacts are not compared, because a diff would be noise rather
    #: than a regression signal.
    nondeterministic: str | None = None
    #: Artifacts to ignore when hashing, relative to ``output``.
    ignore: tuple[str, ...] = field(default_factory=tuple)


def _scenarios() -> list[Scenario]:
    g = GEOMETRY
    return [
        Scenario(
            name="01_prep_splits",
            output="01_prep_splits",
            argv=["prep", "splits",
                  "-c", "{chrom_sizes}",
                  "-tcr", *fixture.TEST_CHROMS,
                  "-vcr", *fixture.VALID_CHROMS,
                  "-op", "{out}/splits"],
        ),
        Scenario(
            name="02_prep_nonpeaks",
            output="02_prep_nonpeaks",
            argv=["prep", "nonpeaks",
                  "-g", "{genome}",
                  "-c", "{chrom_sizes}",
                  "-p", "{peaks}",
                  "-fl", "{folds}",
                  "-il", str(fixture.INPUTLEN),
                  "-st", "200",
                  "-s", "1234",
                  # prep nonpeaks spells this -o/--output-prefix while prep
                  # splits spells it -op/--output_prefix. Not a typo here.
                  "-o", "{out}/neg"],
        ),
        Scenario(
            name="03_bias_train",
            output="03_bias_train",
            argv=["bias", "train",
                  "-g", "{genome}", "-c", "{chrom_sizes}", "-itag", "{reads}",
                  "-o", "{out}", "-d", "ATAC",
                  "-p", "{peaks}", "-n", "{nonpeaks}", "-fl", "{folds}",
                  "-b", "0.5", "-j", "0", "--device", "cpu", *g],
        ),
        Scenario(
            name="04_train",
            output="04_train",
            argv=["train",
                  "-g", "{genome}", "-c", "{chrom_sizes}", "-itag", "{reads}",
                  "-o", "{out}", "-d", "ATAC",
                  "-p", "{peaks}", "-n", "{nonpeaks}", "-fl", "{folds}",
                  "-b", "{root}/03_bias_train/models/bias.pt",
                  "-j", str(fixture.MAX_JITTER), "-sr", "0.5",
                  "--device", "cpu", *g],
        ),
        Scenario(
            name="05_bias_qc",
            nondeterministic=(
                "qc exposes neither --seed nor --benchmark, so "
                "initializers passes benchmark_seed=None and make_generator_rng "
                "returns the global np.random, which nothing in the qc path ever "
                "seeds. subsample_nonpeak_data therefore draws a different test-set "
                "nonpeak subsample on every run."),
            output="05_bias_qc",
            needs_modiscolite=True,
            argv=["bias", "qc",
                  "-bw", "{bigwig}", "-bm", "{root}/03_bias_train/models/bias.pt",
                  "-g", "{genome}", "-c", "{chrom_sizes}",
                  "-o", "{out}", "-d", "ATAC",
                  "-p", "{peaks}", "-n", "{nonpeaks}", "-fl", "{folds}",
                  "-bs", "8", "--device", "cpu"],
        ),
        Scenario(
            name="06_qc",
            nondeterministic=(
                "qc exposes neither --seed nor --benchmark, so "
                "initializers passes benchmark_seed=None and make_generator_rng "
                "returns the global np.random, which nothing in the qc path ever "
                "seeds. subsample_nonpeak_data therefore draws a different test-set "
                "nonpeak subsample on every run."),
            output="06_qc",
            needs_modiscolite=True,
            argv=["qc",
                  "-bw", "{bigwig}",
                  "-cm", "{root}/04_train/models/chrombpnet.pt",
                  "-cmb", "{root}/04_train/models/chrombpnet_nobias.pt",
                  "-g", "{genome}", "-c", "{chrom_sizes}",
                  "-o", "{out}", "-d", "ATAC",
                  "-p", "{peaks}", "-n", "{nonpeaks}", "-fl", "{folds}",
                  "-bs", "8", "--device", "cpu"],
        ),
        Scenario(
            name="07_pred_bw",
            output="07_pred_bw",
            argv=["pred_bw",
                  "-bm", "{root}/03_bias_train/models/bias.pt",
                  "-cm", "{root}/04_train/models/chrombpnet.pt",
                  "-cmb", "{root}/04_train/models/chrombpnet_nobias.pt",
                  "-r", "{peaks}", "-g", "{genome}", "-c", "{chrom_sizes}",
                  "-bw", "{bigwig}",
                  "-op", "{out}/pred", "-os", "{out}/stats",
                  "-bs", "8", "-t", "0", "--device", "cpu"],
        ),
        Scenario(
            name="08_contribs_bw",
            output="08_contribs_bw",
            argv=["contribs_bw",
                  "-m", "{root}/04_train/models/chrombpnet_nobias.pt",
                  "-r", "{peaks}", "-g", "{genome}", "-c", "{chrom_sizes}",
                  "-op", "{out}/contrib",
                  "-pc", "counts", "profile",
                  "-t", "0", "--batch-size", "8", "--num-shuffles", "2",
                  "--seed", "1234", "--device", "cpu"],
        ),
        Scenario(
            name="09_footprints",
            output="09_footprints",
            argv=["footprints",
                  "-m", "{root}/04_train/models/chrombpnet_nobias.pt",
                  "-r", "{nonpeaks}", "-g", "{genome}", "-fl", "{folds}",
                  "-op", "{out}/fp",
                  "-pwm_f", "{motifs_to_pwm}",
                  "-bs", "8", "--device", "cpu"],
        ),
        Scenario(
            name="10_modisco_motifs",
            output="10_modisco_motifs",
            needs_modiscolite=True,
            argv=["modisco_motifs",
                  "-i", "{root}/08_contribs_bw/contrib.profile_scores.h5",
                  "-op", "{out}/modisco",
                  "-n", "500", "-l", "1", "-w", "50"],
        ),
    ]


SCENARIOS = _scenarios()
BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}
