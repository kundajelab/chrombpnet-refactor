import h5py
import numpy as np
import pyBigWig
import pytest
from types import SimpleNamespace

from chrombpnet.evaluation.interpret.interpret import generate_shap_dict
from chrombpnet.evaluation.interpret.input_utils import get_seq as get_interpret_sequences
from chrombpnet.evaluation.make_bigwigs.bigwig_helper import write_bigwig
from chrombpnet.training.predict import write_predictions_h5py
from chrombpnet.training.utils import augment
from chrombpnet.training.utils.one_hot import dna_to_one_hot, one_hot_to_dna
from chrombpnet.model import BiasModel, ModelConfig, PredictionAdapter, save_model
from chrombpnet.evaluation.make_bigwigs.predict_to_bigwig import main as predict_bigwig


def test_one_hot_unknown_and_roundtrip():
    values = dna_to_one_hot(["ACGTN", "tgcan"])
    assert values.shape == (2, 5, 4)
    assert values[:, -1].sum() == 0
    assert one_hot_to_dna(values) == ["ACGTN", "TGCAN"]


def test_interpret_sequence_loading_filters_unknown_bases():
    regions = np.array([
        ("chr1", 0, 2),
        ("chr1", 4, 4),
    ], dtype=[("chr", "U4"), ("start", "i8"), ("summit", "i8")])
    import pandas as pd
    frame = pd.DataFrame.from_records(regions)
    sequences, used = get_interpret_sequences(
        frame, {"chr1": "AACCGGTTNNNN"}, width=4)
    assert used.tolist() == [True, False]
    assert sequences.shape == (1, 4, 4)
    assert np.all(sequences.sum(axis=2) == 1)


def test_crop_and_reverse_complement_semantics():
    seqs = dna_to_one_hot(["AACCGGTT", "ACGTACGT"])
    labels = np.arange(12).reshape(2, 6)
    coords = np.array([["chr1", "10", "f", "1"], ["chr1", "20", "f", "1"]])
    np.random.seed(2)
    cropped, counts, shifted = augment.random_crop(seqs, labels, 6, 4, coords)
    assert cropped.shape == (2, 6, 4)
    assert counts.shape == (2, 4)
    assert np.all(shifted[:, 1].astype(int) >= coords[:, 1].astype(int))
    np.random.seed(1)
    rc_seq, rc_counts, rc_coords = augment.random_rev_comp(
        seqs.copy(), labels.copy(), coords.copy(), frac=1.0
    )
    assert np.array_equal(rc_seq, seqs[:, ::-1, ::-1])
    assert np.array_equal(rc_counts, labels[:, ::-1])
    assert set(rc_coords[:, 2]) == {"r"}


def test_prediction_hdf5_schema(tmp_path):
    prefix = str(tmp_path / "fixture")
    profile = np.full((2, 4), 0.25)
    logcounts = np.array([1.0, 2.0])
    coords = np.array([["chr1", "10", "f", "1"], ["chr2", "20", "f", "0"]])
    write_predictions_h5py(prefix, profile, logcounts, coords)
    with h5py.File(prefix + "_predictions.h5") as handle:
        assert set(handle) == {"coords", "predictions"}
        assert set(handle["coords"]) == {"coords_center", "coords_chrom", "coords_peak"}
        assert set(handle["predictions"]) == {"logcounts", "profs"}
        np.testing.assert_allclose(handle["predictions/profs"], profile)


def test_contribution_hdf5_layout_arrays():
    seqs = dna_to_one_hot(["ACGT"])
    scores = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    values = generate_shap_dict(seqs, scores)
    assert set(values) == {"raw", "shap", "projected_shap"}
    assert values["raw"]["seq"].shape == (1, 4, 4)
    assert values["raw"]["seq"].dtype == np.int8
    assert values["shap"]["seq"].dtype == np.float16
    np.testing.assert_allclose(
        values["projected_shap"]["seq"],
        np.transpose(seqs * scores, (0, 2, 1)),
    )


def test_bigwig_output(tmp_path):
    path = tmp_path / "predictions.bw"
    regions = [["chr1", 10, 14, 12], ["chr1", 20, 24, 22]]
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
    write_bigwig(data, regions, [("chr1", 40)], str(path))
    with pyBigWig.open(str(path)) as handle:
        np.testing.assert_allclose(handle.values("chr1", 10, 14), data[0])
        np.testing.assert_allclose(handle.values("chr1", 20, 24), data[1])


def test_single_region_debug_prediction_uses_expm1_and_unique_stats(tmp_path):
    model = BiasModel(ModelConfig(128, 20, filters=4, n_dilation_layers=1))
    model_path = tmp_path / "bias.pt"
    save_model(model, model_path)
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\n" + "ACGT" * 100 + "\n")
    sizes = tmp_path / "sizes.tsv"
    sizes.write_text("chr1\t400\n")
    regions = tmp_path / "regions.bed"
    regions.write_text("chr1\t100\t200\t.\t.\t.\t.\t.\t.\t50\n")
    prefix = str(tmp_path / "prediction")
    stats = str(tmp_path / "stats")
    args = SimpleNamespace(
        bias_model=str(model_path), chrombpnet_model=None, chrombpnet_model_nb=None,
        regions=str(regions), genome=str(fasta), chrom_sizes=str(sizes),
        output_prefix=prefix, output_prefix_stats=stats, batch_size=1, tqdm=0,
        debug_chr=["chr1"], bigwig=None, device="cpu",
    )
    predict_bigwig(args)
    assert (tmp_path / "stats_bias.txt").is_file()
    with pyBigWig.open(prefix + "_bias.bw") as handle:
        observed_sum = np.sum(handle.values("chr1", 140, 160))
    sequence = dna_to_one_hot(["GTAC" * 32])
    _, logcounts = PredictionAdapter(model).predict(sequence)
    np.testing.assert_allclose(observed_sum, np.expm1(logcounts[0, 0]), rtol=1e-5)


def test_pred_bw_metrics_survive_an_edge_filtered_region(tmp_path):
    """--bigwig metrics must line up when a region is dropped at a chrom edge.

    get_seq drops any region whose window runs off the chromosome and
    get_regions applies the same mask, so predictions cover only the in-bounds
    regions. compare_with_observed used to receive the *unfiltered* frame and
    read one observed profile per input row, so the observed and predicted
    arrays disagreed in length as soon as any region was edge-filtered -- which
    the fixtures never triggered because none of their regions were.
    """
    import pyBigWig
    from types import SimpleNamespace

    from chrombpnet.evaluation.make_bigwigs.predict_to_bigwig import main
    from chrombpnet.model import BiasModel, ModelConfig, save_model

    chrom_len, inputlen, outputlen = 1000, 200, 100
    (tmp_path / "g.fa").write_text(">chr1\n" + "ACGT" * (chrom_len // 4) + "\n")
    (tmp_path / "sizes.tsv").write_text(f"chr1\t{chrom_len}\n")

    handle = pyBigWig.open(str(tmp_path / "obs.bw"), "w")
    handle.addHeader([("chr1", chrom_len)])
    handle.addEntries("chr1", 0, values=[1.0] * chrom_len, span=1, step=1)
    handle.close()

    # Two regions well inside the chromosome, and one whose inputlen window
    # runs off the right edge so get_seq drops it.
    summit = inputlen // 2
    rows = [(300, summit), (500, summit), (chrom_len - 20, summit)]
    with open(tmp_path / "regions.bed", "w") as bed:
        for start, s_off in rows:
            bed.write("\t".join(str(v) for v in
                                 ("chr1", start, start + 2 * s_off,
                                  ".", ".", ".", ".", ".", ".", s_off)) + "\n")

    model_path = tmp_path / "bias.pt"
    save_model(BiasModel(ModelConfig(inputlen, outputlen, filters=2,
                                     n_dilation_layers=1)), model_path)

    main(SimpleNamespace(
        bias_model=str(model_path), chrombpnet_model=None,
        chrombpnet_model_nb=None, regions=str(tmp_path / "regions.bed"),
        genome=str(tmp_path / "g.fa"), chrom_sizes=str(tmp_path / "sizes.tsv"),
        output_prefix=str(tmp_path / "out"), output_prefix_stats=None,
        batch_size=4, tqdm=0, debug_chr=None,
        bigwig=str(tmp_path / "obs.bw"), device="cpu"))

    # Two regions survive the edge filter, and the metrics are written for them.
    kept = (tmp_path / "out_bias_preds.bed").read_text().splitlines()
    assert len(kept) == 2, kept
    with h5py.File(tmp_path / "out_bias_predictions.h5") as handle:
        assert handle["predictions/profs"].shape == (2, outputlen)
    assert (tmp_path / "out_bias_metrics.json").is_file()


def test_pred_bw_shares_inputs_across_models_without_over_sharing(tmp_path):
    """The three model passes reuse one featurization -- unless geometry differs.

    `main` used to re-parse the BED, re-open the FASTA, re-featurize and
    re-read the observed bigwig once per model. Those are now cached per
    invocation, keyed by what they actually depend on. The risk is a key that
    is too coarse: a bias model with a different input length must not be
    handed the accessibility model's sequences.
    """
    import pyBigWig
    from types import SimpleNamespace

    from chrombpnet.evaluation.make_bigwigs import bigwig_helper
    from chrombpnet.evaluation.make_bigwigs.predict_to_bigwig import main
    from chrombpnet.model import BiasModel, BPNet, ModelConfig, save_model

    chrom_len, outputlen = 2000, 100
    (tmp_path / "g.fa").write_text(">chr1\n" + "ACGT" * (chrom_len // 4) + "\n")
    (tmp_path / "sizes.tsv").write_text(f"chr1\t{chrom_len}\n")
    handle = pyBigWig.open(str(tmp_path / "obs.bw"), "w")
    handle.addHeader([("chr1", chrom_len)])
    handle.addEntries("chr1", 0, values=[1.0] * chrom_len, span=1, step=1)
    handle.close()
    with open(tmp_path / "regions.bed", "w") as bed:
        for start in (600, 900, 1200):
            bed.write("\t".join(str(v) for v in
                                ("chr1", start, start + 200, ".", ".", ".",
                                 ".", ".", ".", 100)) + "\n")

    # Deliberately mismatched input lengths: 400 for the pair, 200 for bias.
    save_model(BPNet(ModelConfig(400, outputlen, filters=2, n_dilation_layers=1)),
               tmp_path / "nobias.pt")
    save_model(BiasModel(ModelConfig(200, outputlen, filters=2, n_dilation_layers=1)),
               tmp_path / "bias.pt")

    widths = []
    real_get_seq = bigwig_helper.get_seq

    def recording_get_seq(peaks_df, genome, width):
        widths.append(width)
        return real_get_seq(peaks_df, genome, width)

    bigwig_helper.get_seq = recording_get_seq
    try:
        main(SimpleNamespace(
            bias_model=str(tmp_path / "bias.pt"), chrombpnet_model=None,
            chrombpnet_model_nb=str(tmp_path / "nobias.pt"),
            regions=str(tmp_path / "regions.bed"),
            genome=str(tmp_path / "g.fa"),
            chrom_sizes=str(tmp_path / "sizes.tsv"),
            output_prefix=str(tmp_path / "out"), output_prefix_stats=None,
            batch_size=4, tqdm=0, debug_chr=None,
            bigwig=str(tmp_path / "obs.bw"), device="cpu"))
    finally:
        bigwig_helper.get_seq = real_get_seq

    # One featurization per distinct input length -- not one per model, and
    # not one shared across both geometries.
    assert sorted(widths) == [200, 400], widths
    for name in ("bias", "chrombpnet_nobias"):
        assert (tmp_path / f"out_{name}.bw").is_file()
        assert (tmp_path / f"out_{name}_metrics.json").is_file()


def test_pred_bw_reuses_one_featurization_for_matching_models(tmp_path):
    """Two models of the same geometry featurize once between them."""
    import pyBigWig
    from types import SimpleNamespace

    from chrombpnet.evaluation.make_bigwigs import bigwig_helper
    from chrombpnet.evaluation.make_bigwigs.predict_to_bigwig import main
    from chrombpnet.model import BiasModel, BPNet, ModelConfig, save_model

    chrom_len, inputlen, outputlen = 2000, 200, 100
    (tmp_path / "g.fa").write_text(">chr1\n" + "ACGT" * (chrom_len // 4) + "\n")
    (tmp_path / "sizes.tsv").write_text(f"chr1\t{chrom_len}\n")
    with open(tmp_path / "regions.bed", "w") as bed:
        for start in (600, 900, 1200):
            bed.write("\t".join(str(v) for v in
                                ("chr1", start, start + 200, ".", ".", ".",
                                 ".", ".", ".", 100)) + "\n")
    save_model(BPNet(ModelConfig(inputlen, outputlen, filters=2, n_dilation_layers=1)),
               tmp_path / "nobias.pt")
    save_model(BiasModel(ModelConfig(inputlen, outputlen, filters=2, n_dilation_layers=1)),
               tmp_path / "bias.pt")

    calls = []
    real_get_seq = bigwig_helper.get_seq
    bigwig_helper.get_seq = lambda df, g, w: (calls.append(w), real_get_seq(df, g, w))[1]
    try:
        main(SimpleNamespace(
            bias_model=str(tmp_path / "bias.pt"), chrombpnet_model=None,
            chrombpnet_model_nb=str(tmp_path / "nobias.pt"),
            regions=str(tmp_path / "regions.bed"),
            genome=str(tmp_path / "g.fa"),
            chrom_sizes=str(tmp_path / "sizes.tsv"),
            output_prefix=str(tmp_path / "out"), output_prefix_stats=None,
            batch_size=4, tqdm=0, debug_chr=None, bigwig=None, device="cpu"))
    finally:
        bigwig_helper.get_seq = real_get_seq

    assert calls == [inputlen], f"featurized {len(calls)} times, expected 1"


def _write_bw(tmp_path, name, data, regions, gs, outstats_file=None):
    from chrombpnet.evaluation.make_bigwigs.bigwig_helper import write_bigwig
    path = str(tmp_path / name)
    write_bigwig(data, regions, gs, path, outstats_file=outstats_file)
    return path


def test_write_bigwig_bytes_do_not_depend_on_the_stats_file(tmp_path):
    """Skipping the signal accumulation must not touch the written bigwig."""
    import hashlib

    rng = np.random.default_rng(0)
    gs = [("chr1", 1_000_000)]
    regions = [["chr1", s, s + 100, s + 50] for s in range(1000, 41000, 400)]
    data = rng.random((len(regions), 100)).astype(np.float32)

    without = _write_bw(tmp_path, "a.bw", data, regions, gs, outstats_file=None)
    with_stats = _write_bw(tmp_path, "b.bw", data, regions, gs,
                           outstats_file=str(tmp_path / "stats.txt"))
    digest = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert digest(without) == digest(with_stats)
    assert (tmp_path / "stats.txt").is_file()


def test_write_bigwig_numpy_and_list_argument_forms_agree(tmp_path):
    """The numpy fast path and the list fallback must produce one file."""
    import hashlib

    from chrombpnet.evaluation.make_bigwigs import bigwig_helper

    rng = np.random.default_rng(1)
    gs = [("chr1", 1_000_000)]
    regions = [["chr1", s, s + 100, s + 50] for s in range(1000, 21000, 400)]
    data = rng.random((len(regions), 100)).astype(np.float32)

    fast = _write_bw(tmp_path, "fast.bw", data, regions, gs)
    original = bigwig_helper._PYBIGWIG_NUMPY
    bigwig_helper._PYBIGWIG_NUMPY = False
    try:
        slow = _write_bw(tmp_path, "slow.bw", data, regions, gs)
    finally:
        bigwig_helper._PYBIGWIG_NUMPY = original

    digest = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert digest(fast) == digest(slow)


def test_write_bigwig_still_raises_when_every_region_is_filtered_out(tmp_path):
    """The raise counts emitted regions, not written bases.

    It used to test `if not all_entries`, which a region contributing an empty
    slice would satisfy anyway; the replacement flag must keep that shape and
    must fire whether or not stats were requested.
    """
    from chrombpnet.evaluation.make_bigwigs.bigwig_helper import write_bigwig

    gs = [("chr1", 1_000_000)]
    regions = [["chr1", 1000, 1100, 1050]]
    data = np.zeros((1, 100), dtype=np.float32)
    for outstats in (None, str(tmp_path / "s.txt")):
        with pytest.raises(ValueError, match="No BigWig entries were written"):
            write_bigwig(data, regions, gs, str(tmp_path / "x.bw"),
                         debug_chr=["chrZ"], outstats_file=outstats)
