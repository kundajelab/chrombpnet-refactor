"""Input streams of the preprocessing: gzip handling and subprocess failures.

Ported from kundajelab/chrombpnet#284 tests/test_preprocessing.py, adapted to
this package's functions (check_stream / check_input_streams instead of the
PR's ReadStream, and generate_bigwig's shift pairs).
"""

import gzip
import shutil
import subprocess
import warnings

import numpy as np
import pytest

from chrombpnet.helpers.preprocessing import auto_shift_detect, reads_to_bigwig

CHROM_LEN = 5000


def needs_tools(*tools):
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        pytest.skip("missing tools: " + ", ".join(missing))


@pytest.fixture
def genome(tmp_path):
    rng = np.random.RandomState(0)
    fasta, sizes = tmp_path / "genome.fa", tmp_path / "genome.chrom.sizes"
    with open(fasta, "w") as handle:
        for chrom in ("chr1", "chr2"):
            seq = "".join(rng.choice(list("ACGT"), CHROM_LEN))
            handle.write(">{}\n".format(chrom))
            for i in range(0, CHROM_LEN, 60):
                handle.write(seq[i:i + 60] + "\n")
    sizes.write_text("chr1\t{0}\nchr2\t{0}\n".format(CHROM_LEN))
    return str(fasta), str(sizes)


def random_reads(n, seed=1):
    rng = np.random.RandomState(seed)
    reads = []
    for i in range(n):
        chrom = "chr1" if i % 2 else "chr2"
        start = int(rng.randint(100, CHROM_LEN - 200))
        reads.append((chrom, start, start + 50, "+" if rng.rand() < 0.5 else "-"))
    return reads


def write_input(tmp_path, kind, n, seed=3):
    reads = random_reads(n, seed)
    if kind == "fragment":
        path = tmp_path / "reads.tsv"
        path.write_text("".join("{}\t{}\t{}\tBC{}\t1\n".format(c, s, e, i % 7)
                                for i, (c, s, e, _) in enumerate(reads)))
    else:
        path = tmp_path / "reads.tagAlign"
        path.write_text("".join("{}\t{}\t{}\tN\t1000\t{}\n".format(c, s, e, strand)
                                for c, s, e, strand in reads))
    return str(path)


def gzip_file(path):
    with open(path, "rb") as source, gzip.open(path + ".gz", "wb") as target:
        shutil.copyfileobj(source, target)
    return path + ".gz"


def with_trailing_garbage(path):
    with open(path, "ab") as handle:
        handle.write(b"\0" * 64 + b"not gzip")
    return path


def truncated_copy(path, fraction=0.5):
    data = open(path, "rb").read()
    out = path + ".truncated.gz"
    with open(out, "wb") as handle:
        handle.write(data[:int(len(data) * fraction)])
    return out


def read_stream(stream_fn, path):
    process = stream_fn(path)
    out = process.stdout.read()
    process.stdout.close()
    auto_shift_detect.check_input_streams(process)
    return out


STREAMS = {"fragment": auto_shift_detect.fragment_to_tagalign_stream,
           "tagalign": auto_shift_detect.tagalign_stream}


def test_check_stream():
    auto_shift_detect.check_stream(subprocess.Popen(["true"]))
    with pytest.raises(subprocess.CalledProcessError):
        auto_shift_detect.check_stream(subprocess.Popen(["false"]))


@pytest.mark.parametrize("kind", ["fragment", "tagalign"])
def test_gzipped_and_multi_member_inputs_stream_the_same_bytes(tmp_path, kind):
    needs_tools("awk", "gzip")
    plain = write_input(tmp_path, kind, 500)
    expected = read_stream(STREAMS[kind], plain)
    assert read_stream(STREAMS[kind], gzip_file(plain)) == expected
    data = open(plain, "rb").read()
    multi = plain + ".multi.gz"  # bgzip-style: several gzip members
    with open(multi, "wb") as handle:
        half = len(data) // 2
        handle.write(gzip.compress(data[:half]) + gzip.compress(data[half:]))
    assert read_stream(STREAMS[kind], multi) == expected


@pytest.mark.parametrize("kind", ["fragment", "tagalign"])
def test_gzip_trailing_garbage_is_only_a_warning(tmp_path, kind):
    """gzip exits 2 after writing the whole stream; chrombpnet 1.x accepted these."""
    needs_tools("awk", "gzip")
    plain = write_input(tmp_path, kind, 50)
    garbage = with_trailing_garbage(gzip_file(plain))
    with pytest.warns(RuntimeWarning, match="trailing data"):
        streamed = read_stream(STREAMS[kind], garbage)
    assert streamed == read_stream(STREAMS[kind], plain)


@pytest.mark.parametrize("kind", ["fragment", "tagalign"])
def test_truncated_gzip_input_raises_naming_gzip(tmp_path, genome, kind):
    needs_tools("awk", "gzip")
    fasta, _ = genome
    truncated = truncated_copy(gzip_file(write_input(tmp_path, kind, 3000)))
    with pytest.raises(subprocess.CalledProcessError, match="gzip"):
        read_stream(STREAMS[kind], truncated)
    inputs = {"fragment": (None, truncated, None), "tagalign": (None, None, truncated)}[kind]
    with pytest.raises(subprocess.CalledProcessError):
        auto_shift_detect.sample_reads(*inputs, 50, fasta)


def bigwig_inputs(kind, path):
    return {"fragment": (None, path, None), "tagalign": (None, None, path)}[kind]


@pytest.mark.parametrize("no_st", [False, True])
@pytest.mark.parametrize("kind", ["fragment", "tagalign"])
def test_bigwig_from_input_with_trailing_garbage_matches_clean_input(tmp_path, genome, kind, no_st):
    needs_tools("bedtools", "bedGraphToBigWig", "awk", "sort", "gzip")
    fasta, sizes = genome
    plain = write_input(tmp_path, kind, 300)
    outputs = []
    for name, path in (("plain", plain), ("garbage", with_trailing_garbage(gzip_file(plain)))):
        prefix = str(tmp_path / name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reads_to_bigwig.generate_bigwig(*bigwig_inputs(kind, path), prefix, fasta, False, None,
                                            no_st, sizes, [(4, -4)])
        outputs.append(prefix + "_unstranded.bw")
    assert open(outputs[0], "rb").read() == open(outputs[1], "rb").read()


@pytest.mark.parametrize("no_st", [False, True])
@pytest.mark.parametrize("kind", ["fragment", "tagalign"])
def test_bigwig_from_truncated_gzip_raises(tmp_path, genome, kind, no_st):
    needs_tools("bedtools", "bedGraphToBigWig", "awk", "sort", "gzip")
    fasta, sizes = genome
    truncated = truncated_copy(gzip_file(write_input(tmp_path, kind, 3000)))
    with pytest.raises(subprocess.CalledProcessError, match="gzip"):
        reads_to_bigwig.generate_bigwig(*bigwig_inputs(kind, truncated), str(tmp_path / "out"), fasta,
                                        False, None, no_st, sizes, [(4, -4)])
    assert not (tmp_path / "out_unstranded.bw").exists()


@pytest.mark.parametrize("no_st", [False, True])
@pytest.mark.parametrize("kind", ["fragment", "tagalign"])
def test_a_failed_bigwig_pipeline_leaves_no_process_or_pipe_behind(tmp_path, genome, kind, no_st,
                                                                   monkeypatch):
    """Streaming into awk, the input check raising on a truncated gzip used to
    leave awk's stdin open: awk and the stages after it waited on it forever,
    unreaped, and the pipe surfaced as an unclosed-file warning at exit."""
    needs_tools("bedtools", "bedGraphToBigWig", "awk", "sort", "gzip")
    fasta, sizes = genome
    started = []

    class Recorded(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            started.append(self)

    # The one subprocess module, so this records auto_shift_detect's input
    # processes too.
    monkeypatch.setattr(reads_to_bigwig.subprocess, "Popen", Recorded)
    truncated = truncated_copy(gzip_file(write_input(tmp_path, kind, 3000)))
    with pytest.raises(subprocess.CalledProcessError, match="gzip"):
        reads_to_bigwig.generate_bigwig(*bigwig_inputs(kind, truncated), str(tmp_path / "out"), fasta,
                                        False, None, no_st, sizes, [(4, -4)])
    assert len(started) >= 5
    for process in started:
        assert process.returncode is not None, f"{process.args} was never reaped"
        for stream in (process.stdin, process.stdout):
            assert stream is None or stream.closed, f"a pipe of {process.args} is still open"


def test_uncompressed_fragments_stream_like_the_cat_awk_pipeline(tmp_path):
    """The file is awk's stdin now, instead of `cat -- file | awk`."""
    needs_tools("awk", "sh")
    fragments = tmp_path / "odd.tsv"
    fragments.write_text("chr1\t10\t60\tBC1\t3\nchr2\t5\t105\nchr1  7   77  x\nchr1\t8\n\nchr2\t1\t2\tBC\t1\textra\n")
    legacy = subprocess.run(
        "cat -- " + str(fragments) + """ | awk -v OFS="\\t" '{print $1,$2,$3,1000,0,"+"; """
        """print $1,$2,$3,1000,0,"-"}'""", shell=True, check=True, stdout=subprocess.PIPE).stdout
    assert read_stream(auto_shift_detect.fragment_to_tagalign_stream, str(fragments)) == legacy
    assert read_stream(auto_shift_detect.fragment_to_tagalign_stream, gzip_file(str(fragments))) == legacy


@pytest.mark.parametrize("no_st", [False, True])
def test_contig_missing_from_chrom_sizes_fails_with_a_hint(tmp_path, genome, no_st, capsys):
    """Reads on chr2 with a chrom sizes file lacking it: genomecov only warns,
    so bedGraphToBigWig fails the run, with a hint, and writes no bigwig."""
    needs_tools("bedtools", "bedGraphToBigWig", "awk", "sort")
    fasta, _ = genome
    tagalign = write_input(tmp_path, "tagalign", 200)
    only_chr1 = tmp_path / "chr1.chrom.sizes"
    only_chr1.write_text("chr1\t{}\n".format(CHROM_LEN))
    with pytest.raises(subprocess.CalledProcessError):
        reads_to_bigwig.generate_bigwig(None, None, tagalign, str(tmp_path / "out"), fasta, False,
                                        None, no_st, str(only_chr1), [(4, -4)])
    assert "must be listed in the chrom sizes file" in capsys.readouterr().err
    assert not (tmp_path / "out_unstranded.bw").exists()


def test_no_reads_left_is_a_clear_error(tmp_path, genome):
    needs_tools("bedtools", "awk", "sort")
    fasta, sizes = genome
    elsewhere = tmp_path / "elsewhere.tagAlign"
    elsewhere.write_text("chrUn\t10\t60\tN\t1000\t+\n")
    with pytest.raises(RuntimeError, match="bedGraph built from .* is empty"):
        reads_to_bigwig.generate_bigwig(None, None, str(elsewhere), str(tmp_path / "out"), fasta,
                                        False, None, False, sizes, [(4, -4)])


def test_reservoir_sample_is_seeded_and_uniform():
    lines = [b"line%d\n" % i for i in range(200)]

    def sample(seed, size=20, stream=lines):
        reservoir = auto_shift_detect.ReservoirSample(size, seed)
        for line in stream:
            reservoir.write(line)
        return reservoir.lines

    # A stream no longer than the sample is kept whole, in order.
    assert sample(0, size=200) == lines and sample(0, size=500) == lines
    first = sample(7)
    assert len(first) == len(set(first)) == 20 and set(first) <= set(lines)
    assert sample(7) == first and sample(8) != first

    # Every line is equally likely to be kept: 2000 samples of 20 of 200
    # lines keep each line 200 times on average (sd ~13).
    counts = np.zeros(len(lines))
    for seed in range(2000):
        for line in sample(seed):
            counts[int(line[4:-1])] += 1
    assert counts.sum() == 2000 * 20
    assert counts.min() > 140 and counts.max() < 260
    assert abs(counts[:100].sum() - counts[100:].sum()) < 0.03 * counts.sum()


def test_the_shift_read_sample_is_reproducible_above_its_size(tmp_path, genome):
    """The original sampled with an unseeded `shuf`, so above 2 * num_samples
    reads two runs estimated the shift from different reads."""
    fasta, _ = genome
    reads = write_input(tmp_path, "tagalign", 3000)

    def sample(seed):
        plus, minus = auto_shift_detect.sample_reads(None, None, str(reads), 50, fasta, seed)
        return plus.values.tolist(), minus.values.tolist()

    first = sample(7)
    assert len(first[0]) + len(first[1]) == 100
    assert sample(7) == first and sample(8) != first
