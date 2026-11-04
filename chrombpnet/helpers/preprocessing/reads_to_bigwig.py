import argparse
import subprocess
import sys
import tempfile
import os
import chrombpnet.helpers.preprocessing.auto_shift_detect as auto_shift_detect
from chrombpnet import conventions
from chrombpnet.data import DefaultDataFile, get_default_data_path
from chrombpnet.seed import DEFAULT_SEED

def parse_args():
    parser=argparse.ArgumentParser(description="Convert input BAM/fragment/tagAlign file to appropriately shifted unstranded Bigwig")
    parser.add_argument('-g','--genome', required=True, type=str, help="reference genome fasta file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-ibam', '--input-bam-file', type=str, help="Input BAM file")
    group.add_argument('-ifrag', '--input-fragment-file', type=str, help="Input fragment file")
    group.add_argument('-itag', '--input-tagalign-file', type=str, help="Input tagAlign file")
    parser.add_argument('-c', '--chrom-sizes', type=str, required=True, help="Chrom sizes file")
    parser.add_argument('-op', '--output-prefix', type=str, required=True, help="Output prefix (path/to/prefix)")
    parser.add_argument('-d', '--data-type', required=True, type=str, choices=['ATAC', 'DNASE'], help="assay type")
    parser.add_argument('--bsort', required=False, default=False, action='store_true', help="use bedtools sort (default is unix sort)")
    parser.add_argument('--no-st', required=False,  default=False, action='store_true', help="No streaming in preprocessing")
    parser.add_argument('--tmpdir', required=False,  type=str, default=None, help="tmp dir path for unix sort command")
    parser.add_argument('-ps', '--plus-shift', type=int, default=None, help="Plus strand shift applied to reads. Estimated if not specified")
    parser.add_argument('-ms', '--minus-shift', type=int, default=None, help="Minus strand shift applied to reads. Estimated if not specified")
    parser.add_argument('--ATAC-ref-path', type=str, default=None, help="Path to ATAC reference motifs (chrombpnet/data/ATAC.ref.motifs.txt used by default)")
    parser.add_argument('--DNASE-ref-path', type=str, default=None, help="Path to DNASE reference motifs (chrombpnet/data/DNASE.ref.motifs.txt used by default)")
    parser.add_argument('--num-samples', type=int, default=10000, help="Number of reads to sample from BAM/fragment/tagAlign file for shift estimation")
    parser.add_argument('--dnase-convention', type=str, default=None,
                        choices=list(conventions.DNASE_CONVENTION_CHOICES),
                        help="DNASE only: where a cut is recorded; see chrombpnet.conventions. "
                             "Default: both-half. right-deprecated is the convention before "
                             "this flag, kept for reproducibility only.")
    args = parser.parse_args()

    return args


def _awk_program(n_deposits):
    """awk emitting one record per deposit, each guarded against bedtools.

    ``bedtools genomecov -bg -5`` reports one position per record, so a
    convention that puts a cut on both bases flanking the bond needs two of
    them -- which is the only reason this is generated rather than a literal.

    Both guards exist because bedtools **aborts the whole run** on a record
    it cannot parse, and a shifted convention can produce one. Measured on
    v2.31.1: ``start < 0`` and ``end < start`` each exit 1 and kill the
    pipeline; ``end == start`` is accepted. Hence ``>=`` on both, not ``>``
    -- a strict comparison would discard zero-width records that bedtools
    handles today.

    `left` makes this reachable rather than theoretical: its plus delta is
    -1, so any read aligned at coordinate 0 goes negative, and 171 of the 456
    contigs in a standard hg38 chrom.sizes start with a real base -- chrM
    among them. Guarding can only suppress records that would otherwise abort
    the run, so no run that previously succeeded produces different output.
    """
    plus = " ".join(f'if ($2+plus{i} >= 0) print $1,$2+plus{i},$3,$4,$5,$6;'
                    for i in range(n_deposits))
    minus = " ".join(f'if ($3+minus{i} >= $2) print $1,$2,$3+minus{i},$4,$5,$6;'
                     for i in range(n_deposits))
    return ('BEGIN {OFS="\\t"} '
            '{if ($6=="+") {' + plus + '} '
            'else if ($6=="-") {' + minus + '}}')


def _stop_pipeline(processes):
    """End a pipeline that is being abandoned on an error.

    Without this, a failure while streaming reads into awk -- the input check
    raising on a truncated gzip, say -- left awk's stdin open, so awk and the
    stages after it waited on a pipe that would never close, unreaped, and the
    open pipe surfaced as an unclosed-file ResourceWarning at exit.

    Each stage is killed before its pipes are closed: closing a stdin flushes
    it, which could block on a stage that is still running but no longer
    reading. Closing a pipe to a killed stage raises BrokenPipeError after
    closing it, which is ignored. The caller re-raises the original error.
    """
    processes = [process for process in processes if process is not None]
    for process in processes:
        if process.poll() is None:
            process.kill()
    for process in processes:
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass
    for process in processes:
        process.wait()


def generate_bigwig(input_bam_file, input_fragment_file, input_tagalign_file, output_prefix, genome_fasta_file, bsort, tmpdir, no_st, chrom_sizes_file, delta_pairs, scale=1.0):
    if sum(value is not None for value in (input_bam_file, input_fragment_file, input_tagalign_file)) != 1:
        raise ValueError("Exactly one input file must be provided")

    if input_bam_file:
        p1 = auto_shift_detect.bam_to_tagalign_stream(input_bam_file)
    elif input_fragment_file:
        p1 = auto_shift_detect.fragment_to_tagalign_stream(input_fragment_file)
    elif input_tagalign_file:
        p1 = auto_shift_detect.tagalign_stream(input_tagalign_file)

    if tmpdir is not None and not os.path.isdir(tmpdir):
        raise ValueError(f"Temporary directory does not exist: {tmpdir}")
    sort_args = ["sort"] + (["-T", tmpdir] if tmpdir else []) + ["-k1,1"]
    final_sort = (["bedtools", "sort", "-i", "stdin"] if bsort else
                  ["sort"] + (["-T", tmpdir] if tmpdir else []) + ["-k1,1", "-k2,2n"])
    awk_program = _awk_program(len(delta_pairs))
    awk_cmd = ["awk"]
    for index, (plus_delta, minus_delta) in enumerate(delta_pairs):
        awk_cmd += ["-v", f"plus{index}={plus_delta}",
                    "-v", f"minus{index}={minus_delta}"]
    awk_cmd.append(awk_program)

    tmp_bedgraph = tempfile.NamedTemporaryFile()
    # Every process of the pipeline, so a failure can stop them all.
    pipeline = [p1, getattr(p1, "_chrombpnet_upstream", None)]
    try:
        with open(tmp_bedgraph.name, "w") as output:
            p_awk = subprocess.Popen(awk_cmd, stdin=p1.stdout if no_st else subprocess.PIPE,
                                     stdout=subprocess.PIPE)
            pipeline.append(p_awk)
            if no_st:
                p1.stdout.close()
            p_sort = subprocess.Popen(sort_args, stdin=p_awk.stdout, stdout=subprocess.PIPE)
            pipeline.append(p_sort)
            p_awk.stdout.close()
            genomecov_cmd = ["bedtools", "genomecov", "-bg", "-5", "-i", "stdin",
                             "-g", chrom_sizes_file]
            if scale != 1.0:
                # Half a count on each of the two bases flanking a cut, so the
                # track total matches the single-deposit conventions exactly.
                # bigWig values are float32 and 0.5 is binary-exact, so this is
                # lossless, and genomecov still omits zero intervals under -scale.
                genomecov_cmd += ["-scale", repr(float(scale))]
            p_cov = subprocess.Popen(genomecov_cmd, stdin=p_sort.stdout,
                                     stdout=subprocess.PIPE)
            pipeline.append(p_cov)
            p_sort.stdout.close()
            env = os.environ.copy()
            env["LC_COLLATE"] = "C"
            p_final = subprocess.Popen(final_sort, stdin=p_cov.stdout, stdout=output, env=env)
            pipeline.append(p_final)
            p_cov.stdout.close()
            if not no_st:
                auto_shift_detect.stream_filtered_tagaligns(p1, genome_fasta_file, p_awk.stdin.write)
                p_awk.stdin.close()
            return_codes = [p_final.wait(), p_cov.wait(), p_sort.wait(), p_awk.wait()]
            if any(code != 0 for code in return_codes):
                raise subprocess.CalledProcessError(1, "BedGraph generation pipeline")
            if no_st:
                auto_shift_detect.check_input_streams(p1)
    except BaseException:
        _stop_pipeline(pipeline)
        raise

    if os.path.getsize(tmp_bedgraph.name) == 0:
        source = input_bam_file or input_fragment_file or input_tagalign_file
        raise RuntimeError(
            f"The bedGraph built from {source} is empty: no reads were left after "
            f"filtering them to the reference genome ({genome_fasta_file}) and the chrom "
            f"sizes file ({chrom_sizes_file}). Check that the input has reads and that "
            f"its chromosome names match the reference.")

    print("Making Bigwig")
    try:
        subprocess.run(["bedGraphToBigWig", tmp_bedgraph.name, chrom_sizes_file,
                        output_prefix + "_unstranded.bw"], check=True)
    except subprocess.CalledProcessError:
        # bedtools genomecov only warns about reads on contigs the chrom sizes
        # file lacks; bedGraphToBigWig is where such a run fails.
        print(f"bedGraphToBigWig failed: every chromosome or contig with reads must be "
              f"listed in the chrom sizes file {chrom_sizes_file} (use the full chrom sizes "
              f"of the reference the reads were aligned to).", file=sys.stderr)
        raise

    tmp_bedgraph.close()

def main(args):

    plus_shift, minus_shift = args.plus_shift, args.minus_shift

    if (plus_shift is None) or (minus_shift is None):
        # TODO: validate inputs, check bedtools and ucsc tools
        if args.data_type=="ATAC":
            ref_motifs_file = args.ATAC_ref_path
            if ref_motifs_file is None:
                ref_motifs_file=get_default_data_path(DefaultDataFile.atac_ref_motifs)
        elif args.data_type=="DNASE":
            ref_motifs_file = args.DNASE_ref_path
            if ref_motifs_file is None:
                ref_motifs_file =  get_default_data_path(DefaultDataFile.dnase_ref_motifs)
    
        # The read sample is seeded by --seed where the command has one.
        seed = getattr(args, "seed", None)
        if seed is None:
            seed = DEFAULT_SEED
        print("Estimating enzyme shift in input file")
        plus_shift, minus_shift = auto_shift_detect.compute_shift(args.input_bam_file,
                args.input_fragment_file,
                args.input_tagalign_file,
                args.num_samples,
                args.genome,
                args.data_type,
                ref_motifs_file,
                seed)
    
        print("Current estimated shift: {:+}/{:+}".format(plus_shift, minus_shift))

    else:
        print("The specified shift is: {:+}/{:+}".format(plus_shift, minus_shift))

    # Where each cut is recorded, and on how many bases.
    convention = conventions.from_args(args)
    delta_pairs = conventions.shift_delta_pairs(convention, plus_shift, minus_shift)

    generate_bigwig(args.input_bam_file,
            args.input_fragment_file,
            args.input_tagalign_file,
            args.output_prefix,
            args.genome,
            args.bsort,
            args.tmpdir,
            args.no_st,
            args.chrom_sizes,
            delta_pairs,
            convention.scale)

if __name__=="__main__":
    args = parse_args()
    main(args)
