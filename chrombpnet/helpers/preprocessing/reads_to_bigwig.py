import argparse
import pyBigWig
import pyfaidx
import subprocess
import tempfile
import os
import numpy as np
import chrombpnet.helpers.preprocessing.auto_shift_detect as auto_shift_detect
from chrombpnet.data import DefaultDataFile, get_default_data_path

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
    args = parser.parse_args()

    return args


def generate_bigwig(input_bam_file, input_fragment_file, input_tagalign_file, output_prefix, genome_fasta_file, bsort, tmpdir, no_st, chrom_sizes_file, plus_shift_delta, minus_shift_delta):
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
    awk_program = (
        'BEGIN {OFS="\\t"} '
        '{if ($6=="+") print $1,$2+plus,$3,$4,$5,$6; '
        'else if ($6=="-") print $1,$2,$3+minus,$4,$5,$6}'
    )
    awk_cmd = ["awk", "-v", f"plus={plus_shift_delta}", "-v",
               f"minus={minus_shift_delta}", awk_program]

    tmp_bedgraph = tempfile.NamedTemporaryFile()
    with open(tmp_bedgraph.name, "w") as output:
        p_awk = subprocess.Popen(awk_cmd, stdin=p1.stdout if no_st else subprocess.PIPE,
                                 stdout=subprocess.PIPE)
        if no_st:
            p1.stdout.close()
        p_sort = subprocess.Popen(sort_args, stdin=p_awk.stdout, stdout=subprocess.PIPE)
        p_awk.stdout.close()
        p_cov = subprocess.Popen(["bedtools", "genomecov", "-bg", "-5", "-i", "stdin",
                                  "-g", chrom_sizes_file], stdin=p_sort.stdout,
                                 stdout=subprocess.PIPE)
        p_sort.stdout.close()
        env = os.environ.copy()
        env["LC_COLLATE"] = "C"
        p_final = subprocess.Popen(final_sort, stdin=p_cov.stdout, stdout=output, env=env)
        p_cov.stdout.close()
        if not no_st:
            auto_shift_detect.stream_filtered_tagaligns(p1, genome_fasta_file, p_awk)
            p_awk.stdin.close()
        return_codes = [p_final.wait(), p_cov.wait(), p_sort.wait(), p_awk.wait()]
        if any(code != 0 for code in return_codes):
            raise subprocess.CalledProcessError(1, "BedGraph generation pipeline")
        if no_st and p1.wait() != 0:
            raise subprocess.CalledProcessError(p1.returncode, "input conversion")
        upstream = getattr(p1, "_chrombpnet_upstream", None)
        if no_st and upstream is not None and upstream.wait() != 0:
            raise subprocess.CalledProcessError(upstream.returncode, "input decompression")

    print("Making Bigwig")
    subprocess.run(["bedGraphToBigWig", tmp_bedgraph.name, chrom_sizes_file,
                    output_prefix + "_unstranded.bw"], check=True)

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
    
        print("Estimating enzyme shift in input file")
        plus_shift, minus_shift = auto_shift_detect.compute_shift(args.input_bam_file,
                args.input_fragment_file,
                args.input_tagalign_file,
                args.num_samples,
                args.genome,
                args.data_type,
                ref_motifs_file)
    
        print("Current estimated shift: {:+}/{:+}".format(plus_shift, minus_shift))

    else:
        print("The specified shift is: {:+}/{:+}".format(plus_shift, minus_shift))

    # computing additional shifting to apply
    if args.data_type=="ATAC":
        plus_shift_delta, minus_shift_delta = 4-plus_shift, -4-minus_shift
    elif args.data_type=="DNASE":
        plus_shift_delta, minus_shift_delta = -plus_shift, 1-minus_shift

    generate_bigwig(args.input_bam_file,
            args.input_fragment_file,
            args.input_tagalign_file,
            args.output_prefix,
            args.genome,
            args.bsort,
            args.tmpdir,
            args.no_st,
            args.chrom_sizes,
            plus_shift_delta,
            minus_shift_delta)

if __name__=="__main__":
    args = parse_args()
    main(args)
