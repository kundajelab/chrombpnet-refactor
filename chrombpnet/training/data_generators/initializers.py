import chrombpnet.training.data_generators.batchgen_generator as batchgen_generator
from chrombpnet import conventions
from chrombpnet.training.utils.benchmark_utils import benchmark_enables
import pandas as pd
import json

from chrombpnet.regions import NARROWPEAK_SCHEMA, is_disabled

def fetch_data_and_model_params_based_on_mode(mode, args, parameters, nonpeak_regions, peak_regions, convention):

    if mode=="train": 
        inputlen=int(parameters["inputlen"])
        outputlen=int(parameters["outputlen"])
        negative_sampling_ratio=float(parameters["negative_sampling_ratio"])
        max_jitter=int(parameters["max_jitter"])
        add_revcomp=convention.reverse_complements(mode)
        shuffle_at_epoch_start=not benchmark_enables(args, "fixed_example_order")
        

    elif mode=="valid":
        inputlen=int(parameters["inputlen"])
        outputlen=int(parameters["outputlen"])
        # fix negatives set for validation
        if (nonpeak_regions is not None) and (peak_regions is not None):
            requested = int(float(parameters["negative_sampling_ratio"]) * peak_regions.shape[0])
            nonpeak_regions=nonpeak_regions.sample(
                n=min(requested, len(nonpeak_regions)), replace=False, random_state=args.seed
            )
        negative_sampling_ratio=1.0 # already subsampled
        # do not jitter at valid time - we are testing only at summits
        max_jitter=0
        # normally no reverse complementation at valid time; --data-aug
        # revcomp-only deliberately flips every split, which is the whole point of it
        add_revcomp=convention.reverse_complements(mode)
        # no need to shuffle
        shuffle_at_epoch_start=False

    elif mode=="test":
        # read input/output length
        inputlen=args.inputlen
        outputlen=args.outputlen
        # no subsampling of negatives - test on all positives and negatives
        negative_sampling_ratio=1.0
        # no jitter at valid time - we are testing only at summits
        max_jitter=0
        # normally no reverse complementation at test time; see "valid" above
        add_revcomp=convention.reverse_complements(mode)    
        # no need to shuffle
        shuffle_at_epoch_start=False
        
    else:
        raise ValueError(
            f"mode must be train, valid or test, got {mode!r}")

    return inputlen, outputlen,  nonpeak_regions, negative_sampling_ratio, max_jitter, add_revcomp, shuffle_at_epoch_start


def get_bed_regions_for_fold_split(bed_regions, mode, splits_dict):
    chroms_to_keep=splits_dict[mode]
    bed_regions_to_keep=bed_regions[bed_regions["chr"].isin(chroms_to_keep)]
    print("got split:"+str(mode)+" for bed regions:"+str(bed_regions_to_keep.shape))
    if bed_regions_to_keep.empty:
        raise ValueError(f"No BED regions are available for the {mode} chromosome split")
    return bed_regions_to_keep, chroms_to_keep

def initialize_generators(args, mode, parameters, return_coords):

    # Resolved once, here, so preprocessing and training cannot disagree about
    # what the flags mean, and so a flag undefined for the assay raises before
    # any data is read rather than quietly doing nothing.
    convention = conventions.from_args(args)

    # defaults
    peak_regions=None
    nonpeak_regions=None

    # get only those peak/non peak regions corresponding to train/valid/test set
    splits_dict=json.load(open(args.chr_fold_path))

    if not is_disabled(args.peaks):
        print("loading peaks...")
        peak_regions=pd.read_csv(args.peaks,header=None,sep='\t',names=NARROWPEAK_SCHEMA)
        peak_regions, chroms=get_bed_regions_for_fold_split(peak_regions, mode, splits_dict)

    if not is_disabled(args.nonpeaks):
        print("loading nonpeaks...")
        nonpeak_regions=pd.read_csv(args.nonpeaks,header=None,sep='\t',names=NARROWPEAK_SCHEMA)
        nonpeak_regions, chroms=get_bed_regions_for_fold_split(nonpeak_regions, mode, splits_dict) 

    inputlen, outputlen, \
    nonpeak_regions, negative_sampling_ratio, \
    max_jitter, add_revcomp, shuffle_at_epoch_start  =  fetch_data_and_model_params_based_on_mode(mode, args, parameters, nonpeak_regions, peak_regions, convention)
    generator=batchgen_generator.ChromBPNetBatchGenerator(
                                    peak_regions=peak_regions,
                                    nonpeak_regions=nonpeak_regions,
                                    genome_fasta=args.genome,
                                    batch_size=args.batch_size,
                                    inputlen=inputlen,                                        
                                    outputlen=outputlen,
                                    max_jitter=max_jitter,
                                    negative_sampling_ratio=negative_sampling_ratio,
                                    cts_bw_file=args.bigwig,
                                    add_revcomp=add_revcomp,
                                    return_coords=return_coords,
                                    shuffle_at_epoch_start=shuffle_at_epoch_start,
                                    mode=mode,
                                    benchmark=getattr(args, "benchmark", None),
                                    benchmark_seed=getattr(args, "seed", None),
                                    rc_shift=convention.rc_shift,
                                    rc_frac=convention.rc_frac
                                    )

    return generator
