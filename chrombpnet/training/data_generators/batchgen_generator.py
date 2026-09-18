from chrombpnet.training.utils import augment
from chrombpnet.training.utils import data_utils
from chrombpnet.training.utils.benchmark_utils import (
    benchmark_enables,
    make_generator_rng,
    normalize_benchmark_level,
)
import numpy as np
import math

def subsample_nonpeak_data(nonpeak_seqs, nonpeak_cts, nonpeak_coords, peak_data_size, negative_sampling_ratio, rng=None):
    #Randomly samples a portion of the non-peak data to use in training
    if negative_sampling_ratio < 0:
        raise ValueError("negative_sampling_ratio must be non-negative")
    num_nonpeak_samples = min(
        int(negative_sampling_ratio * peak_data_size), len(nonpeak_seqs)
    )
    stream = np.random if rng is None else rng
    nonpeak_indices_to_keep = stream.choice(len(nonpeak_seqs), size=num_nonpeak_samples, replace=False)
    nonpeak_seqs = nonpeak_seqs[nonpeak_indices_to_keep]
    nonpeak_cts = nonpeak_cts[nonpeak_indices_to_keep]
    nonpeak_coords = nonpeak_coords[nonpeak_indices_to_keep]
    return nonpeak_seqs, nonpeak_cts, nonpeak_coords

class ChromBPNetBatchGenerator:
    """
    This generator randomly crops (=jitter) and revcomps training examples for 
    every epoch, and calls bias model on it, whose outputs (bias profile logits 
    and bias logcounts) are fed as input to the chrombpnet model.
    """
    def __init__(self, peak_regions, nonpeak_regions, genome_fasta, batch_size, inputlen, outputlen, max_jitter, negative_sampling_ratio, cts_bw_file, add_revcomp, return_coords, shuffle_at_epoch_start, mode=None, benchmark=None, benchmark_seed=None):
        """
        seqs: B x L' x 4
        cts: B x M'
        inputlen: int (L <= L'), L' is greater to allow for cropping (= jittering)
        outputlen: int (M <= M'), M' is greater to allow for cropping (= jittering)
        batch_size: int (B)
        mode/benchmark/benchmark_seed: benchmark bookkeeping. From the medium
        level onwards all augmentation is drawn from a dedicated NumPy stream
        derived from (benchmark_seed, mode, epoch) so both frameworks realize
        the same schedule regardless of what else consumes the global RNG.
        """

        peak_seqs, peak_cts, peak_coords, nonpeak_seqs, nonpeak_cts, nonpeak_coords, = data_utils.load_data(peak_regions, nonpeak_regions, genome_fasta, cts_bw_file, inputlen, outputlen, max_jitter)
        self.peak_seqs, self.nonpeak_seqs = peak_seqs, nonpeak_seqs
        self.peak_cts, self.nonpeak_cts = peak_cts, nonpeak_cts
        self.peak_coords, self.nonpeak_coords = peak_coords, nonpeak_coords

        self.negative_sampling_ratio = negative_sampling_ratio
        self.inputlen = inputlen
        self.outputlen = outputlen
        self.batch_size = batch_size
        self.add_revcomp = add_revcomp
        self.return_coords = return_coords
        self.shuffle_at_epoch_start = shuffle_at_epoch_start

        self.mode = mode
        self.benchmark = normalize_benchmark_level(benchmark)
        self.benchmark_seed = benchmark_seed
        self.epoch = 0
        self.rng = self._make_rng()

        # random crop training data to the desired sizes, revcomp augmentation
        self.crop_revcomp_data()

    def _make_rng(self):
        return make_generator_rng(self.benchmark, self.benchmark_seed, self.mode, self.epoch)

    def __len__(self):

        return math.ceil(self.seqs.shape[0]/self.batch_size)


    def crop_revcomp_data(self):
        # random crop training data to inputlen and outputlen (with corresponding offsets), revcomp augmentation
        # shuffle required since otherwise peaks and nonpeaks will be together
        #Sample a fraction of the negative samples according to the specified ratio
        if (self.peak_seqs is not None) and (self.nonpeak_seqs is not None):
            # crop peak data before stacking
            cropped_peaks, cropped_cnts, cropped_coords = augment.random_crop(self.peak_seqs, self.peak_cts, self.inputlen, self.outputlen, self.peak_coords, rng=self.rng)
            requested_nonpeaks = int(self.negative_sampling_ratio * len(self.peak_seqs))
            if requested_nonpeaks < len(self.nonpeak_seqs):
                self.sampled_nonpeak_seqs, self.sampled_nonpeak_cts, self.sampled_nonpeak_coords = subsample_nonpeak_data(self.nonpeak_seqs, self.nonpeak_cts, self.nonpeak_coords, len(self.peak_seqs), self.negative_sampling_ratio, rng=self.rng)
                self.seqs = np.vstack([cropped_peaks, self.sampled_nonpeak_seqs])
                self.cts = np.vstack([cropped_cnts, self.sampled_nonpeak_cts])
                self.coords = np.vstack([cropped_coords, self.sampled_nonpeak_coords])
            else:
                self.seqs = np.vstack([cropped_peaks, self.nonpeak_seqs])
                self.cts = np.vstack([cropped_cnts, self.nonpeak_cts])
                self.coords = np.vstack([cropped_coords, self.nonpeak_coords])

        elif self.peak_seqs is not None:
            # crop peak data before stacking
            cropped_peaks, cropped_cnts, cropped_coords = augment.random_crop(self.peak_seqs, self.peak_cts, self.inputlen, self.outputlen, self.peak_coords, rng=self.rng)

            self.seqs = cropped_peaks
            self.cts = cropped_cnts
            self.coords = cropped_coords

        elif self.nonpeak_seqs is not None:
            self.seqs = self.nonpeak_seqs
            self.cts = self.nonpeak_cts
            self.coords = self.nonpeak_coords
        else :
            print("Both peak and non-peak arrays are empty")

        self.cur_seqs, self.cur_cts, self.cur_coords = augment.crop_revcomp_augment(
                                            self.seqs, self.cts, self.coords, self.inputlen, self.outputlen, 
                                            self.add_revcomp, shuffle=self.shuffle_at_epoch_start,
                                            rng=self.rng
                                          )

    def __getitem__(self, idx):
        batch_seq = self.cur_seqs[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_cts = self.cur_cts[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_coords = self.cur_coords[idx*self.batch_size:(idx+1)*self.batch_size]

        if self.return_coords:
            return (batch_seq, [batch_cts, np.log(1+batch_cts.sum(-1, keepdims=True))], batch_coords)
        else:
            return (batch_seq, [batch_cts, np.log(1+batch_cts.sum(-1, keepdims=True))])

    def on_epoch_end(self):
        self.epoch += 1
        # Keep validation fixed without consuming NumPy RNG state; a zero-width
        # random crop still advances the shared RNG and would shift the next
        # training epoch's schedule.
        if (benchmark_enables(self.benchmark, "frozen_validation") and
                not self.add_revcomp and self.negative_sampling_ratio >= 1.0):
            return
        if benchmark_enables(self.benchmark, "replayable_augmentation"):
            # Re-derive the stream so epoch k depends only on (seed, mode, k).
            self.rng = self._make_rng()
        self.crop_revcomp_data()
