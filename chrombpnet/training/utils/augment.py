import numpy as np


def _stream(rng):
    """Benchmark runs pass a dedicated RandomState; otherwise use global state."""
    return np.random if rng is None else rng


# https://stackoverflow.com/questions/46091111/python-slice-array-at-different-position-on-every-row
def take_per_row(A, indx, num_elem):
    """
    Matrix A, indx is a vector for each row which specifies 
    slice beginning for that row. Each has width num_elem.
    """

    all_indx = indx[:,None] + np.arange(num_elem)
    return A[np.arange(all_indx.shape[0])[:,None], all_indx]


def random_crop(seqs, labels, seq_crop_width, label_crop_width, coords, rng=None):
    """
    Takes sequences and corresponding counts labels. They should have the same
    #examples. The widths would correspond to inputlen and outputlen respectively,
    and any additional flanking width for jittering which should be the same
    for seqs and labels. Each example is cropped starting at a random offset. 

    seq_crop_width - label_crop_width should be equal to seqs width - labels width,
    essentially implying they should have the same flanking width.
    """

    if seqs.shape[1] < seq_crop_width:
        raise ValueError("Sequence crop width exceeds sequence width")
    if labels.shape[1] < label_crop_width:
        raise ValueError("Label crop width exceeds label width")
    if seqs.shape[1] - seq_crop_width != labels.shape[1] - label_crop_width:
        raise ValueError("Sequence and label crop flanks must match")

    max_start = seqs.shape[1] - seq_crop_width # This should be the same for both input and output

    starts = _stream(rng).choice(range(max_start+1), size=seqs.shape[0], replace=True)

    new_coords = coords.copy()
    #new_coords[:,1] = new_coords[:,1].astype(int) - (seqs.shape[1]//2) + starts
    new_coords[:,1] = new_coords[:,1].astype(int) + starts

    return take_per_row(seqs, starts, seq_crop_width), take_per_row(labels, starts, label_crop_width), new_coords

def rc_padding(rc_shift):
    """``(pad_left, pad_right)``: which side the extra label column sits on.

    A shifted flip reads one base beyond the window on the side it shifts
    towards. Fetching that base is what keeps the leading slot from being a
    fabricated zero in every reverse-complemented example.
    """
    return (1 if rc_shift < 0 else 0), (1 if rc_shift > 0 else 0)


def random_rev_comp(seqs, labels, coords, frac=0.5, rng=None, rc_shift=0,
                    output_width=None):
    """
    Data augmentation: applies reverse complement randomly to a fraction of
    sequences and labels.

    Assumes seqs are arranged in ACGT. Then ::-1 gives TGCA which is revcomp.

    ``rc_shift`` re-registers the profile after reversal, which DNase needs
    and ATAC does not. DNase's 0/+1 convention stores a cut at the *bond*
    between two bases, named by that bond's right-hand base. Reversal carries
    the bond to the right place but leaves it named by the wrong base, so the
    whole profile has to move one position: ``+1`` is that correction, ``-1``
    is the mirror for a -1/0 convention, and ``0`` leaves plain reversal
    alone -- correct for ATAC, whose 9 bp stagger centres the event on a
    base, and for the arms that deposit on both bases flanking the bond,
    which are already symmetric about it. See :mod:`chrombpnet.conventions`.

    ``labels`` is one column wider than ``output_width`` when ``rc_shift`` is
    non-zero (see :func:`rc_padding`). The flipped rows read from the shifted
    offset and the forward rows from the unshifted one, so both describe the
    same genomic window and the shifted slot holds real signal.

    NOTE: modifies ``seqs`` and ``coords`` in place, and returns a **new**
    label array -- it has to, because the width changes.
    """

    pos_to_rc = _stream(rng).choice(range(seqs.shape[0]),
            size=int(seqs.shape[0]*frac),
            replace=False)

    width = labels.shape[1] if output_width is None else int(output_width)
    pad_left, pad_right = rc_padding(rc_shift)
    if labels.shape[1] != width + pad_left + pad_right:
        raise ValueError(
            f"labels are {labels.shape[1]} wide; a shift of {rc_shift:+d} "
            f"needs {width + pad_left + pad_right} to return {width}")

    seqs[pos_to_rc] = seqs[pos_to_rc, ::-1, ::-1]

    out_labels = labels[:, pad_left:pad_left + width].copy()
    flipped = labels[:, pad_right:pad_right + width][:, ::-1]
    out_labels[pos_to_rc] = flipped[pos_to_rc]

    coords[pos_to_rc,2] =  "r"

    return seqs, out_labels, coords


def crop_revcomp_augment(seqs, labels, coords, seq_crop_width, label_crop_width, add_revcomp, rc_frac=0.5, shuffle=False, rng=None, rc_shift=0):
    """
    seqs: B x IL x 4
    labels: B x OL, or B x (OL+1) when rc_shift is non-zero

    Applies random crop to seqs and labels and reverse complements rc_frac.
    Always returns labels ``label_crop_width`` wide: the extra column that a
    shifting convention fetches is trimmed here, on both branches, because
    every split fetches it -- including the ones that never flip.
    """

    if seqs.shape[0] != labels.shape[0]:
        raise ValueError("Sequence and label batch sizes must match")

    # this does not modify seqs and labels
    #mod_seqs, mod_labels, mod_coords = random_crop(seqs, labels, seq_crop_width, label_crop_width, coords)
    mod_seqs, mod_labels, mod_coords = seqs, labels, coords

    # this modifies mod_seqs in place and returns a fresh label array
    if add_revcomp:
        mod_seqs, mod_labels, mod_coords = random_rev_comp(mod_seqs, mod_labels, mod_coords, frac=rc_frac, rng=rng, rc_shift=rc_shift, output_width=label_crop_width)
    else:
        pad_left, _ = rc_padding(rc_shift)
        mod_labels = mod_labels[:, pad_left:pad_left + label_crop_width]


    if shuffle:
        perm = _stream(rng).permutation(mod_seqs.shape[0])
        mod_seqs = mod_seqs[perm]
        mod_labels = mod_labels[perm]
        mod_coords = mod_coords[perm]


    return mod_seqs, mod_labels, mod_coords
