"""Where a cut is recorded, and what reverse complementation owes to that.

DNase I is a nuclease: it cuts the backbone *between* two bases, so the event
is a bond. Tn5 leaves a 9 bp staggered footprint, and 9 being odd, that event
is a *base*. The preprocessing conventions differ accordingly, and so must
reverse-complement augmentation.

Under DNase's 0/+1 shift a plus read deposits at ``s`` and a minus read at
``e`` (``reads_to_bigwig``). Writing ``P`` and ``M`` for the enzyme-corrected
5' bases: a plus read's nick lies between ``P-1`` and ``P`` and is stored at
``P``; a minus read's lies between ``M`` and ``M+1`` and is stored at ``M+1``.
Both store **the bond at its right-hand base** -- that is what the ``+1``
buys, and it is what makes the array bond-indexed rather than base-indexed.

The label window is even-width (``data_utils.get_cts``), so its centre is a
bond too, and reversal's fixed point is that bond. Reversal maps base ``i`` to
``W-1-i``, so the bond between ``(i-1, i)`` ends up between ``(W-1-i, W-i)``;
indexed by its right-hand base in the new frame that is ``W-i``. Plain
reversal puts it at ``W-1-i``. Hence::

    reverse, then shift right by one          (rc_shift = +1)

"Shift left by one, then reverse" is the *same* operation -- both send
``i -> W-i`` -- so the order is not a free choice, only the direction paired
with it. The shift does not depend on the read's strand: preprocessing has
already merged both strands onto one bond index.

A -1/0 convention indexes the bond by its **left** base and so needs the
mirror, ``rc_shift = -1``. Depositing on **both** flanking bases is symmetric
about the bond, so plain reversal is already right and ``rc_shift = 0``. ATAC
is base-centred and also needs no shift.

That last case is the DNase default (``both-half``), in preference to the
shift. A shift corrects the misregistration; splitting the cut removes the
asymmetry that causes it, so there is nothing left to correct and no dependence
on getting the sign right. It also costs no edge base -- a shift leaves one
label slot with no source (see :attr:`Convention.widens_label_window`) whereas
a symmetric deposit does not -- at the price of smearing each cut over 2 bp.
The ``0.5`` scale keeps the track total exactly what the single-deposit
conventions give, which is what lets ``counts_loss_weight`` and the
``-oth``/``-b`` ratios carry over unchanged. ATAC has no bond to split, so it
takes no ``--dnase-convention`` at all.

Two flags select all this. ``--dnase-convention`` (DNASE only) picks where a
cut is recorded and the shift that goes with it: ``right``, ``left``,
``both-half`` (the default), ``both-full``, and ``right-deprecated`` -- the
0/+1 recording *without* the shift, i.e. the misregistration described above,
which is what the package did before these flags; kept to reproduce earlier
runs only. ``--data-aug`` picks which examples are reverse complemented, the
same for either assay: ``augment`` (the default: half of training),
``forward-only`` (none) or ``revcomp-only`` (every example of every split).

This is not only derived but measured: the plus- and minus-strand cut-context
PWMs shipped in ``chrombpnet/data/{DNASE,ATAC}.ref.motifs.txt`` align with
each other's reverse complement at lag 0 for DNase and lag -1 for ATAC, and
``rc_shift == lag + 1`` for both. ``tests/test_conventions.py`` asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass


#: How preprocessing records one cut.
BOND_RIGHT = "bond_right"    # DNase 0/+1: the bond, named by its right base
BOND_LEFT = "bond_left"      # DNase -1/0: the bond, named by its left base
BOND_BOTH = "bond_both"      # both bases flanking the bond
BASE = "base"                # ATAC +4/-4: a base

#: ``--dnase-convention`` values: where a DNase cut is recorded, and what
#: reverse complementation does about it. ``right-deprecated`` is the 0/+1
#: convention without the +1 re-registration -- what the package (and
#: chrombpnet 1.x) did before the flags existed, misregistering every
#: reverse-complemented DNase example by one base. Kept for reproducing
#: earlier runs only.
DNASE_CONVENTION_CHOICES = ("right", "left", "both-half", "both-full",
                            "right-deprecated")

#: What an omitted ``--dnase-convention`` means on DNASE. ``both-half``
#: splits the cut across the two bases flanking the bond: symmetric about the
#: bond, so reverse complementation needs no shift and cannot misregister,
#: and scaled by 0.5 so the track total is unchanged. ATAC's cut is already
#: on a base, so the flag has no meaning there and is refused if given.
DNASE_CONVENTION_DEFAULT = "both-half"

#: ``--data-aug`` values: which examples are reverse complemented. They mean
#: the same for either assay; the geometry comes from the assay (and, for
#: DNASE, ``--dnase-convention``).
DATA_AUG_CHOICES = ("augment", "forward-only", "revcomp-only")

#: What an omitted ``--data-aug`` means: half of the training examples
#: reverse complemented, validation and test never.
DATA_AUG_DEFAULT = "augment"


def default_for(data_type):
    """The ``--dnase-convention`` an omitted flag selects for ``data_type``
    (``None`` for ATAC, which has no DNase convention)."""
    if data_type == "DNASE":
        return DNASE_CONVENTION_DEFAULT
    if data_type == "ATAC":
        return None
    raise ValueError(
        f"unknown --data-type {data_type!r}; expected ATAC or DNASE")


_ALL_MODES = frozenset({"train", "valid", "test"})
_TRAIN_ONLY = frozenset({"train"})
_NO_MODES = frozenset()


@dataclass(frozen=True)
class Convention:
    """What one ``(data_type, data_aug, dnase_convention)`` means, end to end."""

    data_type: str
    data_aug: str
    dnase_convention: object    # str for DNASE, None for ATAC
    deposit: str
    scale: float
    rc_shift: int
    rc_modes: frozenset
    rc_frac: float

    def reverse_complements(self, mode):
        """Whether ``mode`` ("train"/"valid"/"test") is augmented at all."""
        return mode in self.rc_modes

    @property
    def widens_label_window(self):
        """True when ``get_cts`` must fetch one extra column.

        The shift leaves one slot with no source. Zero-filling it would plant
        a systematic zero at the same index of every reverse-complemented
        example -- small, but an arm-asymmetric artifact inside the very
        comparison these arms exist to make. Fetching one more base instead
        costs one column and keeps the totals exact. Only the shifted
        conventions pay it, and only when something is reverse complemented,
        so every other combination stays on the original code path.
        """
        return self.rc_shift != 0 and bool(self.rc_modes)


# deposit, scale, rc_shift -- for DNASE. ATAC is handled separately: its
# signal is base-centred, so it needs no convention.
_DNASE = {
    "right":            (BOND_RIGHT, 1.0, +1),
    "left":             (BOND_LEFT,  1.0, -1),
    "both-half":        (BOND_BOTH,  0.5, 0),
    "both-full":        (BOND_BOTH,  1.0, 0),
    "right-deprecated": (BOND_RIGHT, 1.0, 0),
}

# rc_modes, rc_frac -- for either assay.
_RC_MODES = {
    "augment":      (_TRAIN_ONLY, 0.5),
    "forward-only": (_NO_MODES,   0.5),
    "revcomp-only": (_ALL_MODES,  1.0),
}


def resolve(data_type, data_aug=None, dnase_convention=None):
    """The :class:`Convention` for an assay, an augmentation and, for DNASE,
    a cut convention.

    Raises on a DNase convention given for ATAC, rather than quietly doing
    nothing: a silent no-op would let an ATAC sweep return several identical
    numbers with nothing to say why.
    """
    if data_type not in ("ATAC", "DNASE"):
        raise ValueError(
            f"unknown --data-type {data_type!r}; expected ATAC or DNASE")
    data_aug = DATA_AUG_DEFAULT if data_aug is None else str(data_aug)
    if data_aug not in DATA_AUG_CHOICES:
        raise ValueError(
            f"unknown --data-aug {data_aug!r}; expected one of "
            f"{', '.join(DATA_AUG_CHOICES)}")
    rc_modes, rc_frac = _RC_MODES[data_aug]

    if data_type == "ATAC":
        if dnase_convention is not None:
            raise ValueError(
                f"--dnase-convention {dnase_convention!r} is defined only for "
                f"DNASE. ATAC's +4/-4 shift centres the cut on a base, not on a "
                f"bond, so there is no bond to re-index or to split, and plain "
                f"reversal is already correct. Leave the flag out for ATAC.")
        return Convention(data_type, data_aug, None, BASE, 1.0, 0,
                          rc_modes, rc_frac)

    # An omitted flag resolves to the DNase default, so the message below
    # only ever fires for a value the caller actually asked for.
    dnase_convention = (DNASE_CONVENTION_DEFAULT if dnase_convention is None
                        else str(dnase_convention))
    if dnase_convention not in DNASE_CONVENTION_CHOICES:
        raise ValueError(
            f"unknown --dnase-convention {dnase_convention!r}; expected one of "
            f"{', '.join(DNASE_CONVENTION_CHOICES)}")
    deposit, scale, rc_shift = _DNASE[dnase_convention]
    return Convention(data_type, data_aug, dnase_convention, deposit, scale,
                      rc_shift, rc_modes, rc_frac)


def from_args(args):
    """:func:`resolve` for a parsed command line (either flag may be absent)."""
    return resolve(args.data_type, getattr(args, "data_aug", None),
                   getattr(args, "dnase_convention", None))


def shift_delta_pairs(convention, plus_shift, minus_shift):
    """``[(plus_delta, minus_delta), ...]`` for ``reads_to_bigwig``'s awk.

    One pair per deposit. ``BOND_BOTH`` returns two, and they are exactly the
    ``BOND_LEFT`` and ``BOND_RIGHT`` pairs -- depositing on both flanking
    bases *is* applying both single-base conventions at once.

    The deltas are applied as ``$2 += plus`` on a plus read and ``$3 += minus``
    on a minus read, after which ``bedtools genomecov -bg -5`` takes the start
    of a plus record and ``end - 1`` of a minus one.
    """
    left = (-1 - plus_shift, -minus_shift)
    right = (-plus_shift, 1 - minus_shift)
    if convention.deposit == BASE:
        return [(4 - plus_shift, -4 - minus_shift)]
    if convention.deposit == BOND_LEFT:
        return [left]
    if convention.deposit == BOND_RIGHT:
        return [right]
    if convention.deposit == BOND_BOTH:
        return [left, right]
    raise ValueError(f"unknown deposit scheme {convention.deposit!r}")
