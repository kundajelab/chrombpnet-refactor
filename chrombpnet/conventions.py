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
The ``0.5`` scale keeps the track total exactly what ``standard`` would give,
which is what lets ``counts_loss_weight`` and the ``-oth``/``-b`` ratios carry
over unchanged. ATAC has no bond to split, so its default stays ``standard``.

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

#: ``--data-aug`` values.
DATA_AUG_CHOICES = ("standard", "fr-only", "rc-only", "right", "left",
                    "both-half", "both-full")

#: Arms that mean the same thing for either assay. The rest reshape a bond,
#: which ATAC has not got.
ASSAY_AGNOSTIC = ("standard", "fr-only", "rc-only")

#: What an omitted ``--data-aug`` means, per assay.
#:
#: The default has to be assay-dependent, because the DNase default is an arm
#: that does not exist for ATAC. ``both-half`` splits the cut across the two
#: bases flanking the bond: symmetric about the bond, so reverse
#: complementation needs no shift and cannot misregister, and scaled by 0.5 so
#: the track total is unchanged. ATAC's cut is already on a base, so there is
#: no bond to split and ``standard`` is the only thing an omitted flag can
#: mean there.
DATA_AUG_DEFAULT_DNASE = "both-half"
DATA_AUG_DEFAULT_ATAC = "standard"

#: There is deliberately no bare ``DATA_AUG_DEFAULT``. A single name meaning
#: "the default" would have to be one assay's, and using it for the other is
#: the exact mistake this split exists to prevent -- on ATAC it would name an
#: arm that raises.


def default_for(data_type):
    """The arm an omitted ``--data-aug`` selects for ``data_type``."""
    if data_type == "DNASE":
        return DATA_AUG_DEFAULT_DNASE
    if data_type == "ATAC":
        return DATA_AUG_DEFAULT_ATAC
    raise ValueError(
        f"unknown --data-type {data_type!r}; expected ATAC or DNASE")


_ALL_MODES = frozenset({"train", "valid", "test"})
_TRAIN_ONLY = frozenset({"train"})
_NO_MODES = frozenset()


@dataclass(frozen=True)
class Convention:
    """What one ``(data_type, data_aug)`` pair means, end to end."""

    data_type: str
    data_aug: str
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
        costs one column and keeps the totals exact. Only the shifted arms
        pay it, so every other arm stays on the original code path.
        """
        return self.rc_shift != 0 and bool(self.rc_modes)


# deposit, scale, rc_shift -- for DNASE. ATAC is handled separately: its
# signal is base-centred whatever the arm, and only the agnostic arms apply.
_DNASE = {
    "standard":  (BOND_RIGHT, 1.0, 0),
    "fr-only":   (BOND_RIGHT, 1.0, 0),
    "rc-only":   (BOND_RIGHT, 1.0, 0),
    "right":     (BOND_RIGHT, 1.0, +1),
    "left":      (BOND_LEFT,  1.0, -1),
    "both-half": (BOND_BOTH,  0.5, 0),
    "both-full": (BOND_BOTH,  1.0, 0),
}

_RC_MODES = {
    "standard":  (_TRAIN_ONLY, 0.5),
    "fr-only":   (_NO_MODES,   0.5),
    "rc-only":   (_ALL_MODES,  1.0),
    "right":     (_TRAIN_ONLY, 0.5),
    "left":      (_TRAIN_ONLY, 0.5),
    "both-half": (_TRAIN_ONLY, 0.5),
    "both-full": (_TRAIN_ONLY, 0.5),
}


def resolve(data_type, data_aug=None):
    """The :class:`Convention` for an assay and an augmentation arm.

    Raises on an arm that has no meaning for the assay, rather than quietly
    doing nothing: a silent no-op would let an ATAC sweep return several
    identical numbers with nothing to say why.
    """
    if data_type not in ("ATAC", "DNASE"):
        raise ValueError(
            f"unknown --data-type {data_type!r}; expected ATAC or DNASE")
    # An omitted flag resolves per assay, so the message below only ever
    # fires for an arm the caller actually asked for.
    data_aug = default_for(data_type) if data_aug is None else str(data_aug)
    if data_aug not in DATA_AUG_CHOICES:
        raise ValueError(
            f"unknown --data-aug {data_aug!r}; expected one of "
            f"{', '.join(DATA_AUG_CHOICES)}")

    rc_modes, rc_frac = _RC_MODES[data_aug]

    if data_type == "ATAC":
        if data_aug not in ASSAY_AGNOSTIC:
            raise ValueError(
                f"--data-aug {data_aug!r} is defined only for DNASE. ATAC's "
                f"+4/-4 shift centres the cut on a base, not on a bond, so "
                f"there is no bond to re-index or to split, and plain "
                f"reversal is already correct. For ATAC use one of: "
                f"{', '.join(ASSAY_AGNOSTIC)}.")
        return Convention(data_type, data_aug, BASE, 1.0, 0, rc_modes, rc_frac)

    deposit, scale, rc_shift = _DNASE[data_aug]
    return Convention(data_type, data_aug, deposit, scale, rc_shift,
                      rc_modes, rc_frac)


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
