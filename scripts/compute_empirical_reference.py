#!/usr/bin/env python
"""
External empirical reference anchor (rho) of the MPC construct scale from
the high-state runs of a declared held-out subset of participants.

The construct scale ``c = (m - nu) / (rho - nu)`` needs a reference anchor
``rho`` per principle. The default protocols take it from the cohort's own
high state (``cohort_high_state``), which makes the high-state runs part of
their own anchor: their ``c`` averages 1 by construction, so necessity tests
among report-positive (high-state) episodes need an **external** reference.
This script computes one from participants that are declared beforehand and
held out of the evaluation, and writes it with its provenance:

- ``rho[P]`` = mean over the reference participants of the participant's
  mean run excess ``m - nu`` (estimate minus its null mean) over the
  high-state runs in which ``P`` is defined; ``se[P]`` = SD of the
  participant means / sqrt(n). This is the pipeline's cohort reference
  (``synergy_ci._mpc_references``) restricted to the declared subset.
- Anchor rule (as ``impact_pipeline.bench.reference``): a principle gets an
  anchor only with at least ``--min-n`` participants and a positive
  one-sided ``1 - alpha`` Student-t lower bound of ``rho`` (``n - 1``
  degrees of freedom). A principle without an anchor is listed under
  ``no_anchor`` with the reason; under the derived protocol its evidence is
  UNDEFINED (``INVALID_ANCHORS``), never judged against a made-up anchor.
- Channels. An external reference takes the keys ``P`` or ``P:channel``
  (``evidence.Protocol``). A principle with one declared channel gets the key
  ``P``. A principle with several declared channels gets one key
  ``P:channel`` per channel that has an estimator: under the default
  ``mpc_default_v1.json`` RAM declares ``default`` (behavioural, implemented),
  ``perturbational`` and ``endogenous`` (no estimator), so the reference is
  ``RAM:default``, and the two unimplemented channels get none; they are
  listed under ``channels_without_reference`` with the reason
  ``NOT_IMPLEMENTED`` (their evidence is UNDEFINED under any reference, so
  RAM stays never-ABSENT under v1). Per-channel evidence exists only in
  ``--data-dir`` mode (``compute_synergy_ci(record_channel_evidence=True)``).
  A step-2 table holds ``<P>_estimate``/``<P>_null_mean`` for the channel
  that decides the principle's status, and under v1 that is an unimplemented
  RAM channel (NaN) whenever the behavioural channel is ABSENT: the table
  cannot identify the channel of a row's RAM estimate, and using the finite
  rows would drop exactly the runs with the lowest behavioural excess. So in
  ``--step2-table`` mode a principle with several declared channels gets no
  reference, with the reason ``CHANNEL_NOT_IDENTIFIABLE`` (never a guess);
  compute it with ``--data-dir``.

Evidence sources (one of):

- ``--step2-table``: a step-2 table (``<out>/cache/step2_df.csv``) of a
  pipeline run (local or Hunter finalize) that covers the reference
  participants and used ``--protocol``; its ``MPC_protocol_hash`` must equal
  the protocol's hash. One row per run is used (the smallest ``theta``).
- ``--data-dir``: the evidence is computed here with ``compute_synergy_ci`` on
  the reference participants' high-state runs only (the pipeline's
  preprocessed layout ``<data-dir>/<subject>/<session>/<condition>/
  *_<atlas>_ts.npy``; events from ``--bids-root``; estimator parameters from
  ``--dataset-id`` or ``--params-json``; ``--null-surrogates`` default 19).
  It records every run's evidence per principle and declared channel, so it
  is the only mode that gives a per-channel reference (``RAM:default`` under
  v1). It passes the estimator parameters only, not every other step-2
  option of ``run_pipeline.py``; for the single-channel principles an anchor
  computed exactly as the evaluation runs comes from running the pipeline on
  the reference participants and ``--step2-table``.

The held-out subset is declared with ``--reference-subjects`` (a comma list
or ``@file`` with one participant per line); ``--evaluation-subjects``
(optional) is checked to be disjoint from it. Output (``--out``, schema
``impact-empirical-reference/1``): ``reference`` (the
``evidence.Protocol`` external layout: kind, scale ``excess``, values, se,
source), ``summary`` (the evidence mode, ``per_channel_evidence`` or
``step2_columns``; the declared channels; the reference keys; per principle,
and per channel for a principle with several: participants, runs, mean, SD,
SE, lower bound, or the reason for no anchor; ``channels_without_reference``
``{P or P:channel: reason code}``) and ``provenance`` (dataset, evidence
source and its SHA-256,
protocol path, name and hash, the declared subset and its SHA-256, the
participants used and missing, session, estimator ids and versions seen,
null settings, code version, UTC time, command line). ``--write-protocol``
also writes the protocol with this external reference (name
``<name>+external-reference``); use it for the evaluation participants only.

Example::

    # every principle and channel, RAM:default included (per-channel evidence)
    python scripts/compute_empirical_reference.py \\
        --data-dir outputs/ds003171/preprocessed --atlas schaefer400 \\
        --bids-root /data/openneuro/ds003171 --dataset-id ds003171 \\
        --protocol protocols/examples/mpc_default_v1_schaefer400_7networks_hub.json \\
        --reference-subjects @heldout.txt --evaluation-subjects @evaluation.txt \\
        --out outputs/ds003171_reference/empirical_reference.json \\
        --write-protocol outputs/ds003171_reference/mpc_default_v1_hub_external.json
    # from a step-2 table of the pipeline run on the reference participants
    # (single-channel principles only; RAM under v1: CHANNEL_NOT_IDENTIFIABLE)
    python scripts/compute_empirical_reference.py \\
        --step2-table outputs/ds003171_reference/cache/step2_df.csv \\
        --protocol protocols/examples/mpc_default_v1_schaefer400_7networks_hub.json \\
        --reference-subjects @heldout.txt --evaluation-subjects @evaluation.txt \\
        --out outputs/ds003171_reference/empirical_reference.json

The default ``--protocol`` is ``protocols/mpc_default_v1.json`` (the default
empirical protocol); pass the derived protocol the evaluation uses (e.g. one
with a declared NAS hub), since the table's protocol hash must match.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from impact_pipeline import evidence as E  # noqa: E402

SCHEMA = "impact-empirical-reference/1"
PRINCIPLES = E.PRINCIPLES
# the default empirical protocol of run_pipeline.py (preregistered, unchanged
# since the freeze)
DEFAULT_PROTOCOL = REPO_ROOT / "protocols" / "mpc_default_v1.json"
DEFAULT_MIN_N = 2
DEFAULT_ALPHA = 0.05
DEFAULT_NULL_SURROGATES = 19
ANCHOR_RULE = (
    "excess scale: anchor only with >= min_n participants and a positive "
    "one-sided (1 - alpha) Student-t lower bound of the mean participant excess"
)
# Why a declared channel (reference key P or P:channel) has no reference:
# stable codes of summary.channels_without_reference.
NO_REF_NOT_IMPLEMENTED = "NOT_IMPLEMENTED"  # the channel has no estimator
NO_REF_CHANNEL_NOT_IDENTIFIABLE = "CHANNEL_NOT_IDENTIFIABLE"  # step-2 columns
NO_REF_NOT_COMPUTED = "NOT_COMPUTED"  # the principle was not computed
NO_REF_MISSING_CHANNEL = "MISSING_CHANNEL"  # no evidence for this channel
NO_REF_MISSING_COLUMNS = "MISSING_COLUMNS"  # the table lacks the columns
NO_REF_TOO_FEW_PARTICIPANTS = "TOO_FEW_PARTICIPANTS"
NO_REF_NOT_ABOVE_NULL = "NOT_ABOVE_NULL"  # anchor rule not met
EVIDENCE_PER_CHANNEL = "per_channel_evidence"
EVIDENCE_STEP2_COLUMNS = "step2_columns"


# --------------------------------------------------------------------------
# declarations
# --------------------------------------------------------------------------
def _norm_subject(s) -> str:
    return str(s).strip().replace("sub-", "", 1)


def parse_subjects(spec) -> List[str]:
    """Participants from a comma list, a sequence, or ``@file`` (one per line,
    ``#`` comments). Refuses an empty declaration and duplicates."""
    if spec is None:
        return []
    if isinstance(spec, str):
        text = spec.strip()
        if text.startswith("@"):
            lines = Path(text[1:]).read_text(encoding="utf-8").splitlines()
            items = [ln.split("#", 1)[0] for ln in lines]
        else:
            items = text.split(",")
    else:
        items = list(spec)
    out = [_norm_subject(s) for s in items if str(s).strip()]
    if len(set(out)) != len(out):
        raise ValueError("duplicate participants in the declaration")
    return out


def _sha256_text(items: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(items)).encode("utf-8")).hexdigest()


def _sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declared_channels(proto) -> Dict[str, tuple]:
    """Declared channels per principle (``("default",)`` when undeclared)."""
    return {p: tuple(proto.channels_for(p) or ("default",)) for p in PRINCIPLES}


def channel_has_estimator(principle: str, channel: str) -> bool:
    """
    Whether the pipeline computes evidence for ``principle`` on ``channel``:
    RAM on ``default`` (untyped behavioural events) and on
    ``mpc_metrics.RAM_IMPLEMENTED_CHANNELS``; the other principles on
    ``default`` only. RAM's ``perturbational`` and ``endogenous`` channels
    have no estimator (``NOT_IMPLEMENTED``).
    """
    from impact_pipeline.mpc_metrics import RAM_IMPLEMENTED_CHANNELS

    if str(principle) == "RAM":
        return str(channel) in ("default", *RAM_IMPLEMENTED_CHANNELS)
    return str(channel) == "default"


def reference_key(principle: str, channel: str, n_channels: int) -> str:
    """The key of an external reference value: ``P`` for a principle with
    one declared channel, ``P:channel`` with several (``evidence.Protocol``)."""
    return str(principle) if int(n_channels) <= 1 else f"{principle}:{channel}"


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def one_row_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """Step-2 tables have one row per theta and run with identical evidence
    columns; keep the rows of the smallest theta (one per run)."""
    if "theta" not in df.columns or df.empty:
        return df
    theta = pd.to_numeric(df["theta"], errors="coerce")
    if not theta.notna().any():
        return df
    return df[np.isclose(theta, float(theta.min()))]


def anchor_lower_bound(mean: float, se: float, n: int, alpha: float) -> float:
    from scipy.stats import t as t_dist

    if int(n) < 2 or not (math.isfinite(mean) and math.isfinite(se)):
        return float("nan")
    return float(mean - t_dist.ppf(1.0 - float(alpha), int(n) - 1) * se)


def _no_reference(code: str, text: str, **extra) -> dict:
    return {"n": 0, "n_runs": 0, **extra, "reason": code,
            "anchor": f"none ({code}: {text})"}


def _anchor_stats(excess: pd.Series, subjects: pd.Series, min_n: int,
                  alpha: float) -> dict:
    """
    Participant means first (``excess`` per run, finite values only), then
    mean, SD, SE = SD / sqrt(n) and the one-sided Student-t lower bound; the
    anchor rule (module docstring) decides whether it is an anchor.
    """
    values = pd.to_numeric(excess, errors="coerce")
    ok = np.isfinite(values.to_numpy(dtype=float))
    by_subject = values[ok].groupby(subjects[ok]).mean()
    arr = by_subject.to_numpy(dtype=float)
    n = int(arr.size)
    out = {"n": n, "n_runs": int(ok.sum()),
           "participants": sorted(by_subject.index.tolist())}
    if n < int(min_n):
        out["reason"] = NO_REF_TOO_FEW_PARTICIPANTS
        out["anchor"] = (
            f"none (fewer than {int(min_n)} participants with a defined, "
            "null-calibrated estimate)")
        return out
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(n)
    lower = anchor_lower_bound(mean, se, n, alpha)
    out.update(mean=mean, sd=sd, se=se, lower_bound=lower)
    if not lower > 0:
        out["reason"] = NO_REF_NOT_ABOVE_NULL
        out["anchor"] = (
            "none (high-state excess not credibly above its null: "
            f"mean {mean:.6g}, se {se:.6g}, one-sided {1 - float(alpha):g} "
            f"lower bound {lower:.6g})")
    return out


def _check_rule(min_n, alpha):
    if not 0.0 < float(alpha) < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if int(min_n) < 2:
        raise ValueError("min_n must be >= 2 (an SE needs two participants)")


def _collect(per_channel: Dict[str, Dict[str, dict]], chans_by_p: Dict[str, tuple]):
    """``(values, se, per_principle)`` from per-principle, per-channel stats."""
    values, ses, per = {}, {}, {}
    for p, by_ch in per_channel.items():
        chans = chans_by_p[p]
        for ch, st in by_ch.items():
            if "reason" not in st:
                key = reference_key(p, ch, len(chans))
                st["reference_key"] = key
                values[key] = st["mean"]
                ses[key] = st["se"]
        if len(chans) <= 1:
            ch = chans[0]
            per[p] = {"channel": ch, **by_ch[ch]}
            continue
        keys = [st["reference_key"] for st in by_ch.values() if "reference_key" in st]
        per[p] = {"declared_channels": list(chans), "channels": by_ch,
                  "reference_keys": keys}
        if not keys:
            per[p]["anchor"] = "none (no declared channel has an anchor)"
    return values, ses, per


def reference_from_table(
    df: pd.DataFrame,
    subjects: Sequence[str],
    session: str,
    principles: Sequence[str] = PRINCIPLES,
    channels: Optional[Dict[str, tuple]] = None,
    min_n: int = DEFAULT_MIN_N,
    alpha: float = DEFAULT_ALPHA,
) -> tuple:
    """
    ``(values, se, per_principle)`` from a step-2 table: the high-state
    (``session``) runs of ``subjects``, one row per run, excess
    ``<P>_estimate - <P>_null_mean`` where both are finite, participant means
    first (module docstring). The ``<P>_*`` columns describe the channel that
    decides the principle's status, so only a principle with one declared
    channel is computed; with several, every channel is left without a
    reference (``NOT_IMPLEMENTED`` for a channel without an estimator,
    ``CHANNEL_NOT_IDENTIFIABLE`` otherwise).
    """
    _check_rule(min_n, alpha)
    missing = [c for c in ("subject", "session") if c not in df.columns]
    if missing:
        raise ValueError(f"the table lacks the columns {missing}")
    want = {_norm_subject(s) for s in subjects}
    rows = df[df["subject"].map(_norm_subject).isin(want)
              & (df["session"].astype(str) == str(session))]
    rows = one_row_per_run(rows)
    per_channel, chans_by_p = {}, {}
    for p in principles:
        chans = tuple((channels or {}).get(p, ("default",)))
        chans_by_p[p] = chans
        if len(chans) > 1:
            not_identifiable = (
                f"{p} declares the channels {list(chans)}; the step-2 columns "
                f"{p}_estimate/{p}_null_mean hold the channel that decides the "
                "principle's status, which varies by run (e.g. under v1 an "
                "unimplemented RAM channel decides whenever the behavioural "
                "one is ABSENT), so the channel of a row's estimate cannot be "
                "identified; compute the reference with --data-dir")
            per_channel[p] = {
                ch: (_no_reference(NO_REF_CHANNEL_NOT_IDENTIFIABLE, not_identifiable)
                     if channel_has_estimator(p, ch) else _no_reference(
                         NO_REF_NOT_IMPLEMENTED,
                         f"{p} has no estimator for the channel {ch!r}"))
                for ch in chans
            }
            continue
        ch = chans[0]
        if not channel_has_estimator(p, ch):
            per_channel[p] = {ch: _no_reference(
                NO_REF_NOT_IMPLEMENTED,
                f"{p} has no estimator for the channel {ch!r}")}
            continue
        ec, nc = f"{p}_estimate", f"{p}_null_mean"
        if ec not in rows.columns or nc not in rows.columns:
            per_channel[p] = {ch: _no_reference(
                NO_REF_MISSING_COLUMNS, f"{p} evidence columns are missing")}
            continue
        ids = rows.get(f"{p}_estimator")
        if ids is not None and not rows.empty and not (
                ids.fillna("").astype(str).str.strip() != "").any():
            # an empty evidence id in every row: the principle was not computed
            # (as NOT_COMPUTED in --data-dir mode)
            per_channel[p] = {ch: _no_reference(
                NO_REF_NOT_COMPUTED,
                f"no {p} evidence (the principle was not computed)")}
            continue
        excess = (pd.to_numeric(rows[ec], errors="coerce")
                  - pd.to_numeric(rows[nc], errors="coerce"))
        per_channel[p] = {ch: _anchor_stats(
            excess, rows["subject"].map(_norm_subject), min_n, alpha)}
    return _collect(per_channel, chans_by_p)


def reference_from_channel_evidence(
    records: Sequence[dict],
    subjects: Sequence[str],
    session: str,
    principles: Sequence[str] = PRINCIPLES,
    channels: Optional[Dict[str, Optional[tuple]]] = None,
    min_n: int = DEFAULT_MIN_N,
    alpha: float = DEFAULT_ALPHA,
) -> tuple:
    """
    ``(values, se, per_principle)`` from per-channel evidence records
    (``df.attrs['mpc_evidence']['channel_evidence']`` of
    ``compute_synergy_ci(record_channel_evidence=True)``): per principle and
    declared channel, the high-state runs of ``subjects``, excess
    ``estimate - null_mean`` where both are finite, participant means first.
    Keys ``P`` (one declared channel) or ``P:channel`` (several). A channel
    whose records are all ``NOT_IMPLEMENTED`` gets no reference.
    ``channels[P] = None`` (undeclared) uses the channels of the records.
    """
    _check_rule(min_n, alpha)
    want = {_norm_subject(s) for s in subjects}
    ev = pd.DataFrame(list(records or []), columns=[
        "subject", "session", "principle", "channel", "estimate", "null_mean",
        "reason"])
    ev = ev[ev["subject"].map(_norm_subject).isin(want)
            & (ev["session"].astype(str) == str(session))]
    per_channel, chans_by_p = {}, {}
    for p in principles:
        recs_p = ev[ev["principle"] == p]
        declared = (channels or {}).get(p)
        if declared:
            chans = tuple(declared)
        else:
            seen = sorted(set(recs_p["channel"].astype(str)),
                          key=lambda c: (c != "default", c))
            chans = tuple(seen) or ("default",)
        chans_by_p[p] = chans
        per_channel[p] = {}
        for ch in chans:
            recs = recs_p[recs_p["channel"].astype(str) == ch]
            not_impl = (recs["reason"].astype(str) == E.REASON_NOT_IMPLEMENTED)
            if (recs.empty and not channel_has_estimator(p, ch)) or (
                    not recs.empty and not_impl.all()):
                st = _no_reference(
                    NO_REF_NOT_IMPLEMENTED,
                    f"{p} has no estimator for the channel {ch!r}"
                    + (f" ({len(recs)} runs, all NOT_IMPLEMENTED)"
                       if len(recs) else ""))
            elif recs_p.empty:
                st = _no_reference(NO_REF_NOT_COMPUTED,
                                   f"no {p} evidence (the principle was not computed)")
            elif recs.empty:
                st = _no_reference(NO_REF_MISSING_CHANNEL,
                                   f"no {p} evidence for the channel {ch!r}")
            else:
                excess = (pd.to_numeric(recs["estimate"], errors="coerce")
                          - pd.to_numeric(recs["null_mean"], errors="coerce"))
                st = _anchor_stats(excess, recs["subject"].map(_norm_subject),
                                   min_n, alpha)
            per_channel[p][ch] = st
    return _collect(per_channel, chans_by_p)


def channels_without_reference(per: Dict[str, dict]) -> Dict[str, str]:
    """``{P or P:channel: reason code}`` of every channel without a reference."""
    out = {}
    for p, entry in per.items():
        if "channels" in entry:
            for ch, st in entry["channels"].items():
                if "reference_key" not in st:
                    out[f"{p}:{ch}"] = st.get("reason", "")
        elif "reference_key" not in entry:
            out[p] = entry.get("reason", "")
    return out


def _estimators_seen(rows: pd.DataFrame) -> dict:
    out = {}
    for p in PRINCIPLES:
        ec, vc = f"{p}_estimator", f"{p}_estimator_version"
        if ec not in rows.columns:
            continue
        ids = sorted({str(v) for v in rows[ec].dropna() if str(v)})
        vers = (sorted({str(v) for v in rows[vc].dropna() if str(v)})
                if vc in rows.columns else [])
        out[p] = {"estimator": ids, "estimator_version": vers}
    return out


def build_reference(
    df: pd.DataFrame,
    protocol,
    reference_subjects: Sequence[str],
    session: str = "awake",
    evaluation_subjects: Sequence[str] = (),
    min_n: int = DEFAULT_MIN_N,
    alpha: float = DEFAULT_ALPHA,
    source: Optional[dict] = None,
    dataset_id: Optional[str] = None,
    command: Optional[Sequence[str]] = None,
    channel_evidence: Optional[Sequence[dict]] = None,
) -> dict:
    """
    The reference JSON (schema ``impact-empirical-reference/1``) from a
    step-2 table computed under ``protocol``. Refuses an empty or overlapping
    declaration and a table from another protocol. With ``channel_evidence``
    (the per-channel records of the same computation, ``--data-dir`` mode)
    every principle is computed per declared channel
    (:func:`reference_from_channel_evidence`); without it from the step-2
    columns (:func:`reference_from_table`).
    """
    from impact_pipeline.provenance import collect_code_version

    proto = E.resolve_protocol(protocol)
    if proto is None:
        raise ValueError("a protocol is required")
    ref_subj = [_norm_subject(s) for s in reference_subjects]
    if not ref_subj:
        raise ValueError("declare the held-out reference participants "
                         "(--reference-subjects)")
    overlap = sorted(set(ref_subj) & {_norm_subject(s) for s in evaluation_subjects})
    if overlap:
        raise ValueError(
            f"reference participants {overlap} are also evaluation participants; "
            "the reference subset must be held out of the evaluation")
    if "MPC_protocol_hash" not in df.columns:
        raise ValueError("the table lacks MPC_protocol_hash (not a step-2 table "
                         "of the evidence layer)")
    sel = df[df["subject"].map(_norm_subject).isin(set(ref_subj))
             & (df["session"].astype(str) == str(session))]
    hashes = sorted({str(h) for h in sel["MPC_protocol_hash"].dropna()})
    if sel.empty:
        raise ValueError(
            f"no {session!r} rows of the reference participants in the table")
    if hashes != [proto.hash]:
        raise ValueError(
            f"the table was computed under protocol hashes {hashes}, not under "
            f"{proto.name or 'the protocol'} ({proto.hash})")
    channels = declared_channels(proto)
    if channel_evidence is not None:
        mode = EVIDENCE_PER_CHANNEL
        values, ses, per = reference_from_channel_evidence(
            channel_evidence, ref_subj, session,
            channels={p: proto.channels_for(p) for p in PRINCIPLES},
            min_n=min_n, alpha=alpha)
    else:
        mode = EVIDENCE_STEP2_COLUMNS
        values, ses, per = reference_from_table(
            df, ref_subj, session, channels=channels, min_n=min_n, alpha=alpha)
    runs = one_row_per_run(sel)
    used = sorted({_norm_subject(s) for s in runs["subject"]})
    missing = sorted(set(ref_subj) - set(used))
    no_anchor = [p for p in PRINCIPLES
                 if not any(k == p or k.startswith(f"{p}:") for k in values)]
    without = channels_without_reference(per)

    def _counts(p):
        entry = per[p]
        if "channels" not in entry:
            return f"{p} n={entry['n']}"
        return ", ".join(
            f"{p}:{ch} " + (f"n={st['n']}" if st.get("reason") in (
                None, NO_REF_TOO_FEW_PARTICIPANTS, NO_REF_NOT_ABOVE_NULL)
                else st["reason"])
            for ch, st in entry["channels"].items())

    src_text = (
        f"empirical high-state reference: dataset {dataset_id or 'unspecified'}, "
        f"session {session}, held-out participants n={len(used)} "
        f"(declared {len(ref_subj)}, sha256 {_sha256_text(ref_subj)[:12]}), "
        f"protocol {proto.name} {proto.hash[:12]}; "
        + ", ".join(_counts(p) for p in PRINCIPLES)
        + f"; anchor rule: positive one-sided {1 - float(alpha):g} t lower "
        "bound; reference keys: " + (",".join(sorted(values)) or "none")
        + "; no anchor: " + (",".join(no_anchor) or "none")
    )
    reference = None
    if values:
        reference = {"kind": "external", "scale": "excess", "values": values,
                     "se": ses, "source": src_text}
    null_k = sorted({int(v) for v in pd.to_numeric(
        runs.get("MPC_null_surrogates", pd.Series(dtype=float)),
        errors="coerce").dropna()})
    return {
        "schema": SCHEMA,
        "reference": reference,
        "summary": {
            "scale": "excess",
            "min_n": int(min_n),
            "anchor_alpha": float(alpha),
            "anchor_rule": ANCHOR_RULE,
            "evidence": mode,
            "declared_channels": {p: list(c) for p, c in channels.items()},
            "reference_keys": sorted(values),
            "no_anchor": no_anchor,
            "channels_without_reference": without,
            "per_principle": per,
        },
        "provenance": {
            "dataset_id": dataset_id,
            "high_state_session": str(session),
            "reference_subjects_declared": sorted(ref_subj),
            "reference_subjects_sha256": _sha256_text(ref_subj),
            "reference_subjects_used": used,
            "reference_subjects_missing": missing,
            "evaluation_subjects_declared": sorted(
                _norm_subject(s) for s in evaluation_subjects),
            "n_runs": int(len(runs)),
            "protocol_name": proto.name,
            "protocol_hash": proto.hash,
            "protocol_reference": proto.to_dict()["reference"],
            "necessity_set": list(proto.necessity_set),
            "estimators_seen": _estimators_seen(runs),
            "null_surrogates": null_k,
            "evidence_source": dict(source or {}),
            "code_version": collect_code_version(REPO_ROOT),
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"),
            "command": None if command is None else list(command),
        },
    }


def protocol_with_reference(protocol, result: dict, name: Optional[str] = None):
    """The measurement protocol with the result's external reference."""
    proto = E.resolve_protocol(protocol)
    if not result.get("reference"):
        raise ValueError("no principle has a valid anchor; no protocol written")
    return proto.replace(
        reference=result["reference"],
        name=name or f"{proto.name or 'protocol'}+external-reference",
    )


# --------------------------------------------------------------------------
# evidence from a preprocessed layout
# --------------------------------------------------------------------------
def _dataset_params(dataset_id=None, params_json=None, modality=None) -> dict:
    from impact_pipeline.run_synergy_ci import resolve_ram_params
    from impact_pipeline.synergy_ci import PDI_PARAM_DEFAULTS, SRPI_PARAM_DEFAULTS

    params: dict = {}
    if dataset_id is not None:
        # run as a script, sys.path[0] is scripts/: run_pipeline.py (the
        # dataset configurations) lives in the repository root
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        import run_pipeline

        cfg = run_pipeline.DATASET_CONFIGS.get(str(dataset_id))
        if cfg is None and params_json is None:
            # never fall back silently to default estimator parameters
            raise ValueError(
                f"unknown --dataset-id {dataset_id!r} (known: "
                f"{sorted(run_pipeline.DATASET_CONFIGS)}); give --params-json")
        if cfg is not None:
            params = {k: cfg[k] for k in ("pdi_params", "nas_params", "srpi_params")
                      if k in cfg}
            modality = modality or cfg.get("modality")
    if params_json is not None:
        params.update(json.loads(Path(params_json).read_text(encoding="utf-8")))
    modality = str(params.pop("modality", None) or modality or "fmri")
    params.setdefault("pdi_params", dict(PDI_PARAM_DEFAULTS))
    params.setdefault("srpi_params", {**SRPI_PARAM_DEFAULTS, "modality": modality})
    params.setdefault("nas_params", None)
    params["ram_params"] = resolve_ram_params(params.get("ram_params"), modality)
    params["modality"] = modality
    return params


def compute_high_state_table(
    data_dir,
    atlas: str,
    protocol,
    reference_subjects: Sequence[str],
    session: str = "awake",
    condition: str = "audio",
    tr: Optional[float] = None,
    bids_root=None,
    dataset_id: Optional[str] = None,
    params: Optional[dict] = None,
    metrics: Optional[Sequence[str]] = None,
    null_surrogates: int = DEFAULT_NULL_SURROGATES,
    null_seed: int = 0,
    iim_kwargs: Optional[dict] = None,
    hardware_target: str = "cpu",
) -> pd.DataFrame:
    """Step-2 evidence of the reference participants' high-state runs only
    (``compute_synergy_ci`` with ``protocol``; no bootstrap: the reference SE
    is the between-participant SE). The per-channel evidence records are in
    ``df.attrs['mpc_evidence']['channel_evidence']``."""
    from impact_pipeline.synergy_ci import compute_synergy_ci

    p = dict(params or _dataset_params(dataset_id))
    modality = p.pop("modality", None)
    subjects = [_norm_subject(s) for s in reference_subjects]
    onsets = None
    if bids_root is not None:
        from impact_pipeline.run_synergy_ci import load_onsets

        onsets = {s: {session: load_onsets(bids_root, s, session,
                                           condition=condition,
                                           dataset_id=dataset_id)}
                  for s in subjects}
    if tr is None:
        from impact_pipeline.run_synergy_ci import _infer_sample_interval_seconds

        tr = _infer_sample_interval_seconds(bids_root)
    if tr is None:
        raise ValueError("an explicit --tr is required (no BIDS sidecar found)")
    df = compute_synergy_ci(
        str(data_dir), atlas, [0.5], sessions=(session,), condition=condition,
        tr=float(tr), stimulus_onsets=onsets, mpc_metrics=metrics,
        compute_ci=False, subjects=subjects, protocol=protocol,
        null_surrogates=int(null_surrogates), null_seed=int(null_seed),
        bootstrap_se=0, dataset_id=dataset_id, modality=modality,
        hardware_target=hardware_target, record_channel_evidence=True,
        **p, **dict(iim_kwargs or {}),
    )
    return df


def channel_evidence_of(df: pd.DataFrame) -> Optional[List[dict]]:
    """The per-channel evidence records of a computed table (None for a table
    read from a file, which has only the step-2 columns)."""
    ev = dict(getattr(df, "attrs", {}) or {}).get("mpc_evidence") or {}
    recs = ev.get("channel_evidence")
    return None if recs is None else list(recs)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="compute_empirical_reference.py",
        description=__doc__.split("\n\n")[0],
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--step2-table", help="step-2 table (CSV) of a pipeline run")
    src.add_argument("--data-dir", help="preprocessed layout to compute from")
    ap.add_argument("--protocol", default=str(DEFAULT_PROTOCOL),
                    help="measurement protocol of the evidence (default "
                         "protocols/mpc_default_v1.json, the default empirical "
                         "protocol; pass the derived protocol the evaluation "
                         "uses)")
    ap.add_argument("--reference-subjects", required=True,
                    help="held-out participants: comma list or @file")
    ap.add_argument("--evaluation-subjects", default=None,
                    help="evaluation participants (checked to be disjoint)")
    ap.add_argument("--session", default=None,
                    help="high-state session (default: the protocol's cohort "
                         "reference session, else 'awake')")
    ap.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--dataset-id", default=None)
    ap.add_argument("--out", required=True, help="reference JSON to write")
    ap.add_argument("--write-protocol", default=None,
                    help="also write the protocol with the external reference")
    ap.add_argument("--protocol-name", default=None)
    g = ap.add_argument_group("--data-dir mode")
    g.add_argument("--atlas", default=None)
    g.add_argument("--condition", default="audio")
    g.add_argument("--tr", type=float, default=None)
    g.add_argument("--bids-root", default=None)
    g.add_argument("--params-json", default=None,
                   help="pdi/nas/srpi/ram params and modality (JSON object)")
    g.add_argument("--modality", default=None)
    g.add_argument("--metrics", nargs="+", default=None)
    g.add_argument("--null-surrogates", type=int, default=DEFAULT_NULL_SURROGATES)
    g.add_argument("--null-seed", type=int, default=0)
    g.add_argument("--iim-bins", type=int, default=None)
    g.add_argument("--iim-lag-trs", type=int, default=None)
    g.add_argument("--iim-max-nodes", type=int, default=None)
    g.add_argument("--iim-max-timepoints", type=int, default=None)
    g.add_argument("--hardware-target", default="cpu")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    proto = E.Protocol.from_json(args.protocol)
    session = args.session or (
        proto.reference.get("session") if proto.reference["kind"] == "cohort_high_state"
        else "awake")
    ref_subj = parse_subjects(args.reference_subjects)
    eval_subj = parse_subjects(args.evaluation_subjects)
    channel_evidence = None
    if args.step2_table is not None:
        df = pd.read_csv(args.step2_table, dtype={"subject": str, "session": str})
        source = {"kind": "step2_table",
                  "path": str(Path(args.step2_table).resolve()),
                  "sha256": _sha256_file(args.step2_table)}
    else:
        if args.atlas is None:
            print("--data-dir needs --atlas", file=sys.stderr)
            return 2
        iim = {k: v for k, v in (
            ("iim_bins", args.iim_bins), ("iim_lag_trs", args.iim_lag_trs),
            ("iim_max_nodes", args.iim_max_nodes),
            ("iim_max_timepoints", args.iim_max_timepoints),
        ) if v is not None}
        df = compute_high_state_table(
            args.data_dir, args.atlas, proto, ref_subj, session=session,
            condition=args.condition, tr=args.tr, bids_root=args.bids_root,
            dataset_id=args.dataset_id,
            params=_dataset_params(args.dataset_id, args.params_json, args.modality),
            metrics=args.metrics, null_surrogates=args.null_surrogates,
            null_seed=args.null_seed, iim_kwargs=iim,
            hardware_target=args.hardware_target,
        )
        channel_evidence = channel_evidence_of(df)
        source = {"kind": "computed", "data_dir": str(Path(args.data_dir).resolve()),
                  "atlas": args.atlas, "condition": args.condition,
                  "bids_root": args.bids_root, "metrics": args.metrics,
                  "null_seed": int(args.null_seed), "iim": iim}
    result = build_reference(
        df, proto, ref_subj, session=session, evaluation_subjects=eval_subj,
        min_n=args.min_n, alpha=args.alpha, source=source,
        dataset_id=args.dataset_id, command=[sys.argv[0], *(argv or sys.argv[1:])],
        channel_evidence=channel_evidence,
    )
    if args.write_protocol and result["reference"]:
        derived = protocol_with_reference(proto, result, args.protocol_name)
        derived.to_json(args.write_protocol)
        result["provenance"]["derived_protocol"] = {
            "path": str(Path(args.write_protocol).resolve()),
            "name": derived.name, "hash": derived.hash}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(out)
    if not result["reference"]:
        print("no principle has a valid anchor (see summary.per_principle)",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
