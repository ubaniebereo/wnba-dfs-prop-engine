"""Scanner service: pull props + context, compute features, write current scan.

Decoupled from the Dash UI (Section 3A). Run once (`run_scan`) or on a schedule
(`run_loop` via APScheduler). Writes `current_prop_scan` + `current_pair_scan` to
feeds.sqlite and appends odds history to `odds_snapshots` for line movement.
"""

from __future__ import annotations

import time
import uuid

import pandas as pd

import anchor_lines
import pair_engine
from feeds.normalizers import norm_name
from feeds.storage import ENGINE as FEEDS_DB, init_storage, insert
from feeds.x_client import XClient
from feeds.normalizers import PlayerResolver
from propmodel import injuries, models, prop_model
from propmodel.featureset import STATS, build
from propmodel import live_edge
from scanner import compute_scan_features as csf
from src.database import get_engine
from src.utils import get_logger

log = get_logger(__name__)

SEVERITY = {"out": 1.0, "doubtful": 0.8, "minutes_cap": 0.7, "questionable": 0.6,
            "starting": 0.3, "available": 0.2}

import joblib
from src.config import DATA_DIR

# The heavy bundle (models, params, projections, correlations) only changes when
# new games complete -> cache it to DISK so repeated scans reuse it. Only odds
# change minute-to-minute, and those are fetched fresh each scan (fast).
_BUNDLE_PATH = DATA_DIR / "scan_bundle.joblib"
_BUNDLE_TTL = 1800          # 30 min; bump after a slate completes
_BUNDLE_MEM: dict = {}


def _heavy_bundle(eng):
    """models + params + projections + teammate correlations, disk-cached."""
    now = time.time()
    if _BUNDLE_MEM.get("ts", 0) > now - _BUNDLE_TTL:
        return _BUNDLE_MEM["data"]
    if _BUNDLE_PATH.exists() and _BUNDLE_PATH.stat().st_mtime > now - _BUNDLE_TTL:
        data = joblib.load(_BUNDLE_PATH)
        _BUNDLE_MEM.update(ts=data["built_at"], data=data)
        log.info("loaded scan bundle from disk")
        return data
    log.info("rebuilding scan bundle (models + projections + correlations)...")
    from src.config import MODELS_DIR
    from propmodel.family_calibration import sim_params_from_store
    fam_path = MODELS_DIR / "family_calibration.joblib"
    store = None
    if fam_path.exists():                       # prefer 7-family calibrated models
        store = joblib.load(fam_path)
        mm, rms = store["minutes"], store["rates"]
        params = sim_params_from_store(store)
        log.info("using per-family calibrated models (%d families)", len(rms))
    else:                                       # fallback: 3-stat models
        feat = build(eng)
        d = feat[feat.history >= 5].dropna(subset=["minutes"])
        tr = d.iloc[:int(len(d) * 0.8)]
        mm = models.train_minutes(d)
        rms = {s: models.train_rate(d, s) for s in STATS}
        params = prop_model.build_params(tr, models.project(tr, mm, rms), float(mm["sd"] ** 2))
    proj = live_edge.project_upcoming(eng, mm, rms, days_ahead=2,
                                      out_ids=injuries.out_player_ids())
    corr = pair_engine.teammate_correlations("points")
    data = {"mm": mm, "rms": rms, "params": params, "proj": proj, "corr": corr,
            "store": store, "built_at": now}
    joblib.dump(data, _BUNDLE_PATH)
    _BUNDLE_MEM.update(ts=now, data=data)
    return data


def _news_map() -> dict:
    try:
        x = XClient()
        resolver = PlayerResolver()
        posts = x.fetch_recent_posts(max_results=25)
        out = {}
        for ev in x.to_news_events(posts, resolver):
            if not ev.get("player_name"):
                continue
            k = norm_name(ev["player_name"])
            out[k] = {"status": ev["parsed_status"],
                      "severity": SEVERITY.get(ev["parsed_status"], 0.3)}
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("news fetch failed: %s", e)
        return {}


def _open_lines() -> dict:
    """Earliest snapshot odds per (player, market, side, line) today -> open line."""
    try:
        snaps = pd.read_sql("SELECT player_name, market_type, side, line_value, "
                            "odds_american, captured_at FROM odds_snapshots "
                            "ORDER BY captured_at", FEEDS_DB)
        if snaps.empty:
            return {}
        first = snaps.groupby(["player_name", "market_type", "side", "line_value"]).first()
        return {(i[0], i[1].replace("player_", ""), i[2], i[3]): r["odds_american"]
                for i, r in first.iterrows()}
    except Exception:
        return {}


def run_scan() -> dict:
    init_storage()
    eng = get_engine()
    scan_id = uuid.uuid4().hex
    scanned_at = pd.Timestamp.utcnow().isoformat()

    props = anchor_lines.fetch_props_multi_region()
    if props.empty:
        log.info("no live props"); return {"props": 0}
    # append snapshot history (for movement) via the feeds client schema
    from feeds.odds_api_client import snapshot_props
    insert("odds_snapshots", snapshot_props())

    anchor = anchor_lines.build_anchor(props)
    bundle = _heavy_bundle(eng)              # disk-cached: models + proj + corr
    proj, params, corr = bundle["proj"], bundle["params"], bundle["corr"]
    news_map = _news_map()

    # pairs — DIFFERENT-TEAM only (same-game opponents carry real cross-team corr).
    # cross-team (game-environment) correlation is smaller than same-team, so the
    # mispricing threshold is lower (~1.5%) than the old same-team 3%.
    pairs = pair_engine.find_pairs(anchor, proj, props,
                                   opp_corr=pair_engine.opponent_correlation(),
                                   min_abs_mispricing=0.015)
    pair_players = {}
    if not pairs.empty:
        for r in pairs.itertuples():
            pair_players[norm_name(r.player_a)] = r.direction.split("(")[0].strip()
            pair_players[norm_name(r.player_b)] = r.direction.split("(")[0].strip()
        pairs = pairs.assign(scan_id=scan_id, scanned_at=scanned_at)
        pairs.to_sql("current_pair_scan", FEEDS_DB, if_exists="replace", index=False)

    scan = csf.compute(props, anchor, proj, params, news_map, pair_players,
                       _open_lines(), scanned_at)
    if scan.empty:
        return {"props": len(props), "scan_rows": 0}
    scan.insert(0, "scan_id", scan_id)
    scan["reason_tags_json"] = scan["reason_tags_json"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else x)
    # reasoning layer: grounded English summaries + evidence-based final decision
    try:
        from reasoning.decision_engine import annotate
        scan = annotate(scan)
    except Exception as e:  # noqa: BLE001
        log.warning("reasoning annotate failed: %s", e)
    scan.to_sql("current_prop_scan", FEEDS_DB, if_exists="replace", index=False)

    # --- PrizePicks (DFS pick'em) comparison vs sharp market + model ---
    n_pp = 0
    try:
        from feeds.prizepicks_client import fetch_prizepicks_wnba
        from scanner.prizepicks_scan import compute_pp_scan
        pp = fetch_prizepicks_wnba()
        pp_scan = compute_pp_scan(pp, props, proj, params, scanned_at,
                                  store=bundle.get("store"))
        if not pp_scan.empty:
            pp_scan.to_sql("prizepicks_scan", FEEDS_DB, if_exists="replace", index=False)
            n_pp = len(pp_scan)
            _build_entries(pp_scan, scan_id, scanned_at)
    except Exception as e:  # noqa: BLE001
        log.warning("prizepicks scan failed: %s", e)

    log.info("scan written: %d prop rows, %d pairs, %d PrizePicks", len(scan), len(pairs), n_pp)
    return {"props": len(props), "scan_rows": len(scan), "pairs": len(pairs),
            "standouts": int((scan["standout_score"] >= 20).sum()), "prizepicks": n_pp}


def _build_entries(pp_scan, scan_id, scanned_at):
    """Build ranked 2-4 leg PrizePicks entries from priced standard lines."""
    import json
    from dfs_ev import build_entries
    df = pp_scan[(pp_scan["demon_goblin_flag"] == 0)
                 & pp_scan["model_prob_over"].notna()].copy()
    # CALIBRATION PLACEHOLDER: the model is ~market-efficient, so its deviations
    # from 0.5 are mostly noise. Shrink hard toward the market before building
    # entries, else uncalibrated overconfidence manufactures fake +EV.
    # TODO: replace with per-prop-family isotonic calibration vs realized outcomes.
    SHRINK = 0.30
    legs = []
    for r in df.itertuples():
        p_cal = 0.5 + (float(r.model_prob_over) - 0.5) * SHRINK
        side = "more" if p_cal >= 0.5 else "less"
        legs.append({"player": r.player_name, "market": r.market, "side": side,
                     "line": r.pp_line, "prob": max(p_cal, 1 - p_cal)})
    entries = build_entries(legs, max_size=4, payout_style="power", top=20)
    if not entries:
        return
    rows = [{"scan_id": scan_id, "scanned_at": scanned_at, "entry_size": e["entry_size"],
             "legs_json": json.dumps(e["legs"]), "payout_style": e["payout_style"],
             "payout_multiple": e.get("payout_if_perfect"),
             "joint_prob_all": e["joint_prob_all"], "entry_ev": e["EV_per_$1"],
             "risk_tier": e["risk_tier"],
             "entry_verdict": "+EV" if e["EV_per_$1"] > 0 else "-EV"} for e in entries]
    pd.DataFrame(rows).to_sql("current_entries", FEEDS_DB, if_exists="replace", index=False)


def run_loop(interval_seconds=120):
    from apscheduler.schedulers.blocking import BlockingScheduler
    sched = BlockingScheduler()
    sched.add_job(run_scan, "interval", seconds=interval_seconds,
                  next_run_time=pd.Timestamp.now())
    log.info("scanner loop every %ds (Ctrl-C to stop)", interval_seconds)
    sched.start()


if __name__ == "__main__":
    print(run_scan())
