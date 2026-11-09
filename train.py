"""Forecast next month's demand per product, turn it into a stock level, and
test it against simple ordering rules in dollars.

    python3 train.py --work data/work --out site

The decision: at the end of each month, how many units of each product should
be on the shelf next month. Being short loses the profit on each missed sale.
Having extra costs money: perishable food may spoil or be marked down, other
products tie up cash and space. Because those two costs differ, the right
stock level is not the forecast itself but a point on the forecast range:

    stock level = the demand we expect to cover in a share CR of months,
    CR = cost of being short / (cost of being short + cost of extra)

Rules compared, on the same months, scored on estimated demand:
  last month        order what sold last month (common rule of thumb)
  last month +20%   the same with a buffer
  model, sales      forecast trained on recorded sales
  model, demand     forecast trained on estimated demand (stockouts added back)
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

# Assumptions (M5 has no unit costs). Change them here; every result follows.
COSTS = {
    #            share of price that is profit, cost of one leftover unit for a month as share of unit cost
    "FOODS":     {"margin": 0.30, "extra": 0.25, "why": "some food spoils or is marked down"},
    "HOUSEHOLD": {"margin": 0.35, "extra": 0.03, "why": "keeps; cost is cash and space"},
    "HOBBIES":   {"margin": 0.40, "extra": 0.03, "why": "keeps; cost is cash and space"},
}
for c in COSTS.values():
    short, extra = c["margin"], c["extra"] * (1 - c["margin"])  # both per dollar of price
    c["cr"] = short / (short + extra)

P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0)
BUFFER = 0.20


def features(p):
    """One row per product and target month, using only what is known the
    day before the month starts (sales up to last month, the calendar, and the
    store's own price for the month)."""
    p = p.sort_values(["item_id", "month"]).copy()
    g = p.groupby("item_id")
    for col in ("sales", "demand"):
        lg = np.log1p(p[col])
        for k in (1, 2, 3, 12):
            p[f"{col}_lag{k}"] = g[col].shift(k)
        p[f"{col}_m3"] = lg.groupby(p.item_id).transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        p[f"{col}_m6"] = lg.groupby(p.item_id).transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
        p[f"{col}_m12"] = lg.groupby(p.item_id).transform(lambda s: s.shift(1).rolling(12, min_periods=1).mean())
        for k in (1, 2, 3, 12):
            p[f"{col}_lag{k}"] = np.log1p(p[f"{col}_lag{k}"])
    p["oos_last"] = g.oos_days.shift(1)
    p["oos_3"] = g.oos_days.transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    p["zero_6"] = g.sales.transform(lambda s: (s.shift(1) == 0).rolling(6, min_periods=1).sum())
    p["age"] = g.live_days.transform(lambda s: s.shift(1).cumsum()) / 30
    p["live_last"] = g.live_days.shift(1)
    p["price_prev3"] = g.price.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    p["price_change"] = p.price / p.price_prev3 - 1
    p["moy"] = p.month.dt.month
    p["dept"] = p.dept_id.astype("category").cat.codes
    p["sales_last"] = g.sales.shift(1)
    p["rev_12"] = (p.sales * p.price.fillna(0)).groupby(p.item_id).transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum())
    return p


F = (["sales_lag1", "sales_lag2", "sales_lag3", "sales_lag12", "sales_m3", "sales_m6", "sales_m12",
      "demand_lag1", "demand_lag2", "demand_lag3", "demand_lag12", "demand_m3", "demand_m6", "demand_m12",
      "oos_last", "oos_3", "zero_6", "age", "price", "price_change", "moy", "days", "events", "snap", "dept"])


def fit(X, y, alpha=None):
    kw = dict(objective="quantile", alpha=alpha) if alpha else {}
    return lgb.LGBMRegressor(**P, **kw).fit(X, y)


def fit_mean(X, y):
    """Expected units (not the median), so monthly totals are not biased low."""
    return lgb.LGBMRegressor(**P, objective="tweedie", tweedie_variance_power=1.2).fit(X, y)


FRONTIER = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def cost_table(d, q, truth="demand"):
    """Dollars lost to being short and to extra stock, for stock levels q."""
    D = d[truth].to_numpy()
    price = d.price.to_numpy()
    m = d.cat_id.map(lambda c: COSTS[c]["margin"]).to_numpy()
    e = d.cat_id.map(lambda c: COSTS[c]["extra"]).to_numpy()
    short = np.maximum(D - q, 0)
    extra = np.maximum(q - D, 0)
    return {
        "short_cost": float((short * price * m).sum()),
        "extra_cost": float((extra * price * (1 - m) * e).sum()),
        "fill_rate": float(np.minimum(q, D).sum() / D.sum()),
        "stock_value": float((q * price * (1 - m)).sum()),
        "units": float(q.sum()),
    }


def tiers(d):
    """A: products making the top 80% of last year's revenue, B: next 15%, C: the rest."""
    r = d.rev_12.fillna(0).to_numpy()
    order = np.argsort(-r)
    cum = np.cumsum(r[order]) / max(r.sum(), 1)
    t = np.empty(len(d), dtype=object)
    t[order] = np.where(cum <= 0.80, "A", np.where(cum <= 0.95, "B", "C"))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="data/work")
    ap.add_argument("--out", default="site")
    ap.add_argument("--first", default="2015-05", help="first test month")
    ap.add_argument("--last", default="2016-04", help="last test month; also the month on the order sheet")
    ap.add_argument("--store", default="CA_4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    p = features(pd.read_parquet(os.path.join(a.work, "monthly.parquet")))
    # a product is on the order sheet for a month if it was carried last month and has a price this month
    p = p[p.full_month & (p.live_last > 0) & p.price.notna() & (p.month >= "2012-01")].reset_index(drop=True)
    p["cr"] = p.cat_id.map(lambda c: COSTS[c]["cr"])

    test_months = pd.date_range(a.first, a.last, freq="MS")
    rows = []
    for t in test_months:
        tr = p[p.month <= t - pd.DateOffset(months=2)]   # a month's gap: the last month's stockouts are not settled yet
        te = p[p.month == t].copy()
        te["tier"] = tiers(te)
        te["q_last"] = np.ceil(te.sales_last)
        te["q_buffer"] = np.ceil(te.sales_last * (1 + BUFFER))
        te["mid"] = fit_mean(tr[F], tr.demand).predict(te[F])
        te["mid_sales"] = fit_mean(tr[F], tr.sales).predict(te[F])
        for al in FRONTIER:  # one stock rule for every product, to trace stock dollars against lost profit
            te[f"q_a{int(al * 100)}"] = np.ceil(np.expm1(fit(tr[F], np.log1p(tr.demand), alpha=al).predict(te[F])).clip(0))
        for target, name in (("sales", "q_model_sales"), ("demand", "q_model")):
            te[name] = 0.0
            for cat, c in COSTS.items():
                k, kt = tr.cat_id == cat, te.cat_id == cat
                m = fit(tr.loc[k, F], np.log1p(tr.loc[k, target]), alpha=c["cr"])
                te.loc[kt, name] = np.ceil(np.expm1(m.predict(te.loc[kt, F])).clip(0))
        if t == test_months[-1]:  # the order sheet also shows a likely range
            te["lo"] = np.floor(np.expm1(fit(tr[F], np.log1p(tr.demand), alpha=0.1).predict(te[F])).clip(0))
        rows.append(te)
        print(f"{t:%Y-%m} done, {len(te)} products")
    bt = pd.concat(rows, ignore_index=True)
    bt.to_parquet(os.path.join(a.work, "backtest.parquet"))

    rules = {"last": "Last month", "buffer": "Last month +20%", "model_sales": "Model on recorded sales", "model": "Model on estimated demand"}
    col = {"last": "q_last", "buffer": "q_buffer", "model_sales": "q_model_sales", "model": "q_model"}

    def score(d, truth="demand"):
        return {k: cost_table(d, d[col[k]].to_numpy(), truth) for k in rules}

    def wape(y, f):
        return float(np.abs(f - y).sum() / y.sum())

    metrics = {
        "store": a.store, "test": [a.first, a.last], "rules": rules, "buffer": BUFFER,
        "costs": {k: {kk: v[kk] for kk in ("margin", "extra", "cr", "why")} for k, v in COSTS.items()},
        "rows": int(len(bt)), "products": int(bt.item_id.nunique()),
        "overall": score(bt), "overall_on_sales": score(bt, "sales"),
        "by_tier": {t: score(bt[bt.tier == t]) for t in "ABC"},
        "by_cat": {c: score(bt[bt.cat_id == c]) for c in COSTS},
        "frontier": [{"cover": al, **cost_table(bt, bt[f"q_a{int(al * 100)}"].to_numpy())} for al in FRONTIER],
        "same_stock": None,
        "tier_products": {t: float(bt[bt.tier == t].groupby("month").size().mean()) for t in "ABC"},
        "by_month": [{"month": f"{m:%Y-%m}", **{k: (lambda r: r["short_cost"] + r["extra_cost"])(cost_table(g, g[col[k]].to_numpy())) for k in rules}}
                     for m, g in bt.groupby("month")],
        "accuracy": {
            "wape_last": wape(bt.demand, bt.sales_last), "wape_model": wape(bt.demand, bt.mid),
            "bias_last": float(bt.sales_last.sum() / bt.demand.sum() - 1), "bias_model": float(bt.mid.sum() / bt.demand.sum() - 1),
            "bias_model_sales": float(bt.mid_sales.sum() / bt.demand.sum() - 1),
            "by_tier": {t: {"wape_last": wape(g.demand, g.sales_last), "wape_model": wape(g.demand, g.mid), "products": int(g.item_id.nunique())}
                        for t, g in bt.groupby("tier")},
        },
        "stockouts": {
            "sales": float(p[p.month < a.first].sales.sum()), "demand": float(p[p.month < a.first].demand.sum()),
            "share_test_rows_with_oos": float((bt.oos_days > 0).mean()),
        },
    }
    metrics["same_stock"] = same_stock(metrics)
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    export(a, metrics, bt[bt.month == test_months[-1]].copy(), pd.read_parquet(os.path.join(a.work, "monthly.parquet")))
    o = metrics["overall"]
    for k, n in rules.items():
        print(f"{n:28s} short ${o[k]['short_cost']:>10,.0f}  extra ${o[k]['extra_cost']:>9,.0f}  "
              f"total ${o[k]['short_cost'] + o[k]['extra_cost']:>10,.0f}  fill {o[k]['fill_rate']:.1%}  stock ${o[k]['stock_value']:>11,.0f}")
    for f in metrics["frontier"]:
        print(f"cover {f['cover']:.2f}  short ${f['short_cost']:>9,.0f}  extra ${f['extra_cost']:>8,.0f}  fill {f['fill_rate']:.1%}  stock ${f['stock_value']:>11,.0f}")
    for tier, sc in metrics["by_tier"].items():
        print(tier, {k: round(v["fill_rate"], 3) for k, v in sc.items()}, {k: round(v["short_cost"] + v["extra_cost"]) for k, v in sc.items()})
    print("on recorded sales:", {k: round(v["short_cost"] + v["extra_cost"]) for k, v in metrics["overall_on_sales"].items()})
    print(json.dumps(metrics["accuracy"], indent=1), metrics["tier_products"])


def same_stock(metrics):
    """Lost profit and extra-stock cost the model would have at the same stock
    dollars as each simple rule (straight line between the two nearest points)."""
    fr = sorted(metrics["frontier"], key=lambda f: f["stock_value"])
    xs = [f["stock_value"] for f in fr]
    out = {}
    for k in ("last", "buffer"):
        b = metrics["overall"][k]
        x = b["stock_value"]
        if not xs[0] <= x <= xs[-1]:
            continue
        j = max(i for i in range(len(xs) - 1) if xs[i] <= x)
        w = (x - xs[j]) / (xs[j + 1] - xs[j])
        mix = {m: fr[j][m] + w * (fr[j + 1][m] - fr[j][m]) for m in ("short_cost", "extra_cost", "fill_rate")}
        out[k] = {"stock_value": x, "rule_total": b["short_cost"] + b["extra_cost"], "rule_fill": b["fill_rate"],
                  "model_total": mix["short_cost"] + mix["extra_cost"], "model_fill": mix["fill_rate"]}
    return out


def reasons(r):
    """Plain reasons for an order, from what the owner can check themselves."""
    out = []
    last = r.sales_last
    if r.oos_last >= 3:
        out.append(f"Likely out of stock for about {int(r.oos_last)} days last month, so last month's sales understate demand")
    if r.age < 6:
        out.append("New product with under 6 months of sales, so the range is wide")
    if abs(r.price_change) >= 0.08:
        out.append(f"Price {'drops' if r.price_change < 0 else 'rises'} {abs(r.price_change):.0%} against the last 3 months")
    ly = np.expm1(r.demand_lag12) if pd.notna(r.demand_lag12) else None
    m3 = np.expm1(r.demand_m3)
    if ly is not None and m3 > 2 and ly > 1.3 * m3:
        out.append(f"Same month last year sold {ly:.0f}, above the recent pace of {m3:.0f} a month")
    elif ly is not None and m3 > 2 and ly < 0.7 * m3:
        out.append(f"Same month last year sold only {ly:.0f}, below the recent pace of {m3:.0f} a month")
    if r.cat_id == "FOODS":
        out.append("Food: ordering closer to the forecast, because leftovers may spoil")
    else:
        out.append("Keeps on the shelf: ordering above the forecast, because a leftover unit costs little")
    if last == 0 and r.q_model > 0:
        out.insert(0, "Sold nothing last month, but the model expects some demand")
    return out


def export(a, metrics, sheet, monthly):
    asof = pd.Timestamp(a.last)
    hist = monthly[(monthly.month >= asof - pd.DateOffset(months=23)) & (monthly.month <= asof)]
    hist = {k: g.sort_values("month") for k, g in hist.groupby("item_id")}
    items = []
    for _, r in sheet.iterrows():
        h = hist.get(r.item_id)
        cost = r.price * (1 - COSTS[r.cat_id]["margin"])
        items.append({
            "id": r.item_id, "dept": r.dept_id, "cat": r.cat_id, "tier": r.tier, "price": round(float(r.price), 2),
            "unit_cost": round(float(cost), 2), "cover": round(COSTS[r.cat_id]["cr"], 3),
            "last": float(r.sales_last), "forecast": round(float(r.mid), 1), "low": float(r.lo), "high": float(r.q_a90),
            "order": float(r.q_model), "covers": {str(int(al * 100)): float(r[f"q_a{int(al * 100)}"]) for al in FRONTIER}, "order_last": float(r.q_last), "order_buffer": float(r.q_buffer),
            "actual_sales": float(r.sales), "actual_demand": round(float(r.demand), 1), "oos_days": float(r.oos_days),
            "reasons": reasons(r),
            "history": [] if h is None else [{"m": f"{m:%Y-%m}", "s": int(s), "d": round(float(d), 1), "o": int(o)}
                                             for m, s, d, o in zip(h.month, h.sales, h.demand, h.oos_days)],
        })
    items.sort(key=lambda d: -d["order"] * d["unit_cost"])
    data = {"store": a.store, "asof": f"{asof:%Y-%m}", "metrics": metrics, "items": items}
    with open(os.path.join(a.out, "data.json"), "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"order sheet: {len(items)} products, {os.path.getsize(os.path.join(a.out, 'data.json')) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
