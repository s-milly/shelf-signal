"""Turn the M5 daily sales of one store into a monthly table, one row per
product per month.

    python3 prep.py --store CA_4

Why the extra work: a day with zero sales can mean nobody wanted the product,
or that the shelf was empty. If we forecast from raw sales, every stockout
teaches the model that demand was low, and the next order is too small again.

So we flag "likely out of stock" days: a run of zero-sales days that is too
long to be bad luck for that product. If a product had no sales on a share z
of the 91 days before the run, the chance of L zero days in a row by luck is
about z ** L. Runs where that chance is under 1% are flagged. Using the
product's own zero-day share (not its average sales) keeps slow, lumpy
sellers from being flagged just for being lumpy. Runs that last
past the end of the data or longer than 120 days are treated as the product
not being carried, not as a stockout.

Estimated demand = sales + r * (flagged days). Both raw sales and estimated
demand are kept so the two can be compared.
"""
import argparse
import os

import numpy as np
import pandas as pd

MAX_RUN = 120   # longer zero runs: product not carried, not a stockout
LUCK = 0.01     # flag a zero run if a run that long had under 1% chance by luck


def zero_runs(row, first):
    """Start and end (inclusive) of every zero run after the first sale."""
    z = np.zeros(len(row) + 2, dtype=np.int8)
    z[1 + first:-1] = row[first:] == 0
    d = np.diff(z)
    return np.where(d == 1)[0], np.where(d == -1)[0] - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--store", default="CA_4")
    ap.add_argument("--out", default="data/work")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cal = pd.read_csv(os.path.join(a.raw, "calendar.csv"), parse_dates=["date"])
    s = pd.read_csv(os.path.join(a.raw, "sales_train_evaluation.csv"))
    s = s[s.store_id == a.store].reset_index(drop=True)
    days = [c for c in s.columns if c.startswith("d_")]
    cal = cal[cal.d.isin(days)].reset_index(drop=True)
    X = s[days].to_numpy(dtype=np.float32)
    n_items, n_days = X.shape

    # ---- flag likely out-of-stock days, item by item
    oos = np.zeros_like(X, dtype=bool)        # flagged day
    oos_rate = np.zeros_like(X)               # demand per day we add back
    runs = []
    for i in range(n_items):
        row = X[i]
        if not (row > 0).any():
            continue
        first = int((row > 0).argmax())
        for st, en in zip(*zero_runs(row, first)):
            L = en - st + 1
            if en == n_days - 1 or L > MAX_RUN:
                continue
            before = row[max(first, st - 91):st]
            if len(before) < 28 or L < 3:
                continue
            z = ((before == 0).sum() + 1) / (len(before) + 2)  # never exactly 0 or 1
            if z ** L >= LUCK:
                continue
            r = before.mean()
            oos[i, st:en + 1] = True
            oos_rate[i, st:en + 1] = r
            runs.append((i, st, en, r))
    print(f"{a.store}: {n_items} products, {len(runs)} likely stockouts, {int(oos.sum())} product-days flagged")

    # ---- daily to monthly
    month = cal.date.dt.to_period("M")
    months = month.unique()
    mid = pd.factorize(month)[0]
    M = len(months)

    def by_month(arr):
        out = np.zeros((n_items, M), dtype=np.float32)
        for k in range(M):
            out[:, k] = arr[:, mid == k].sum(1)
        return out

    sales_m = by_month(X)
    oos_m = by_month(oos.astype(np.float32))
    demand_m = sales_m + by_month(oos_rate * oos)
    first_day = np.where((X > 0).any(1), (X > 0).argmax(1), n_days)
    live_m = np.zeros((n_items, M), dtype=np.float32)
    for k in range(M):
        idx = np.where(mid == k)[0]
        live_m[:, k] = np.clip(idx[-1] + 1 - first_day, 0, len(idx))

    # ---- calendar facts for each month (known in advance)
    c = cal.assign(month=month)
    cm = c.groupby("month").agg(
        days=("d", "size"),
        events=("event_name_1", lambda x: x.notna().sum()),
        snap=(f"snap_{a.store[:2]}", "sum"))

    # ---- monthly price (the store sets it, so next month's price is known)
    pr = pd.read_csv(os.path.join(a.raw, "sell_prices.csv"))
    pr = pr[pr.store_id == a.store]
    wk = c.groupby("wm_yr_wk").month.agg(lambda x: x.mode().iloc[0])
    pr["month"] = pr.wm_yr_wk.map(wk)
    price = pr.groupby(["item_id", "month"]).sell_price.mean()

    rows = []
    for i in range(n_items):
        for k in range(M):
            rows.append((s.item_id[i], s.dept_id[i], s.cat_id[i], months[k], sales_m[i, k],
                         demand_m[i, k], oos_m[i, k], live_m[i, k]))
    p = pd.DataFrame(rows, columns=["item_id", "dept_id", "cat_id", "month", "sales", "demand", "oos_days", "live_days"])
    p = p.join(cm, on="month")
    p["price"] = p.set_index(["item_id", "month"]).index.map(price.to_dict().get)
    p["full_month"] = p.days >= 28  # the data starts on Jan 29, 2011 and ends on Jun 19, 2016
    p["month"] = p.month.dt.to_timestamp()
    p.to_parquet(os.path.join(a.out, "monthly.parquet"))
    pd.DataFrame(runs, columns=["item", "start", "end", "rate"]).assign(
        item_id=lambda d: s.item_id.to_numpy()[d.item]).to_parquet(os.path.join(a.out, "stockouts.parquet"))

    tot = p[p.full_month]
    print(f"months {months[0]} to {months[-1]}; full-month sales {tot.sales.sum():,.0f}, "
          f"estimated demand {tot.demand.sum():,.0f} (+{tot.demand.sum() / tot.sales.sum() - 1:.1%})")


if __name__ == "__main__":
    main()
