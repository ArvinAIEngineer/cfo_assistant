"""
CFO analytics engine: exact calculations, no AI involved.

Input : items (from Business Central) and posted sales invoice lines for a period.
Output: P&L by product group and by line of business, and deterministic answers to
        a small set of question types. The AI only chooses WHICH question type;
        every number is computed here.
"""

import re
from typing import Any, Dict, List, Optional

import pandas as pd

CODE_RE = re.compile(r"^(?P<lob>[A-Za-z]+\d+)-(?P<pg>[A-Za-z]+\d+)$")

METRICS = {
    "revenue": "Revenue",
    "cogs": "COGS",
    "gross_profit": "Gross profit",
    "margin": "Gross margin %",
    "avg_margin": "Average gross margin %",
    "margin_range": "Margin spread (max - min)",
}


def items_frame(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for it in items:
        no = str(it.get("number") or "")
        m = CODE_RE.match(no)
        if not m:
            continue
        cost, price = float(it.get("unitCost") or 0), float(it.get("unitPrice") or 0)
        rows.append({
            "item_no": no,
            "description": it.get("displayName") or no,
            "lob": m.group("lob").upper(),
            "pg": m.group("pg").upper(),
            "unit_cost": cost,
            "unit_price": price,
            "list_margin_pct": (price - cost) / price * 100 if price else None,
        })
    return pd.DataFrame(rows, columns=["item_no", "description", "lob", "pg", "unit_cost", "unit_price", "list_margin_pct"])


def sales_frame(lines: List[Dict[str, Any]], items: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(lines)
    if df.empty or items.empty:
        return pd.DataFrame(columns=["item_no", "description", "lob", "pg", "quantity", "revenue", "cogs", "gross_profit"])
    df = df.merge(items, on="item_no", how="inner")
    df["quantity"] = df["quantity"].astype(float)
    df["revenue"] = df["net_amount"].astype(float)
    df["cogs"] = df["quantity"] * df["unit_cost"]
    df["gross_profit"] = df["revenue"] - df["cogs"]
    return df


def pg_summary(sales: pd.DataFrame) -> pd.DataFrame:
    if sales.empty:
        return pd.DataFrame(columns=["lob", "pg", "item_no", "description", "quantity", "revenue", "cogs", "gross_profit", "margin_pct"])
    g = (sales.groupby(["lob", "pg", "item_no", "description"], as_index=False)
              [["quantity", "revenue", "cogs", "gross_profit"]].sum())
    g["margin_pct"] = g.apply(lambda r: r.gross_profit / r.revenue * 100 if r.revenue else None, axis=1)
    return g.sort_values(["lob", "pg"]).reset_index(drop=True)


def lob_summary(pg: pd.DataFrame) -> pd.DataFrame:
    if pg.empty:
        return pd.DataFrame(columns=["lob", "revenue", "cogs", "gross_profit", "avg_margin_simple_pct",
                                     "margin_weighted_pct", "min_margin_pct", "max_margin_pct", "margin_spread_pts", "product_groups"])
    g = pg.groupby("lob").agg(
        revenue=("revenue", "sum"), cogs=("cogs", "sum"), gross_profit=("gross_profit", "sum"),
        avg_margin_simple_pct=("margin_pct", "mean"), min_margin_pct=("margin_pct", "min"),
        max_margin_pct=("margin_pct", "max"), product_groups=("pg", "count")).reset_index()
    g["margin_weighted_pct"] = g["gross_profit"] / g["revenue"] * 100
    g["margin_spread_pts"] = g["max_margin_pct"] - g["min_margin_pct"]
    cols = ["lob", "revenue", "cogs", "gross_profit", "avg_margin_simple_pct", "margin_weighted_pct",
            "min_margin_pct", "max_margin_pct", "margin_spread_pts", "product_groups"]
    return g[cols].sort_values("lob").reset_index(drop=True)


def _label(metric: str) -> str:
    return METRICS[metric] if metric == "cogs" else METRICS[metric].lower()


def _lob_name(code: str) -> str:
    digits = re.sub(r"\D", "", code)
    return f"Line of Business {int(digits)}" if digits else code


def _n(v: float) -> str:
    return f"{v:,.0f}" if abs(v - round(v)) < 0.005 else f"{v:,.2f}"


def _p(v: float) -> str:
    return f"{v:.2f}%"


def answer(spec: Dict[str, Any], pg: pd.DataFrame, lob: pd.DataFrame, period_label: str) -> Dict[str, Any]:
    metric = spec.get("metric")
    level = spec.get("level") or "lob"
    order = spec.get("order") or "highest"
    n = max(1, min(int(spec.get("n") or 1), 20))
    only_lob = (spec.get("lob") or "").upper() or None

    if metric not in METRICS:
        return {"error": True, "text": f"I can't calculate '{metric}'. I can answer revenue, COGS, gross profit and margin questions."}
    if level not in ("lob", "product_group") or order not in ("highest", "lowest"):
        return {"error": True, "text": "I couldn't tell what to rank or in which direction. Try e.g. 'which line of business has the highest profit?'"}
    if pg.empty:
        return {"error": True, "text": f"No posted sales for the POC items in {period_label}."}

    if level == "product_group":
        if metric in ("avg_margin", "margin_range"):
            metric = "margin"
        df = pg if not only_lob else pg[pg["lob"] == only_lob]
        col = {"revenue": "revenue", "cogs": "cogs", "gross_profit": "gross_profit", "margin": "margin_pct"}[metric]
        df = df.sort_values(col, ascending=(order == "lowest")).head(n)
        parts = []
        for r in df.itertuples():
            val = _p(r.margin_pct) if col == "margin_pct" else _n(getattr(r, col))
            detail = f"gross margin {_p(r.margin_pct)}" if col != "margin_pct" else f"gross profit {_n(r.gross_profit)} on revenue {_n(r.revenue)}"
            parts.append(f"{r.description} ({r.item_no}): {val} - {detail}")
        word = "highest" if order == "highest" else "lowest"
        head = (f"Product group with the {word} {_label(metric)}, {period_label}" if len(df) == 1
                else f"{len(df)} product groups with the {word} {_label(metric)}, {period_label}")
        table = df[["lob", "pg", "item_no", "description", "quantity", "revenue", "cogs", "gross_profit", "margin_pct"]]
        return {"text": head + ":\n" + "\n".join(f"- {p}" for p in parts), "table": table,
                "entities": [{"code": r.item_no, "name": r.description} for r in df.itertuples()], "sort_col": col}

    col = {"revenue": "revenue", "cogs": "cogs", "gross_profit": "gross_profit",
           "margin": "avg_margin_simple_pct", "avg_margin": "avg_margin_simple_pct",
           "margin_range": "margin_spread_pts"}[metric]
    df = lob.sort_values(col, ascending=(order == "lowest")).head(n)
    parts = []
    for r in df.itertuples():
        if metric in ("margin", "avg_margin"):
            parts.append(f"{r.lob}: simple average of product-group margins {_p(r.avg_margin_simple_pct)}; "
                         f"total profit / total revenue {_p(r.margin_weighted_pct)}")
        elif metric == "margin_range":
            parts.append(f"{r.lob}: margins range from {_p(r.min_margin_pct)} to {_p(r.max_margin_pct)} "
                         f"(spread {r.margin_spread_pts:.2f} points)")
        elif metric == "cogs":
            parts.append(f"{r.lob}: COGS {_n(r.cogs)} against revenue {_n(r.revenue)}")
        elif metric == "revenue":
            parts.append(f"{r.lob}: revenue {_n(r.revenue)} (gross profit {_n(r.gross_profit)})")
        else:
            parts.append(f"{r.lob}: gross profit {_n(r.gross_profit)} (revenue {_n(r.revenue)}, COGS {_n(r.cogs)})")
    word = "highest" if order == "highest" else "lowest"
    head = (f"Line of business with the {word} {_label(metric)}, {period_label}" if len(df) == 1
            else f"{len(df)} lines of business with the {word} {_label(metric)}, {period_label}")
    return {"text": head + ":\n" + "\n".join(f"- {p}" for p in parts), "table": df,
            "entities": [{"code": r.lob, "name": _lob_name(r.lob)} for r in df.itertuples()],
            "sort_col": col}


def matches_expected(entities: List[Dict[str, str]], expected: str) -> bool:
    exp = (expected or "").lower()
    if not entities:
        return False
    return all(e["code"].lower() in exp or e["name"].lower() in exp for e in entities)


SCOPE_RE = re.compile(r"^[A-Za-z]+\d+(-[A-Za-z]+\d+)?$")


def _scope(scope: Optional[str], items: pd.DataFrame):
    code = (scope or "all").strip().upper()
    if code in ("ALL", "", "COMPANY", "TOTAL"):
        return "all", "ALL", "All lines of business"
    if not SCOPE_RE.match(code):
        raise ValueError(f"'{scope}' is not a line of business or product group code (e.g. LOB02 or LOB02-PG01).")
    if "-" in code:
        hit = items[items["item_no"].str.upper() == code]
        if hit.empty:
            raise ValueError(f"There is no product group '{code}' among the items.")
        return "pg", code, hit.iloc[0]["description"]
    if code not in set(items["lob"]):
        raise ValueError(f"There is no line of business '{code}' among the items.")
    return "lob", code, _lob_name(code)


def totals(pg: pd.DataFrame, kind: str, code: str) -> Dict[str, Any]:
    df = pg if kind == "all" else (pg[pg["lob"] == code] if kind == "lob" else pg[pg["item_no"].str.upper() == code])
    rev, cogs, gp = df["revenue"].sum(), df["cogs"].sum(), df["gross_profit"].sum()
    return {"revenue": rev, "cogs": cogs, "gross_profit": gp, "quantity": df["quantity"].sum(),
            "margin_weighted_pct": gp / rev * 100 if rev else None,
            "avg_margin_simple_pct": df["margin_pct"].mean() if (kind != "pg" and not df.empty) else None,
            "rows": df}


def _fmt_metric(metric: str, t: Dict[str, Any], kind: str) -> str:
    if metric in ("margin", "avg_margin"):
        if t["margin_weighted_pct"] is None:
            return "no margin (no revenue)"
        if kind == "pg" or t["avg_margin_simple_pct"] is None:
            return f"gross margin {_p(t['margin_weighted_pct'])}"
        return (f"total profit / total revenue {_p(t['margin_weighted_pct'])}; "
                f"simple average of product-group margins {_p(t['avg_margin_simple_pct'])}")
    if metric == "cogs":
        return f"COGS {_n(t['cogs'])} (revenue {_n(t['revenue'])})"
    if metric == "revenue":
        return f"revenue {_n(t['revenue'])} (gross profit {_n(t['gross_profit'])})"
    return f"gross profit {_n(t['gross_profit'])} (revenue {_n(t['revenue'])}, COGS {_n(t['cogs'])})"


def answer_total(spec: Dict[str, Any], pg: pd.DataFrame, items: pd.DataFrame, period_label: str) -> Dict[str, Any]:
    metric = spec.get("metric")
    if metric not in ("revenue", "cogs", "gross_profit", "margin", "avg_margin"):
        return {"error": True, "text": f"I can't give a total for '{metric}'. I can total revenue, COGS, gross profit and margin."}
    try:
        kind, code, name = _scope(spec.get("scope"), items)
    except ValueError as e:
        return {"error": True, "text": str(e)}
    if pg.empty:
        return {"error": True, "text": f"No posted sales for the POC items in {period_label}."}
    t = totals(pg, kind, code)
    if t["rows"].empty:
        return {"error": True, "text": f"No posted sales for {name} in {period_label}."}
    who = "Company total (all lines of business)" if kind == "all" else f"{name} ({code})"
    table = t["rows"][["lob", "pg", "item_no", "description", "quantity", "revenue", "cogs", "gross_profit", "margin_pct"]]
    return {"text": f"{who}, {period_label}: {_fmt_metric(metric, t, kind)}.", "table": table,
            "entities": [{"code": code, "name": name}]}


def _change(metric: str, cur: Dict[str, Any], prev: Dict[str, Any]) -> str:
    if metric in ("margin", "avg_margin"):
        a, b = prev["margin_weighted_pct"], cur["margin_weighted_pct"]
        if a is None or b is None:
            return "change can't be calculated (no revenue in one of the months)"
        return f"{_p(a)} -> {_p(b)} ({b - a:+.2f} points)"
    a, b = prev[metric], cur[metric]
    pct = f", {(b - a) / abs(a) * 100:+.1f}%" if a else ""
    return f"{_n(a)} -> {_n(b)} ({b - a:+,.2f}{pct})"


def answer_compare(spec: Dict[str, Any], cur_pg: pd.DataFrame, prev_pg: pd.DataFrame, items: pd.DataFrame,
                   cur_label: str, prev_label: str) -> Dict[str, Any]:
    metric = spec.get("metric")
    if metric not in ("revenue", "cogs", "gross_profit", "margin", "avg_margin"):
        return {"error": True, "text": f"I can't compare '{metric}' between months."}
    try:
        kind, code, name = _scope(spec.get("scope"), items)
    except ValueError as e:
        return {"error": True, "text": str(e)}
    cur, prev = totals(cur_pg, kind, code), totals(prev_pg, kind, code)
    who = "Company total" if kind == "all" else f"{name} ({code})"
    if cur["rows"].empty and prev["rows"].empty:
        return {"error": True, "text": f"No posted sales for {who} in {prev_label} or {cur_label}."}
    if prev["rows"].empty:
        return {"error": True, "text": f"No posted sales for {who} in {prev_label}, so there is nothing to compare "
                                       f"{cur_label} against. {cur_label}: {_fmt_metric(metric, cur, kind)}."}
    if cur["rows"].empty:
        return {"error": True, "text": f"No posted sales for {who} in {cur_label}. {prev_label}: {_fmt_metric(metric, prev, kind)}."}
    label = "gross margin (profit / revenue)" if metric in ("margin", "avg_margin") else _label(metric)
    table = pd.DataFrame([
        {"period": prev_label, "revenue": prev["revenue"], "cogs": prev["cogs"], "gross_profit": prev["gross_profit"], "margin_pct": prev["margin_weighted_pct"]},
        {"period": cur_label, "revenue": cur["revenue"], "cogs": cur["cogs"], "gross_profit": cur["gross_profit"], "margin_pct": cur["margin_weighted_pct"]},
    ])
    return {"text": f"{who}, {label}: {prev_label} vs {cur_label}: {_change(metric, cur, prev)}.", "table": table,
            "entities": [{"code": code, "name": name}]}
