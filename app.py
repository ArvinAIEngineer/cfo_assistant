"""
CFO Analytics Demo on Business Central (Streamlit).
Compatible with local execution & Streamlit Cloud deployment.
"""

import datetime
import io
import json
import os
import time

import pandas as pd
import requests
import streamlit as st

import cfo_engine as E
from mcp_client import BusinessCentralMCPClient

st.set_page_config(page_title="CFO Assistant - Business Central", layout="wide")


# ============================================================ connection & data
@st.cache_resource(show_spinner="Connecting to Business Central...")
def client():
    c = BusinessCentralMCPClient()
    auth = c.authenticate()
    if not auth.get("success"):
        raise RuntimeError(f"Login failed: {auth.get('error')}")
    c.get_company_id()
    return c


def item_prefix() -> str:
    return (client().config.get("cfo") or {}).get("item_prefix") or "LOB"


def _get_all(path: str):
    """GET an API v2.0 collection, following @odata.nextLink pages."""
    c, out, url = client(), [], c_path(path)
    while url:
        res = c._rest("GET", url)
        if res.status_code != 200:
            raise RuntimeError(f"Business Central API {res.status_code} on {path.split('?')[0]}: {res.text[:300]}")
        data = res.json()
        out.extend(data.get("value", []))
        url = data.get("@odata.nextLink")
    return out


def c_path(path: str) -> str:
    return client()._company_path(path)


def q(v: str) -> str:
    return "'" + str(v).replace("'", "''") + "'"


@st.cache_data(ttl=600, show_spinner="Reading items from Business Central...")
def load_items(prefix: str):
    return _get_all(f"items?$filter=startswith(number,{q(prefix)})"
                    "&$select=id,number,displayName,unitCost,unitPrice,itemCategoryCode")


@st.cache_data(ttl=600, show_spinner="Reading sales invoices from Business Central...")
def load_sales(start: str, end: str, item_numbers: tuple):
    invoices = _get_all(f"salesInvoices?$filter=postingDate ge {start} and postingDate le {end}"
                        "&$select=id,number,postingDate,customerNumber,customerName,status"
                        "&$expand=salesInvoiceLines")
    wanted, lines = set(item_numbers), []
    stats = {"invoices_in_period": len(invoices), "skipped_status": 0, "lines_in_period": 0, "poc_items": len(wanted)}
    for inv in invoices:
        if str(inv.get("status", "")).lower() in ("canceled", "cancelled"):
            stats["skipped_status"] += 1
            continue
        for ln in inv.get("salesInvoiceLines", []) or []:
            stats["lines_in_period"] += 1
            no = ln.get("lineObjectNumber")
            if str(ln.get("lineType", "")).lower() == "item" and no in wanted:
                lines.append({"item_no": no, "quantity": ln.get("quantity") or 0, "net_amount": ln.get("netAmount") or 0,
                              "invoice_no": inv.get("number"), "posting_date": inv.get("postingDate"),
                              "customer": inv.get("customerName")})
    stats["poc_lines"] = len(lines)
    return lines, stats


def month_bounds(ym: str):
    y, m = map(int, ym.split("-"))
    start = datetime.date(y, m, 1)
    end = (datetime.date(y + (m == 12), m % 12 + 1, 1) - datetime.timedelta(days=1))
    return start.isoformat(), end.isoformat(), start.strftime("%b %Y")


def build(ym: str):
    start, end, label = month_bounds(ym)
    items = E.items_frame(load_items(item_prefix()))
    lines, stats = load_sales(start, end, tuple(items["item_no"]))
    pg = E.pg_summary(E.sales_frame(lines, items))
    return {"items": items, "lines": pd.DataFrame(lines), "pg": pg, "lob": E.lob_summary(pg),
            "label": label, "stats": stats, "ym": ym}


# ============================================================ AI: question -> calculation spec
PROMPT = """You map a CFO's question to ONE calculation. Return ONLY JSON:
{"intent": "rank" | "total" | "compare" | "unsupported",
 "metric": "revenue"|"cogs"|"gross_profit"|"margin"|"avg_margin"|"margin_range",
 "level": "lob"|"product_group"            (rank only),
 "order": "highest"|"lowest"              (rank only),
 "n": integer, default 1                  (rank only),
 "lob": "LOB01"-style code or null        (rank only: restrict product groups to one LOB),
 "scope": "all" | "LOB02" | "LOB02-PG01"  (total/compare: whole company, one LOB, or one product group),
 "period": "YYYY-MM" or null,
 "compare_period": "YYYY-MM" or null      (compare only: the earlier month; null = month before period),
 "reason": short text                     (unsupported only)}

Choose intent:
- rank: asks WHICH / top / bottom / highest / lowest / best / worst LOB or product group.
- total: asks for an AMOUNT or % without ranking ("what was revenue", "LOB02's margin", "total COGS").
- compare: asks how something CHANGED between two months, or compares two months.
- unsupported: needs anything else - customers, quantities/units, trends over 3+ months, reasons ("why"),
  forecasts, budgets, cash, receivables, or anything not revenue/COGS/gross profit/margin.
  When unsure, choose unsupported. Never force a question into the nearest calculation.

Meanings:
- line of business = LOB; product group = PG; product group codes look like LOB02-PG01.
- profit / earnings = gross_profit. margin / profitability % = margin. "average gross margin" of a LOB = avg_margin.
- COGS / cost of goods sold / cost booking = cogs. sales / turnover = revenue.
- "pricing potential" = rank, highest margin, product_group.
- "varying / spread / range of margin between product groups" = rank, margin_range, lob.
- Month written like "Sep 26" means September 2026 -> "2026-09". "last month" = null. Use null if no month is named."""


@st.cache_data(ttl=3600, show_spinner=False)
def understand(question: str) -> dict:
    cfg = client().config
    key = os.environ.get("GROQ_API_KEY") or cfg.get("groq_api_key")
    if not key and hasattr(st, "secrets"):
        key = st.secrets.get("groq_api_key")
    if not key:
        raise RuntimeError("Set GROQ_API_KEY in environment or Streamlit secrets.")

    model = os.environ.get("GROQ_MODEL") or cfg.get("GROQ_MODEL") or "openai/gpt-oss-120b"
    body = {"model": model, "temperature": 0, "max_completion_tokens": 1024,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": question}]}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    for attempt in range(4):
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {key}"}, json=body, timeout=60)
        if r.status_code == 429 and attempt < 3:
            time.sleep(min(float(r.headers.get("retry-after", 5) or 5), 30))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"Groq HTTP {r.status_code}: {r.text[:300]}")
        return json.loads(r.json()["choices"][0]["message"]["content"])
    raise RuntimeError("Groq kept rate-limiting. Wait a minute and try again.")


def prev_month(ym: str) -> str:
    y, m = map(int, ym.split("-"))
    return f"{y - (m == 1)}-{(m - 2) % 12 + 1:02d}"


VALID_YM = __import__("re").compile(r"^\d{4}-(0[1-9]|1[0-2])$")
SUPPORTED = ("I can answer: which line of business or product group is highest/lowest on revenue, COGS, "
             "gross profit or margin; totals for the company, a line of business or a product group; "
             "and comparisons between two months.")


def ask(question: str, default_ym: str) -> dict:
    spec = understand(question)
    intent = spec.get("intent")
    if intent not in ("rank", "total", "compare"):
        reason = spec.get("reason") or "that question isn't about revenue, COGS, gross profit or margin"
        return {"spec": spec, "error": True, "text": f"I can't answer that yet ({reason}). {SUPPORTED}"}

    ym = spec.get("period") or default_ym
    if not VALID_YM.match(str(ym)):
        return {"spec": spec, "error": True, "text": f"I didn't understand the month '{ym}'. Try e.g. 'Sep 26'."}
    data = build(ym)
    st_ = data["stats"]
    if intent != "compare" and st_["poc_lines"] == 0:
        if st_["poc_items"] == 0:
            why = (f"no items in Business Central have a number starting with '{item_prefix()}' "
                   "(check the item numbers, or set cfo.item_prefix in bc_config.json)")
        else:
            why = (f"found {st_['poc_items']} POC items, but none of the {st_['invoices_in_period']} sales invoices "
                   f"in {data['label']} contain them")
        return {"spec": spec, "data_label": data["label"], "stats": st_, "error": True,
                "text": f"No sales data to answer from for {data['label']}: {why}."}

    if intent == "rank":
        res = E.answer(spec, data["pg"], data["lob"], data["label"])
    elif intent == "total":
        res = E.answer_total(spec, data["pg"], data["items"], data["label"])
    else:
        pym = spec.get("compare_period") or prev_month(ym)
        if not VALID_YM.match(str(pym)) or pym == ym:
            return {"spec": spec, "error": True, "text": "I need two different months to compare, e.g. 'Sep 26 vs Aug 26'."}
        if pym > ym:
            pym, ym = ym, pym
            data = build(ym)
        before = build(pym)
        res = E.answer_compare(spec, data["pg"], before["pg"], data["items"], data["label"], before["label"])
        data["stats"] = {k: f"{before['stats'][k]} / {data['stats'][k]}" for k in data["stats"]}
    return {"spec": spec, "data_label": data["label"], "stats": data["stats"], **res}


# ============================================================ formatting
MONEY = ["revenue", "cogs", "gross_profit", "unit_cost", "unit_price"]
PCT = ["margin_pct", "avg_margin_simple_pct", "margin_weighted_pct", "min_margin_pct", "max_margin_pct", "list_margin_pct"]
NAMES = {"lob": "Line of business", "pg": "Product group", "item_no": "Item", "description": "Description",
         "quantity": "Qty", "revenue": "Revenue", "cogs": "COGS (std cost)", "gross_profit": "Gross profit",
         "margin_pct": "Gross margin %", "avg_margin_simple_pct": "Avg margin % (simple)",
         "margin_weighted_pct": "Margin % (profit / revenue)", "min_margin_pct": "Lowest PG margin %",
         "max_margin_pct": "Highest PG margin %", "margin_spread_pts": "Spread (pts)", "product_groups": "PGs",
         "unit_cost": "Unit cost", "unit_price": "Unit price", "list_margin_pct": "List margin %"}


def show_table(df: pd.DataFrame):
    cfg = {}
    for c in df.columns:
        if c in MONEY:
            cfg[c] = st.column_config.NumberColumn(NAMES.get(c, c), format="%.2f")
        elif c in PCT or c == "margin_spread_pts":
            cfg[c] = st.column_config.NumberColumn(NAMES.get(c, c), format="%.2f")
        else:
            cfg[c] = st.column_config.Column(NAMES.get(c, c))
    st.dataframe(df, hide_index=True, use_container_width=True, column_config=cfg)


def excel_report(data: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        notes = pd.DataFrame({"Item": [
            f"Period: {data['label']}",
            "Source: Business Central API v2.0 - items and sales invoice lines",
            "Revenue = invoice line net amount (excl. tax)",
            "COGS = quantity x item unit cost (standard cost)",
            "Gross margin % = gross profit / revenue",
            "Avg margin % (simple) = average of product-group margins",
            "Margin % (profit / revenue) = total gross profit / total revenue",
            f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}"]})
        data["lob"].rename(columns=NAMES).to_excel(xw, sheet_name="By line of business", index=False)
        data["pg"].rename(columns=NAMES).to_excel(xw, sheet_name="By product group", index=False)
        data["items"].rename(columns=NAMES).to_excel(xw, sheet_name="Items", index=False)
        if not data["lines"].empty:
            data["lines"].to_excel(xw, sheet_name="Source invoice lines", index=False)
        notes.to_excel(xw, sheet_name="Notes", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(45, max(12, max(len(str(c.value or "")) for c in col) + 2))
    return buf.getvalue()


# ============================================================ UI
try:
    client()
except Exception as ex:
    st.error(f"Could not connect to Business Central: {ex}")
    st.stop()

today = datetime.date.today()
prev = (today.replace(day=1) - datetime.timedelta(days=1))
with st.sidebar:
    st.subheader("Period")
    year = st.selectbox("Year", list(range(today.year - 3, today.year + 1))[::-1], index=0 if prev.year == today.year else 1)
    month = st.selectbox("Month", list(range(1, 13)), index=prev.month - 1, format_func=lambda m: datetime.date(2000, m, 1).strftime("%B"))
    ym = f"{year}-{month:02d}"
    st.caption(f"Questions that name a month (e.g. 'Sep 26') use that month instead.")
    st.subheader("Data")
    bc = client().bc_cfg
    st.caption(f"{bc.get('environment')} · {bc.get('company_name')}")
    st.caption(f"Items: number starts with '{item_prefix()}'")
    st.caption("Source: Business Central API v2.0 (items, sales invoice lines)")
    if st.button("Refresh data from Business Central", use_container_width=True):
        st.cache_data.clear()

st.title("CFO Assistant")
st.caption("Ask about revenue, COGS, gross profit and margins by line of business or product group. "
           "Numbers are calculated from live Business Central data; the AI only interprets the question.")

try:
    data = build(ym)
except Exception as ex:
    st.error(str(ex))
    st.stop()

tab_ask, tab_report, tab_qa = st.tabs(["Ask", "P&L report", "Q&A check"])

# ---------------------------------------------------------------- Ask
with tab_ask:
    st.session_state.setdefault("chat", [])
    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input("Your question", placeholder="e.g. Which line of business made the most profit in Sep 26?")
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted and question.strip():
        try:
            st.session_state.chat.append({"q": question, **ask(question, ym)})
        except Exception as ex:
            st.session_state.chat.append({"q": question, "text": str(ex), "error": True})
    if not st.session_state.chat:
        st.info("Try: *Line of business with maximum profit in Sep 26* · *What was total revenue in Sep 26?* · "
                "*How did LOB02's margin change from Aug 26 to Sep 26?* · *List top 2 least profit making product groups*")
    for h in st.session_state.chat:
        with st.chat_message("user"):
            st.write(h["q"])
        with st.chat_message("assistant"):
            (st.warning if h.get("error") else st.success)(h["text"])
            if h.get("table") is not None and not h.get("error"):
                show_table(h["table"])
            if h.get("spec"):
                with st.expander("How this was answered"):
                    st.write("**AI interpreted the question as:**")
                    st.json(h["spec"])
                    if h.get("stats"):
                        s_ = h["stats"]
                        st.write(f"**Data:** {h['data_label']} · {s_.get('poc_items')} POC items · "
                                 f"{s_['invoices_in_period']} invoices in period · {s_['poc_lines']} lines for POC items")
                    st.caption("All figures computed in code from Business Central data.")

# ---------------------------------------------------------------- P&L report
with tab_report:
    lob, pg = data["lob"], data["pg"]
    st.subheader(f"Gross profit report - {data['label']}")
    if pg.empty:
        s_ = data["stats"]
        st.warning(f"No sales for the POC items in {data['label']}. Found {s_['poc_items']} items starting with "
                   f"'{item_prefix()}', and {s_['invoices_in_period']} invoices ({s_['lines_in_period']} lines) "
                   "in the period, none containing those items.")
    else:
        rev, cogs, gp = lob["revenue"].sum(), lob["cogs"].sum(), lob["gross_profit"].sum()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Revenue", f"{rev:,.0f}")
        k2.metric("COGS (std cost)", f"{cogs:,.0f}")
        k3.metric("Gross profit", f"{gp:,.0f}")
        k4.metric("Gross margin", f"{gp / rev * 100:.2f}%" if rev else "-")

        st.markdown("**By line of business**")
        show_table(lob)
        st.caption("Avg margin % (simple) = average of the product groups' margins. "
                   "Margin % (profit / revenue) = total gross profit ÷ total revenue.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Gross profit by line of business**")
            st.bar_chart(lob.set_index("lob")["gross_profit"])
        with c2:
            st.markdown("**Gross margin % by product group**")
            chart = pg.assign(label=pg["item_no"]).set_index("label")["margin_pct"].sort_values()
            st.bar_chart(chart)

        st.markdown("**By product group**")
        show_table(pg)
        with st.expander("Source invoice lines"):
            st.dataframe(data["lines"], hide_index=True, use_container_width=True)

        st.download_button("Download report (Excel)", excel_report(data),
                           file_name=f"gross_profit_report_{data['ym']}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           type="primary")

# ---------------------------------------------------------------- Q&A check
with tab_qa:
    st.write("Upload the Q&A workbook. Each question is put through the same pipeline as the Ask tab, "
             "and the answer is compared with the expected answer.")
    up = st.file_uploader("Q&A workbook (.xlsx with a 'Q&A' sheet)", type=["xlsx"])
    if up and st.button("Run all questions", type="primary"):
        try:
            sheet = pd.read_excel(up, sheet_name="Q&A")
            qcol = next(c for c in sheet.columns if "question" in str(c).lower())
            acol = next(c for c in sheet.columns if "answer" in str(c).lower())
        except Exception as ex:
            st.error(f"Could not read a 'Q&A' sheet with Question and Answer columns: {ex}")
        else:
            rows = []
            prog = st.progress(0.0)
            for i, r in enumerate(sheet.itertuples(index=False)):
                qtext, expected = str(getattr(r, qcol)), str(getattr(r, acol)).strip()
                try:
                    res = ask(qtext, ym)
                    ok = (not res.get("error")) and E.matches_expected(res.get("entities", []), expected)
                    rows.append({"Question": qtext, "Expected (answer key)": expected,
                                 "AI answer (computed)": res["text"], "Same answer?": "Yes" if ok else "Check"})
                except Exception as ex:
                    rows.append({"Question": qtext, "Expected (answer key)": expected,
                                 "AI answer (computed)": f"Error: {ex}", "Same answer?": "Error"})
                prog.progress((i + 1) / len(sheet))
            res_df = pd.DataFrame(rows)
            n_ok = (res_df["Same answer?"] == "Yes").sum()
            st.metric("Matching answers", f"{n_ok} / {len(res_df)}")
            st.dataframe(res_df, hide_index=True, use_container_width=True,
                         column_config={"AI answer (computed)": st.column_config.TextColumn(width="large"),
                                        "Expected (answer key)": st.column_config.TextColumn(width="medium")})
            st.caption("'Same answer?' compares which line of business / product group was named. "
                       "Check the numbers too: the answer key's Q3 figure (45.41%) is the sum of LOB02's margins; "
                       "the average is shown by the AI.")
