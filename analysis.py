"""Data loading, cleaning and analysis for Task 4 (Data Engineering)."""
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

EUR_TO_USD = 1.2
DATA_DIR = Path(__file__).parent / "data"

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


# ---------- loading ----------

def load_books(folder):
    with open(folder / "books.yaml", encoding="utf-8") as f:
        books = pd.DataFrame(yaml.safe_load(f))
    books.columns = [c.lstrip(":") for c in books.columns]  # ":title" -> "title"
    books["year"] = pd.to_numeric(books["year"], errors="coerce").astype("Int64")
    books["authors"] = books["author"].map(
        lambda a: tuple(sorted({x.strip() for x in str(a).split(",") if x.strip()})))
    return books


def load_users(folder):
    users = pd.read_csv(folder / "users.csv", dtype=str, keep_default_na=False)
    users["id"] = users["id"].astype(int)
    for col in ["name", "address", "email"]:
        users[col] = users[col].str.strip()
    users["email"] = users["email"].str.lower()
    users["phone"] = users["phone"].str.replace(r"\D", "", regex=True)  # keep digits only
    return users


def load_orders(folder):
    orders = pd.read_parquet(folder / "orders.parquet")
    orders = orders.drop_duplicates()
    orders["price_usd"] = orders["unit_price"].map(parse_price)
    orders["ts"] = orders["timestamp"].map(parse_timestamp)
    orders = orders.dropna(subset=["price_usd", "ts"])
    orders["paid_price"] = (orders["quantity"] * orders["price_usd"]).round(2)
    orders["date"] = orders["ts"].dt.normalize()
    orders["year"] = orders["ts"].dt.year
    orders["month"] = orders["ts"].dt.month
    orders["day"] = orders["ts"].dt.day
    return orders


# ---------- parsing helpers ----------

def parse_price(raw):
    """'$ 35', '30.0USD', 'EUR37.50', '€30¢99', '23€25¢', '41.$' -> amount in USD."""
    s = str(raw).strip()
    is_eur = "EUR" in s.upper() or "€" in s
    numbers = re.findall(r"\d+", s)
    if not numbers:
        return None
    if "¢" in s:                                   # dollars¢cents
        value = int(numbers[0]) + int(numbers[1]) / 100
    else:
        match = re.search(r"\d+(?:\.\d*)?", s)
        value = float(match.group().rstrip("."))
    return round(value * EUR_TO_USD, 2) if is_eur else value


TIME_RE = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?\s*([AaPp])?\.?\s*[Mm]?\.?")


def parse_timestamp(raw):
    """Parse the many mixed timestamp formats found in the data."""
    s = str(raw).strip()
    m = TIME_RE.search(s)
    if not m:
        return None
    hour, minute, second = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    ampm = (m.group(4) or "").upper()
    if ampm == "P" and hour != 12:
        hour += 12
    elif ampm == "A" and hour == 12:
        hour = 0
    date_part = (s[:m.start()] + " " + s[m.end():]).strip(" ,;T")
    date = parse_date(date_part)
    if date is None:
        return None
    return pd.Timestamp(datetime(date[0], date[1], date[2], hour, minute, second))


def parse_date(s):
    s = s.strip(" ,;T")
    if m := re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s):          # 2025-06-25
        return int(m[1]), int(m[2]), int(m[3])
    if m := re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s):        # 12/31/25 (MM/DD/YY)
        y = int(m[3]); y = y + 2000 if y < 100 else y
        return y, int(m[1]), int(m[2])
    if m := re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", s):        # 30.06.2025 (DD.MM.YYYY)
        return int(m[3]), int(m[2]), int(m[1])
    if m := re.search(r"(\d{1,2})-([A-Za-z]+)-(\d{4})", s):        # 10-Dec-2025, 15-September-2025
        return int(m[3]), MONTHS[m[2][:3].lower()], int(m[1])
    if m := re.search(r"([A-Za-z]{3})[a-z]*\s+(\d{1,2})\s+(\d{4})", s):  # Fri Apr 10 ... 2026
        return int(m[3]), MONTHS[m[1].lower()], int(m[2])
    return None


# ---------- user reconciliation ----------

USER_FIELDS = ["name", "address", "phone", "email"]


def reconcile_users(users):
    """Return a Series user_id -> person_id. Two records are the same person when
    all fields except at most one are equal (chains are merged with union-find)."""
    parent = {uid: uid for uid in users["id"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for skip in USER_FIELDS:
        keep = [f for f in USER_FIELDS if f != skip]
        for ids in users.groupby(keep)["id"]:
            ids = list(ids[1])
            for other in ids[1:]:
                parent[find(other)] = find(ids[0])

    return users["id"].map(find).rename("person")


# ---------- analysis ----------

def analyze(store):
    folder = DATA_DIR / store
    books, users, orders = load_books(folder), load_users(folder), load_orders(folder)

    users["person"] = reconcile_users(users)
    orders = orders.merge(users[["id", "person"]], left_on="user_id", right_on="id",
                          suffixes=("", "_user")).drop(columns="id_user")
    orders = orders.merge(books[["id", "authors"]], left_on="book_id", right_on="id",
                          suffixes=("", "_book")).drop(columns="id_book")

    daily = orders.groupby("date")["paid_price"].sum().round(2).sort_index()
    top5 = daily.sort_values(ascending=False).head(5)

    author_sales = orders.groupby("authors")["quantity"].sum()
    best_qty = author_sales.max()
    popular = [list(a) for a, q in author_sales.items() if q == best_qty]

    spend = orders.groupby("person")["paid_price"].sum()
    best_person = spend.idxmax()
    best_ids = sorted(users.loc[users["person"] == best_person, "id"].tolist())

    return {
        "store": store,
        "orders": orders,
        "users": users,
        "books": books,
        "daily": daily,
        "top5_days": [(d.strftime("%Y-%m-%d"), round(v, 2)) for d, v in top5.items()],
        "unique_users": int(users["person"].nunique()),
        "user_records": len(users),
        "author_sets": int(books["authors"].nunique()),
        "popular_authors": popular,
        "popular_qty": int(best_qty),
        "best_buyer_ids": best_ids,
        "best_buyer_total": round(float(spend.max()), 2),
        "orders_total": len(pd.read_parquet(folder / "orders.parquet")),
        "orders_clean": len(orders),
    }


def stores():
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("STORE"))
