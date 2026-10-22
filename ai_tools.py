"""Functions the AI model may call. All numbers are computed here with pandas,
the model only chooses which function to call and phrases the answer."""
import pandas as pd


def make_tools(r):
    orders, users, books = r["orders"], r["users"], r["books"]
    daily = r["daily"]

    def _dates(start_date, end_date):
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        return orders[(orders["date"] >= start) & (orders["date"] <= end)]

    def get_summary() -> dict:
        """Key facts about the selected store: total revenue, number of orders,
        date range of the data, unique users, unique author sets, most popular
        author set and best buyer."""
        return {
            "store": r["store"],
            "total_revenue_usd": round(float(daily.sum()), 2),
            "orders": int(len(orders)),
            "first_date": daily.index.min().strftime("%Y-%m-%d"),
            "last_date": daily.index.max().strftime("%Y-%m-%d"),
            "unique_users": r["unique_users"],
            "user_records": r["user_records"],
            "unique_author_sets": r["author_sets"],
            "most_popular_authors": r["popular_authors"],
            "most_popular_books_sold": r["popular_qty"],
            "best_buyer_ids": r["best_buyer_ids"],
            "best_buyer_total_usd": r["best_buyer_total"],
        }

    def revenue_between(start_date: str, end_date: str) -> dict:
        """Total revenue in USD, number of orders and books sold between two dates
        (inclusive). Dates in YYYY-MM-DD format."""
        part = _dates(start_date, end_date)
        return {"start_date": start_date, "end_date": end_date,
                "revenue_usd": round(float(part["paid_price"].sum()), 2),
                "orders": int(len(part)), "books_sold": int(part["quantity"].sum())}

    def monthly_revenue(year: int) -> dict:
        """Revenue in USD for every month of the given year."""
        part = orders[orders["year"] == year]
        by_month = part.groupby("month")["paid_price"].sum().round(2)
        return {"year": year, "revenue_by_month_usd": {int(m): float(v) for m, v in by_month.items()}}

    def top_days(n: int, ascending: bool = False) -> list:
        """The n days with the highest revenue (or the lowest if ascending is true)."""
        s = daily.sort_values(ascending=ascending).head(n)
        return [{"date": d.strftime("%Y-%m-%d"), "revenue_usd": round(float(v), 2)} for d, v in s.items()]

    def top_customers(n: int) -> list:
        """The n customers (merged persons) with the highest total spending,
        with all their user ids."""
        spend = orders.groupby("person")["paid_price"].sum().sort_values(ascending=False).head(n)
        result = []
        for person, total in spend.items():
            ids = sorted(users.loc[users["person"] == person, "id"].tolist())
            name = users.loc[users["id"] == person, "name"].iloc[0]
            result.append({"name": name, "user_ids": ids, "total_spent_usd": round(float(total), 2)})
        return result

    def customer_info(user_id: int) -> dict:
        """Everything about the person that owns this user id: all aliases (ids,
        names, e-mails, phones), number of orders and total spending."""
        row = users[users["id"] == user_id]
        if row.empty:
            return {"error": f"user {user_id} not found"}
        person = row["person"].iloc[0]
        group = users[users["person"] == person]
        mine = orders[orders["person"] == person]
        return {"user_ids": sorted(group["id"].tolist()),
                "names": sorted(set(group["name"])), "emails": sorted(set(group["email"])),
                "phones": sorted(set(group["phone"])), "orders": int(len(mine)),
                "total_spent_usd": round(float(mine["paid_price"].sum()), 2)}

    def top_authors(n: int) -> list:
        """The n author sets with the most books sold."""
        s = orders.groupby("authors")["quantity"].sum().sort_values(ascending=False).head(n)
        return [{"authors": list(a), "books_sold": int(q)} for a, q in s.items()]

    def author_sales(name: str) -> dict:
        """Books sold and revenue for every author set that contains an author
        whose name contains the given text (case-insensitive). Sales are counted
        per author set: the set with only this author is their own books."""
        mask = orders["authors"].map(lambda a: any(name.lower() in x.lower() for x in a))
        part = orders[mask]
        if part.empty:
            return {"error": f"no author matching '{name}'"}
        g = part.groupby("authors").agg(books_sold=("quantity", "sum"), revenue=("paid_price", "sum"))
        return {"matches": [{"authors": list(a), "books_sold": int(row.books_sold),
                             "revenue_usd": round(float(row.revenue), 2)} for a, row in g.iterrows()]}

    def count_books(genre: str = "", publisher: str = "", year: int = 0) -> dict:
        """Number of books in the catalogue, optionally filtered by genre,
        publisher (substring, case-insensitive) and publication year."""
        part = books
        if genre:
            part = part[part["genre"].astype(str).str.contains(genre, case=False)]
        if publisher:
            part = part[part["publisher"].astype(str).str.contains(publisher, case=False)]
        if year:
            part = part[part["year"] == year]
        return {"books": int(len(part))}

    return [get_summary, revenue_between, monthly_revenue, top_days, top_customers,
            customer_info, top_authors, author_sales, count_books]
