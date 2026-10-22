# Bookstore Analytics (Task 4, Data Engineering)

Streamlit dashboard for five bookstore datasets (`data/STORE1` ... `data/STORE5`).

## Run locally

```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data cleaning

- `books.yaml`: Ruby-style keys (`:title`) are renamed; authors are split by comma and stored as a sorted tuple, so `['John', 'Paul']` and `['Paul', 'John']` are the same set.
- `orders.parquet`: prices come in many formats (`$ 35`, `30.0USD`, `EUR37.50`, `€30¢99`, `23€25¢`, `41.$`) and are converted to USD (EUR 1 = USD 1.2). Timestamps come in dozens of formats (ISO, `MM/DD/YY`, `DD.MM.YYYY`, `10-Dec-2025`, `Fri Apr 10 15:32:26 2026`, 12/24-hour with `AM`, `A.M.`, `pm`) and are parsed explicitly.
- `users.csv`: phones are reduced to digits, e-mails lower-cased. Records that differ in at most one field (name, address, phone, e-mail) are merged into one person with union-find.

## Files

- `analysis.py` - loading, cleaning and all calculations.
- `app.py` - dashboard.
