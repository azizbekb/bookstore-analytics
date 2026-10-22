import re

import altair as alt
import httpx
import pandas as pd
import streamlit as st

from ai_tools import make_tools
from analysis import analyze, stores

st.set_page_config(page_title="Bookstore Analytics", page_icon="📚", layout="wide")


@st.cache_data
def get_result(store):
    return analyze(store)


st.title("📚 Bookstore Analytics")

with st.sidebar:
    st.header("Data source")
    store = st.radio("Store", stores())
    st.caption("Each store folder is processed separately.")

r = get_result(store)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total revenue", f"${r['daily'].sum():,.2f}")
c2.metric("Orders", f"{r['orders_clean']:,}")
c3.metric("Unique users", f"{r['unique_users']:,}", help=f"{r['user_records']:,} user records before reconciliation")
c4.metric("Unique author sets", f"{r['author_sets']:,}")

tab_rev, tab_cust, tab_books, tab_ai = st.tabs(["📈 Revenue", "👤 Customers", "📖 Authors", "💬 Ask AI"])

with tab_rev:
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Top 5 days by revenue")
        top = pd.DataFrame(r["top5_days"], columns=["Date", "Revenue, $"])
        top.index = range(1, len(top) + 1)
        st.dataframe(top.style.format({"Revenue, $": "{:,.2f}"}), width="stretch")
    with right:
        st.subheader("Daily revenue")
        daily = r["daily"].reset_index()
        daily.columns = ["date", "revenue"]
        top_dates = pd.to_datetime([d for d, _ in r["top5_days"]])
        line = alt.Chart(daily).mark_line(strokeWidth=1.2).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("revenue:Q", title="Revenue, $"),
            tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), alt.Tooltip("revenue:Q", format=",.2f")],
        )
        points = alt.Chart(daily[daily["date"].isin(top_dates)]).mark_point(
            size=80, filled=True, color="#e4572e").encode(x="date:T", y="revenue:Q")
        st.altair_chart((line + points).properties(height=380), width="stretch")
        st.caption("Red points mark the top 5 days.")

with tab_cust:
    a, b = st.columns(2)
    a.metric("Real unique users", f"{r['unique_users']:,}")
    a.caption(f"{r['user_records']:,} records merged: records that differ in at most one field "
              "(name, address, phone or e-mail) belong to one person.")
    b.metric("Best buyer total spending", f"${r['best_buyer_total']:,.2f}")
    b.markdown("**Best buyer IDs (all aliases)**")
    b.code(str(r["best_buyer_ids"]), language=None)

with tab_books:
    a, b = st.columns(2)
    a.metric("Unique author sets", f"{r['author_sets']:,}")
    a.caption("['John', 'Paul'] and ['Paul', 'John'] count as one set.")
    b.metric("Books sold by the most popular author set", f"{r['popular_qty']:,}")
    b.markdown("**Most popular author(s)**")
    for authors in r["popular_authors"]:
        b.markdown("- " + ", ".join(authors))

SYSTEM_PROMPT = """You are a data analyst assistant for a bookstore dashboard.
Answer questions about the currently selected store ONLY by calling the provided
functions. Never calculate, estimate or invent numbers yourself: every number in
your answer must come from a function result. If no function can answer the
question, say that you cannot answer it with the available data.
Do not add, subtract or combine numbers: report them exactly as the functions
return them. Sales are counted per author set, so report each set separately.
When you name customers, also give their user ids.
Use dates in YYYY-MM-DD format and money in US dollars. Keep answers short."""


def secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:  # no secrets.toml at all
        return default


SKIP_WORDS = ("image", "tts", "audio", "live", "embedding", "vision", "robotics", "computer-use")


def version_key(name):
    nums = [int(x) for x in re.findall(r"\d+", name)]
    return nums[:2] + [0] * (2 - len(nums[:2]))


def gemini_client(api_key):
    """Gemini client that connects over IPv4. On this network an IPv6 connection
    to Google takes 20-60 s (IPv4: 0.1 s) and httpx tries IPv6 first; binding to
    0.0.0.0 makes it use IPv4 only."""
    from google import genai
    from google.genai import types
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(
        timeout=30_000, retry_options=types.HttpRetryOptions(attempts=1),  # no slow retries
        client_args={"transport": httpx.HTTPTransport(local_address="0.0.0.0")}))


@st.cache_data(ttl=3600, show_spinner=False)
def available_models(api_key):
    """Ask Gemini which text models this key can use. Order: 'flash-lite' first
    (answers in ~2 s, the bigger 'flash' models took 1-2 minutes), then newest
    first, stable before preview."""
    client = gemini_client(api_key)
    names = []
    for m in client.models.list():
        name = m.name.removeprefix("models/")
        if "generateContent" not in (m.supported_actions or []):
            continue
        if "flash" not in name or any(w in name for w in SKIP_WORDS):
            continue
        names.append(name)
    return sorted(names, key=lambda n: ("lite" not in n, [-v for v in version_key(n)],
                                        "preview" in n or "exp" in n, n))


def model_chain(api_key):
    """Models to try, in order. GEMINI_MODELS in secrets (comma-separated) wins;
    otherwise GEMINI_MODEL first and then every other flash model of the key.
    It is a generator, so the model list is only fetched if the first model fails."""
    if custom := secret("GEMINI_MODELS"):
        yield from [m.strip() for m in custom.split(",") if m.strip()]
        return
    first = secret("GEMINI_MODEL", "gemini-3.5-flash-lite")
    yield first
    try:
        others = available_models(api_key)
    except Exception:
        others = []
    yield from (m for m in others if m != first)


def ask_ai(question, history):
    from google.genai import errors, types

    api_key = secret("GEMINI_API_KEY")
    client = gemini_client(api_key)
    past = [types.Content(role="user" if m["role"] == "user" else "model",
                          parts=[types.Part(text=m["content"])]) for m in history]
    thinking = types.ThinkingConfig(thinking_level=secret("GEMINI_THINKING", "minimal"))

    def run(model, with_thinking):
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, tools=make_tools(r),
            thinking_config=thinking if with_thinking else None)
        chat = client.chats.create(model=model, config=config, history=past)
        return chat.send_message(question).text or "I could not find an answer."

    tried = []
    for model in model_chain(api_key):
        try:
            try:
                answer = run(model, True)
            except errors.ClientError as e:
                if "think" not in str(e).lower():
                    raise
                answer = run(model, False)  # model without thinking levels
            st.session_state["last_model"] = model
            return answer
        except errors.APIError as e:
            # 429 = quota used up, 404 = model not available, 5xx = overloaded: try the next one
            if getattr(e, "code", None) in (404, 429, 500, 503):
                tried.append(f"{model} ({e.code})")
                continue
            raise
        except httpx.TimeoutException:  # model too slow: try the next one
            tried.append(f"{model} (timeout)")
    raise RuntimeError("all models are unavailable or out of quota: " + ", ".join(tried))


with tab_ai:
    st.subheader(f"Ask about {store}")
    st.caption("Examples: What was the revenue in August 2025? · Who are the top 3 customers? · "
               "How many books did Ethan Cronin sell? · Show all ids of user 48403.")
    if not secret("GEMINI_API_KEY"):
        st.info("Add GEMINI_API_KEY to .streamlit/secrets.toml to enable the chat.")
    else:
        key = f"chat_{store}"
        history = st.session_state.setdefault(key, [])
        for m in history:
            st.chat_message(m["role"]).write(m["content"])
        if question := st.chat_input("Ask a question about the data"):
            st.chat_message("user").write(question)
            with st.chat_message("assistant"), st.spinner("Calculating..."):
                try:
                    answer = ask_ai(question, history)
                except Exception as e:
                    answer = f"Error while calling the AI service: {e}"
                st.write(answer)
                if model := st.session_state.get("last_model"):
                    st.caption(f"model: {model}")
            history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
