"""Tweet Analytics Dashboard — Streamlit app for exploring tweet export CSVs
with engagement charts and OpenAI-powered vibe / personality / tweet generation."""

import glob
import json
import os
import re
from datetime import datetime

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "gpt-5.6"
ACCENT = "#1d9bf0"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(APP_DIR, "sample_data")

st.set_page_config(
    page_title="Tweet Analytics Dashboard",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling — dark sidebar / light main, X-inspired, accent #1d9bf0
# --------------------------------------------------------------------------

st.markdown(
    f"""
    <style>
        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}

        .stApp {{
            background-color: #ffffff;
        }}

        section[data-testid="stMain"] {{
            background-color: #ffffff;
            color: #0f1419;
        }}

        section[data-testid="stSidebar"] {{
            background-color: #0b0e11;
            border-right: 1px solid #202327;
        }}
        section[data-testid="stSidebar"] * {{
            color: #e7e9ea !important;
        }}
        section[data-testid="stSidebar"] .stRadio > label,
        section[data-testid="stSidebar"] label {{
            color: #e7e9ea !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: #202327;
        }}
        section[data-testid="stSidebar"] .stButton button {{
            background-color: #16181c;
            color: #e7e9ea !important;
            border: 1px solid #2f3336;
            border-radius: 9999px;
            width: 100%;
        }}
        section[data-testid="stSidebar"] .stButton button:hover {{
            border-color: {ACCENT};
            color: {ACCENT} !important;
        }}

        h1, h2, h3, h4 {{
            color: #0f1419;
        }}

        .stButton button {{
            background-color: {ACCENT};
            color: #ffffff;
            border: none;
            border-radius: 9999px;
            padding: 0.5rem 1.4rem;
            font-weight: 600;
        }}
        .stButton button:hover {{
            background-color: #1a8cd8;
            color: #ffffff;
        }}

        .metric-card {{
            background: #f7f9f9;
            border: 1px solid #eff3f4;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            text-align: left;
        }}
        .metric-label {{
            font-size: 0.8rem;
            color: #536471;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        .metric-value {{
            font-size: 1.8rem;
            font-weight: 800;
            color: #0f1419;
            margin-top: 0.15rem;
        }}

        .vibe-card {{
            background: #ffffff;
            border: 1px solid #eff3f4;
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 1px 3px rgba(15, 20, 25, 0.06);
        }}
        .vibe-card h2, .vibe-card h3 {{
            color: {ACCENT};
        }}

        .tweet-preview {{
            background: #ffffff;
            border: 1px solid #eff3f4;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            max-width: 520px;
            box-shadow: 0 1px 3px rgba(15, 20, 25, 0.08);
        }}
        .tweet-avatar {{
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: {ACCENT};
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.1rem;
            flex-shrink: 0;
        }}
        .tweet-name {{
            font-weight: 700;
            color: #0f1419;
        }}
        .tweet-handle {{
            color: #536471;
        }}
        .tweet-body {{
            color: #0f1419;
            font-size: 1.05rem;
            margin: 0.6rem 0 0.8rem 0;
            white-space: pre-wrap;
        }}
        .tweet-metrics {{
            display: flex;
            gap: 1.6rem;
            color: #536471;
            font-size: 0.9rem;
        }}

        .empty-state {{
            border: 1px dashed #cfd9de;
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            color: #536471;
        }}

        .badge {{
            display: inline-block;
            background: rgba(29, 155, 240, 0.1);
            color: {ACCENT};
            border-radius: 9999px;
            padding: 0.15rem 0.7rem;
            font-size: 0.75rem;
            font-weight: 600;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def strip_code_fences(text: str) -> str:
    """Remove ```html / ```json / ``` fences the model sometimes wraps output in."""
    if text is None:
        return ""
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned.strip()


@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    return OpenAI(api_key=api_key)


def call_openai(messages, json_mode: bool = False, temperature: float = 0.8):
    """Call the chat completions endpoint and return cleaned text content."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "No OPENAI_API_KEY found. Add it to a .env file in the project root."
        )
    client = get_client(OPENAI_API_KEY)
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return strip_code_fences(response.choices[0].message.content)


@st.cache_data(show_spinner=False)
def load_and_clean(raw_bytes: bytes) -> pd.DataFrame:
    import io

    df = pd.read_csv(io.BytesIO(raw_bytes))
    df.columns = [c.strip().lower() for c in df.columns]

    required = ["text", "view_count", "created_at", "favorite_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(missing)}. "
            f"Expected at least: {', '.join(required)}."
        )

    df["text"] = df["text"].fillna("").astype(str)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.tz_localize(None)
    df["view_count"] = pd.to_numeric(df["view_count"], errors="coerce").fillna(0)
    df["favorite_count"] = pd.to_numeric(df["favorite_count"], errors="coerce").fillna(0)

    for optional in ["retweet_count", "reply_count"]:
        if optional in df.columns:
            df[optional] = pd.to_numeric(df[optional], errors="coerce").fillna(0)
        else:
            df[optional] = 0

    df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)

    df["engagement"] = df.apply(
        lambda r: (r["favorite_count"] / r["view_count"]) if r["view_count"] > 0 else 0.0,
        axis=1,
    )
    df["engagement_pct"] = df["engagement"] * 100

    return df


def demo_files():
    if not os.path.isdir(SAMPLE_DIR):
        return []
    return sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.csv")))


def pretty_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return base.replace("_", " ").replace("-", " ").title()


def build_tweet_sample(df: pd.DataFrame, n_top: int = 20, n_recent: int = 20) -> pd.DataFrame:
    top = df.sort_values("engagement", ascending=False).head(n_top)
    recent = df.sort_values("created_at", ascending=False).head(n_recent)
    combined = pd.concat([top, recent]).drop_duplicates(subset=["text", "created_at"])
    return combined.sort_values("created_at")


def tweets_to_block(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"- \"{row['text']}\" (favorites: {int(row['favorite_count'])}, "
            f"views: {int(row['view_count'])}, engagement: {row['engagement_pct']:.2f}%)"
        )
    return "\n".join(lines)


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------

for key, default in {
    "df": None,
    "source_label": None,
    "vibe_html": None,
    "personality": None,
    "generated_tweet": None,
    "account_name": "Your Account",
    "account_handle": "@youraccount",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------------------------------------------------------------
# Sidebar — branding, data loading, nav
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom: 0.3rem;">
            <span style="font-size:1.6rem;">🐦</span>
            <span style="font-size:1.25rem; font-weight:800;">Tweet Analytics</span>
        </div>
        <div style="color:#71767b; font-size:0.85rem; margin-bottom:1.2rem;">
            Engagement insights + AI vibe analysis
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Load data**")
    uploaded = st.file_uploader("Upload tweet export CSV", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        try:
            st.session_state.df = load_and_clean(uploaded.getvalue())
            st.session_state.source_label = uploaded.name
        except Exception as exc:
            st.error(f"Couldn't load that CSV: {exc}")

    samples = demo_files()
    if samples:
        st.markdown("**Or try a demo**")
        for path in samples:
            label = pretty_name(path)
            if st.button(label, key=f"demo_{label}", use_container_width=True):
                with open(path, "rb") as f:
                    raw = f.read()
                try:
                    st.session_state.df = load_and_clean(raw)
                    st.session_state.source_label = os.path.basename(path)
                    st.session_state.account_name = label
                    st.session_state.account_handle = "@" + re.sub(r"[^a-z0-9]", "", label.lower())
                except Exception as exc:
                    st.error(f"Couldn't load demo file: {exc}")

    st.markdown("---")

    section = st.radio(
        "Section",
        ["Overview", "Engagement", "Vibe Report", "Personality", "Generate Tweet"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    if OPENAI_API_KEY:
        st.markdown(f'<span class="badge">Model: {MODEL}</span>', unsafe_allow_html=True)
    else:
        st.warning("No OPENAI_API_KEY set. AI sections will be disabled until one is added to .env.")

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------

df = st.session_state.df

st.title("Tweet Analytics Dashboard")

if df is None:
    st.markdown(
        """
        <div class="empty-state">
            <h3>No data loaded yet</h3>
            <p>Upload a tweet export CSV from the sidebar, or click one of the demo datasets
            to explore the dashboard instantly.</p>
            <p style="font-size:0.85rem;">Required columns: <code>text</code>, <code>view_count</code>,
            <code>created_at</code>, <code>favorite_count</code>. Optional: <code>retweet_count</code>,
            <code>reply_count</code>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

st.caption(f"Source: {st.session_state.source_label}  ·  {len(df)} tweets loaded")

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------

if section == "Overview":
    total_tweets = len(df)
    total_favs = int(df["favorite_count"].sum())
    avg_favs = df["favorite_count"].mean()
    max_favs = int(df["favorite_count"].max())
    avg_engagement = df["engagement_pct"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Tweets", f"{total_tweets:,}")
    with c2:
        metric_card("Total Favorites", f"{total_favs:,}")
    with c3:
        metric_card("Avg Favorites", f"{avg_favs:,.0f}")
    with c4:
        metric_card("Max Favorites", f"{max_favs:,}")
    with c5:
        metric_card("Avg Engagement", f"{avg_engagement:.2f}%")

    st.write("")
    st.subheader("Tweets")
    display_cols = ["created_at", "text", "favorite_count", "view_count", "engagement_pct"]
    if (df["retweet_count"] != 0).any():
        display_cols.insert(4, "retweet_count")
    if (df["reply_count"] != 0).any():
        display_cols.insert(4, "reply_count")

    table = df[display_cols].rename(
        columns={
            "created_at": "Date",
            "text": "Tweet",
            "favorite_count": "Favorites",
            "view_count": "Views",
            "retweet_count": "Retweets",
            "reply_count": "Replies",
            "engagement_pct": "Engagement %",
        }
    ).sort_values("Date", ascending=False)

    st.dataframe(
        table,
        use_container_width=True,
        height=520,
        column_config={
            "Engagement %": st.column_config.NumberColumn(format="%.2f%%"),
            "Date": st.column_config.DatetimeColumn(format="MMM D, YYYY · h:mm A"),
        },
    )

# --------------------------------------------------------------------------
# Engagement
# --------------------------------------------------------------------------

elif section == "Engagement":
    st.subheader("Engagement over time")

    base = alt.Chart(df).encode(
        tooltip=[
            alt.Tooltip("created_at:T", title="Date"),
            alt.Tooltip("text:N", title="Tweet"),
            alt.Tooltip("favorite_count:Q", title="Favorites"),
            alt.Tooltip("view_count:Q", title="Views"),
            alt.Tooltip("engagement_pct:Q", title="Engagement %", format=".2f"),
        ]
    )

    engagement_chart = base.mark_circle(size=90, color=ACCENT, opacity=0.75).encode(
        x=alt.X("created_at:T", title="Date"),
        y=alt.Y("engagement_pct:Q", title="Engagement %"),
    ).properties(height=380).interactive()

    st.altair_chart(engagement_chart, use_container_width=True)

    st.subheader("Favorites over time")
    favorites_chart = base.mark_circle(size=90, color="#0f1419", opacity=0.7).encode(
        x=alt.X("created_at:T", title="Date"),
        y=alt.Y("favorite_count:Q", title="Favorites"),
    ).properties(height=380).interactive()

    st.altair_chart(favorites_chart, use_container_width=True)

# --------------------------------------------------------------------------
# Vibe Report
# --------------------------------------------------------------------------

elif section == "Vibe Report":
    st.subheader("Vibe Report")
    st.caption("GPT reads a mix of your top-engagement and most recent tweets to describe the account's persona.")

    disabled = not OPENAI_API_KEY
    if st.button("Generate Vibe Report", disabled=disabled):
        sample = build_tweet_sample(df)
        prompt = f"""You are a sharp social media analyst. Below is a sample of tweets from one account
(a mix of their top-engagement and most recent tweets).

{tweets_to_block(sample)}

Write a vibe report as **raw HTML only** (no markdown, no code fences, no <html>/<body> wrapper —
just semantic elements like <h3>, <p>, <ul>/<li>). Cover exactly these three sections, each with an
<h3> heading:
1. Persona — who this account seems to be and what they care about.
2. Writing Style — tone, sentence structure, recurring habits, use of humor/punctuation/etc.
3. Engagement Insights — what seems to drive their best-performing tweets.

Keep it concise, specific, and grounded in the tweets shown. Output HTML only."""

        with st.spinner("Reading the timeline and figuring out the vibe..."):
            try:
                html = call_openai(
                    [
                        {"role": "system", "content": "You write concise, insightful HTML reports. You output raw HTML only, never markdown or code fences."},
                        {"role": "user", "content": prompt},
                    ]
                )
                st.session_state.vibe_html = html
            except Exception as exc:
                st.error(f"Vibe report failed: {exc}")

    if st.session_state.vibe_html:
        st.markdown(f'<div class="vibe-card">{st.session_state.vibe_html}</div>', unsafe_allow_html=True)
    elif not disabled:
        st.info("Click **Generate Vibe Report** to analyze this account's persona and style.")

# --------------------------------------------------------------------------
# Personality
# --------------------------------------------------------------------------

elif section == "Personality":
    st.subheader("Personality — Big Five")
    st.caption("GPT estimates Big Five personality traits (0–100) from the account's tweets.")

    disabled = not OPENAI_API_KEY
    if st.button("Analyze Personality", disabled=disabled):
        sample = build_tweet_sample(df)
        prompt = f"""Below is a sample of tweets from one account:

{tweets_to_block(sample)}

Estimate this account's Big Five personality traits based purely on writing style and content.
Return a JSON object with exactly these keys:
{{
  "openness": <0-100 integer>,
  "conscientiousness": <0-100 integer>,
  "extraversion": <0-100 integer>,
  "agreeableness": <0-100 integer>,
  "neuroticism": <0-100 integer>,
  "summary": "<2-3 sentence summary of the personality profile>"
}}
Return JSON only."""

        with st.spinner("Profiling the account's personality..."):
            try:
                raw = call_openai(
                    [
                        {"role": "system", "content": "You are a careful personality-profiling assistant. You always return valid JSON matching the requested schema."},
                        {"role": "user", "content": prompt},
                    ],
                    json_mode=True,
                    temperature=0.4,
                )
                st.session_state.personality = json.loads(raw)
            except Exception as exc:
                st.error(f"Personality analysis failed: {exc}")

    profile = st.session_state.personality
    if profile:
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        labels = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
        values = [float(profile.get(t, 0)) for t in traits]

        col1, col2 = st.columns([3, 2])
        with col1:
            fig = go.Figure()
            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=labels + [labels[0]],
                    fill="toself",
                    fillcolor="rgba(29, 155, 240, 0.25)",
                    line=dict(color=ACCENT, width=2),
                    name="Personality",
                )
            )
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#eff3f4"),
                    bgcolor="#ffffff",
                ),
                showlegend=False,
                height=440,
                margin=dict(l=40, r=40, t=40, b=40),
                paper_bgcolor="#ffffff",
                font=dict(color="#0f1419"),
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Summary**")
            st.write(profile.get("summary", ""))
            st.markdown("**Scores**")
            for label, trait in zip(labels, traits):
                st.write(f"{label}: {profile.get(trait, 0)}")
    elif not disabled:
        st.info("Click **Analyze Personality** to generate a Big Five radar chart.")

# --------------------------------------------------------------------------
# Generate Tweet
# --------------------------------------------------------------------------

elif section == "Generate Tweet":
    st.subheader("Generate a Tweet in This Account's Voice")

    colname, colhandle = st.columns(2)
    with colname:
        st.session_state.account_name = st.text_input("Display name", st.session_state.account_name)
    with colhandle:
        st.session_state.account_handle = st.text_input("Handle", st.session_state.account_handle)

    disabled = not OPENAI_API_KEY
    if st.button("Generate Tweet", disabled=disabled):
        top = df.sort_values("engagement", ascending=False).head(15)
        prompt = f"""Here are this account's top-performing tweets (few-shot examples of their voice):

{tweets_to_block(top)}

Write one brand-new original tweet in this exact voice, tone, and style — matching their typical
sentence length, punctuation habits, and topics. Do not reuse any line verbatim.
Keep it under 280 characters. Return ONLY the tweet text, with no quotes, labels, or commentary."""

        with st.spinner("Drafting a tweet in this account's voice..."):
            try:
                tweet = call_openai(
                    [
                        {"role": "system", "content": "You are a ghostwriter who perfectly mimics a given account's tweeting voice. You output only the tweet text."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.9,
                )
                st.session_state.generated_tweet = tweet.strip().strip('"')
            except Exception as exc:
                st.error(f"Tweet generation failed: {exc}")

    if st.session_state.generated_tweet:
        avg_favs = int(df["favorite_count"].mean())
        avg_views = int(df["view_count"].mean())
        avg_rt = int(df["retweet_count"].mean()) if (df["retweet_count"] != 0).any() else max(1, avg_favs // 10)
        avg_reply = int(df["reply_count"].mean()) if (df["reply_count"] != 0).any() else max(1, avg_favs // 20)
        initials = "".join([w[0] for w in st.session_state.account_name.split()][:2]).upper() or "?"

        st.markdown(
            f"""
            <div class="tweet-preview">
                <div style="display:flex; gap:0.7rem;">
                    <div class="tweet-avatar">{initials}</div>
                    <div>
                        <span class="tweet-name">{st.session_state.account_name}</span>
                        &nbsp;<span class="tweet-handle">{st.session_state.account_handle} · now</span>
                        <div class="tweet-body">{st.session_state.generated_tweet}</div>
                        <div class="tweet-metrics">
                            <span>💬 {avg_reply:,}</span>
                            <span>🔁 {avg_rt:,}</span>
                            <span>❤️ {avg_favs:,}</span>
                            <span>👁️ {avg_views:,}</span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif not disabled:
        st.info("Click **Generate Tweet** to draft a new tweet in this account's voice.")
