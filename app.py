"""Tweet Analytics Dashboard — Lecture 2 vibe-coding starter."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import altair as alt
import openai
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

MODEL = "gpt-5.6"
DEMO_FILES = {
    "Sam Altman": APP_DIR / "TwExportly_sama_tweets_2026_07_14.csv",
    "Elon Musk": APP_DIR / "TwExportly_elonmusk_tweets_2026_07_14.csv",
}

REQUIRED_COLS = ["text", "view_count", "created_at", "favorite_count"]

st.set_page_config(
    page_title="Tweet Analytics Dashboard",
    page_icon="𝕏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', 'DM Sans', sans-serif;
        color: #0a2540;
    }
    .stApp {
        background: linear-gradient(165deg, #f7f9f9 0%, #eef3f7 45%, #e8eef5 100%);
        color: #0a2540;
    }
    .stApp [data-testid="stMain"] p,
    .stApp [data-testid="stMain"] span,
    .stApp [data-testid="stMain"] label,
    .stApp [data-testid="stMain"] li,
    .stApp [data-testid="stMain"] h1,
    .stApp [data-testid="stMain"] h2,
    .stApp [data-testid="stMain"] h3,
    .stApp [data-testid="stMain"] .stMarkdown,
    .stApp [data-testid="stMain"] [data-testid="stCaptionContainer"],
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #0a2540 !important;
    }
    [data-testid="stSidebar"] {
        background: #0f1419;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stMarkdown {
        color: #e7e9ea;
    }
    [data-testid="stSidebar"] .stRadio label {
        background: #1a2330 !important;
        border: 1px solid #2f3b4a !important;
        border-radius: 12px !important;
        padding: 0.65rem 0.85rem !important;
        margin-bottom: 0.35rem;
    }
    .hero-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 2.35rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #0a2540;
        margin-bottom: 0.15rem;
    }
    .hero-sub {
        color: #1e3a5f;
        font-size: 1.05rem;
        margin-bottom: 1.25rem;
    }
    .accent {
        color: #1d9bf0;
    }
    .tweet-card {
        background: #ffffff;
        border: 1px solid #cfd9de;
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 8px 24px rgba(15, 20, 25, 0.06);
    }
    .tweet-meta {
        color: #1e3a5f;
        font-size: 0.92rem;
    }
    .tweet-body {
        font-size: 1.15rem;
        line-height: 1.45;
        color: #0a2540;
        margin: 0.65rem 0 0.85rem 0;
        white-space: pre-wrap;
    }
    .tweet-stats {
        color: #1e3a5f;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e1e8ed;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        box-shadow: 0 2px 8px rgba(15, 20, 25, 0.04);
    }
    .stButton > button {
        border-radius: 999px !important;
        font-weight: 600 !important;
        border: none !important;
        background: #1d9bf0 !important;
        color: white !important;
    }
    .stButton > button:hover {
        background: #1a8cd8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

for key, default in [
    ("df", None),
    ("data_label", None),
    ("analysis_result", None),
    ("personality_data", None),
    ("generated_tweet", None),
    ("selected_tab", "Overview"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def get_client() -> openai.OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not found. Add it to a `.env` file.")
        return None
    return openai.OpenAI(api_key=api_key)


def chat_complete(messages: list[dict], *, json_mode: bool = False, max_tokens: int = 2000) -> str | None:
    client = get_client()
    if client is None:
        return None

    kwargs = {
        "model": MODEL,
        "messages": messages,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    # gpt-5.x prefers max_completion_tokens; fall back for older SDKs/models
    try:
        kwargs["max_completion_tokens"] = max_tokens
        response = client.chat.completions.create(**kwargs)
    except TypeError:
        kwargs.pop("max_completion_tokens", None)
        kwargs["max_tokens"] = max_tokens
        response = client.chat.completions.create(**kwargs)
    except Exception as first_err:
        # Some snapshots still want max_tokens
        kwargs.pop("max_completion_tokens", None)
        kwargs["max_tokens"] = max_tokens
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            st.error(f"OpenAI error: {e or first_err}")
            return None

    return response.choices[0].message.content


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return None

    out = df.copy()
    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
    for col in ["favorite_count", "view_count", "retweet_count", "reply_count"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    views = out["view_count"].replace(0, pd.NA)
    out["engagement"] = pd.to_numeric(out["favorite_count"] / views, errors="coerce").fillna(0.0)
    out = out.sort_values("engagement", ascending=False).reset_index(drop=True)
    return out


def load_csv(source) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(source)
        return process_dataframe(df)
    except Exception as e:
        st.error(f"Could not load CSV: {e}")
        return None


def sample_tweets(df: pd.DataFrame, n: int = 40) -> pd.DataFrame:
    """Mix top engagement + most recent so vibe isn't only viral hits."""
    n = min(n, len(df))
    top_n = max(n // 2, 1)
    recent_n = n - top_n
    top = df.nlargest(top_n, "engagement")
    recent = df.nlargest(recent_n, "created_at") if recent_n else df.iloc[0:0]
    mixed = pd.concat([top, recent]).drop_duplicates(subset=["text"]).head(n)
    return mixed


def strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:html|json|markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def analyze_vibe(df: pd.DataFrame) -> str | None:
    sample = sample_tweets(df, 40)
    payload = [
        {"text": row.text, "favorites": int(row.favorite_count), "engagement": round(float(row.engagement), 4)}
        for row in sample.itertuples()
    ]
    prompt = f"""You are a sharp social media strategist. Analyze these posts and engagement stats.

Tweets:
{json.dumps(payload, ensure_ascii=False)}

Return ONLY raw HTML (no markdown, no code fences) in this structure:

<div style="font-family: IBM Plex Sans, sans-serif;">
  <h2 style="color:#1d9bf0;margin:0 0 0.75rem 0;">Account Vibe Report</h2>
  <h3 style="color:#0f1419;">Persona</h3>
  <p style="color:#536471;line-height:1.65;">...</p>
  <h3 style="color:#0f1419;">Writing Style</h3>
  <p style="color:#536471;line-height:1.65;">...</p>
  <h3 style="color:#0f1419;">Engagement Insights</h3>
  <p style="color:#536471;line-height:1.65;">...</p>
  <h3 style="color:#0f1419;">What To Post Next</h3>
  <ul style="color:#536471;line-height:1.65;"><li>...</li></ul>
</div>
"""
    return chat_complete(
        [
            {
                "role": "system",
                "content": "Return pure HTML only. No markdown. No fences. Be specific and evidence-based.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=4000,
    )


def analyze_personality(df: pd.DataFrame) -> dict | None:
    sample = sample_tweets(df, 40)
    tweets = "\n".join(f"- {t}" for t in sample["text"].tolist())
    prompt = f"""Infer personality from these posts. Score each trait 0-100.

Posts:
{tweets}

Return JSON with keys:
Extraversion, Openness, Conscientiousness, Agreeableness, Neuroticism,
Assertiveness, Creativity, Analytical, summary
(summary = 2-3 sentences, punchy, no jargon dump).
"""
    raw = chat_complete(
        [
            {
                "role": "system",
                "content": "You are a careful personality analyst. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        json_mode=True,
        max_tokens=2000,
    )
    if not raw:
        return None
    try:
        return json.loads(strip_fences(raw))
    except json.JSONDecodeError as e:
        st.error(f"Could not parse personality JSON: {e}")
        return None


def create_radar(personality: dict) -> go.Figure:
    traits, scores = [], []
    for k, v in personality.items():
        if k == "summary" or not isinstance(v, (int, float)):
            continue
        traits.append(k)
        scores.append(float(v))

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=scores + ([scores[0]] if scores else []),
                theta=traits + ([traits[0]] if traits else []),
                fill="toself",
                name="Profile",
                line=dict(color="#1d9bf0", width=2),
                fillcolor="rgba(29, 155, 240, 0.28)",
            )
        ]
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title=dict(text="Personality Radar", x=0.5, font=dict(size=18, color="#0f1419")),
        height=520,
        margin=dict(t=60, b=40, l=60, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def generate_tweet(df: pd.DataFrame) -> str | None:
    sample = sample_tweets(df.nlargest(min(30, len(df)), "favorite_count"), 25)
    examples = [
        {"text": row.text, "favorites": int(row.favorite_count)}
        for row in sample.itertuples()
    ]
    now = datetime.now()
    prompt = f"""Write ONE new post in this account's exact voice.

Now: {now.strftime('%A %Y-%m-%d %H:%M')}

High-performing examples:
{json.dumps(examples, ensure_ascii=False)}

Rules:
- Match tone, length, and rhetorical habits
- Timely if possible, but stay authentic
- Return ONLY the tweet text (no quotes, no preface)
"""
    raw = chat_complete(
        [
            {
                "role": "system",
                "content": "Return only the tweet text. Nothing else.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=800,
    )
    if not raw:
        return None
    text = strip_fences(raw).strip().strip('"').strip("'")
    return text


def engagement_chart(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_circle(size=90, opacity=0.7)
        .encode(
            x=alt.X("created_at:T", title="Posted"),
            y=alt.Y("engagement:Q", title="Engagement (likes / views)", scale=alt.Scale(zero=False)),
            color=alt.Color("engagement:Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(title="Engagement")),
            tooltip=[
                alt.Tooltip("text:N", title="Tweet"),
                alt.Tooltip("created_at:T", title="When", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("engagement:Q", title="Engagement", format=".4f"),
                alt.Tooltip("favorite_count:Q", title="Likes"),
                alt.Tooltip("view_count:Q", title="Views"),
            ],
        )
        .properties(height=420, title="Engagement over time")
        .interactive()
    )


def favorites_chart(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_circle(size=90, opacity=0.7)
        .encode(
            x=alt.X("created_at:T", title="Posted"),
            y=alt.Y("favorite_count:Q", title="Likes", scale=alt.Scale(zero=False)),
            color=alt.Color("favorite_count:Q", scale=alt.Scale(scheme="tealblues"), legend=alt.Legend(title="Likes")),
            tooltip=[
                alt.Tooltip("text:N", title="Tweet"),
                alt.Tooltip("created_at:T", title="When", format="%Y-%m-%d %H:%M"),
                alt.Tooltip("favorite_count:Q", title="Likes"),
                alt.Tooltip("view_count:Q", title="Views"),
                alt.Tooltip("engagement:Q", title="Engagement", format=".4f"),
            ],
        )
        .properties(height=420, title="Likes over time")
        .interactive()
    )


# ——— Sidebar ———
with st.sidebar:
    st.markdown("### 𝕏 Data")
    uploaded = st.file_uploader("Upload tweet CSV", type=["csv"])
    if uploaded is not None and st.session_state.get("_upload_name") != uploaded.name:
        loaded = load_csv(uploaded)
        if loaded is not None:
            st.session_state.df = loaded
            st.session_state.data_label = uploaded.name
            st.session_state._upload_name = uploaded.name
            st.session_state.analysis_result = None
            st.session_state.personality_data = None
            st.session_state.generated_tweet = None

    st.caption("Or load a demo export:")
    for label, path in DEMO_FILES.items():
        if path.exists() and st.button(f"Load {label}", use_container_width=True, key=f"demo_{label}"):
            loaded = load_csv(path)
            if loaded is not None:
                st.session_state.df = loaded
                st.session_state.data_label = label
                st.session_state.analysis_result = None
                st.session_state.personality_data = None
                st.session_state.generated_tweet = None
                st.rerun()

    if st.session_state.df is not None:
        st.divider()
        st.markdown("### Navigate")
        tabs = ["Overview", "Engagement", "Vibe Report", "Personality", "Generate Tweet"]
        selected = st.radio(
            "Section",
            tabs,
            index=tabs.index(st.session_state.selected_tab)
            if st.session_state.selected_tab in tabs
            else 0,
            label_visibility="collapsed",
        )
        st.session_state.selected_tab = selected
        st.caption(f"Model: `{MODEL}`")

# ——— Main ———
st.markdown(
    '<p class="hero-title">Tweet Analytics <span class="accent">Dashboard</span></p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-sub">Upload an export, chart engagement, then let GPT read the room.</p>',
    unsafe_allow_html=True,
)

df = st.session_state.df
if df is None:
    st.info("Upload a CSV in the sidebar, or load a demo account to start.")
    st.markdown(
        """
        **Expected columns:** `text`, `view_count`, `created_at`, `favorite_count`

        TwExportly-style exports work out of the box.
        """
    )
else:
    label = st.session_state.data_label or "your account"
    st.success(f"Loaded **{len(df):,}** tweets from **{label}**", icon="✅")

    tab = st.session_state.selected_tab

    if tab == "Overview":
        st.subheader("Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Tweets", f"{len(df):,}")
        c2.metric("Total likes", f"{int(df['favorite_count'].sum()):,}")
        c3.metric("Avg likes", f"{df['favorite_count'].mean():,.0f}")
        c4, c5, c6 = st.columns(3)
        c4.metric("Max likes", f"{int(df['favorite_count'].max()):,}")
        c5.metric("Avg engagement", f"{df['engagement'].mean() * 100:.2f}%")
        if "retweet_count" in df.columns:
            c6.metric("Total reposts", f"{int(df['retweet_count'].sum()):,}")
        else:
            c6.metric("Date range", f"{df['created_at'].min().strftime('%b %Y')}–{df['created_at'].max().strftime('%b %Y')}")

        st.markdown("#### Tweet table")
        left, right = st.columns([3, 1])
        with left:
            sort_col = st.selectbox(
                "Sort by",
                ["engagement", "favorite_count", "view_count", "created_at"],
                format_func=lambda x: {
                    "engagement": "Engagement",
                    "favorite_count": "Likes",
                    "view_count": "Views",
                    "created_at": "Date",
                }[x],
            )
        with right:
            ascending = st.toggle("Ascending", value=False)

        show = df.sort_values(sort_col, ascending=ascending)
        st.dataframe(
            show[["text", "created_at", "engagement", "favorite_count", "view_count"]],
            use_container_width=True,
            height=560,
            hide_index=True,
            column_config={
                "text": st.column_config.TextColumn("Tweet", width="large"),
                "created_at": st.column_config.DatetimeColumn("Posted", format="YYYY-MM-DD HH:mm"),
                "engagement": st.column_config.NumberColumn("Engagement", format="%.4f"),
                "favorite_count": st.column_config.NumberColumn("Likes", format="%d"),
                "view_count": st.column_config.NumberColumn("Views", format="%d"),
            },
        )

    elif tab == "Engagement":
        st.subheader("Engagement charts")
        st.altair_chart(engagement_chart(df), use_container_width=True)
        st.altair_chart(favorites_chart(df), use_container_width=True)

    elif tab == "Vibe Report":
        st.subheader("Vibe report")
        st.caption("GPT reads a mix of top-engagement and recent posts, then writes a strategist-style HTML brief.")
        if st.button("Analyze my vibe", type="primary", use_container_width=True):
            with st.spinner(f"Asking {MODEL} to read the timeline…"):
                result = analyze_vibe(df)
                if result:
                    st.session_state.analysis_result = strip_fences(result)

        if st.session_state.analysis_result:
            st.markdown(
                f'<div class="tweet-card">{st.session_state.analysis_result}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Hit the button to generate the report.")

    elif tab == "Personality":
        st.subheader("Personality profile")
        if st.button("Analyze personality", type="primary", use_container_width=True):
            with st.spinner(f"Scoring traits with {MODEL}…"):
                result = analyze_personality(df)
                if result:
                    st.session_state.personality_data = result

        data = st.session_state.personality_data
        if data:
            st.plotly_chart(create_radar(data), use_container_width=True)
            if summary := data.get("summary"):
                st.markdown(
                    f'<div class="tweet-card"><strong class="accent">Summary</strong>'
                    f'<p style="margin:0.5rem 0 0 0;color:#0f1419;line-height:1.6;">{summary}</p></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("#### Trait scores")
            scores = [(k, v) for k, v in data.items() if k != "summary" and isinstance(v, (int, float))]
            cols = st.columns(4)
            for i, (trait, score) in enumerate(scores):
                cols[i % 4].metric(trait, f"{int(score)}/100")
        else:
            st.info("Run personality analysis to see the radar.")

    elif tab == "Generate Tweet":
        st.subheader("Generate a tweet")
        st.caption("Few-shot from high performers — same voice, new post.")
        if st.button("Generate tweet", type="primary", use_container_width=True):
            with st.spinner(f"Drafting with {MODEL}…"):
                tweet = generate_tweet(df)
                if tweet:
                    st.session_state.generated_tweet = tweet

        if st.session_state.generated_tweet:
            handle = re.sub(r"[^a-zA-Z0-9_]", "", str(label).split()[0].lower()) or "account"
            left, mid, right = st.columns([1, 2.2, 1])
            with mid:
                st.markdown(
                    f"""
                    <div class="tweet-card">
                      <div style="display:flex;gap:0.85rem;align-items:flex-start;">
                        <img src="https://abs.twimg.com/sticky/default_profile_images/default_profile_400x400.png"
                             width="48" height="48" style="border-radius:50%;" />
                        <div style="flex:1;">
                          <div><strong>{label}</strong>
                            <span class="tweet-meta"> @{handle} · just now</span></div>
                          <div class="tweet-body">{st.session_state.generated_tweet}</div>
                          <div class="tweet-stats">💬 18 &nbsp; 🔁 42 &nbsp; ❤️ 310 &nbsp; 📊 24K</div>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Generate a draft to preview the post card.")
