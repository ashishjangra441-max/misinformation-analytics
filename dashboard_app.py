"""
Social Media Misinformation Analytics Dashboard
================================================
Interactive Streamlit app for analyzing fake vs real news with:
- Dataset overview & filters
- Word clouds
- Sentiment & emotion analysis
- Advanced topic modeling (LDA + interactive explorer)
- Network graph (nodes & edges) of topic-keyword relationships
- Document similarity graph (TF-IDF + cosine similarity)
"""

import os
import ast
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS

# ML / NLP
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

# Network analysis
import networkx as nx


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Misinformation Analytics",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — explicit colors so cards stay readable on both light & dark themes
st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    h1, h2, h3 { font-family: 'Georgia', serif; }

    /* Metric cards — force readable text on any theme */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #4c6ef5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label p {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #94a3b8 !important;
    }

    /* Insight callout box — readable on dark backgrounds */
    .insight-box {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-left: 4px solid #4c6ef5;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #e2e8f0 !important;
        line-height: 1.6;
    }
    .insight-box b { color: #ffffff; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────
st.title("📱 Social Media Misinformation Dashboard")
st.markdown(
    "Explore patterns in **fake vs real news** through word clouds, sentiment, "
    "topic modeling, and network graphs. Use the sidebar to navigate and filter."
)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file_path_or_buffer):
    return pd.read_csv(file_path_or_buffer)


local_file_name = "dashboard_data.csv"
df = None

if os.path.exists(local_file_name):
    df = load_data(local_file_name)
    st.sidebar.success(f"✅ Auto-connected to: `{local_file_name}`")
else:
    st.sidebar.header("Data Upload")
    st.sidebar.warning(f"`{local_file_name}` not found in repo.")
    uploaded_file = st.sidebar.file_uploader("Upload processed dataset (CSV)", type="csv")
    if uploaded_file is not None:
        df = load_data(uploaded_file)


# ─────────────────────────────────────────────────────────────
# CACHED COMPUTE HELPERS
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fitting LDA topic model...")
def fit_lda(texts, n_topics=6, max_features=2000):
    """Fit LDA on a list of cleaned text documents. Returns model, vectorizer, doc-topic matrix."""
    texts = [str(t) for t in texts if isinstance(t, str) and t.strip()]
    if len(texts) < n_topics:
        return None, None, None, None

    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=5,
        max_df=0.85,
    )
    dtm = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="online",
        max_iter=15,
    )
    doc_topic = lda.fit_transform(dtm)
    feature_names = vectorizer.get_feature_names_out()
    return lda, vectorizer, doc_topic, feature_names


def get_top_words(lda_model, feature_names, n_top=10):
    """Return list[list[(word, weight)]] for each topic."""
    topics = []
    for comp in lda_model.components_:
        top_idx = comp.argsort()[: -n_top - 1 : -1]
        topics.append([(feature_names[i], float(comp[i])) for i in top_idx])
    return topics


# ─────────────────────────────────────────────────────────────
# TOPIC AUTO-NAMING
# Maps LDA topic keywords to human-readable theme names.
# ─────────────────────────────────────────────────────────────
THEME_KEYWORDS = {
    "Political Campaigns":     ["trump", "clinton", "election", "campaign", "vote",
                                "voters", "candidate", "republican", "democrat",
                                "obama", "hillary", "primary", "ballot", "poll"],
    "Violence & Crime":        ["police", "killed", "attack", "shooting", "gun",
                                "victim", "crime", "violence", "suspect", "arrested",
                                "death", "murder", "assault", "officer"],
    "Economy & Taxes":         ["tax", "economy", "economic", "jobs", "business",
                                "market", "trade", "money", "billion", "million",
                                "budget", "fiscal", "wages", "growth", "deal"],
    "Immigration":             ["immigration", "immigrants", "border", "refugees",
                                "wall", "mexican", "mexico", "deportation", "illegal",
                                "migrants", "asylum", "ice", "dreamers"],
    "International Politics":  ["foreign", "country", "nations", "diplomatic",
                                "ambassador", "embassy", "summit", "leaders",
                                "international", "global", "minister", "european"],
    "Military Conflict":       ["war", "military", "syria", "isis", "troops",
                                "forces", "weapons", "missile", "army", "soldiers",
                                "combat", "defense", "iran", "iraq", "north korea"],
    "Russia Investigation":    ["russia", "russian", "putin", "fbi", "mueller",
                                "investigation", "probe", "collusion", "comey",
                                "intelligence", "kremlin", "interference"],
    "Social Media Narratives": ["twitter", "facebook", "social", "media", "post",
                                "viral", "online", "internet", "video", "tweet",
                                "shared", "platform"],
    "Healthcare & Policy":     ["health", "healthcare", "obamacare", "insurance",
                                "medical", "patients", "hospital", "doctors",
                                "medicare", "medicaid", "drug"],
    "Legal & Courts":          ["court", "judge", "ruling", "supreme", "lawsuit",
                                "legal", "law", "justice", "attorney", "constitutional",
                                "federal court"],
}


def auto_name_topic(topic_words, theme_dict=THEME_KEYWORDS):
    """
    Score each theme by how many of its keywords appear in the topic's top words,
    weighted by the LDA word importance. Return the best-matching theme name.
    """
    word_weights = {w.lower(): wt for w, wt in topic_words}
    scores = {}
    for theme, kws in theme_dict.items():
        score = 0.0
        hits = 0
        for kw in kws:
            kw_lower = kw.lower()
            # Exact match or substring match (handles "russian" matching "russia")
            for word, wt in word_weights.items():
                if kw_lower == word or kw_lower in word or word in kw_lower:
                    score += wt
                    hits += 1
                    break
        scores[theme] = (score, hits)

    # Pick best theme — needs at least 2 keyword hits to qualify
    best = max(scores.items(), key=lambda x: (x[1][1], x[1][0]))
    if best[1][1] >= 2:
        return best[0]
    # Fallback: use top 2 distinctive keywords
    top_two = [w for w, _ in topic_words[:2]]
    return f"{top_two[0].title()} & {top_two[1].title()}" if len(top_two) == 2 else "Misc"


def get_topic_names(topics, custom_names=None):
    """Return list of names for each topic, with optional manual overrides."""
    names = []
    used = {}
    for i, topic in enumerate(topics):
        if custom_names and i < len(custom_names) and custom_names[i].strip():
            name = custom_names[i].strip()
        else:
            name = auto_name_topic(topic)
        # Disambiguate duplicates
        if name in used:
            used[name] += 1
            name = f"{name} ({used[name]})"
        else:
            used[name] = 1
        names.append(name)
    return names


@st.cache_data(show_spinner="Computing similarity matrix...")
def compute_similarity(texts, max_docs=200, max_features=1000):
    """TF-IDF + cosine similarity on a sample of documents."""
    texts = [str(t) for t in texts if isinstance(t, str) and t.strip()]
    if len(texts) > max_docs:
        # Reproducible sample
        rng = np.random.default_rng(42)
        idx = rng.choice(len(texts), size=max_docs, replace=False)
        texts = [texts[i] for i in idx]
    else:
        idx = np.arange(len(texts))

    if len(texts) < 2:
        return None, None

    vec = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    tfidf = vec.fit_transform(texts)
    sim = cosine_similarity(tfidf)
    return sim, idx


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
if df is not None:
    if "label" in df.columns:
        df["Label_Name"] = df["label"].map({0: "Real News", 1: "Fake News"})

    # ── Sidebar Navigation ─────────────────────────────────
    st.sidebar.header("🧭 Navigation")
    page = st.sidebar.radio(
        "Go to:",
        [
            "1. Dataset Overview",
            "2. Word Clouds",
            "3. Sentiment Analysis",
            "4. Topic Modeling",
            "5. Topic–Keyword Network",
            "6. Document Similarity Graph",
        ],
    )

    # ── Global Filters ─────────────────────────────────────
    st.sidebar.header("🎛️ Global Filters")
    if "subject" in df.columns:
        subjects = ["All"] + sorted(df["subject"].dropna().unique().tolist())
        selected_subject = st.sidebar.selectbox("Filter by Subject", subjects)
        filtered_df = df if selected_subject == "All" else df[df["subject"] == selected_subject]
    else:
        filtered_df = df

    if "Label_Name" in df.columns:
        label_filter = st.sidebar.multiselect(
            "Include Labels",
            options=["Real News", "Fake News"],
            default=["Real News", "Fake News"],
        )
        filtered_df = filtered_df[filtered_df["Label_Name"].isin(label_filter)]

    st.sidebar.caption(f"📦 **{len(filtered_df):,}** articles selected")

    # ─────────────────────────────────────────────────────
    # PAGE 1: DATASET OVERVIEW
    # ─────────────────────────────────────────────────────
    if page == "1. Dataset Overview":
        st.header("📊 Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Articles", f"{len(filtered_df):,}")
        if "label" in filtered_df.columns:
            fake_count = int((filtered_df["label"] == 1).sum())
            real_count = len(filtered_df) - fake_count
            c2.metric("Fake News", f"{fake_count:,}")
            c3.metric("Real News", f"{real_count:,}")
            ratio = (fake_count / len(filtered_df) * 100) if len(filtered_df) else 0
            c4.metric("Fake %", f"{ratio:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            if "Label_Name" in filtered_df.columns:
                fig = px.pie(
                    filtered_df, names="Label_Name", title="News Type Distribution",
                    hole=0.5,
                    color="Label_Name",
                    color_discrete_map={"Real News": "#2ca02c", "Fake News": "#d62728"},
                )
                fig.update_traces(textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "subject" in filtered_df.columns:
                sc = filtered_df["subject"].value_counts().reset_index()
                sc.columns = ["Subject", "Count"]
                fig = px.bar(sc, x="Subject", y="Count", title="Articles by Subject",
                             color="Subject", text_auto=True)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔎 Data Preview")
        display_cols = [c for c in ["title", "subject", "Label_Name", "clean_content"]
                        if c in filtered_df.columns]
        st.dataframe(filtered_df[display_cols].head(100), use_container_width=True)

    # ─────────────────────────────────────────────────────
    # PAGE 2: WORD CLOUDS
    # ─────────────────────────────────────────────────────
    elif page == "2. Word Clouds":
        st.header("☁️ Interactive Word Clouds")

        with st.expander("⚙️ Word Cloud Settings", expanded=True):
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                max_words = st.slider("Max words", 10, 500, 150, step=10)
                bg_color = st.selectbox("Background", ["white", "black", "lightgrey"])
            with cc2:
                fake_cmap = st.selectbox("Fake colormap", ["Reds", "magma", "inferno", "Oranges"])
                real_cmap = st.selectbox("Real colormap", ["Blues", "viridis", "ocean", "Greens"])
            with cc3:
                custom_stop = st.text_input("Extra stopwords (comma-separated)",
                                            "said, will, one, new")

        current_stopwords = set(STOPWORDS)
        if custom_stop:
            current_stopwords.update(w.strip().lower() for w in custom_stop.split(","))

        if "label" in filtered_df.columns and "clean_content" in filtered_df.columns:
            col1, col2 = st.columns(2)
            for col, lbl, title, cmap in [
                (col1, 1, "Fake News Vocabulary", fake_cmap),
                (col2, 0, "Real News Vocabulary", real_cmap),
            ]:
                with col:
                    st.markdown(f"<h4 style='text-align:center;'>{title}</h4>",
                                unsafe_allow_html=True)
                    text = " ".join(
                        filtered_df[filtered_df["label"] == lbl]["clean_content"]
                        .dropna().astype(str).tolist()[:1500]
                    )
                    if text:
                        wc = WordCloud(
                            width=800, height=500, max_words=max_words,
                            stopwords=current_stopwords, background_color=bg_color,
                            colormap=cmap,
                        ).generate(text)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.imshow(wc, interpolation="bilinear")
                        ax.axis("off")
                        st.pyplot(fig)

    # ─────────────────────────────────────────────────────
    # PAGE 3: SENTIMENT ANALYSIS
    # ─────────────────────────────────────────────────────
    elif page == "3. Sentiment Analysis":
        st.header("🎭 Sentiment Analysis")

        if "sentiment_label" in filtered_df.columns and "Label_Name" in filtered_df.columns:
            sd = (filtered_df.groupby(["sentiment_label", "Label_Name"])
                  .size().reset_index(name="Count"))
            fig = px.bar(
                sd, x="sentiment_label", y="Count", color="Label_Name",
                barmode="group", text_auto=True,
                title="Sentiment Distribution: Fake vs Real",
                color_discrete_map={"Real News": "#2ca02c", "Fake News": "#d62728"},
            )
            st.plotly_chart(fig, use_container_width=True)

        if ("dominant_emotion" in filtered_df.columns
                and "Label_Name" in filtered_df.columns):
            ed = (filtered_df.groupby(["dominant_emotion", "Label_Name"])
                  .size().reset_index(name="Count"))
            fig2 = px.bar(
                ed, x="dominant_emotion", y="Count", color="Label_Name",
                barmode="group", title="Dominant Emotions in Articles",
                color_discrete_map={"Real News": "#2ca02c", "Fake News": "#d62728"},
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Sentiment polarity by subject heatmap (bonus)
        if ("sentiment_score" in filtered_df.columns
                and "subject" in filtered_df.columns
                and "Label_Name" in filtered_df.columns):
            st.subheader("Average Sentiment Score by Subject")
            piv = (filtered_df.groupby(["subject", "Label_Name"])["sentiment_score"]
                   .mean().reset_index())
            fig3 = px.density_heatmap(
                piv, x="subject", y="Label_Name", z="sentiment_score",
                color_continuous_scale="RdYlGn", text_auto=".2f",
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ─────────────────────────────────────────────────────
    # PAGE 4: TOPIC MODELING (LDA)
    # ─────────────────────────────────────────────────────
    elif page == "4. Topic Modeling":
        st.header("🧠 Topic Modeling — Latent Dirichlet Allocation")
        st.markdown(
            "<div class='insight-box'>LDA discovers latent themes by finding "
            "co-occurring word patterns across documents. Adjust the number of topics "
            "below and inspect each topic's vocabulary.</div>",
            unsafe_allow_html=True,
        )

        if "clean_content" not in filtered_df.columns:
            st.error("`clean_content` column is required for topic modeling.")
        else:
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                n_topics = st.slider("Number of topics", 3, 15, 6)
            with cc2:
                n_words = st.slider("Top words per topic", 5, 25, 10)
            with cc3:
                sample_size = st.slider("Documents to fit on", 200, 5000,
                                        min(2000, len(filtered_df)), step=100)

            sample = filtered_df["clean_content"].dropna().astype(str).head(sample_size).tolist()
            lda, vec, doc_topic, feature_names = fit_lda(sample, n_topics=n_topics)

            if lda is None:
                st.warning("Not enough data to fit a topic model.")
            else:
                topics = get_top_words(lda, feature_names, n_top=n_words)

                # ── Auto-name topics + allow manual overrides ──
                auto_names = get_topic_names(topics)

                with st.expander("🏷️ Topic Names (auto-detected — click to rename)",
                                 expanded=False):
                    st.caption(
                        "Names are inferred from each topic's keywords. "
                        "Edit any field to override the auto-name."
                    )
                    name_cols = st.columns(min(3, len(auto_names)))
                    custom_names = []
                    for i, auto_n in enumerate(auto_names):
                        with name_cols[i % len(name_cols)]:
                            preview = ", ".join(w for w, _ in topics[i][:5])
                            user_n = st.text_input(
                                f"Topic {i} — *{preview}*",
                                value=auto_n,
                                key=f"tname_{i}_{n_topics}",
                            )
                            custom_names.append(user_n)

                topic_names = get_topic_names(topics, custom_names)

                # Topic word weights as horizontal bar chart grid
                st.subheader("📚 Topic Vocabularies")
                cols = st.columns(min(3, n_topics))
                for i, topic in enumerate(topics):
                    with cols[i % len(cols)]:
                        words, weights = zip(*topic)
                        topic_df = pd.DataFrame({"word": words, "weight": weights})
                        fig = px.bar(
                            topic_df, x="weight", y="word", orientation="h",
                            title=f"#{i} — {topic_names[i]}",
                            color="weight", color_continuous_scale="Viridis",
                        )
                        fig.update_layout(
                            yaxis={"categoryorder": "total ascending"},
                            showlegend=False, coloraxis_showscale=False,
                            height=350, margin=dict(l=10, r=10, t=40, b=10),
                            title_font_size=14,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Topic distribution by label
                st.subheader("📈 Topic Distribution: Fake vs Real")
                doc_assign = doc_topic.argmax(axis=1)
                # Align with the sampled subset
                sub = filtered_df.dropna(subset=["clean_content"]).head(sample_size).copy()
                sub["topic"] = doc_assign[: len(sub)]
                sub["topic_name"] = sub["topic"].apply(
                    lambda x: topic_names[x] if 0 <= x < len(topic_names) else "Unknown"
                )
                if "Label_Name" in sub.columns:
                    td = (sub.groupby(["topic_name", "Label_Name"]).size()
                          .reset_index(name="Count"))
                    fig = px.bar(
                        td, x="topic_name", y="Count", color="Label_Name",
                        barmode="group",
                        title="Document Counts per Topic",
                        labels={"topic_name": "Topic"},
                        color_discrete_map={"Real News": "#2ca02c",
                                            "Fake News": "#d62728"},
                    )
                    fig.update_xaxes(tickangle=-30)
                    st.plotly_chart(fig, use_container_width=True)

                # Save context for later pages
                st.session_state["topics"] = topics
                st.session_state["topic_names"] = topic_names
                st.session_state["doc_topic"] = doc_topic

                # Topic explorer
                st.subheader("🔍 Topic Explorer")
                topic_pick_label = st.selectbox(
                    "Select a topic",
                    options=list(range(n_topics)),
                    format_func=lambda i: f"#{i} — {topic_names[i]}",
                )
                top_words_str = ", ".join(w for w, _ in topics[topic_pick_label])
                st.markdown(
                    f"**Theme:** `{topic_names[topic_pick_label]}`  \n"
                    f"**Top words:** *{top_words_str}*"
                )
                if "title" in sub.columns:
                    matches = sub[sub["topic"] == topic_pick_label][
                        [c for c in ["title", "Label_Name", "subject"] if c in sub.columns]
                    ].head(8)
                    st.dataframe(matches, use_container_width=True)

    # ─────────────────────────────────────────────────────
    # PAGE 5: TOPIC–KEYWORD NETWORK GRAPH
    # ─────────────────────────────────────────────────────
    elif page == "5. Topic–Keyword Network":
        st.header("🕸️ Topic–Keyword Network Graph")
        st.markdown(
            "<div class='insight-box'>This network shows <b>topics as large nodes</b> "
            "and their <b>top keywords as smaller nodes</b>. Edges connect topics to "
            "their characteristic words; edge thickness reflects word importance. "
            "Shared keywords between topics surface as cross-cutting hubs.</div>",
            unsafe_allow_html=True,
        )

        if "clean_content" not in filtered_df.columns:
            st.error("`clean_content` column is required.")
        else:
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                n_topics = st.slider("Number of topics", 3, 12, 6, key="net_n_topics")
            with cc2:
                n_words = st.slider("Top words per topic", 5, 15, 8, key="net_n_words")
            with cc3:
                sample_size = st.slider("Sample size", 200, 5000,
                                        min(1500, len(filtered_df)),
                                        step=100, key="net_sample")

            sample = (filtered_df["clean_content"].dropna().astype(str)
                      .head(sample_size).tolist())
            lda, vec, doc_topic, feature_names = fit_lda(sample, n_topics=n_topics)

            if lda is None:
                st.warning("Not enough data to build the network.")
            else:
                topics = get_top_words(lda, feature_names, n_top=n_words)
                topic_names = get_topic_names(topics)

                # Show the auto-detected names so the user knows what's on the graph
                with st.expander("🏷️ Auto-detected topic names", expanded=True):
                    name_df = pd.DataFrame({
                        "Topic ID": list(range(len(topic_names))),
                        "Auto Name": topic_names,
                        "Top Keywords": [", ".join(w for w, _ in t[:6])
                                         for t in topics],
                    })
                    st.dataframe(name_df, use_container_width=True,
                                 hide_index=True)

                # Build NetworkX graph using NAMED topics
                G = nx.Graph()

                # Topic nodes — use names instead of "Topic 0"
                for ti in range(n_topics):
                    G.add_node(topic_names[ti], node_type="topic", topic_id=ti)

                # Keyword nodes + edges
                for ti, topic_words in enumerate(topics):
                    max_w = max(w for _, w in topic_words) if topic_words else 1.0
                    for word, weight in topic_words:
                        if word not in G:
                            G.add_node(word, node_type="word")
                        G.add_edge(topic_names[ti], word, weight=weight / max_w)

                # Layout
                pos = nx.spring_layout(G, k=0.7, iterations=100, seed=42)

                # Build edge traces
                edge_x, edge_y = [], []
                for u, v, d in G.edges(data=True):
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.8, color="rgba(150,150,150,0.5)"),
                    hoverinfo="none", mode="lines",
                )

                # Node traces (topics vs words separately for color/size)
                topic_x, topic_y, topic_text = [], [], []
                word_x, word_y, word_text, word_degree = [], [], [], []

                for node, data in G.nodes(data=True):
                    x, y = pos[node]
                    if data["node_type"] == "topic":
                        topic_x.append(x); topic_y.append(y); topic_text.append(node)
                    else:
                        word_x.append(x); word_y.append(y); word_text.append(node)
                        word_degree.append(G.degree(node))

                topic_trace = go.Scatter(
                    x=topic_x, y=topic_y, mode="markers+text",
                    text=topic_text, textposition="middle center",
                    textfont=dict(color="white", size=10, family="Arial Black"),
                    marker=dict(size=70, color="#4c6ef5",
                                line=dict(width=2, color="#1c3faa"),
                                symbol="circle"),
                    hovertext=[f"<b>{t}</b>" for t in topic_text],
                    hoverinfo="text", name="Topics",
                )

                word_trace = go.Scatter(
                    x=word_x, y=word_y, mode="markers+text",
                    text=word_text, textposition="top center",
                    textfont=dict(size=10, color="#333"),
                    marker=dict(
                        size=[10 + d * 4 for d in word_degree],
                        color=word_degree,
                        colorscale="Sunset",
                        showscale=True,
                        colorbar=dict(title="Topic<br>Connections", thickness=12),
                        line=dict(width=1, color="white"),
                    ),
                    hovertext=[f"<b>{w}</b><br>Connected to {d} topics"
                               for w, d in zip(word_text, word_degree)],
                    hoverinfo="text", name="Keywords",
                )

                fig = go.Figure(data=[edge_trace, word_trace, topic_trace])
                fig.update_layout(
                    title=dict(text="Topic–Keyword Network",
                               font=dict(size=20)),
                    showlegend=True,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=60),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor="#fafbfc",
                    height=750,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Bridge keywords (connected to multiple topics)
                bridges = [(n, G.degree(n)) for n, d in G.nodes(data=True)
                           if d["node_type"] == "word" and G.degree(n) > 1]
                bridges.sort(key=lambda x: -x[1])
                if bridges:
                    st.subheader("🌉 Bridge Keywords")
                    st.write(
                        "These keywords appear across multiple topics — they "
                        "represent shared vocabulary or potential topic overlap:"
                    )
                    bdf = pd.DataFrame(bridges[:15],
                                       columns=["Keyword", "# Topics Connected"])
                    st.dataframe(bdf, use_container_width=True)

    # ─────────────────────────────────────────────────────
    # PAGE 6: DOCUMENT SIMILARITY GRAPH
    # ─────────────────────────────────────────────────────
    elif page == "6. Document Similarity Graph":
        st.header("🔗 Document Similarity Network")
        st.markdown(
            "<div class='insight-box'>Each <b>node is an article</b>, colored by "
            "Real/Fake. An <b>edge connects two articles</b> if their TF-IDF cosine "
            "similarity exceeds the threshold below. Tight clusters reveal groups "
            "of articles with similar language — often the signature of "
            "coordinated narratives or repeated talking points.</div>",
            unsafe_allow_html=True,
        )

        if "clean_content" not in filtered_df.columns:
            st.error("`clean_content` column is required.")
        else:
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                max_docs = st.slider("Documents to analyze", 50, 500, 150, step=25)
            with cc2:
                threshold = st.slider("Similarity threshold", 0.10, 0.90, 0.35, step=0.05)
            with cc3:
                layout_choice = st.selectbox(
                    "Layout algorithm",
                    ["spring", "kamada_kawai", "circular"],
                )

            # Sample, keeping titles/labels aligned
            base = filtered_df.dropna(subset=["clean_content"]).reset_index(drop=True)
            n = min(max_docs, len(base))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(base), size=n, replace=False)
            sub = base.iloc[idx].reset_index(drop=True)

            sim, _ = compute_similarity(sub["clean_content"].astype(str).tolist(),
                                        max_docs=n)

            if sim is None:
                st.warning("Need at least 2 documents.")
            else:
                # Build graph
                G = nx.Graph()
                for i in range(len(sub)):
                    title = (str(sub.iloc[i]["title"])[:80]
                             if "title" in sub.columns else f"Doc {i}")
                    label = (sub.iloc[i]["Label_Name"]
                             if "Label_Name" in sub.columns else "Unknown")
                    G.add_node(i, title=title, label=label)

                edge_count = 0
                for i in range(len(sub)):
                    for j in range(i + 1, len(sub)):
                        if sim[i, j] >= threshold:
                            G.add_edge(i, j, weight=float(sim[i, j]))
                            edge_count += 1

                # Layout
                if layout_choice == "spring":
                    pos = nx.spring_layout(G, k=0.5, iterations=80, seed=42)
                elif layout_choice == "kamada_kawai":
                    pos = (nx.kamada_kawai_layout(G) if edge_count > 0
                           else nx.spring_layout(G, seed=42))
                else:
                    pos = nx.circular_layout(G)

                # Edge trace
                edge_x, edge_y = [], []
                for u, v in G.edges():
                    x0, y0 = pos[u]; x1, y1 = pos[v]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color="rgba(120,120,120,0.4)"),
                    hoverinfo="none", mode="lines",
                )

                # Node trace
                node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
                color_map = {"Real News": "#2ca02c", "Fake News": "#d62728",
                             "Unknown": "#888"}
                for nid, data in G.nodes(data=True):
                    x, y = pos[nid]
                    node_x.append(x); node_y.append(y)
                    node_color.append(color_map.get(data["label"], "#888"))
                    deg = G.degree(nid)
                    node_size.append(8 + deg * 1.5)
                    node_text.append(
                        f"<b>{data['title']}</b><br>Label: {data['label']}<br>"
                        f"Connections: {deg}"
                    )

                node_trace = go.Scatter(
                    x=node_x, y=node_y, mode="markers",
                    marker=dict(size=node_size, color=node_color,
                                line=dict(width=1, color="white"), opacity=0.85),
                    hovertext=node_text, hoverinfo="text",
                )

                fig = go.Figure(data=[edge_trace, node_trace])
                fig.update_layout(
                    title=dict(
                        text=f"Similarity Network ({len(G.nodes())} docs, "
                             f"{edge_count} connections, threshold={threshold:.2f})",
                        font=dict(size=18),
                    ),
                    showlegend=False, hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=60),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor="#fafbfc", height=700,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Network stats
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Nodes", len(G.nodes()))
                m2.metric("Edges", edge_count)
                density = nx.density(G) if len(G.nodes()) > 1 else 0
                m3.metric("Density", f"{density:.4f}")
                if edge_count > 0:
                    components = nx.number_connected_components(G)
                    m4.metric("Clusters", components)

                # Top connected docs
                if edge_count > 0:
                    st.subheader("🏆 Most Connected Articles (Hubs)")
                    degrees = sorted(G.degree, key=lambda x: -x[1])[:10]
                    hub_rows = []
                    for nid, deg in degrees:
                        d = G.nodes[nid]
                        hub_rows.append({
                            "Title": d["title"],
                            "Label": d["label"],
                            "Connections": deg,
                        })
                    st.dataframe(pd.DataFrame(hub_rows), use_container_width=True)

                # Similarity distribution
                with st.expander("📉 Similarity Score Distribution"):
                    upper = sim[np.triu_indices_from(sim, k=1)]
                    fig_h = px.histogram(
                        upper, nbins=40,
                        title="Pairwise Cosine Similarity Distribution",
                        labels={"value": "Cosine Similarity", "count": "Pair Count"},
                    )
                    fig_h.add_vline(x=threshold, line_dash="dash", line_color="red",
                                    annotation_text=f"Threshold = {threshold:.2f}")
                    fig_h.update_layout(showlegend=False)
                    st.plotly_chart(fig_h, use_container_width=True)

else:
    st.info("👈 Upload a CSV or place `dashboard_data.csv` in the repo root to begin.")
    st.markdown("""
    **Expected columns:**
    - `title`, `clean_content`, `subject`, `label` (0=Real, 1=Fake)
    - Optional: `sentiment_label`, `sentiment_score`, `dominant_emotion`, `dominant_topic`
    """)