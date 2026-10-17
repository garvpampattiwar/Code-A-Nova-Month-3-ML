"""
UI Components for CineMatch: The Last Row Ticket Booth.
Renders marquee headers, vintage cinema ticket stubs, SVG match rings,
and vintage theatre UI elements with HTML escaping.
"""

import os
import html
import inspect
from typing import Dict, Any, List, Optional
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.path.join(CURRENT_DIR, "styles.css")


def load_css() -> None:
    """Read and inject ui/styles.css into the Streamlit document."""
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def _render_html(markup: str) -> None:
    """Safely render HTML without triggering markdown indented code blocks."""
    cleaned = inspect.cleandoc(markup)
    st.markdown(cleaned, unsafe_allow_html=True)


def render_marquee_hero() -> None:
    """Render vintage cinema marquee sign with pulsing amber bulb track."""
    bulbs_top = "".join(
        [f'<div class="bulb bulb-{"even" if i % 2 == 0 else "odd"}"></div>' for i in range(18)]
    )
    bulbs_bottom = "".join(
        [f'<div class="bulb bulb-{"odd" if i % 2 == 0 else "even"}"></div>' for i in range(18)]
    )

    hero_html = f"""
<div class="marquee-container">
<div class="marquee-bulbs-track">{bulbs_top}</div>
<div class="marquee-badge">★ THE LAST ROW CINEMA · REELS &amp; ALGORITHMS ★</div>
<h1 class="marquee-title">CINE<span class="highlight-amber">MATCH</span></h1>
<p class="marquee-tagline">Tell us one film you love. We'll find your next obsession.</p>
<div class="marquee-bulbs-track" style="margin-top:14px; margin-bottom:0;">{bulbs_bottom}</div>
</div>
"""
    _render_html(hero_html)


def render_match_ring(pct: int, size: int = 56, stroke_width: int = 5) -> str:
    """Generate inline circular SVG progress ring with amber/violet gradient."""
    radius = (size - stroke_width) / 2
    circumference = 2 * 3.14159 * radius
    clamped_pct = max(0, min(100, pct))
    offset = circumference - (clamped_pct / 100.0) * circumference

    svg = f"""<div class="match-ring-box">
<svg width="{size}" height="{size}" class="match-ring-svg">
<defs>
<linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#8A2BE2" />
<stop offset="100%" stop-color="#FFB800" />
</linearGradient>
</defs>
<circle cx="{size/2}" cy="{size/2}" r="{radius}" stroke="#1E2536" stroke-width="{stroke_width}" fill="transparent" />
<circle cx="{size/2}" cy="{size/2}" r="{radius}" stroke="url(#ringGrad)" stroke-width="{stroke_width}" fill="transparent" stroke-dasharray="{circumference}" stroke-dashoffset="{offset:.2f}" stroke-linecap="round" />
</svg>
<div>
<div class="match-ring-label">{clamped_pct}%</div>
<div class="match-ring-sub">MATCH</div>
</div>
</div>"""
    return svg


def render_ticket(
    movie: Dict[str, Any],
    stub_num: int = 1,
    target_genres: Optional[List[str]] = None,
) -> None:
    """Render a standard vintage cinema ticket stub."""
    title_esc = html.escape(movie.get("clean_title", "Untitled"))
    year_str = str(movie.get("year")) if movie.get("year") else "N/A"
    rating_mean = movie.get("rating_mean", 0.0)
    rating_count = movie.get("rating_count", 0)
    match_pct = movie.get("match_percentage", 50)
    serial_str = f"ADMIT ONE · NO. {stub_num:04d}"

    target_g_set = set(target_genres or [])
    genres_list = movie.get("genres_list", [])
    genre_pills_html = []
    for g in genres_list:
        g_esc = html.escape(str(g))
        cls = "ticket-genre-pill ticket-genre-highlight" if g in target_g_set else "ticket-genre-pill"
        genre_pills_html.append(f'<span class="{cls}">{g_esc}</span>')
    genres_markup = "".join(genre_pills_html)

    ring_markup = render_match_ring(match_pct, size=50, stroke_width=4.5)

    explanation = movie.get("explanation", "")
    why_markup = ""
    if explanation:
        shared_tags = movie.get("shared_tags", [])
        tags_chips = "".join([f'<span class="why-tag-chip">#{html.escape(t)}</span>' for t in shared_tags[:3]])
        why_markup = f"""<div class="ticket-why-box">
<span style="color:#FFB800; font-weight:700;">WHY THIS:</span> {html.escape(explanation)}
<div style="margin-top:4px;">{tags_chips}</div>
</div>"""

    imdb_btn = ""
    if movie.get("imdb_url"):
        imdb_btn = f'<a href="{movie["imdb_url"]}" target="_blank" class="ticket-imdb-btn">🎟️ IMDb PASS ↗</a>'

    ticket_html = f"""<div class="ticket-stub">
<div class="ticket-notch-left"></div>
<div class="ticket-notch-right"></div>
<div>
<div class="ticket-header">
<span class="ticket-admit">CINEMA ADMISSION</span>
<span class="ticket-serial">{serial_str}</span>
</div>
<div class="ticket-perforation"></div>
<h3 class="ticket-title">{title_esc}</h3>
<div class="ticket-meta">
<span>📅 {year_str}</span>
<span class="ticket-rating">★ {rating_mean:.1f} <span style="color:#5E6773; font-weight:normal;">({rating_count:,})</span></span>
</div>
<div class="ticket-genres">{genres_markup}</div>
{ring_markup}
{why_markup}
</div>
<div style="display:flex; justify-content:flex-end; margin-top:8px;">
{imdb_btn}
</div>
</div>"""
    _render_html(ticket_html)


def render_double_feature(
    movie: Dict[str, Any],
    target_genres: Optional[List[str]] = None,
) -> None:
    """Render the #1 recommendation as a wide featured Double Feature ticket."""
    title_esc = html.escape(movie.get("clean_title", "Untitled"))
    year_str = str(movie.get("year")) if movie.get("year") else "N/A"
    rating_mean = movie.get("rating_mean", 0.0)
    rating_count = movie.get("rating_count", 0)
    match_pct = movie.get("match_percentage", 50)

    target_g_set = set(target_genres or [])
    genres_list = movie.get("genres_list", [])
    genre_pills_html = []
    for g in genres_list:
        g_esc = html.escape(str(g))
        cls = "ticket-genre-pill ticket-genre-highlight" if g in target_g_set else "ticket-genre-pill"
        genre_pills_html.append(f'<span class="{cls}">{g_esc}</span>')
    genres_markup = "".join(genre_pills_html)

    ring_markup = render_match_ring(match_pct, size=64, stroke_width=6)

    explanation = movie.get("explanation", "")
    shared_tags = movie.get("shared_tags", [])
    tags_chips = "".join([f'<span class="why-tag-chip">#{html.escape(t)}</span>' for t in shared_tags[:5]])
    why_markup = f"""<div class="ticket-why-box" style="margin-top:12px;">
<span style="color:#FFB800; font-weight:700;">PROJECTIONIST'S PICK:</span> {html.escape(explanation)}
<div style="margin-top:5px;">{tags_chips}</div>
</div>"""

    imdb_btn = ""
    if movie.get("imdb_url"):
        imdb_btn = f'<a href="{movie["imdb_url"]}" target="_blank" class="ticket-imdb-btn" style="padding:7px 16px; font-size:0.78rem;">🎟️ ADMIT ONE ON IMDb ↗</a>'

    card_html = f"""<div class="double-feature-card">
<div class="double-feature-banner">★ DOUBLE FEATURE · TOP MATCH ★</div>
<div class="double-feature-grid">
<div>
<h2 class="double-feature-title">{title_esc}</h2>
<div class="ticket-meta" style="margin-bottom:12px; font-size:0.85rem;">
<span>📅 RELEASED: {year_str}</span>
<span class="ticket-rating">★ {rating_mean:.1f} <span style="color:#5E6773; font-weight:normal;">({rating_count:,} reviews)</span></span>
<span style="color:#FFB800;">• BAYESIAN SCORE: {movie.get('weighted_rating', 0.0):.2f}</span>
</div>
<div class="ticket-genres">{genres_markup}</div>
{why_markup}
</div>
<div style="display:flex; flex-direction:column; align-items:flex-end; justify-content:space-between; gap:16px;">
{ring_markup}
{imdb_btn}
</div>
</div>
</div>"""
    _render_html(card_html)


def render_target_ticket(movie: Dict[str, Any]) -> None:
    """Render the currently selected target film ticket header."""
    title_esc = html.escape(movie.get("clean_title", "Untitled"))
    year_str = str(movie.get("year")) if movie.get("year") else "N/A"
    genres_markup = "".join([f'<span class="ticket-genre-pill">{html.escape(g)}</span>' for g in movie.get("genres_list", [])])

    imdb_btn = ""
    if movie.get("imdb_url"):
        imdb_btn = f'<a href="{movie["imdb_url"]}" target="_blank" class="ticket-imdb-btn">IMDb ↗</a>'

    target_html = f"""<div class="target-ticket-box">
<div class="target-badge">🎯 NOW PROJECTING IN THEATRE 1</div>
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
<div>
<h3 style="font-family:'Bebas Neue', sans-serif; font-size:2rem; color:#F4E9D8; margin:0 0 4px 0; letter-spacing:0.04em;">
{title_esc} <span style="font-size:1.3rem; color:#8B949E;">({year_str})</span>
</h3>
<div class="ticket-meta" style="margin-bottom:8px;">
<span class="ticket-rating">★ {movie.get('rating_mean', 0.0):.1f} <span style="color:#5E6773; font-weight:normal;">({movie.get('rating_count', 0):,} ratings)</span></span>
<span>• Weighted Bayesian: {movie.get('weighted_rating', 0.0):.2f}</span>
</div>
<div class="ticket-genres">{genres_markup}</div>
</div>
<div>{imdb_btn}</div>
</div>
</div>"""
    _render_html(target_html)


def render_empty_state() -> None:
    """Render tasteful vintage theatre empty state."""
    html_markup = """<div class="theatre-empty-state">
<div class="empty-state-icon">📽️</div>
<h2 class="empty-state-heading">THE SCREEN IS DARK. PICK A FILM TO BEGIN.</h2>
<p class="empty-state-sub">
The projector is loaded with 9,742 authentic films from the MovieLens archives.
Select a title above or click one of our Now Showing classics to illuminate the theatre.
</p>
</div>"""
    _render_html(html_markup)


def render_not_found(query: str, suggestions: Optional[List[Dict[str, Any]]] = None) -> None:
    """Render friendly ticket counter not-found notice with fuzzy alternatives."""
    q_esc = html.escape(query)
    sugg_list = suggestions or []
    sugg_html = ""
    if sugg_list:
        pills = "".join([f'<span class="why-tag-chip" style="font-size:0.78rem;">{html.escape(s.get("clean_title", ""))} ({s.get("year", "")})</span> ' for s in sugg_list[:4]])
        sugg_html = f"<div style='margin-top:10px;'><b>Did you mean:</b><br>{pills}</div>"

    html_markup = f"""<div class="theatre-empty-state" style="border-color:#8F6600;">
<div class="empty-state-icon" style="filter:none;">🎟️</div>
<h2 class="empty-state-heading" style="color:#FFB800;">TICKET COUNTER NOTICE</h2>
<p class="empty-state-sub">
We could not find <b>"{q_esc}"</b> in our 35mm collection.<br>
Please check your spelling or choose from our suggested archive titles below.
{sugg_html}
</p>
</div>"""
    _render_html(html_markup)


def render_footer() -> None:
    """Render vintage theatre footer with How It Works expander."""
    st.markdown("---")
    with st.expander("🎞️ HOW THE PROJECTION ENGINE WORKS · 3-LINE ARCHITECTURE", expanded=False):
        st.markdown("""
        1. **Sparse TF-IDF Representation**: Vectorizes genres (weighted $2\\times$), crowd-sourced user tags, and title tokens into a 25,000-dimensional sparse feature space across 9,742 MovieLens films.
        2. **On-Demand Cosine Similarity**: Measures pairwise angular proximity via Scikit-Learn `linear_kernel` in under **2 milliseconds** with zero dense memory allocation.
        3. **Bayesian Quality Re-Ranking**: Combines content similarity with an IMDb Bayesian weighted formula ($WR = \\frac{v}{v+m}R + \\frac{m}{v+m}C$) to promote critically acclaimed cinema without compromising semantic relevance.
        """)

    footer_html = """<div style="text-align:center; padding:18px 0; font-family:'Space Mono', monospace; font-size:0.75rem; color:#5E6773; letter-spacing:0.06em;">
BUILT BY <span style="color:#FFB800; font-weight:700;">GARV PAMPATTIWAR</span> · CODE-A-NOVA MONTH 3 MACHINE LEARNING TRACK · MOVIELENS 100K ARCHIVE
</div>"""
    _render_html(footer_html)
