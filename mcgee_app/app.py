"""
Streamlit Web Application for McGee EBM Physical Exam Strategy.

This application provides an educational interface for generating 
evidence-based physical examination strategies using Likelihood Ratio data.

Run with: streamlit run mcgee_app/app.py
"""

import time
import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Must be first Streamlit command
st.set_page_config(
    page_title="AI-PEx",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import application modules
try:
    from config import (
        APP_TITLE,
        IRB_WARNING,
        validate_config,
        HIGH_YIELD_LR_POSITIVE_THRESHOLD,
        HIGH_YIELD_LR_NEGATIVE_THRESHOLD,
        MONGODB_URI,
    )
    from rag_engine import (
        run_rag_pipeline,
        run_rag_pipeline_with_sample_data,
        MongoDBClient,
        PYMONGO_AVAILABLE,
        OPENAI_AVAILABLE,
    )
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)


# Custom CSS for streamlined styling
def load_custom_css():
    st.markdown("""
    <style>
    :root {
        --clinic-blue: #1f3b5b;
        --clinic-blue-strong: #16324f;
        --clinic-gray: #475569;
        --clinic-border: #e2e8f0;
        --clinic-bg: #f8fafc;
        --clinic-bg-alt: #f1f5f9;
        --clinic-accent: #2563eb;
        --clinic-positive: #1e7b57;
        --clinic-negative: #b91c1c;
        --sidebar-width: 0px;
        --chat-input-font-size: 1rem;
        --chat-input-line-height: 1.4;
        --chat-input-padding-y: 0.8rem;
        --chat-input-min-height: 3.25rem;
        --chat-input-max-height: 7.2rem;
        --chat-input-container-max-height: 9.5rem;
    }

    body {
        --sidebar-width: 0px;
    }
    body:has(section[data-testid="stSidebar"][aria-expanded="true"]) {
        --sidebar-width: 21rem;
    }
    body:has(section[data-testid="stSidebar"][aria-expanded="false"]) {
        --sidebar-width: 0px;
    }

    section.main > div {
        padding-top: 3.25rem;
    }
    .block-container {
        padding-top: 3.25rem;
        padding-bottom: var(--chat-input-container-max-height);
        max-width: 1200px;
    }

    .app-header {
        margin: 0 0 1rem 0;
    }
    .app-title {
        font-size: 2.25rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--clinic-blue);
        margin: 0;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: var(--clinic-gray);
        margin-top: 0.4rem;
        max-width: 760px;
        line-height: 1.5;
    }

    .disclaimer-card {
        background: #ffffff;
        border: 1px solid var(--clinic-border);
        border-left: 4px solid var(--clinic-accent);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin: 0.75rem 0 1.25rem 0;
        box-shadow: 0 1px 0 rgba(15, 23, 42, 0.06);
        max-width: 900px;
    }
    .disclaimer-card p {
        margin: 0 0 0.6rem 0;
        color: #0f172a;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .disclaimer-card p:last-child {
        margin-bottom: 0;
    }
    .disclaimer-card strong {
        color: var(--clinic-blue-strong);
    }

    .banner {
        background: var(--clinic-bg);
        border: 1px solid var(--clinic-border);
        border-left: 4px solid var(--clinic-accent);
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        color: #0f172a;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }

    .chat-empty-state {
        color: var(--clinic-gray);
        font-size: 0.95rem;
        padding: 0.5rem 0 1rem 0;
    }

    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 0;
        left: var(--sidebar-width, 0px);
        right: 0;
        width: calc(100% - var(--sidebar-width, 0px));
        background: #ffffff;
        border-top: 1px solid var(--clinic-border);
        padding: 0.5rem;
        z-index: 100;
        transition: left 0.2s ease, width 0.2s ease;
    }
    div[data-testid="stChatInput"] > div {
        max-width: 1200px;
        width: 100%;
        margin: 0 auto;
        display: flex !important;
        align-items: flex-end !important;
    }
    div[data-testid="stChatInput"] form {
        display: flex !important;
        align-items: flex-end !important;
        width: 100%;
        position: relative;
    }
    div[data-testid="stChatInput"] form > div {
        display: flex !important;
        align-items: flex-end !important;
        width: 100%;
    }
    div[data-testid="stChatInput"] [data-testid="stChatInputContainer"],
    div[data-testid="stChatInput"] form > div > div {
        display: flex !important;
        align-items: flex-end !important;
    }
    div[data-testid="stChatInput"] textarea {
        border-radius: 10px;
        border: 1px solid var(--clinic-border);
        background: var(--clinic-bg);
        font-size: var(--chat-input-font-size);
        line-height: var(--chat-input-line-height);
        padding: 0.8rem 0.7rem !important;
        min-height: var(--chat-input-min-height);
        max-height: var(--chat-input-max-height);
        overflow-y: auto;
        resize: none;
        align-self: flex-end;
        box-sizing: border-box;
        width: calc(100% - 4rem) !important;
    }
    div[data-testid="stChatInput"] textarea:focus {
        border-color: var(--clinic-accent);
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.12);
    }
    div[data-testid="stChatInput"] button {
        position: absolute !important;
        right: 1rem;
        bottom: 0.5rem;
        top: 50%;
        transform: translateY(-50%);
        flex-shrink: 0;
        height: var(--chat-input-min-height);
        width: var(--chat-input-min-height);
        border-radius: 10px;
    }

    div[data-testid="stChatMessage"] {
        padding: 0.35rem 0;
    }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0.35rem;
    }

    .streamlit-expanderHeader {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: var(--clinic-blue-strong) !important;
    }
    div[data-testid="stExpander"] > details {
        border: 1px solid var(--clinic-border);
        border-radius: 10px;
        background: #ffffff;
    }
    div[data-testid="stExpander"] > details > summary {
        padding: 0.5rem 0.7rem;
        background: var(--clinic-bg);
        border-radius: 10px;
    }

    hr {
        margin: 1rem 0;
    }

    /* Sidebar compactness */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    section[data-testid="stSidebar"] h3 {
        font-size: 0.9rem;
        margin: 0.5rem 0 0.25rem 0;
    }
    section[data-testid="stSidebar"] p {
        font-size: 0.82rem;
        line-height: 1.35;
        margin-bottom: 0.4rem;
    }
    section[data-testid="stSidebar"] hr {
        margin: 0.5rem 0;
    }
    section[data-testid="stSidebar"] [data-testid="stAlert"] {
        padding: 0.4rem 0.6rem;
        font-size: 0.82rem;
    }
    </style>
    """, unsafe_allow_html=True)


def display_header():
    """Display the application header."""
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-title">{APP_TITLE}</div>
            <div class="app-subtitle">Evidence-Based Physical Exam Strategist</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def format_disclaimer_html(disclaimer_text: str) -> str:
    """Convert the disclaimer markdown into HTML for structured styling."""
    def bold_to_html(text: str) -> str:
        parts = text.split("**")
        for index in range(1, len(parts), 2):
            parts[index] = f"<strong>{parts[index]}</strong>"
        return "".join(parts)

    paragraphs = [
        paragraph.strip()
        for paragraph in disclaimer_text.strip().split("\n\n")
        if paragraph.strip()
    ]
    html_paragraphs = "".join(
        f"<p>{bold_to_html(paragraph)}</p>" for paragraph in paragraphs
    )
    return f'<div class="disclaimer-card">{html_paragraphs}</div>'


def display_irb_warning():
    """Display the IRB compliance warning."""
    disclaimer_html = format_disclaimer_html(IRB_WARNING)
    st.markdown(disclaimer_html, unsafe_allow_html=True)


def display_config_status():
    """Display configuration validation status in sidebar."""
    with st.sidebar:
        st.markdown("### System Status")
        
        validation = validate_config()
        
        if validation['valid']:
            st.success("Configuration valid")
        else:
            st.error("Configuration issues")
            for error in validation['errors']:
                st.error(f"• {error}")
        
        for warning in validation.get('warnings', []):
            st.warning(warning)
        
        # Check MongoDB connection
        st.markdown("### Database Status")
        
        if PYMONGO_AVAILABLE:
            try:
                client = MongoDBClient()
                if client.connect():
                    diseases = client.get_all_diseases()
                    record_count = client.collection.count_documents({})
                    st.success("MongoDB connected")
                    st.info(f"{len(diseases)} diseases, {record_count} total records")
                    client.close()
                else:
                    st.error("MongoDB not connected")
                    st.error("""
                    **Required:** MongoDB must be running and accessible.
                    
                    **Quick Setup:**
                    1. Install: `brew tap mongodb/brew && brew install mongodb-community`
                    2. Start: `brew services start mongodb-community`
                    3. Load data: `python mcgee_app/load_sample.py`
                    """)
            except Exception as e:
                st.error(f"MongoDB error: {str(e)[:100]}")
                st.error("Please ensure MongoDB is installed and running.")
        else:
            st.error("pymongo not installed")
            st.error("Install with: `pip install pymongo`")
        
        # OpenAI status
        st.markdown("### AI Status")
        
        if OPENAI_AVAILABLE:
            from config import OPENAI_API_KEY
            if OPENAI_API_KEY:
                st.success("OpenAI configured")
            else:
                st.error("OPENAI_API_KEY not set")
        else:
            st.error("openai library not installed")


def display_sidebar_info():
    """Display educational information in sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Understanding LRs")
        
        st.markdown("""
        **Likelihood Ratios (LRs)** help quantify how a test result changes 
        disease probability.
        
        **LR+ (Positive LR):**
        - LR+ > 10: Large increase in probability
        - LR+ 5-10: Moderate increase
        - LR+ 2-5: Small increase
        - LR+ < 2: Minimal change
        
        **LR- (Negative LR):**
        - LR- < 0.1: Large decrease in probability
        - LR- 0.1-0.2: Moderate decrease
        - LR- 0.2-0.5: Small decrease
        - LR- > 0.5: Minimal change
        """)
        
        st.markdown("---")
        st.markdown("### High-Yield Criteria")
        st.markdown(f"""
        This tool highlights maneuvers meeting:
        - **LR+ ≥ {HIGH_YIELD_LR_POSITIVE_THRESHOLD}** (strong positive predictor)
        - **LR- ≤ {HIGH_YIELD_LR_NEGATIVE_THRESHOLD}** (strong negative predictor)
        """)


def display_example_cases():
    """Display example cases for users to try."""
    with st.sidebar:
        with st.expander("Example Cases", expanded=False):
            st.markdown("""
            **Case 1: Possible DVT**
            ```
            45-year-old woman with left leg swelling and pain for 3 days. 
            Redness and warmth in calf. Recent 12-hour flight one week ago. 
            No fever. On oral contraceptives.
            ```
            
            **Case 2: Suspected Pneumonia**
            ```
            68-year-old man with productive cough for 5 days, fever to 101°F,
            shortness of breath with exertion. History of COPD. 
            Increased sputum production, now greenish.
            ```
            
            **Case 3: Heart Failure Evaluation**
            ```
            72-year-old woman with progressive dyspnea over 2 weeks. 
            Orthopnea (uses 3 pillows). Bilateral ankle swelling. 
            History of hypertension and diabetes.
            ```
            
            **Case 4: Acute Abdomen**
            ```
            23-year-old male with right lower quadrant pain for 18 hours.
            Pain started around umbilicus, now localized to RLQ.
            Low-grade fever, nausea, loss of appetite.
            ```
            """)


def create_evidence_dataframe(evidence: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a formatted DataFrame from evidence data.
    
    Handles nested schema from MongoDB where:
    - source.ebm_box_label contains the diagnosis
    - result_buckets[0].lr_positive/lr_negative contain LR values
    - original_finding contains the physical finding
    """
    if not evidence:
        return pd.DataFrame()
    
    # Flatten nested schema for display
    flattened_data = []
    for item in evidence:
        # Extract from nested schema
        source = item.get('source', {})
        result_buckets = item.get('result_buckets', [{}])
        bucket = result_buckets[0] if result_buckets else {}
        maneuver = item.get('maneuver', {})
        
        flattened_data.append({
            'Diagnosis': source.get('ebm_box_label') or item.get('ebm_box_label', '-'),
            'Physical Finding': item.get('original_finding', '-'),
            'Maneuver': maneuver.get('name') or item.get('maneuver_base', '-'),
            'LR+': bucket.get('lr_positive') or item.get('pos_lr_numeric'),
            'LR-': bucket.get('lr_negative') or item.get('neg_lr_numeric'),
            'Pretest %': bucket.get('pretest_prob') or item.get('pretest_prob_numeric'),
        })
    
    df = pd.DataFrame(flattened_data)
    
    # Format numeric columns
    if 'Pretest %' in df.columns:
        df['Pretest %'] = df['Pretest %'].apply(
            lambda x: f"{x:.0f}%" if pd.notna(x) and x is not None else "-"
        )
    if 'LR+' in df.columns:
        df['LR+'] = df['LR+'].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) and x is not None else "-"
        )
    if 'LR-' in df.columns:
        df['LR-'] = df['LR-'].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) and x is not None else "-"
        )
    
    return df


def parse_lr_value(lr_string: str) -> Optional[float]:
    """
    Parse a likelihood ratio string value to a float.
    
    Handles various formats like "15.0", "0.05", "2.5", etc.
    Returns None if parsing fails.
    """
    if not lr_string:
        return None
    try:
        # Handle string representations
        lr_str = str(lr_string).strip()
        return float(lr_str)
    except (ValueError, TypeError):
        return None


def init_session_state():
    """Initialize chat session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def format_differential_markdown(differential_items: List[Dict[str, str]]) -> str:
    """Format differential diagnosis items as chat-friendly markdown."""
    if not differential_items:
        return "**Differential diagnosis**\n\nNo differential diagnoses were generated."

    lines = ["**Differential diagnosis**"]
    for item in differential_items:
        name = (item.get("name") or "").strip()
        rationale = (item.get("rationale") or "").strip()
        if not rationale:
            rationale = "Considered given the presenting symptoms and common etiologies."
        lines.append(f"- **{name}** — {rationale}")
    return "\n".join(lines)


def stream_markdown(text: str, delay: float = 0.012):
    """Stream markdown text progressively for a chat-like effect."""
    placeholder = st.empty()
    if not text:
        return
    words = text.split(" ")
    buffer = ""
    for word in words:
        buffer = f"{buffer} {word}".strip()
        placeholder.markdown(buffer)
        time.sleep(delay)
    placeholder.markdown(buffer)


def render_strategy_message(result: Dict[str, Any]):
    """Render the structured strategy and evidence in a chat message."""
    st.markdown("**Physical exam strategy**")
    display_compact_summary(result)

    strategy_structured = result.get("strategy_structured", {})
    parse_error = strategy_structured.get("parse_error")
    if parse_error:
        st.warning(
            "The structured response could not be parsed reliably. "
            "Showing the fallback strategy below."
        )

    if strategy_structured.get("sections"):
        display_structured_strategy(strategy_structured)
    else:
        strategy = result.get("strategy", "No strategy generated")
        st.markdown(strategy)

    display_raw_evidence(result)


def display_compact_summary(result: Dict[str, Any]):
    """Display a compact inline summary of the analysis."""
    categories = result.get('categories', {})
    high_yield_count = (
        len(categories.get('high_yield_positive', [])) +
        len(categories.get('high_yield_negative', []))
    )
    evidence_count = len(result.get('evidence', []))
    processing_time = result.get('processing_time', 0)
    
    st.markdown(
        f'<div style="color: var(--clinic-gray); font-size: 0.9rem; margin: 0.25rem 0 0.75rem 0;">'
        f'Analyzed <strong>{evidence_count}</strong> evidence records · '
        f'<span style="color: var(--clinic-positive); font-weight: 600;">{high_yield_count} high-yield maneuvers</span> · '
        f'{processing_time:.1f}s</div>',
        unsafe_allow_html=True
    )


def display_structured_strategy(strategy_structured: Dict[str, Any]):
    """
    Display the physical exam strategy as collapsible body system sections,
    with diseases grouped underneath and maneuvers categorized as Rule In or Rule Out.
    
    Hierarchy: Body System -> Disease -> Rule In/Rule Out -> Maneuvers
    
    Args:
        strategy_structured: Dictionary with 'sections' list containing disease groupings
    """
    sections = strategy_structured.get('sections', [])
    
    if not sections:
        st.info("No exam recommendations generated. Try providing more specific symptoms.")
        return
    
    # Custom CSS for the hierarchical display
    st.markdown("""
    <style>
    /* Body system container */
    .system-block {
        background: transparent;
        border-radius: 12px;
        padding: 4px 2px 12px 2px;
        margin: 0;
        width: 100%;
        box-sizing: border-box;
        
    }
    .streamlit-expanderHeader {
        font-size: 1.2rem !important;
    }

    /* Disease block styling */
    .disease-block {
        background: transparent;
        border: none;
        padding: 0;
        margin: 16px 0;
    }
    .disease-header {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 0;
        padding: 8px 10px;
        border-radius: 8px;
        border: none;
        background: #e2e8f0;
    }
    .disease-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--clinic-blue-strong);
    }
    .disease-meta {
        font-size: 0.8rem;
        color: var(--clinic-gray);
        white-space: nowrap;
    }
    .disease-body {
        padding-left: 2px;
    }
    
    /* Rule category headers - neutral styling */
    .rule-category {
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0 0 6px 0;
        padding: 4px 8px;
        border-radius: 6px;
        color: var(--clinic-gray);
        background-color: #eef2f7;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        border: none;
    }
    
    
    /* Maneuver card styling - neutral by default */
    .maneuver-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
        margin-left: 4px;
        border: none;
        box-shadow: 0 1px 0 rgba(15, 23, 42, 0.06);
    }
    /* High-yield positive: LR+ > 10 only */
    .maneuver-card.high-yield-positive {
        background: #eef7f1;
    }
    /* High-yield negative: LR- < 0.1 only */
    .maneuver-card.high-yield-negative {
        background: #fef2f2;
    }
    .maneuver-name {
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--clinic-blue-strong);
        margin-bottom: 2px;
    }
    .maneuver-lr {
        font-size: 0.85rem;
        font-weight: 600;
    }
    .maneuver-lr.positive {
        color: var(--clinic-positive);
    }
    .maneuver-lr.negative {
        color: var(--clinic-negative);
    }
    .maneuver-technique {
        font-size: 0.85rem;
        color: var(--clinic-gray);
        font-style: italic;
        margin-top: 4px;
    }
    
    /* Empty state */
    .no-maneuvers {
        font-size: 0.85rem;
        color: var(--clinic-gray);
        font-style: italic;
        margin-left: 4px;
        padding: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    for section in sections:
        system_name = section.get('system', 'General Exam')
        diseases = section.get('diseases', [])
        
        # Count total maneuvers across all diseases
        total_rule_in = sum(len(d.get('rule_in', [])) for d in diseases)
        total_rule_out = sum(len(d.get('rule_out', [])) for d in diseases)
        total_maneuvers = total_rule_in + total_rule_out
        
        # Create section header with counts
        section_label = f"{system_name} ({len(diseases)} diseases, {total_maneuvers} maneuvers)"
        
        with st.expander(section_label, expanded=True):
            if not diseases:
                st.markdown(
                    '<div class="system-block">'
                    '<p class="no-maneuvers">No specific diseases identified for this system.</p>'
                    '</div>',
                    unsafe_allow_html=True
                )
                continue

            system_html = ['<div class="system-block">']

            for disease in diseases:
                disease_name = disease.get('name', 'Unknown Disease')
                rule_in_maneuvers = disease.get('rule_in', [])
                rule_out_maneuvers = disease.get('rule_out', [])
                
                # Sort Rule In maneuvers by LR+ descending (highest first = most significant)
                rule_in_maneuvers = sorted(
                    rule_in_maneuvers,
                    key=lambda m: parse_lr_value(m.get('lr_positive', '')) or 0,
                    reverse=True
                )
                
                # Sort Rule Out maneuvers by LR- ascending (lowest first = most significant)
                rule_out_maneuvers = sorted(
                    rule_out_maneuvers,
                    key=lambda m: parse_lr_value(m.get('lr_negative', '')) or float('inf'),
                    reverse=False
                )
                
                disease_meta = f"{len(rule_in_maneuvers)} rule-in · {len(rule_out_maneuvers)} rule-out"
                system_html.append(
                    f'<div class="disease-block">'
                    f'<div class="disease-header">'
                    f'<div class="disease-name">{disease_name}</div>'
                    f'<div class="disease-meta">{disease_meta}</div>'
                    f'</div>'
                    f'<div class="disease-body">'
                )
                
                # Rule In section
                if rule_in_maneuvers:
                    system_html.append(
                        f'<div class="rule-category rule-in">Rule In - {len(rule_in_maneuvers)} maneuvers</div>'
                    )
                    for maneuver in rule_in_maneuvers:
                        name = maneuver.get('name', 'Unknown')
                        lr_positive = maneuver.get('lr_positive', '')
                        technique = maneuver.get('technique', '')
                        
                        # Parse LR+ and determine if high-yield (LR+ > 10)
                        lr_value = parse_lr_value(lr_positive)
                        is_high_yield = lr_value is not None and lr_value > HIGH_YIELD_LR_POSITIVE_THRESHOLD
                        card_class = "maneuver-card high-yield-positive" if is_high_yield else "maneuver-card"
                        
                        lr_html = f'<div class="maneuver-lr positive">LR+ {lr_positive}</div>' if lr_positive else ''
                        technique_html = f'<div class="maneuver-technique">{technique}</div>' if technique else ''
                        
                        system_html.append(
                            f'<div class="{card_class}">'
                            f'<div class="maneuver-name">{name}</div>'
                            f'{lr_html}'
                            f'{technique_html}'
                            f'</div>'
                        )
                
                # Rule Out section
                if rule_out_maneuvers:
                    system_html.append(
                        f'<div class="rule-category rule-out">Rule Out - {len(rule_out_maneuvers)} maneuvers</div>'
                    )
                    for maneuver in rule_out_maneuvers:
                        name = maneuver.get('name', 'Unknown')
                        lr_negative = maneuver.get('lr_negative', '')
                        technique = maneuver.get('technique', '')
                        
                        # Parse LR- and determine if high-yield (LR- < 0.1)
                        lr_value = parse_lr_value(lr_negative)
                        is_high_yield = lr_value is not None and lr_value < HIGH_YIELD_LR_NEGATIVE_THRESHOLD
                        card_class = "maneuver-card high-yield-negative" if is_high_yield else "maneuver-card"
                        
                        lr_html = f'<div class="maneuver-lr negative">LR- {lr_negative}</div>' if lr_negative else ''
                        technique_html = f'<div class="maneuver-technique">{technique}</div>' if technique else ''
                        
                        system_html.append(
                            f'<div class="{card_class}">'
                            f'<div class="maneuver-name">{name}</div>'
                            f'{lr_html}'
                            f'{technique_html}'
                            f'</div>'
                        )
                
                # If disease has no maneuvers in either category
                if not rule_in_maneuvers and not rule_out_maneuvers:
                    system_html.append(
                        '<p class="no-maneuvers">No high-yield maneuvers available for this disease.</p>'
                    )
                
                system_html.append("</div></div>")

            system_html.append("</div>")
            st.markdown("".join(system_html), unsafe_allow_html=True)


def display_raw_evidence(result: Dict[str, Any]):
    """Display raw evidence data in a single expandable section."""
    all_evidence = result.get('evidence', [])
    
    if not all_evidence:
        return
    
    with st.expander(f"View Evidence Data ({len(all_evidence)} records)", expanded=False):
        df = create_evidence_dataframe(all_evidence)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(
                f"Evidence filtered by LR thresholds: "
                f"LR+ ≥ {HIGH_YIELD_LR_POSITIVE_THRESHOLD} (rules in), "
                f"LR- ≤ {HIGH_YIELD_LR_NEGATIVE_THRESHOLD} (rules out)"
            )


def display_results(result: Dict[str, Any]):
    """Display the pipeline results with collapsible body system sections."""
    if not result.get('success'):
        st.error(f"Error: {result.get('error', 'Unknown error occurred')}")
        return
    
    # Compact summary line
    display_compact_summary(result)
    
    # Display structured strategy (collapsible body system sections)
    strategy_structured = result.get('strategy_structured', {})
    parse_error = strategy_structured.get('parse_error')
    if parse_error:
        st.warning(
            "The structured response could not be parsed reliably. "
            "Showing the fallback strategy below."
        )
    
    if strategy_structured.get('sections'):
        display_structured_strategy(strategy_structured)
    else:
        # Fallback to legacy markdown display if structured not available
        st.markdown("### Physical Exam Strategy")
        strategy = result.get('strategy', 'No strategy generated')
        st.markdown(strategy)
    
    st.markdown("---")
    
    # Single expandable evidence section
    display_raw_evidence(result)


def main():
    """Main application entry point."""
    # Load custom CSS
    load_custom_css()
    
    # Check imports
    if not IMPORTS_SUCCESSFUL:
        st.error(f"Failed to load application modules: {IMPORT_ERROR}")
        st.info("Please ensure all dependencies are installed: pip install -r requirements.txt")
        return
    
    # Initialize session state for chat history
    init_session_state()

    prompt = st.chat_input("Describe the patient's presenting symptoms and context...")
    if prompt and not prompt.strip():
        prompt = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Display header and disclaimers
    display_header()
    if not st.session_state.messages:
        display_irb_warning()

    # Sidebar content
    display_config_status()
    display_sidebar_info()
    display_example_cases()

    # Render chat history
    if not st.session_state.messages:
        st.markdown(
            '<div class="chat-empty-state">Enter a patient presentation to start.</div>',
            unsafe_allow_html=True
        )
    else:
        for message in st.session_state.messages:
            with st.chat_message(message.get("role", "assistant")):
                message_type = message.get("type")
                if message.get("role") == "user":
                    st.markdown(message.get("content", ""))
                elif message_type == "differential":
                    st.markdown(message.get("content", ""))
                elif message_type == "strategy":
                    result = message.get("result", {})
                    render_strategy_message(result)
                elif message_type == "error":
                    st.error(message.get("content", "An error occurred."))
                else:
                    st.markdown(message.get("content", ""))

    # Process new request
    if prompt:
        with st.chat_message("assistant"):
            status_box = st.status("Preparing response...", expanded=False)
            
            # Placeholder for differential - will be populated by callback
            diff_placeholder = st.empty()
            # Store rendered markdown for session state
            rendered_differential = {"markdown": "", "items": []}

            def status_cb(message: str):
                status_box.update(label=message, state="running")

            def on_differential_ready(diagnoses, details):
                """Callback to display differential immediately when generated."""
                items = details or [{"name": dx, "rationale": ""} for dx in diagnoses]
                markdown = format_differential_markdown(items)
                rendered_differential["markdown"] = markdown
                rendered_differential["items"] = items
                with diff_placeholder.container():
                    stream_markdown(markdown)

            # Validate configuration
            status_cb("Validating configuration...")
            validation = validate_config()
            if not validation["valid"]:
                status_box.update(label="Configuration error", state="error")
                st.error("Configuration error")
                st.markdown("\n".join([f"- {err}" for err in validation["errors"]]))
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "error",
                    "content": "Configuration error: " + "; ".join(validation["errors"])
                })
                return

            # Require MongoDB - no fallback to sample data
            status_cb("Checking database connection...")
            client = MongoDBClient()
            if not client.connect():
                status_box.update(label="Database connection failed", state="error")
                st.error("MongoDB connection failed")
                st.markdown(
                    "Ensure MongoDB is installed, running, and initialized with data."
                )
                st.markdown(f"Connection URI: `{MONGODB_URI}`")
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "error",
                    "content": "MongoDB connection failed. Ensure MongoDB is running and initialized."
                })
                return
            client.close()

            result = run_rag_pipeline(
                prompt,
                status_callback=status_cb,
                differential_callback=on_differential_ready
            )
            if not result.get("success"):
                error_message = result.get("error", "Unknown error occurred")
                status_box.update(label="Request failed", state="error")
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "error",
                    "content": error_message
                })
                return

            status_box.update(label="Response ready", state="complete")

            # Get differential items for session state (already displayed by callback)
            differential_items = rendered_differential["items"] or result.get("differential_details") or [
                {"name": dx, "rationale": ""} for dx in result.get("differential", [])
            ]
            differential_markdown = rendered_differential["markdown"] or format_differential_markdown(differential_items)

        with st.chat_message("assistant"):
            render_strategy_message(result)

        st.session_state.messages.append({
            "role": "assistant",
            "type": "differential",
            "content": differential_markdown,
            "items": differential_items
        })
        st.session_state.messages.append({
            "role": "assistant",
            "type": "strategy",
            "result": result
        })

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--clinic-gray); font-size: 0.85rem;">
        <p>
            Data source: McGee's Evidence-Based Physical Diagnosis, 3rd Edition<br>
            Educational use only. Always consult qualified healthcare professionals for patient care.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()



