"""
Streamlit Web Application for McGee EBM Physical Exam Strategy.

This application provides an educational interface for generating 
evidence-based physical examination strategies using Likelihood Ratio data.

Run with: streamlit run mcgee_app/app.py
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Must be first Streamlit command
st.set_page_config(
    page_title="EBM Physical Exam Strategist",
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
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .warning-box {
        background-color: #FFF3CD;
        border: 1px solid #FFECB5;
        border-left: 4px solid #FFC107;
        padding: 0.75rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    .lr-high-positive {
        color: #198754;
        font-weight: 600;
    }
    .lr-high-negative {
        color: #DC3545;
        font-weight: 600;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E3A5F;
        color: white;
        font-weight: 600;
        padding: 0.6rem;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #2C5282;
    }
    /* Style expanders for body system sections */
    .streamlit-expanderHeader {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #1E3A5F !important;
    }
    /* Compact text area */
    .stTextArea textarea {
        font-size: 0.95rem;
    }
    </style>
    """, unsafe_allow_html=True)


def display_header():
    """Display the application header."""
    st.markdown(f'<h1 class="main-header">🩺 {APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Evidence-Based Physical Examination Learning Tool</p>',
        unsafe_allow_html=True
    )


def display_irb_warning():
    """Display the IRB compliance warning."""
    st.warning(IRB_WARNING)


def display_config_status():
    """Display configuration validation status in sidebar."""
    with st.sidebar:
        st.markdown("### ⚙️ System Status")
        
        validation = validate_config()
        
        if validation['valid']:
            st.success("✅ Configuration valid")
        else:
            st.error("❌ Configuration issues")
            for error in validation['errors']:
                st.error(f"• {error}")
        
        for warning in validation.get('warnings', []):
            st.warning(f"⚠️ {warning}")
        
        # Check MongoDB connection
        st.markdown("---")
        st.markdown("### 📊 Database Status")
        
        if PYMONGO_AVAILABLE:
            try:
                client = MongoDBClient()
                if client.connect():
                    diseases = client.get_all_diseases()
                    record_count = client.collection.count_documents({})
                    st.success(f"✅ MongoDB connected")
                    st.info(f"📋 {len(diseases)} diseases, {record_count} total records")
                    client.close()
                else:
                    st.error("❌ MongoDB not connected")
                    st.error("""
                    **Required:** MongoDB must be running and accessible.
                    
                    **Quick Setup:**
                    1. Install: `brew tap mongodb/brew && brew install mongodb-community`
                    2. Start: `brew services start mongodb-community`
                    3. Load data: `python mcgee_app/load_sample.py`
                    """)
            except Exception as e:
                st.error(f"❌ MongoDB Error: {str(e)[:100]}")
                st.error("Please ensure MongoDB is installed and running.")
        else:
            st.error("❌ pymongo not installed")
            st.error("Install with: `pip install pymongo`")
        
        # OpenAI status
        st.markdown("---")
        st.markdown("### 🤖 AI Status")
        
        if OPENAI_AVAILABLE:
            from config import OPENAI_API_KEY
            if OPENAI_API_KEY:
                st.success("✅ OpenAI configured")
            else:
                st.error("❌ OPENAI_API_KEY not set")
        else:
            st.error("❌ openai library not installed")


def display_sidebar_info():
    """Display educational information in sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📚 Understanding LRs")
        
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
        st.markdown("### 🔬 High-Yield Criteria")
        st.markdown(f"""
        This tool highlights maneuvers meeting:
        - **LR+ ≥ {HIGH_YIELD_LR_POSITIVE_THRESHOLD}** (strong positive predictor)
        - **LR- ≤ {HIGH_YIELD_LR_NEGATIVE_THRESHOLD}** (strong negative predictor)
        """)


def display_example_cases():
    """Display example cases for users to try."""
    with st.expander("📋 Example Cases to Try", expanded=False):
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
    """Create a formatted DataFrame from evidence data."""
    if not evidence:
        return pd.DataFrame()
    
    df = pd.DataFrame(evidence)
    
    # Select and rename columns - using exam_evidence_free schema field names
    column_map = {
        'ebm_box_label': 'Diagnosis',
        'original_finding': 'Physical Finding',
        'maneuver_base': 'Maneuver',
        'result_modifier': 'Result',
        'pretest_prob_numeric': 'Pretest %',
        'pos_lr_numeric': 'LR+',
        'neg_lr_numeric': 'LR-',
    }
    
    # Filter to existing columns
    cols_to_use = [c for c in column_map.keys() if c in df.columns]
    df = df[cols_to_use].rename(columns=column_map)
    
    # Format numeric columns
    if 'Pretest %' in df.columns:
        df['Pretest %'] = df['Pretest %'].apply(
            lambda x: f"{x:.0f}%" if pd.notna(x) else "-"
        )
    if 'LR+' in df.columns:
        df['LR+'] = df['LR+'].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "-"
        )
    if 'LR-' in df.columns:
        df['LR-'] = df['LR-'].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "-"
        )
    
    return df


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
        f'<p style="color: #666; font-size: 0.95rem; margin-bottom: 1.5rem;">'
        f'Analyzed <strong>{evidence_count}</strong> evidence records · '
        f'<span style="color: #198754; font-weight: 600;">{high_yield_count} high-yield maneuvers</span> · '
        f'{processing_time:.1f}s</p>',
        unsafe_allow_html=True
    )


def display_top_picks(strategy_structured: Dict[str, Any]):
    """
    Display the top 2-3 highest-yield maneuvers as a quick reference.
    
    Args:
        strategy_structured: Dictionary with 'sections' list from GPT
    """
    sections = strategy_structured.get('sections', [])
    
    if not sections:
        return
    
    # Collect all high-yield maneuvers across all sections
    all_high_yield = []
    for section in sections:
        for maneuver in section.get('maneuvers', []):
            if maneuver.get('high_yield', False):
                all_high_yield.append({
                    **maneuver,
                    'system': section.get('system', 'General')
                })
    
    if not all_high_yield:
        return
    
    # Sort by LR value and take top 3
    def extract_lr_for_sort(m: Dict[str, Any]) -> float:
        lr_info = m.get('lr_info', '')
        match = re.search(r'[\d.]+', lr_info)
        if match:
            val = float(match.group())
            if 'LR-' in lr_info:
                return 1 / val if val > 0 else 0
            return val
        return 0
    
    top_picks = sorted(all_high_yield, key=extract_lr_for_sort, reverse=True)[:3]
    
    if not top_picks:
        return
    
    # Display top picks section
    st.markdown("""
    <style>
    .top-picks-container {
        background: linear-gradient(135deg, #1E3A5F 0%, #2C5282 100%);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .top-picks-header {
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .top-pick-item {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .top-pick-item:last-child {
        margin-bottom: 0;
    }
    .top-pick-name {
        font-weight: 600;
        color: #1E3A5F;
        font-size: 0.95rem;
    }
    .top-pick-purpose {
        color: #495057;
        font-size: 0.85rem;
        margin-top: 2px;
    }
    .top-pick-lr {
        color: #198754;
        font-weight: 600;
        font-size: 0.8rem;
        margin-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    picks_html = '<div class="top-picks-container">'
    picks_html += '<div class="top-picks-header">🎯 Quick Reference - Top High-Yield Maneuvers</div>'
    
    for i, pick in enumerate(top_picks, 1):
        name = pick.get('name', 'Unknown')
        purpose = pick.get('purpose', '')
        lr_info = pick.get('lr_info', '')
        
        # Use suggestive language
        if purpose:
            # Make the purpose more suggestive if it contains absolute language
            purpose = purpose.replace('diagnoses', 'may suggest')
            purpose = purpose.replace('confirms', 'may indicate')
            purpose = purpose.replace('rules out', 'may help rule out')
            purpose = purpose.replace('rules in', 'may support')
        
        picks_html += f'''
        <div class="top-pick-item">
            <div class="top-pick-name">{i}. {name}</div>
            <div class="top-pick-purpose">{purpose}</div>
            <div class="top-pick-lr">{lr_info}</div>
        </div>
        '''
    
    picks_html += '</div>'
    st.markdown(picks_html, unsafe_allow_html=True)


def display_structured_strategy(strategy_structured: Dict[str, Any]):
    """
    Display the physical exam strategy as collapsible body system sections.
    
    Args:
        strategy_structured: Dictionary with 'sections' list from GPT
    """
    sections = strategy_structured.get('sections', [])
    
    if not sections:
        st.info("No exam recommendations generated. Try providing more specific symptoms.")
        return
    
    # Custom CSS for maneuver cards
    st.markdown("""
    <style>
    .maneuver-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-left: 4px solid #6c757d;
    }
    .maneuver-card.high-yield {
        border-left-color: #198754;
        background: linear-gradient(135deg, #d1e7dd 0%, #c3e6cb 100%);
    }
    .maneuver-name {
        font-weight: 600;
        font-size: 1rem;
        color: #1E3A5F;
        margin-bottom: 4px;
    }
    .maneuver-purpose {
        font-size: 0.9rem;
        color: #495057;
        margin-bottom: 4px;
    }
    .maneuver-lr {
        font-size: 0.85rem;
        color: #198754;
        font-weight: 600;
    }
    .maneuver-technique {
        font-size: 0.85rem;
        color: #6c757d;
        font-style: italic;
    }
    .high-yield-badge {
        display: inline-block;
        background-color: #198754;
        color: white;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 10px;
        margin-left: 8px;
        vertical-align: middle;
    }
    </style>
    """, unsafe_allow_html=True)
    
    def extract_lr_value(maneuver: Dict[str, Any]) -> float:
        """Extract numeric LR value for sorting (higher is better)."""
        lr_info = maneuver.get('lr_info', '')
        match = re.search(r'[\d.]+', lr_info)
        if match:
            val = float(match.group())
            # For LR-, lower values are more useful, so invert for sorting
            if 'LR-' in lr_info:
                return 1 / val if val > 0 else 0
            return val
        return 0
    
    for section in sections:
        system_name = section.get('system', 'General Exam')
        maneuvers = section.get('maneuvers', [])
        
        # Sort maneuvers by LR value (highest first)
        maneuvers = sorted(maneuvers, key=extract_lr_value, reverse=True)
        
        # Count high-yield maneuvers for this section
        high_yield_in_section = sum(1 for m in maneuvers if m.get('high_yield', False))
        
        # Create section header with maneuver count
        section_label = f"{system_name} ({len(maneuvers)} maneuvers"
        if high_yield_in_section > 0:
            section_label += f", {high_yield_in_section} high-yield"
        section_label += ")"
        
        with st.expander(section_label, expanded=True):
            for maneuver in maneuvers:
                is_high_yield = maneuver.get('high_yield', False)
                card_class = "maneuver-card high-yield" if is_high_yield else "maneuver-card"
                
                name = maneuver.get('name', 'Unknown')
                purpose = maneuver.get('purpose', '')
                lr_info = maneuver.get('lr_info', '')
                technique = maneuver.get('technique', '')
                
                badge_html = '<span class="high-yield-badge">HIGH YIELD</span>' if is_high_yield else ''
                lr_html = f'<div class="maneuver-lr">{lr_info}</div>' if lr_info else ''
                technique_html = f'<div class="maneuver-technique">{technique}</div>' if technique else ''
                
                st.markdown(f"""
                <div class="{card_class}">
                    <div class="maneuver-name">{name}{badge_html}</div>
                    <div class="maneuver-purpose">{purpose}</div>
                    {lr_html}
                    {technique_html}
                </div>
                """, unsafe_allow_html=True)


def display_raw_evidence(result: Dict[str, Any]):
    """Display raw evidence data in a single expandable section."""
    all_evidence = result.get('evidence', [])
    
    if not all_evidence:
        return
    
    with st.expander(f"📊 View Raw Evidence Data ({len(all_evidence)} records)", expanded=False):
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
        st.error(f"❌ Error: {result.get('error', 'Unknown error occurred')}")
        return
    
    # Compact summary line
    display_compact_summary(result)
    
    # Display structured strategy (collapsible body system sections)
    strategy_structured = result.get('strategy_structured', {})
    
    if strategy_structured.get('sections'):
        # Show top picks first for quick reference
        display_top_picks(strategy_structured)
        
        # Then show all body system sections
        display_structured_strategy(strategy_structured)
    else:
        # Fallback to legacy markdown display if structured not available
        st.markdown("### 📋 Physical Exam Strategy")
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
    
    # Display header
    display_header()
    
    # Display IRB warning
    display_irb_warning()
    
    # Display sidebar
    display_config_status()
    display_sidebar_info()
    
    # Main content area
    st.markdown("### 🩺 Enter Patient Symptoms")
    
    # Display example cases
    display_example_cases()
    
    # Symptom input
    symptoms = st.text_area(
        "Describe the patient's presenting symptoms:",
        height=150,
        placeholder="Enter patient symptoms, history, and relevant clinical context...\n\n"
                    "Example: 45-year-old woman with left leg swelling and pain for 3 days. "
                    "Redness and warmth in calf. Recent 12-hour flight one week ago.",
        help="Provide relevant clinical details including onset, duration, "
             "associated symptoms, and pertinent history."
    )
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_clicked = st.button(
            "🔬 Generate Physical Exam Strategy",
            type="primary",
            use_container_width=True
        )
    
    # Process request
    if generate_clicked:
        if not symptoms.strip():
            st.warning("⚠️ Please enter patient symptoms before generating a strategy.")
            return
        
        # Validate configuration
        validation = validate_config()
        if not validation['valid']:
            for error in validation['errors']:
                st.error(f"Configuration error: {error}")
            return
        
        # Run the pipeline
        with st.spinner("🔄 Analyzing symptoms and generating strategy..."):
            try:
                # Require MongoDB - no fallback to sample data
                client = MongoDBClient()
                if not client.connect():
                    st.error("❌ **MongoDB Connection Failed**")
                    st.error("""
                    **Cannot connect to MongoDB database.**
                    
                    Please ensure:
                    1. MongoDB is installed and running
                    2. MongoDB is accessible at: `{}`
                    3. The database has been initialized with data
                    
                    **To set up MongoDB:**
                    ```bash
                    # Install MongoDB (macOS)
                    brew tap mongodb/brew
                    brew install mongodb-community
                    brew services start mongodb-community
                    
                    # Load data
                    python mcgee_app/load_sample.py
                    ```
                    
                    See `mcgee_app/MONGODB_EXPLANATION.md` for detailed setup instructions.
                    """.format(MONGODB_URI))
                    return
                
                # Run pipeline with MongoDB
                result = run_rag_pipeline(symptoms)
                
                # Display results
                display_results(result)
                
            except Exception as e:
                logger.exception("Application error")
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Please check your configuration and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>
            📚 Data source: McGee's Evidence-Based Physical Diagnosis, 3rd Edition<br>
            🔬 This tool is for educational purposes only<br>
            ⚕️ Always consult qualified healthcare professionals for patient care
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()



