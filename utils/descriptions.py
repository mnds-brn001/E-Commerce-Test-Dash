import streamlit as st


def render_glass_card(content: str) -> None:
    """
    Renders content inside a glass-effect card.

    Args:
        content (str): The content of the card.
    """
    is_dark_theme = st.get_option("theme.base") == "dark"
    text_color = "rgba(255,255,255,0.9)" if is_dark_theme else "rgba(0,0,0,0.9)"
    border_color = "rgba(255, 255, 255, 0.3)"  # Semi-transparent white border
    bg_color = "rgba(255, 255, 255, 0.1)"  # Semi-transparent white background
    shadow_color = "rgba(0, 0, 0, 0.1)"  # Light shadow

    st.markdown(
        f"""
        <div style="
            backdrop-filter: blur(10px);
            background: {bg_color};
            padding: 20px;
            border-radius: 15px;
            text-align: left;
            font-size: 16px;
            font-family: 'Inter', sans-serif;
            border: 1px solid {border_color};
            box-shadow: 0 4px 30px {shadow_color};">
            <p style= ;">{content}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_page_title(title: str, icon: str = None) -> None:
    """
    Renders a centered and styled page title with optional icon.

    Args:
        title (str): The title text to display
        icon (str, optional): An emoji or icon to display before the title
    """
    is_dark_theme = st.get_option("theme.base") == "dark"
    text_color = "rgba(255,255,255,0.9)" if is_dark_theme else "rgba(0,0,0,0.9)"
    border_color = "rgba(255, 255, 255, 0.3)"  # Semi-transparent white border
    bg_color = "rgba(255, 255, 255, 0.1)"  # Semi-transparent white background
    shadow_color = "rgba(0, 0, 0, 0.1)"  # Light shadow

    # Add icon if provided
    title_text = f"{icon} {title}" if icon else title

    st.markdown(
        f"""
        <div style="
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            background: linear-gradient(135deg, {bg_color}, rgba(255, 255, 255, 0.05));
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            font-family: 'Inter', sans-serif;
            border: 1px solid {border_color};
            box-shadow: 0 4px 30px {shadow_color};
            margin-bottom: 30px;
            ">
            <h1 style="
                margin: 0;
                font-size: 2.5em;
                font-weight: 700;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                letter-spacing: -0.5px;
                ">
                {title_text}
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )