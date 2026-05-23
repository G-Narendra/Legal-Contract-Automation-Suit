"""
Legal Contract Automation Suite - Entry point.

Run with: streamlit run app.py
Or API: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

if __name__ == "__main__":
    import streamlit.web.bootstrap as bootstrap
    import sys
    sys.argv = ["streamlit", "run", "app.py", "--server.port=8501"]
    bootstrap.run("app.py", "", [], [])
