#!/usr/bin/env python3
"""
Main entry point for the Dengue Forecasting Dashboard on Streamlit Cloud
"""
import streamlit as st
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main dashboard
if __name__ == "__main__":
    from dengue_dashboard import main
    main() 