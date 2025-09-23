#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/22 22:10
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   about.py
# @Desc     :   

from streamlit import title, expander, caption

title("**Application Information**")
with expander("About this application", expanded=True):
    caption("- This platform is built with Streamlit for interactive digit recognition.")
    caption("- It allows training, testing, and real-time prediction using MNIST.")
    caption("- Designed for beginners and enthusiasts to explore deep learning visually.")
    caption("- Provides visualization tools like scatter plots and decision boundaries for experiments.")
