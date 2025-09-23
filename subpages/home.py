#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/22 22:10
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :

from streamlit import title, expander, caption, empty

empty_message = empty()
empty_message.info("Please check the details at the different pages of core functions.")

title("ML - Make Moons")
with expander("**INTRODUCTION**", expanded=True):
    caption("+ 📂 Load MNIST dataset and preprocess for model training.")
    caption("+ 🧠 Train a Multi-Layer Perceptron with custom epochs and batch size.")
    caption("+ 📊 Visualize training metrics in real-time (loss, accuracy, precision, recall, AUC).")
    caption("+ 🧪 Test the trained model on the MNIST test dataset.")
    caption("+ ✏️ Draw digits on a canvas and get instant predictions.")
