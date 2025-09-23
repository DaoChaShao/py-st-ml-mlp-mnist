#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/23 15:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   test.py
# @Desc     :

from os import path
from sklearn.metrics import accuracy_score, r2_score
from streamlit import (empty, sidebar, subheader, session_state, button,
                       spinner, rerun, columns, metric)

from subpages.train import MODEL_PATH
from utils.helper import Timer

empty_messages: empty = empty()
empty_result_title: empty = empty()
col_acc, col_r2 = columns(2, gap="small")

pre_sessions: list[str] = ["X_train", "X_test", "y_train", "y_test"]
for session in pre_sessions:
    session_state.setdefault(session, None)
preprocess_sessions: list[str] = ["X_train_flat", "X_test_flat", "y_train_cat"]
for session in preprocess_sessions:
    session_state.setdefault(session, None)
model_sessions: list[str] = ["model"]
for session in model_sessions:
    session_state.setdefault(session, None)
test_sessions: list[str] = ["tTimer", "y_pred"]
for session in test_sessions:
    session_state.setdefault(session, None)

with sidebar:
    if session_state["X_train"] is None:
        empty_messages.error("Please load the data on the Home page first.")
    else:
        if session_state["X_train_flat"] is None:
            empty_messages.error("Please preprocess the data on the Data Preparation page first.")
        else:
            if not path.exists(MODEL_PATH):
                empty_messages.error("Please train the model on the Model Training page and save it first.")
            else:
                subheader("Model Testing Settings")

                if session_state["y_pred"] is None:
                    empty_messages.info("Model is ready. Please test the model first.")

                    if button("Test the Model", type="primary", width="stretch"):
                        with spinner("Testing the Model...", show_time=True, width="stretch"):
                            with Timer("Model Testing") as session_state["tTimer"]:
                                y_pred_probabilities = session_state["model"].predict(session_state["X_test_flat"])
                                session_state["y_pred"] = y_pred_probabilities.argmax(axis=1)

                        rerun()
                else:
                    empty_messages.success(f"{session_state['tTimer']} Model has been tested.")

                    acc = accuracy_score(session_state["y_test"], session_state["y_pred"])
                    r2 = r2_score(session_state["y_test"], session_state["y_pred"])

                    with col_acc:
                        metric("Accuracy", f"{acc:.4%}")
                    with col_r2:
                        metric("R² Score", f"{r2:.4f}")

                    if button("Retest the Test", type="secondary", width="stretch"):
                        for session in test_sessions:
                            session_state[session] = None
                        rerun()
