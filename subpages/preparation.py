#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/22 22:11
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preparation.py
# @Desc     :

from streamlit import (empty, sidebar, subheader, session_state, button,
                       spinner, rerun, columns, slider, caption,
                       markdown, image)
from tensorflow.keras.datasets import mnist

from utils.helper import Timer

empty_messages: empty = empty()
train_col, test_col = columns(2, gap="small")

pre_sessions: list[str] = ["X_train", "X_test", "y_train", "y_test", "pTimer"]
for session in pre_sessions:
    session_state.setdefault(session, None)

with sidebar:
    subheader("Data Preparation Settings")

    if session_state["X_train"] is None:
        empty_messages.error("Please load the data on the Home page first.")

        if button("Load Mnist Data", type="primary", width="stretch"):
            with spinner("Loading Mnist Data...", show_time=True, width="stretch"):
                with Timer("Mnist Data Loading") as session_state["pTimer"]:
                    (
                        (session_state["X_train"], session_state["y_train"]),
                        (session_state["X_test"], session_state["y_test"]),
                    ) = mnist.load_data()
            rerun()
    else:
        empty_messages.success(
            f"{session_state["pTimer"]} Train Data has been loaded {session_state["X_train"].shape}."
        )

        print(type(session_state["X_train"]), type(session_state["X_test"]))
        print(type(session_state["y_train"]), type(session_state["y_test"]))
        print(session_state["X_train"].shape, session_state["X_test"].shape)
        print(session_state["y_train"].shape, session_state["y_test"].shape)

        index_train = slider(
            "Select the number of samples to display",
            min_value=0,
            max_value=len(session_state["X_train"]) - 1,
            value=27,
            step=1,
            help="Select the number of samples to display (max 60,000)",
        )
        caption(f"The maximum number of train samples is {len(session_state['X_train'])}.")
        index_test = slider(
            "Select the number of samples to display",
            min_value=0,
            max_value=len(session_state["X_test"]) - 1,
            value=27,
            step=1,
            help="Select the number of samples to display (max 10,000)",
        )
        caption(f"The maximum number of test samples is {len(session_state['X_test'])}.")

        with train_col:
            markdown(f"### Training Data Sample")
            image(
                session_state["X_train"][index_train],
                caption=f"**{session_state["y_train"][index_train]}**",
                width="stretch"
            )
        with test_col:
            markdown(f"### Testing Data Sample")
            image(
                session_state["X_test"][index_test],
                caption=f"**{session_state["y_test"][index_test]}**",
                width="stretch"
            )

        if button("Clear Mnist Data", type="secondary", width="stretch"):
            for session in pre_sessions:
                session_state[session] = None
            rerun()
