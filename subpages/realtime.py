#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/23 16:14
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   realtime.py
# @Desc     :

from cv2 import cvtColor, resize, COLOR_BGR2GRAY, INTER_AREA
from keras.models import load_model
from numpy import uint8
from os import path
from streamlit import (empty, sidebar, subheader, session_state, button,
                       spinner, rerun, slider, selectbox, columns, image)
from streamlit_drawable_canvas import st_canvas
from subpages.train import MODEL_PATH

from utils.helper import Timer

empty_messages: empty = empty()
empty_write_title: empty = empty()
col_write, col_small = columns(2, gap="small")
empty_pred_result: empty = empty()

pre_sessions: list[str] = ["X_train", "X_test", "y_train", "y_test"]
for session in pre_sessions:
    session_state.setdefault(session, None)
test_sessions: list[str] = ["rtTimer", "img_flat", "y_prediction"]
for session in test_sessions:
    session_state.setdefault(session, None)

with sidebar:
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

                    drawing_modes: list[str] = ["point", "freedraw", "line", "rect", "circle", "transform"]
                    drawing_mode: str = selectbox(
                        "Drawing tool: ",
                        options=drawing_modes,
                        index=1,
                        disabled=True,
                        help="Select the drawing tool",
                    )

                    stroke_width = slider(
                        "Stroke width: ",
                        min_value=5,
                        max_value=30,
                        value=15,
                        step=1,
                        help="Width of the stroke",
                    )

                    canvas_size: int = 280

                    empty_write_title.markdown("#### Real-time Digit Recognition")
                    with col_write:
                        canvas_result = st_canvas(
                            stroke_width=stroke_width,
                            stroke_color="white",
                            background_color="black",
                            update_streamlit=True,
                            height=canvas_size,
                            width=canvas_size,
                            drawing_mode=drawing_mode,
                        )

                    if canvas_result.image_data is not None:
                        # The image data is in RGBA format, we need to convert it with uint8
                        img = canvas_result.image_data.astype(uint8)
                        # Convert the color image to grayscale
                        # img_gray = img[:, :, 0]
                        img_gray = cvtColor(img, COLOR_BGR2GRAY)
                        # Resize the image to 28x28
                        small: int = canvas_size // 10
                        img_small = resize(img_gray, (small, small), interpolation=INTER_AREA)
                        # Normalize the image to [0, 1]
                        img_normalized = img_small.astype("float32") / 255.0
                        # Flatten the image to (1, 784) with reshape
                        session_state["img_flat"] = img_normalized.reshape(1, -1)

                        with col_small:
                            image(
                                img_small,
                                caption="The zoom-out image after resizing to 28x28",
                                width="stretch"
                            )

                        if session_state["y_prediction"] is None:
                            empty_messages.warning("Model is ready. Please draw a digit (0-9) on the canvas.")

                            if button("Predict the Number", type="primary", width="stretch"):
                                with spinner("Predicting the Number...", show_time=True, width="stretch"):
                                    with Timer("Real-time Prediction") as session_state["rtTimer"]:
                                        model = load_model(MODEL_PATH)
                                        # Make predictions
                                        y_pred_probabilities = model.predict(session_state["img_flat"])
                                        session_state["y_prediction"] = y_pred_probabilities.argmax(axis=1)[0]

                                rerun()
                        else:
                            empty_messages.success(f"{session_state["rtTimer"]} Prediction completed.")
                            empty_pred_result.markdown(f"🧠 Predicted Number: **{session_state["y_prediction"]}**")

                            if button("Clear Drawing Canvas", type="secondary", width="stretch"):
                                canvas_result.image_data = None
                                for session in test_sessions:
                                    session_state[session] = None
                                rerun()
