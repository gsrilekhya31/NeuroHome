import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import tempfile
import os
import mne

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="NeuroHome",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# LOAD MODEL + CONFIG
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("neurohome_eeg_model.pkl")


model = load_model()

with open("neurohome_config.json", "r") as f:
    config = json.load(f)

# Threshold selected during your experiments
threshold = float(config.get("threshold", 0.35))

# =========================================================
# SESSION STATE
# =========================================================
default_values = {
    "light": False,
    "fan": False,
    "tv": False,
    "door": False,
    "p300_ready": False,
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# SAVE UPLOADED FILE TEMPORARILY
# =========================================================
def save_uploaded_file(uploaded_file, suffix):

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temp_file.write(uploaded_file.getbuffer())
    temp_file.close()

    return temp_file.name


# =========================================================
# PROCESS REAL EEG DATA
# =========================================================
def process_eeg(bdf_file, events_file):

    bdf_path = save_uploaded_file(
        bdf_file,
        ".bdf"
    )

    events_path = save_uploaded_file(
        events_file,
        ".tsv"
    )

    try:

        # =================================================
        # 1. LOAD BDF EEG FILE
        # =================================================
        raw = mne.io.read_raw_bdf(
            bdf_path,
            preload=True,
            verbose=False
        )

        # Keep EEG channels
        raw.pick("eeg")

        # =================================================
        # 2. FILTER EEG
        # =================================================
        raw.filter(
            l_freq=0.5,
            h_freq=30.0,
            verbose=False
        )

        # =================================================
        # 3. READ EVENTS FILE
        # =================================================
        event_df = pd.read_csv(
            events_path,
            sep="\t"
        )

        if "trial_type" not in event_df.columns:
            raise ValueError(
                "The events TSV file does not contain "
                "a 'trial_type' column."
            )

        if "sample" not in event_df.columns:
            raise ValueError(
                "The events TSV file does not contain "
                "a 'sample' column."
            )

        # Keep only Target / NonTarget events
        valid_events = event_df[
            event_df["trial_type"].isin(
                ["Target", "NonTarget"]
            )
        ].copy()

        if len(valid_events) == 0:
            raise ValueError(
                "No Target or NonTarget events were found."
            )

        # Event mapping
        #
        # NonTarget = 1
        # Target    = 2
        event_codes = valid_events[
            "trial_type"
        ].map(
            {
                "NonTarget": 1,
                "Target": 2
            }
        ).astype(int)

        events = np.column_stack(
            [
                valid_events["sample"].astype(int),
                np.zeros(
                    len(valid_events),
                    dtype=int
                ),
                event_codes
            ]
        )

        event_id = {
            "NonTarget": 1,
            "Target": 2
        }

        # =================================================
        # 4. CREATE EEG EPOCHS
        # =================================================
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=-0.20,
            tmax=0.80,
            baseline=(-0.20, 0),
            preload=True,
            verbose=False
        )

        if len(epochs) == 0:
            raise ValueError(
                "No usable EEG epochs were created."
            )

        # =================================================
        # 5. EXACT ML PREPROCESSING USED IN COLAB
        #
        # This matches:
        #
        # epochs_19 = combined_epochs_19.copy()
        # epochs_19.crop(tmin=0.10, tmax=0.70)
        # epochs_19.resample(64)
        # X_19 = epochs_19.get_data(picks='eeg')
        # X_19 = X_19.reshape(X_19.shape[0], -1)
        # =================================================
        ml_epochs = epochs.copy()

        # Same ERP/P300 interval as training
        ml_epochs.crop(
            tmin=0.10,
            tmax=0.70
        )

        # Same sampling frequency as training
        ml_epochs.resample(
            64,
            verbose=False
        )

        # EEG data:
        # epochs × channels × time
        X = ml_epochs.get_data(
            picks="eeg"
        )

        # Flatten exactly like Colab:
        #
        # trials × channels × time
        #                ↓
        # trials × features
        features = X.reshape(
            X.shape[0],
            -1
        )

        # Labels
        #
        # NonTarget = 0
        # Target    = 1
        labels = (
            ml_epochs.events[:, -1] == 2
        ).astype(int)

        # =================================================
        # 6. VERIFY MODEL INPUT FEATURES
        # =================================================
        expected_features = getattr(
            model,
            "n_features_in_",
            None
        )

        if expected_features is None:
            raise ValueError(
                "Could not determine how many features "
                "the trained model expects."
            )

        if features.shape[1] != expected_features:

            raise ValueError(
                f"Feature mismatch.\n\n"
                f"Model expects: {expected_features}\n\n"
                f"Uploaded dataset produced: "
                f"{features.shape[1]}\n\n"
                f"EEG channels: {X.shape[1]}\n\n"
                f"Time samples: {X.shape[2]}"
            )

        return (
            features,
            labels,
            ml_epochs,
            valid_events,
            expected_features
        )

    finally:

        if os.path.exists(bdf_path):
            os.remove(bdf_path)

        if os.path.exists(events_path):
            os.remove(events_path)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🧠 NeuroHome")

st.sidebar.caption(
    "P300 EEG Smart Home Prototype"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🧠 Real EEG Dataset",
        "🧠 Brain Signal Demo",
        "💡 Smart Home Control",
        "📊 Model Results",
    ]
)


# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    st.title("🧠 NeuroHome")

    st.subheader(
        "P300 EEG Based Smart Home Control Prototype"
    )

    st.write(
        "NeuroHome analyzes P300 EEG brain responses "
        "using a trained machine-learning model and "
        "uses detected target responses to enable "
        "simulated smart-home controls."
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "EEG Runs",
        "19"
    )

    c2.metric(
        "Usable Epochs",
        "9600"
    )

    c3.metric(
        "Target Recall",
        "90%"
    )

    c4.metric(
        "Balanced Accuracy",
        "89.15%"
    )

    st.divider()

    st.subheader(
        "⚙️ NeuroHome System Flow"
    )

    st.info(
        "EEG Signal → Preprocessing → "
        "P300 Detection → ML Classification → "
        "Smart Home Command"
    )

    st.subheader(
        "🧠 ML Preprocessing"
    )

    st.write(
        "P300 Epoch → Crop 0.10–0.70 s → "
        "Resample 64 Hz → EEG Channels → "
        "Flatten → 1216 Features"
    )

    st.success(
        "✅ NeuroHome ML model loaded successfully."
    )


# =========================================================
# REAL EEG DATASET
# =========================================================
elif page == "🧠 Real EEG Dataset":

    st.title(
        "📂 Real P300 EEG Dataset"
    )

    st.write(
        "Upload one real EEG `.bdf` recording and "
        "its matching `_events.tsv` file."
    )

    st.info(
        "The BDF file and Events TSV file must belong "
        "to the same subject, session and run."
    )

    bdf_file = st.file_uploader(
        "1️⃣ Upload EEG BDF file",
        type=["bdf"],
        key="bdf_uploader"
    )

    events_file = st.file_uploader(
        "2️⃣ Upload matching Events TSV file",
        type=["tsv"],
        key="events_uploader"
    )

    if (
        bdf_file is not None
        and events_file is not None
    ):

        st.success(
            "✅ Both dataset files selected"
        )

        st.write(
            "**EEG file:**",
            bdf_file.name
        )

        st.write(
            "**Events file:**",
            events_file.name
        )

        if st.button(
            "🧠 Analyze EEG and Detect P300",
            type="primary"
        ):

            try:

                with st.spinner(
                    "Reading and processing real EEG data..."
                ):

                    (
                        features,
                        labels,
                        ml_epochs,
                        event_df,
                        expected_features
                    ) = process_eeg(
                        bdf_file,
                        events_file
                    )

                    # =====================================
                    # PREDICT TARGET PROBABILITY
                    # =====================================
                    if hasattr(
                        model,
                        "predict_proba"
                    ):

                        probabilities = (
                            model.predict_proba(
                                features
                            )[:, 1]
                        )

                        # Use threshold = 0.35
                        predictions = (
                            probabilities >= threshold
                        ).astype(int)

                    else:

                        predictions = model.predict(
                            features
                        )

                        probabilities = (
                            predictions.astype(float)
                        )

                # =========================================
                # PROCESSING SUCCESS
                # =========================================
                st.success(
                    "✅ EEG processing completed"
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "EEG Channels",
                    len(ml_epochs.ch_names)
                )

                c2.metric(
                    "Epochs",
                    len(features)
                )

                c3.metric(
                    "Model Features",
                    expected_features
                )

                target_count = int(
                    np.sum(
                        predictions == 1
                    )
                )

                c4.metric(
                    "Targets Detected",
                    target_count
                )

                st.caption(
                    f"Classification threshold: "
                    f"{threshold:.2f}"
                )

                # =========================================
                # PREDICTION TABLE
                # =========================================
                st.subheader(
                    "🧠 P300 Predictions"
                )

                result_df = pd.DataFrame(
                    {
                        "Trial": np.arange(
                            1,
                            len(predictions) + 1
                        ),

                        "Actual": np.where(
                            labels == 1,
                            "Target",
                            "NonTarget"
                        ),

                        "Prediction": np.where(
                            predictions == 1,
                            "Target",
                            "NonTarget"
                        ),

                        "Target Probability": np.round(
                            probabilities,
                            4
                        )
                    }
                )

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True
                )

                # =========================================
                # TARGET PROBABILITY CHART
                # =========================================
                st.subheader(
                    "📈 Target Probability"
                )

                chart_df = pd.DataFrame(
                    {
                        "Target Probability":
                            probabilities
                    }
                )

                st.line_chart(
                    chart_df
                )

                # =========================================
                # NEUROHOME DECISION
                # =========================================
                st.subheader(
                    "🏠 NeuroHome Decision"
                )

                max_probability = float(
                    np.max(probabilities)
                )

                st.write(
                    "Highest Target Probability: "
                    f"**{max_probability * 100:.2f}%**"
                )

                if target_count > 0:

                    st.session_state[
                        "p300_ready"
                    ] = True

                    st.success(
                        "🧠 P300 Target detected!"
                    )

                    st.write(
                        "Smart-home control has been enabled."
                    )

                else:

                    st.session_state[
                        "p300_ready"
                    ] = False

                    st.warning(
                        "No P300 Target detected."
                    )

                # =========================================
                # RUN PERFORMANCE
                # =========================================
                accuracy = accuracy_score(
                    labels,
                    predictions
                )

                balanced_accuracy = (
                    balanced_accuracy_score(
                        labels,
                        predictions
                    )
                )

                st.subheader(
                    "📊 Uploaded Run Performance"
                )

                p1, p2 = st.columns(2)

                p1.metric(
                    "Accuracy",
                    f"{accuracy * 100:.2f}%"
                )

                p2.metric(
                    "Balanced Accuracy",
                    f"{balanced_accuracy * 100:.2f}%"
                )

                # =========================================
                # CONFUSION MATRIX
                # =========================================
                cm = confusion_matrix(
                    labels,
                    predictions,
                    labels=[0, 1]
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=[
                        "Actual NonTarget",
                        "Actual Target"
                    ],
                    columns=[
                        "Predicted NonTarget",
                        "Predicted Target"
                    ]
                )

                st.subheader(
                    "Confusion Matrix"
                )

                st.dataframe(
                    cm_df,
                    use_container_width=True
                )

                # =========================================
                # CLASSIFICATION REPORT
                # =========================================
                report = classification_report(
                    labels,
                    predictions,
                    target_names=[
                        "NonTarget",
                        "Target"
                    ],
                    output_dict=True,
                    zero_division=0
                )

                report_df = pd.DataFrame(
                    report
                ).transpose()

                st.subheader(
                    "Classification Report"
                )

                st.dataframe(
                    report_df,
                    use_container_width=True
                )

                st.caption(
                    "Note: if this uploaded run was also "
                    "used to train the saved model, this "
                    "run-level performance is not an "
                    "independent test result."
                )

            except Exception as e:

                st.session_state[
                    "p300_ready"
                ] = False

                st.error(
                    "❌ EEG processing failed."
                )

                st.exception(e)


# =========================================================
# SIMULATED BRAIN SIGNAL DEMO
# =========================================================
elif page == "🧠 Real EEG Dataset":

    st.title(
        "🧠 Brain Signal Prediction Demo"
    )

    st.warning(
        "This page is a simulated demonstration. "
        "Use 'Real EEG Dataset' for actual EEG files."
    )

    probability = st.slider(
        "Simulated P300 Target Probability",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.01
    )

    st.write(
        f"Decision Threshold: **{threshold:.2f}**"
    )

    prediction = (
        1
        if probability >= threshold
        else 0
    )

    if prediction == 1:

        st.session_state[
            "p300_ready"
        ] = True

        st.success(
            "🧠 P300 Target Detected — "
            f"Probability {probability * 100:.2f}%"
        )

    else:

        st.session_state[
            "p300_ready"
        ] = False

        st.warning(
            "NonTarget — "
            f"Probability {probability * 100:.2f}%"
        )


# =========================================================
# SMART HOME CONTROL
# =========================================================
elif page == "💡 Smart Home Control":

    st.title(
        "🏠 NeuroHome Smart Home Control"
    )

    ready = st.session_state.get(
        "p300_ready",
        False
    )

    if ready:

        st.success(
            "🧠 P300 Target detected. "
            "Device control enabled."
        )

    else:

        st.warning(
            "⚠️ No P300 Target is currently active. "
            "Analyze a real EEG dataset or use "
            "the Brain Signal Demo first."
        )

    left, right = st.columns(2)

    # -----------------------------------------------------
    # LIGHT
    # -----------------------------------------------------
    with left:

        st.subheader(
            "💡 Light"
        )

        if st.button(
            "Light ON / OFF",
            disabled=not ready
        ):

            st.session_state.light = (
                not st.session_state.light
            )

        if st.session_state.light:

            st.success(
                "🟢 Light ON"
            )

        else:

            st.error(
                "🔴 Light OFF"
            )

        # -------------------------------------------------
        # FAN
        # -------------------------------------------------
        st.subheader(
            "🌀 Fan"
        )

        if st.button(
            "Fan ON / OFF",
            disabled=not ready
        ):

            st.session_state.fan = (
                not st.session_state.fan
            )

        if st.session_state.fan:

            st.success(
                "🟢 Fan ON"
            )

        else:

            st.error(
                "🔴 Fan OFF"
            )

    # -----------------------------------------------------
    # TV
    # -----------------------------------------------------
    with right:

        st.subheader(
            "📺 TV"
        )

        if st.button(
            "TV ON / OFF",
            disabled=not ready
        ):

            st.session_state.tv = (
                not st.session_state.tv
            )

        if st.session_state.tv:

            st.success(
                "🟢 TV ON"
            )

        else:

            st.error(
                "🔴 TV OFF"
            )

        # -------------------------------------------------
        # DOOR
        # -------------------------------------------------
        st.subheader(
            "🚪 Door"
        )

        if st.button(
            "Open / Close Door",
            disabled=not ready
        ):

            st.session_state.door = (
                not st.session_state.door
            )

        if st.session_state.door:

            st.success(
                "🟢 Door OPEN"
            )

        else:

            st.error(
                "🔒 Door CLOSED"
            )

    st.info(
        "These controls simulate smart-home devices. "
        "They are not connected to physical appliances."
    )


# =========================================================
# MODEL RESULTS
# =========================================================
elif page == "📊 Model Results":

    st.title(
        "📊 NeuroHome Model Performance"
    )

    st.write(
        "**Classifier:** Logistic Regression"
    )

    st.write(
        "**Training data:** 19 P300 EEG runs"
    )

    st.write(
        "**Total usable epochs:** 9600"
    )

    st.write(
        "**Target epochs:** 800"
    )

    st.write(
        "**NonTarget epochs:** 8800"
    )

    st.write(
        "**Feature size:** 1216"
    )

    st.write(
        "**Decision threshold:** "
        f"{threshold:.2f}"
    )

    st.divider()

    a, b, c = st.columns(3)

    a.metric(
        "Group-wise Accuracy",
        "92.07%"
    )

    b.metric(
        "Threshold-Tuned Balanced Accuracy",
        "89.15%"
    )

    c.metric(
        "Target Recall",
        "90%"
    )

    st.subheader(
        "Threshold-Tuned Results"
    )

    results = pd.DataFrame(
        {
            "Class": [
                "NonTarget",
                "Target"
            ],

            "Precision": [
                0.99,
                0.42
            ],

            "Recall": [
                0.89,
                0.90
            ],

            "F1 Score": [
                0.93,
                0.57
            ]
        }
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "ML Feature Pipeline"
    )

    st.code(
        "P300 Epoch\n"
        "   ↓\n"
        "Crop: 0.10–0.70 sec\n"
        "   ↓\n"
        "Resample: 64 Hz\n"
        "   ↓\n"
        "32 EEG Channels\n"
        "   ↓\n"
        "Flatten\n"
        "   ↓\n"
        "1216 Features\n"
        "   ↓\n"
        "Logistic Regression\n"
        "   ↓\n"
        "Target / NonTarget"
    )

    st.warning(
        "The smart-home section is a software "
        "prototype. Physical IoT appliances are "
        "not directly controlled by this application."
    )