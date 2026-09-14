import streamlit as st
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="P300 EEG Analysis",
    page_icon="🧠",
    layout="wide"
)
st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] * {
    color: white;
}

h1 {
    font-size: 42px !important;
    font-weight: 700 !important;
}

h2 {
    font-size: 30px !important;
}

h3 {
    font-size: 22px !important;
}

.stMetric {
    background-color: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
}

div[data-testid="stMetricValue"] {
    font-size: 30px;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

</style>
""", unsafe_allow_html=True)
BASE = Path(r"C:\sub-1\ses-0\eeg")

MODEL_FILE = BASE / "p300_model.joblib"
EPOCH_FILE = BASE / "p300_all_runs-epo.fif"


# =========================================================
# HELPER FUNCTIONS
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_resource
def load_epochs():
    return mne.read_epochs(
        EPOCH_FILE,
        preload=True,
        verbose=False
    )


def get_run_files(run_number):
    prefix = f"sub-1_ses-0_task-p300_run-{run_number}"

    return {
        "bdf": BASE / f"{prefix}_eeg.bdf",
        "events": BASE / f"{prefix}_events.tsv",
        "channels": BASE / f"{prefix}_channels.tsv",
    }


def prepare_features(epochs, model_data):
    """
    Convert EEG epochs into exactly the feature format
    expected by the saved P300 model.
    """

    X = epochs.get_data()

    sfreq = model_data["sampling_frequency"]
    epoch_start = model_data["epoch_start"]
    p300_start = model_data["p300_start"]
    p300_end = model_data["p300_end"]
    downsample_factor = model_data["downsample_factor"]

    start_sample = int(
        (p300_start - epoch_start) * sfreq
    )

    end_sample = int(
        (p300_end - epoch_start) * sfreq
    )

    X = X[
        :,
        :,
        start_sample:end_sample:downsample_factor
    ]

    X = X.reshape(X.shape[0], -1)

    return X


def get_labels(epochs):
    """
    Convert event labels into:
    0 = NonTarget
    1 = Target
    """

    labels = []

    for event_code in epochs.events[:, 2]:

        if event_code == epochs.event_id.get("NonTarget"):
            labels.append(0)

        elif event_code == epochs.event_id.get("Target"):
            labels.append(1)

    return np.array(labels)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🧠 P300 EEG")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "EEG Information",
        "EEG Signal",
        "Event Information",
        "Target vs NonTarget",
        "P300 Analysis",
        "Model Performance",
        "About Project"
    ]
)


# =========================================================
# HOME
# =========================================================

if page == "Home":

    st.title("🧠 P300 EEG Analysis")

    st.subheader(
        "EEG-based Target and NonTarget Signal Analysis"
    )

    st.write(
        """
        This application analyses P300 EEG recordings and
        separates Target and NonTarget responses using
        EEG signal processing and machine learning.
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Channels", "32")

    with col2:
        st.metric("Sampling Rate", "256 Hz")

    with col3:
        st.metric("Duration", "105 sec")

    with col4:
        st.metric("Samples", "26,880")

    st.divider()

    st.header("Project Pipeline")

    st.markdown(
        """
        **EEG Acquisition**

        ↓

        **Preprocessing**

        ↓

        **Event Detection**

        ↓

        **Target / NonTarget Epoching**

        ↓

        **P300 Component Analysis**

        ↓

        **Machine Learning Classification**

        ↓

        **Prediction & Evaluation**
        """
    )


# =========================================================
# EEG INFORMATION
# =========================================================

elif page == "EEG Information":

    st.title("📊 EEG Information")

    st.write("Dataset information and available EEG recordings.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Channels", "32")

    with col2:
        st.metric("Sampling Rate", "256 Hz")

    with col3:
        st.metric("Samples", "26,880")

    with col4:
        st.metric("Runs", "19")

    st.divider()

    st.subheader("Available Runs")

    rows = []

    for run in range(19):

        files = get_run_files(run)

        if files["bdf"].exists():

            rows.append({
                "Run": run,
                "EEG File": files["bdf"].name,
                "Size (MB)": round(
                    files["bdf"].stat().st_size / (1024 * 1024),
                    2
                )
            })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# EEG SIGNAL
# =========================================================

elif page == "EEG Signal":

    st.title("📈 EEG Signal")

    run = st.selectbox(
        "Select EEG Run",
        list(range(19))
    )

    files = get_run_files(run)

    if not files["bdf"].exists():

        st.error("EEG file not found.")

    else:

        with st.spinner("Loading EEG signal..."):

            raw = mne.io.read_raw_bdf(
                files["bdf"],
                preload=True,
                verbose=False
            )

        st.success("EEG recording loaded.")

        channel = st.selectbox(
            "Select Channel",
            raw.ch_names
        )

        max_time = min(
            10.0,
            raw.times[-1]
        )

        duration = st.slider(
            "Signal Duration (seconds)",
            1.0,
            float(max_time),
            min(5.0, float(max_time))
        )

        data, times = raw.copy().pick(
            [channel]
        ).get_data(
            start=0,
            stop=int(duration * raw.info["sfreq"])
        )

        signal_uv = data[0] * 1e6

        fig, ax = plt.subplots(
            figsize=(12, 4)
        )

        ax.plot(times[:len(signal_uv)], signal_uv)

        ax.set_title(
            f"EEG Signal — {channel}"
        )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude (µV)")

        ax.grid(True)

        st.pyplot(fig)

        st.info(
            f"Sampling Frequency: {raw.info['sfreq']:.0f} Hz"
        )


# =========================================================
# EVENT INFORMATION
# =========================================================

elif page == "Event Information":

    st.title("⚡ Event Information")

    run = st.selectbox(
        "Select Run",
        list(range(19))
    )

    files = get_run_files(run)

    if not files["events"].exists():

        st.error("Events file not found.")

    else:

        events = pd.read_csv(
            files["events"],
            sep="\t"
        )

        st.subheader("Event Table")

        st.dataframe(
            events,
            use_container_width=True,
            hide_index=True
        )

        if "trial_type" in events.columns:

            counts = (
                events["trial_type"]
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "Event Type",
                "Count"
            ]

            st.subheader("Event Counts")

            st.dataframe(
                counts,
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# TARGET VS NON TARGET
# =========================================================

elif page == "Target vs NonTarget":

    st.title("🎯 Target vs NonTarget")

    if not EPOCH_FILE.exists():

        st.error(
            "p300_all_runs-epo.fif was not found."
        )

    else:

        with st.spinner(
            "Loading processed EEG epochs..."
        ):

            epochs = load_epochs()

        st.success(
            f"{len(epochs)} EEG epochs loaded."
        )

        st.subheader("Event Classes")

        st.write(
            epochs.event_id
        )

        event_counts = {
            name: len(epochs[name])
            for name in epochs.event_id
        }

        cols = st.columns(
            len(event_counts)
        )

        for col, (name, count) in zip(
            cols,
            event_counts.items()
        ):

            with col:
                st.metric(
                    name,
                    count
                )

        st.divider()

        st.subheader(
            "Target vs NonTarget ERP"
        )

        channel = st.selectbox(
            "Select EEG Channel",
            epochs.ch_names
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        for event_name in [
            "Target",
            "NonTarget"
        ]:

            if event_name in epochs.event_id:

                evoked = (
                    epochs[event_name]
                    .average()
                )

                channel_index = (
                    evoked.ch_names.index(channel)
                )

                times = evoked.times

                signal = (
                    evoked.data[channel_index]
                    * 1e6
                )

                ax.plot(
                    times,
                    signal,
                    label=event_name
                )

        ax.axvline(
            0,
            linestyle="--"
        )

        ax.axvspan(
            0.2,
            0.8,
            alpha=0.15
        )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title(
            f"Target vs NonTarget — {channel}"
        )

        ax.legend()
        ax.grid(True)

        st.pyplot(fig)


# =========================================================
# P300 ANALYSIS
# =========================================================

elif page == "P300 Analysis":

    st.title("🧠 P300 Component Analysis")

    st.write(
        """
        The P300 component is an event-related potential
        associated with attention and detection of relevant
        or target stimuli.
        """
    )

    st.divider()

    epochs = load_epochs()

    channel = st.selectbox(
        "Select Channel",
        epochs.ch_names
    )

    event_type = st.selectbox(
        "Select Event",
        [
            x for x in
            ["Target", "NonTarget"]
            if x in epochs.event_id
        ]
    )

    evoked = epochs[event_type].average()

    channel_index = (
        evoked.ch_names.index(channel)
    )

    times = evoked.times

    signal = (
        evoked.data[channel_index]
        * 1e6
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        times,
        signal
    )

    ax.axvline(
        0,
        linestyle="--"
    )

    ax.axvspan(
        0.2,
        0.8,
        alpha=0.15,
        label="P300 Window"
    )

    ax.set_title(
        f"P300 Analysis — {event_type} — {channel}"
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (µV)")

    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.info(
        "Highlighted region: 200–800 ms after stimulus."
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.title("🤖 Model Performance")

    if not MODEL_FILE.exists():

        st.error(
            "p300_model.joblib was not found."
        )

    elif not EPOCH_FILE.exists():

        st.error(
            "p300_all_runs-epo.fif was not found."
        )

    else:

        with st.spinner(
            "Running model evaluation..."
        ):

            model_data = load_model()
            epochs = load_epochs()

            X = prepare_features(
                epochs,
                model_data
            )

            # Make sure feature count matches
            if X.shape[1] != model_data["feature_count"]:

                st.error(
                    f"Feature mismatch: "
                    f"created {X.shape[1]} features, "
                    f"but model expects "
                    f"{model_data['feature_count']}."
                )

                st.stop()

            y = get_labels(epochs)

            if len(y) != len(X):

                st.error(
                    "Number of labels does not match "
                    "number of epochs."
                )

                st.stop()

            scaler = model_data["scaler"]
            model = model_data["model"]

            X_scaled = scaler.transform(X)

            predictions = model.predict(
                X_scaled
            )

        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        accuracy = accuracy_score(
            y,
            predictions
        )

        precision = precision_score(
            y,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0
        )

        st.success(
            "Model evaluation completed."
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Accuracy",
                f"{accuracy * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Precision",
                f"{precision * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Recall",
                f"{recall * 100:.2f}%"
            )

        with col4:
            st.metric(
                "F1 Score",
                f"{f1 * 100:.2f}%"
            )

        st.divider()

        # -------------------------------------------------
        # CONFUSION MATRIX
        # -------------------------------------------------

        st.subheader(
            "Confusion Matrix"
        )

        cm = confusion_matrix(
            y,
            predictions,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        ax.imshow(cm)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(
            ["NonTarget", "Target"]
        )

        ax.set_yticklabels(
            ["NonTarget", "Target"]
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )

        ax.set_title(
            "P300 Classification Confusion Matrix"
        )

        for i in range(2):
            for j in range(2):

                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center"
                )

        st.pyplot(fig)

        st.divider()

        # -------------------------------------------------
        # PREDICTION COUNTS
        # -------------------------------------------------

        st.subheader(
            "Prediction Distribution"
        )

        predicted_names = [
            "NonTarget" if p == 0 else "Target"
            for p in predictions
        ]

        prediction_counts = (
            pd.Series(predicted_names)
            .value_counts()
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted NonTarget",
                int(
                    np.sum(predictions == 0)
                )
            )

        with col2:

            st.metric(
                "Predicted Target",
                int(
                    np.sum(predictions == 1)
                )
            )

        st.divider()

        st.subheader(
            "Model Details"
        )

        st.write(
            f"**Algorithm:** Logistic Regression"
        )

        st.write(
            f"**Features:** {model_data['feature_count']}"
        )

        st.write(
            f"**P300 Window:** "
            f"{model_data['p300_start']}–"
            f"{model_data['p300_end']} seconds"
        )

        st.write(
            f"**Downsample Factor:** "
            f"{model_data['downsample_factor']}"
        )

        st.write(
            "**Label Mapping:** "
            "0 = NonTarget, 1 = Target"
        )


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "About Project":

    st.title("📚 About the Project")

    st.subheader(
        "P300 EEG Signal Classification"
    )

    st.write(
        """
        This project analyses EEG signals associated with
        P300 event-related potentials.

        EEG recordings are processed into epochs corresponding
        to Target and NonTarget events. Relevant EEG features
        are then provided to a machine learning classifier
        for automatic classification.
        """
    )

    st.subheader("Processing Pipeline")

    st.markdown(
        """
        1. EEG Data Acquisition
        2. Signal Preprocessing
        3. Event Detection
        4. Epoch Extraction
        5. P300 Feature Extraction
        6. Feature Scaling
        7. Logistic Regression Classification
        8. Performance Evaluation
        """
    )

    st.subheader("Model")

    st.write(
        """
        The saved classification model uses Logistic
        Regression with StandardScaler preprocessing.
        """
    )