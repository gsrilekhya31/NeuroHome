import React, { useMemo, useState } from "react";

const API = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const MODEL = {
  accuracy: "70.07%",
  balanced: "69.25%",
  targetF1: "43.59%",
  threshold: "0.40",
  channels: 32,
  runs: 6,
  epochs: 838,
  window: "100-700 ms"
};

const DEVICES = [
  { id: "tv", name: "Television" },
  { id: "telephone", name: "Telephone" },
  { id: "light", name: "Lamp" },
  { id: "door", name: "Door" },
  { id: "window", name: "Window" },
  { id: "radio", name: "Radio" }
];

function safePct(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(1) : "-";
}

export default function App() {
  const [page, setPage] = useState("Dashboard");
  const [eegFile, setEegFile] = useState(null);
  const [eventsFile, setEventsFile] = useState(null);
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const [devices, setDevices] = useState({
    tv: false,
    telephone: false,
    light: false,
    door: false,
    window: false,
    radio: false
  });

  async function predict() {
    if (!eegFile || !eventsFile) {
      setError("Please select both the EEG BDF and EVENTS TSV files.");
      return;
    }

    setBusy(true);
    setError("");

    const form = new FormData();
    form.append("eeg_file", eegFile);
    form.append("events_file", eventsFile);

    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        body: form
      });

      const text = await res.text();

      let json;
      try {
        json = JSON.parse(text);
      } catch {
        throw new Error(
          `Backend returned an invalid response (${res.status}).`
        );
      }

      if (!res.ok) {
        throw new Error(json.detail || "Prediction failed.");
      }

      const synced = {
        tv: false,
        telephone: false,
        light: false,
        door: false,
        window: false,
        radio: false,
        ...(json.appliance_state || {})
      };

      setData(json);
      setDevices(synced);
      setPage("Smart Home");
    } catch (e) {
      setError(
        e?.message || "Unable to connect to the NeuroHome backend."
      );
    } finally {
      setBusy(false);
    }
  }

  const predictions = data?.predictions || [];

  const latest =
    predictions.length > 0
      ? predictions[predictions.length - 1]
      : null;

  const actions = data?.appliance_actions || [];

  const lastAccepted =
    actions.length > 0
      ? actions[actions.length - 1]
      : null;

  const targetPct = data?.epochs
    ? (
        (Number(data.predicted_target_count || 0) /
          Number(data.epochs)) *
        100
      ).toFixed(1)
    : "-";

  const activeCount = Object.values(devices).filter(Boolean).length;

  const content = useMemo(() => {
    if (page === "EEG Signal") {
      return (
        <EEGPage
          eegFile={eegFile}
          eventsFile={eventsFile}
          data={data}
          busy={busy}
          error={error}
          setEegFile={setEegFile}
          setEventsFile={setEventsFile}
          predict={predict}
        />
      );
    }

    if (page === "Brain Response") {
      return <BrainPage data={data} />;
    }

    if (page === "Smart Home") {
      return (
        <SmartHome
          devices={devices}
          activeCount={activeCount}
          lastAccepted={lastAccepted}
          data={data}
        />
      );
    }

    if (page === "Model Results") {
      return <ModelPage />;
    }

    return (
      <Dashboard
        data={data}
        targetPct={targetPct}
        latest={latest}
        setPage={setPage}
      />
    );
  }, [
    page,
    data,
    latest,
    lastAccepted,
    eegFile,
    eventsFile,
    busy,
    error,
    devices,
    activeCount,
    targetPct
  ]);

  return (
    <div className="appShell">
      <aside className="sidebar">
        <div className="logo">
          NEURO<span>HOME</span>
        </div>

        <div className="sideTag">P300 BCI SYSTEM</div>

        <nav>
          {[
            "Dashboard",
            "EEG Signal",
            "Brain Response",
            "Smart Home",
            "Model Results"
          ].map((item) => (
            <button
              key={item}
              className={`navItem ${page === item ? "active" : ""}`}
              onClick={() => setPage(item)}
            >
              {item}
            </button>
          ))}
        </nav>

        <div className="sideBottom">
          <div className="backend">
            <i />
            Backend connected
          </div>

          <div className="modelMini">
            EEGNet - 32 channels
            <br />
            100-700 ms - adaptive threshold 40%-70%
          </div>
        </div>
      </aside>

      <main className="mainArea">
        <header className="topbar">
          <div>
            <div className="pageKicker">
              NEUROHOME / {page.toUpperCase()}
            </div>
            <h2>{page}</h2>
          </div>

          <div className="live">
            <i /> SYSTEM ONLINE
          </div>
        </header>

        <div className="content">{content}</div>
      </main>
    </div>
  );
}

function Dashboard({ data, targetPct, latest, setPage }) {
  return (
    <>
      <section className="heroGrid">
        <div className="heroCopy">
          <div className="eyebrow">BRAIN-COMPUTER INTERFACE</div>

          <h1>NEUROHOME - BRAIN-CONTROLLED SMART HOME</h1>

          <p>P300-based real-time appliance control.</p>

          <button
            className="primary"
            onClick={() => setPage("EEG Signal")}
          >
            UPLOAD REAL EEG <b>â†’</b>
          </button>
        </div>

        <div className="brainCard">
          <div className="brainRing">
            <div>
              EEG
              <br />
              <strong>P300</strong>
            </div>
          </div>

          <div className="brainMeta">
            <span>MODEL</span>
            <strong>EEGNet</strong>

            <span>STATUS</span>
            <strong className="green">READY</strong>
          </div>
        </div>
      </section>

      <section className="statsGrid">
        <Stat
          label="ACCURACY"
          value={MODEL.accuracy}
          sub="Untouched Run 5"
        />

        <Stat
          label="BALANCED ACC."
          value={MODEL.balanced}
          sub="Untouched Run 5"
        />

        <Stat
          label="TARGET F1"
          value={MODEL.targetF1}
          sub="P300 target class"
        />

        <Stat
          label="PREDICTED TARGETS"
          value={data?.predicted_target_count ?? "-"}
          sub={
            data ? `${targetPct}% of epochs` : "Run an EEG analysis"
          }
        />
      </section>

      <section className="twoCol">
        <Panel title="SYSTEM FLOW">
          <div className="flow">
            <Flow n="01" t="REAL EEG" />
            <b>â†’</b>
            <Flow n="02" t="PREPROCESS" />
            <b>â†’</b>
            <Flow n="03" t="EEGNet" />
            <b>â†’</b>
            <Flow n="04" t="TARGET / NON-TARGET" />
            <b>â†’</b>
            <Flow n="05" t="ONE APPLIANCE" />
          </div>
        </Panel>

        <Panel title="LATEST COMMAND">
          <div className="commandBox">
            <span>{latest?.command || "WAITING FOR EEG"}</span>

            <small>
              {latest
                ? `${safePct(Number(latest.probability) * 100)}% model probability`
                : "No prediction yet"}
            </small>
          </div>
        </Panel>
      </section>

      <section className="datasetStrip">
        <div>
          <span>DATASET</span>
          <strong>6 runs</strong>
        </div>

        <div>
          <span>USABLE EPOCHS</span>
          <strong>838</strong>
        </div>

        <div>
          <span>CHANNELS</span>
          <strong>32</strong>
        </div>

        <div>
          <span>MODEL WINDOW</span>
          <strong>100-700 ms</strong>
        </div>

        <div>
          <span>ADAPTIVE THRESHOLD</span>
          <strong>40%-70%</strong>
        </div>
      </section>

      <NoveltyPanel data={data} />
    </>
  );
}

function EEGPage({
  eegFile,
  eventsFile,
  data,
  busy,
  error,
  setEegFile,
  setEventsFile,
  predict
}) {
  return (
    <>
      <section className="sectionIntro">
        <div className="eyebrow">REAL EEG DATASET</div>

        <h1>Run the trained model.</h1>

        <p>
          Upload the matching BDF and events file. EEG predictions
          are evaluated inside 1.5-second decision windows using
          an adaptive 40%-70% confidence threshold.
        </p>
      </section>

      <div className="uploadPanel">
        <label>
          <span>EEG BDF</span>

          <input
            type="file"
            accept=".bdf"
            onChange={(e) =>
              setEegFile(e.target.files?.[0] || null)
            }
          />

          {eegFile && <em>{eegFile.name}</em>}
        </label>

        <label>
          <span>EVENTS TSV</span>

          <input
            type="file"
            accept=".tsv,.txt,.csv"
            onChange={(e) =>
              setEventsFile(e.target.files?.[0] || null)
            }
          />

          {eventsFile && <em>{eventsFile.name}</em>}
        </label>

        <button
          className="primary analyze"
          onClick={predict}
          disabled={!eegFile || !eventsFile || busy}
        >
          {busy ? "ANALYZING..." : "ANALYZE EEG â†’"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="infoGrid">
        <Info label="RAW SAMPLING" value="2048 Hz" />
        <Info label="MODEL SAMPLING" value="256 Hz" />
        <Info label="EPOCH" value="-0.2 to 0.8 s" />
        <Info label="MODEL WINDOW" value="100-700 ms" />
      </div>

      {data && <ResultSummary data={data} />}
    </>
  );
}

function BrainPage({ data }) {
  const selected = data?.appliance_actions?.length
    ? data.appliance_actions[data.appliance_actions.length - 1]
    : null;

  return (
    <>
      <section className="sectionIntro">
        <div className="eyebrow">BRAIN RESPONSE</div>

        <h1>P300 prediction timeline.</h1>

        <p>
          The model classifies each EEG epoch as Target or
          Non-Target.
        </p>
      </section>

      <div className="responseCards">
        <Stat
          label="TARGETS"
          value={data?.predicted_target_count ?? "-"}
          sub="P300 detected"
        />

        <Stat
          label="NON-TARGETS"
          value={data?.predicted_nontarget_count ?? "-"}
          sub="background events"
        />

        <Stat
          label="CONFIDENCE"
          value={
            selected
              ? `${safePct(selected.confidence)}%`
              : "-"
          }
          sub={
            selected
              ? "Final selected command"
              : "Awaiting EEG"
          }
        />
      </div>

      <NoveltyPanel data={data} />
    </>
  );
}

function NoveltyPanel({ data }) {
  const windows =
    data?.decision_windows ||
    data?.windows ||
    [];

  return (
    <section className="panel noveltyPanel">
      <div className="panelTitle">
        1.5-SECOND DECISION WINDOW
      </div>

      <p style={{ marginBottom: "18px" }}>
        All accepted EEG signals are shown with their confidence.
      </p>

      {windows.length === 0 ? (
        <div className="empty">
          Analyze EEG to display signals from the
          1.5-second window.
        </div>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontSize: "13px"
            }}
          >
            <thead>
              <tr>
                <th style={{ textAlign: "left", padding: "12px 10px" }}>
                  1.5 S WINDOW
                </th>

                <th style={{ textAlign: "left", padding: "12px 10px" }}>
                  ADAPTIVE THRESHOLD
                </th>

                <th style={{ textAlign: "left", padding: "12px 10px" }}>
                  WINDOW MEAN
                </th>

                <th style={{ textAlign: "left", padding: "12px 10px" }}>
                  CONFIDENCE
                </th>

                <th style={{ textAlign: "left", padding: "12px 10px" }}>
                  EPOCH
                </th>

                <th style={{ textAlign: "left", padding: "12px 10px" }}>
                  STATUS
                </th>
              </tr>
            </thead>

            <tbody>
              {windows.map((w, i) => {
                const selectedConfidence =
                  w.selected_confidence !== null &&
                  w.selected_confidence !== undefined
                    ? Number(w.selected_confidence)
                    : null;

                return (
                  <tr key={i}>
                    <td
                      style={{
                        padding: "12px 10px",
                        fontWeight: "700"
                      }}
                    >
                      {w.start_sec ?? "-"}s - {w.end_sec ?? "-"}s
                    </td>

                    <td style={{ padding: "12px 10px" }}>
                      {w.adaptive_threshold ?? "-"}%
                    </td>

                    <td style={{ padding: "12px 10px" }}>
                      {w.window_mean_confidence ?? "-"}%
                    </td>

                    <td
                      style={{
                        padding: "12px 10px",
                        fontWeight: "800"
                      }}
                    >
                      {selectedConfidence === null
                        ? "-"
                        : `${safePct(selectedConfidence)}%`}
                    </td>

                    <td
                      style={{
                        padding: "12px 10px",
                        fontWeight: "800"
                      }}
                    >
                      {w.selected_epoch
                        ? `EPOCH ${w.selected_epoch}`
                        : "NONE"}
                    </td>

                    <td
                      style={{
                        padding: "12px 10px",
                        fontWeight: "800"
                      }}
                    >
                      {w.selected_epoch
                        ? "EXECUTED ONCE"
                        : "NO ACCEPTED TARGET"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function SmartHome({
  devices,
  activeCount,
  lastAccepted,
  data
}) {
  const activeDevices = DEVICES.filter(
    (d) => Boolean(devices[d.id])
  );

  return (
    <>
      <section className="sectionIntro">
        <div className="eyebrow">AUTOMATIC SMART HOME</div>

        <h1>Six appliances. One EEG command at a time.</h1>

        <p>
          The selected EEG Target controls only its mapped
          appliance. One execution is made per 1.5-second
          decision window.
        </p>
      </section>

      <div className="homeHeader">
        <div>
          <span>ACTIVE DEVICES</span>
          <strong>{activeCount} / 6</strong>
        </div>

        <div>
          <span>LAST BCI COMMAND</span>
          <strong>
            {lastAccepted?.command || "WAITING FOR EEG"}
          </strong>
        </div>

        <div>
          <span>CONFIDENCE</span>
          <strong>
            {lastAccepted
              ? `${safePct(lastAccepted.confidence)}%`
              : "-"}
          </strong>
        </div>
      </div>

      <section className="activeNow">
        <div className="panelTitle">ACTIVE NOW</div>

        {activeDevices.length ? (
          <div className="activeList">
            {activeDevices.map((d) => (
              <div className="activeItem" key={d.id}>
                <b>{d.name}</b>
                <span>
                  {d.id === "door" || d.id === "window"
                    ? "OPEN"
                    : "ON"}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty">
            No appliance is active. All six appliances are
            OFF/CLOSED.
          </div>
        )}
      </section>

      <div className="deviceGrid">
        {DEVICES.map((d) => (
          <div
            className={`device device-${d.id} ${
              devices[d.id] ? "on" : ""
            }`}
            key={d.id}
          >
            <DeviceVisual
              type={d.id}
              on={devices[d.id]}
            />

            <div>
              <h3>{d.name}</h3>

              <small>
                {d.id === "door" || d.id === "window"
                  ? devices[d.id]
                    ? "OPEN"
                    : "CLOSED"
                  : devices[d.id]
                    ? "ON"
                    : "OFF"}
              </small>
            </div>

            <div
              className={`autoState ${
                devices[d.id] ? "active" : ""
              }`}
            >
              {devices[d.id] ? "ACTIVE" : "STANDBY"}
            </div>
          </div>
        ))}
      </div>

      <NoveltyPanel data={data} />

      <div className="mapping">
        <span>RUN 1 <b>TELEVISION</b></span>
        <span>RUN 2 <b>TELEPHONE</b></span>
        <span>RUN 3 <b>LIGHT</b></span>
        <span>RUN 4 <b>DOOR</b></span>
        <span>RUN 5 <b>WINDOW</b></span>
        <span>RUN 6 <b>RADIO</b></span>
      </div>
    </>
  );
}

function DeviceVisual({ type, on }) {
  if (type === "light") {
    return (
      <div className={`deviceVisual bulbVisual ${on ? "active" : ""}`}>
        <div className="bulb">
          <span />
        </div>
        <div className="bulbBase" />
      </div>
    );
  }

  if (type === "tv") {
    return (
      <div className={`deviceVisual tvVisual ${on ? "active" : ""}`}>
        <div className="tvScreen">
          <span>{on ? "LIVE" : "OFF"}</span>
        </div>
        <div className="tvStand" />
      </div>
    );
  }

  if (type === "telephone") {
    return (
      <div
        className={`deviceVisual telephoneVisual ${
          on ? "active" : ""
        }`}
      >
        <div className="phoneBody">
          <span>â˜Ž</span>
        </div>

        {on && (
          <div className="callWaves">
            <i />
            <i />
            <i />
          </div>
        )}
      </div>
    );
  }

  if (type === "window") {
    return (
      <div
        className={`deviceVisual windowVisual ${
          on ? "active" : ""
        }`}
      >
        <div className="windowFrame">
          <i />
          <i />
          <i />
          <i />
        </div>
      </div>
    );
  }

  if (type === "radio") {
    return (
      <div
        className={`deviceVisual radioVisual ${
          on ? "active" : ""
        }`}
      >
        <div className="radioBox">
          <span className="radioDial" />
          <span className="radioLabel">RADIO</span>

          <div className="songSignal">
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
            <i />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`deviceVisual doorVisual ${
        on ? "open" : "closed"
      }`}
    >
      <div className="doorFrame">
        <div className="doorLeaf">
          <span />
        </div>
      </div>
    </div>
  );
}

function ModelPage() {
  return (
    <>
      <section className="sectionIntro">
        <div className="eyebrow">MODEL RESULTS</div>

        <h1>EEGNet performance.</h1>

        <p>
          Final evaluation on the untouched Run 5. These are
          the model metrics integrated into this prototype.
        </p>
      </section>

      <section className="metricsLarge">
        <Metric title="Accuracy" value={MODEL.accuracy} />
        <Metric
          title="Balanced Accuracy"
          value={MODEL.balanced}
        />
        <Metric title="Target F1" value={MODEL.targetF1} />
      </section>

      <div className="twoCol">
        <Panel title="MODEL CONFIGURATION">
          <div className="details">
            <Info label="ARCHITECTURE" value="EEGNet" />
            <Info label="INPUT" value="32 x 154 x 1" />
            <Info label="WINDOW" value={MODEL.window} />
            <Info label="THRESHOLD" value={MODEL.threshold} />
            <Info label="TRAINING RUNS" value="0-4" />
            <Info label="TEST RUN" value="5" />
          </div>
        </Panel>

        <Panel title="CONFUSION MATRIX">
          <div className="matrix">
            <div>TRUE / PRED</div>
            <b>NonTarget</b>
            <b>Target</b>
            <b>NonTarget</b>
            <strong>86</strong>
            <strong>36</strong>
            <b>Target</b>
            <strong>8</strong>
            <strong>17</strong>
          </div>
        </Panel>
      </div>
    </>
  );
}

function ResultSummary({ data }) {
  return (
    <div className="resultSummary">
      <span>ANALYSIS COMPLETE</span>
      <strong>{data?.epochs ?? 0} epochs</strong>
      <b>{data?.predicted_target_count ?? 0} Target</b>
      <b>{data?.predicted_nontarget_count ?? 0} Non-Target</b>
    </div>
  );
}

function Stat({ label, value, sub }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{sub}</small>
    </div>
  );
}

function Metric({ title, value }) {
  const width = Number.parseFloat(value) || 0;

  return (
    <div className="metric">
      <span>{title}</span>
      <strong>{value}</strong>

      <div className="meter">
        <i style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}

function Info({ label, value }) {
  return (
    <div className="info">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Flow({ n, t }) {
  return (
    <div className="flowItem">
      <small>{n}</small>
      <strong>{t}</strong>
    </div>
  );
}

function Panel({ title, children }) {
  return (
    <section className="panel">
      <div className="panelTitle">{title}</div>
      {children}
    </section>
  );
}
