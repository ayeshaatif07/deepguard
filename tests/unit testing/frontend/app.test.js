/**
 * Frontend unit tests for static/js/app.js.
 *
 * app.js is a plain browser script (loaded via <script src="...">, no
 * ES modules) - it exports a handful of testable functions at its own
 * bottom via a `typeof module !== 'undefined'` guard that is a no-op in
 * the browser and only active here under Jest/Node. Each test resets
 * Jest's module registry and re-requires app.js fresh, so app.js's own
 * top-level state (e.g. `let currentWaveformPeaks`) never leaks between
 * tests.
 */

function loadApp() {
  jest.resetModules();
  return require("../../../static/js/app.js");
}

function mockNavigationType(type) {
  window.performance.getEntriesByType = jest.fn().mockReturnValue([{ type }]);
}

beforeEach(() => {
  document.body.innerHTML = "";
  mockNavigationType("navigate"); // most tests are not exercising a real reload
});

// ---------------------------------------------------------------------
// chipDotClassForVerdict() - pure string -> CSS class mapping
// ---------------------------------------------------------------------
describe("chipDotClassForVerdict", () => {
  test.each([
    ["Real", "chip-green"],
    ["Coherent", "chip-green"],
    ["Non-manipulative", "chip-green"],
    ["Artificial", "chip-amber"],
    ["Suspicious", "chip-amber"],
    ["Deepfake", "chip-red"],
    ["Cloned", "chip-red"],
    ["Incoherent", "chip-red"],
    ["Manipulative", "chip-red"],
  ])("%s -> %s", (verdict, expected) => {
    const { chipDotClassForVerdict } = loadApp();
    expect(chipDotClassForVerdict(verdict)).toBe(expected);
  });

  test("missing verdict defaults to amber, not a crash", () => {
    const { chipDotClassForVerdict } = loadApp();
    expect(chipDotClassForVerdict(null)).toBe("chip-amber");
    expect(chipDotClassForVerdict(undefined)).toBe("chip-amber");
  });
});

// ---------------------------------------------------------------------
// isGenuinePageReload() - Navigation Timing API detection
// ---------------------------------------------------------------------
describe("isGenuinePageReload", () => {
  test("returns true for a real reload", () => {
    mockNavigationType("reload");
    const { isGenuinePageReload } = loadApp();
    expect(isGenuinePageReload()).toBe(true);
  });

  test("returns false for a normal navigation", () => {
    mockNavigationType("navigate");
    const { isGenuinePageReload } = loadApp();
    expect(isGenuinePageReload()).toBe(false);
  });
});

// ---------------------------------------------------------------------
// restoreStoredState() - sessionStorage read/parse, reload-aware
// ---------------------------------------------------------------------
describe("restoreStoredState", () => {
  test("restores and calls back with valid stored data", () => {
    const { restoreStoredState } = loadApp();
    const payload = { verdict: "Real", score: 80 };
    sessionStorage.setItem("dg_test_state", JSON.stringify(payload));

    const showFn = jest.fn();
    const restored = restoreStoredState("dg_test_state", showFn);

    expect(restored).toBe(true);
    expect(showFn).toHaveBeenCalledWith(payload, false);
  });

  test("discards corrupted (non-JSON) stored data instead of throwing", () => {
    const { restoreStoredState } = loadApp();
    sessionStorage.setItem("dg_test_state", "{not valid json");

    const showFn = jest.fn();
    expect(() => restoreStoredState("dg_test_state", showFn)).not.toThrow();
    expect(showFn).not.toHaveBeenCalled();
    expect(sessionStorage.getItem("dg_test_state")).toBeNull();
  });

  test("a genuine reload clears the key instead of restoring it", () => {
    mockNavigationType("reload");
    const { restoreStoredState } = loadApp();
    sessionStorage.setItem("dg_test_state", JSON.stringify({ verdict: "Real" }));

    const showFn = jest.fn();
    const restored = restoreStoredState("dg_test_state", showFn);

    expect(restored).toBe(false);
    expect(showFn).not.toHaveBeenCalled();
    expect(sessionStorage.getItem("dg_test_state")).toBeNull();
  });
});

// ---------------------------------------------------------------------
// renderWaveform() - draws one bar per peak, in the right colour
// ---------------------------------------------------------------------
describe("renderWaveform", () => {
  function attachCanvas() {
    const canvas = document.createElement("canvas");
    canvas.id = "waveformCanvas";
    document.body.appendChild(canvas);
    const ctx = { fillRect: jest.fn(), clearRect: jest.fn(), setTransform: jest.fn() };
    canvas.getContext = jest.fn().mockReturnValue(ctx);
    return { canvas, ctx };
  }

  test("draws exactly one bar per peak", () => {
    const { renderWaveform } = loadApp();
    const { ctx } = attachCanvas();
    const peaks = new Array(100).fill(0.5);

    renderWaveform(peaks);

    expect(ctx.fillRect).toHaveBeenCalledTimes(100);
  });

  test("bars before the playback progress are drawn in the played colour", () => {
    const { renderWaveform } = loadApp();
    const { ctx } = attachCanvas();
    const fillStyles = [];
    Object.defineProperty(ctx, "fillStyle", {
      set(value) { fillStyles.push(value); },
      get() { return fillStyles[fillStyles.length - 1]; },
    });
    const peaks = new Array(10).fill(0.5);

    renderWaveform(peaks, 0.5); // 50% played -> first 5 bars should be the "played" colour

    expect(fillStyles.slice(0, 5)).toEqual(new Array(5).fill("rgb(0, 255, 255)"));
    expect(fillStyles.slice(5)).toEqual(new Array(5).fill("rgb(42, 179, 142)"));
  });

  test("does nothing if the canvas is not on the page", () => {
    const { renderWaveform } = loadApp();
    expect(() => renderWaveform([0.1, 0.2])).not.toThrow();
  });
});

// ---------------------------------------------------------------------
// seekWaveform() - click position -> audio.currentTime
// ---------------------------------------------------------------------
describe("seekWaveform", () => {
  test("clicking at 75% across the waveform seeks to 75% of the duration", () => {
    const app = loadApp();

    const canvas = document.createElement("canvas");
    canvas.id = "waveformCanvas";
    canvas.getContext = jest.fn().mockReturnValue({ fillRect: jest.fn(), clearRect: jest.fn(), setTransform: jest.fn() });
    canvas.getBoundingClientRect = jest.fn().mockReturnValue({ left: 0, width: 400 });
    document.body.appendChild(canvas);

    const audio = document.createElement("audio");
    audio.id = "voiceAudioPlayer";
    audio.src = "data:audio/wav;base64,AAAA";
    Object.defineProperty(audio, "duration", { value: 40, configurable: true });
    Object.defineProperty(audio, "currentTime", { value: 0, writable: true, configurable: true });
    document.body.appendChild(audio);

    app.renderWaveform(new Array(10).fill(0.5)); // establishes currentWaveformPeaks

    app.seekWaveform({ clientX: 300 }); // 300 / 400 = 75%

    expect(audio.currentTime).toBe(30); // 75% of 40s
  });

  test("does nothing if the audio has no duration yet", () => {
    const app = loadApp();
    const canvas = document.createElement("canvas");
    canvas.id = "waveformCanvas";
    document.body.appendChild(canvas);
    const audio = document.createElement("audio");
    audio.id = "voiceAudioPlayer";
    document.body.appendChild(audio); // no src, no duration

    expect(() => app.seekWaveform({ clientX: 100 })).not.toThrow();
  });
});

// ---------------------------------------------------------------------
// showCoherenceResult() - updates the right elements for each case
// ---------------------------------------------------------------------
describe("showCoherenceResult", () => {
  function attachCoherenceUI() {
    document.body.innerHTML = `
      <div id="coherenceLoading"></div>
      <div id="coherenceVerdictBadge" style="display:none"></div>
      <div id="coherenceUnavailable" style="display:none"></div>
      <div id="coherenceUnavailableText"></div>
    `;
  }

  test("a real verdict shows the badge with the right text and colour class", () => {
    attachCoherenceUI();
    const { showCoherenceResult } = loadApp();

    showCoherenceResult("Coherent", 78, null);

    const badge = document.getElementById("coherenceVerdictBadge");
    expect(badge.style.display).toBe("inline-block");
    expect(badge.className).toBe("verdict-badge coherent");
    expect(badge.innerText).toBe("Coherent (78%)");
    expect(document.getElementById("coherenceUnavailable").style.display).toBe("none");
  });

  test("a null verdict shows the unavailable message instead", () => {
    attachCoherenceUI();
    const { showCoherenceResult } = loadApp();

    showCoherenceResult(null, null, "No caption available.");

    expect(document.getElementById("coherenceVerdictBadge").style.display).toBe("none");
    expect(document.getElementById("coherenceUnavailable").style.display).toBe("block");
    expect(document.getElementById("coherenceUnavailableText").innerText).toBe("No caption available.");
  });

  test("does nothing if the page has no coherence UI at all", () => {
    document.body.innerHTML = "";
    const { showCoherenceResult } = loadApp();
    expect(() => showCoherenceResult("Coherent", 90, null)).not.toThrow();
  });
});

// ---------------------------------------------------------------------
// resetDashboard() - clears every dg_*_state sessionStorage key
// ---------------------------------------------------------------------
describe("resetDashboard", () => {
  test("removes all dashboard-related sessionStorage keys", () => {
    const { resetDashboard } = loadApp();
    const keys = ["dg_video_state", "dg_image_state", "dg_voice_state", "dg_caption_state", "dg_pending_audio_check"];
    keys.forEach((k) => sessionStorage.setItem(k, "x"));

    // jsdom doesn't implement real page navigation; assigning
    // window.location.href just needs to not throw here.
    delete window.location;
    window.location = { href: "" };

    resetDashboard({ preventDefault: jest.fn() });

    keys.forEach((k) => expect(sessionStorage.getItem(k)).toBeNull());
  });
});
