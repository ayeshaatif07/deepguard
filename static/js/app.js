let scoreChart = null;

// Picking a file only selects it; analysis starts when the user presses
// "Analyse Video". This matches the link path, which has always needed its
// own button - user testing showed the two paths behaving differently was
// the main source of hesitation on a first video check.
function showVideoConfirm() {
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];
    const box = document.getElementById('videoConfirm');
    const name = document.getElementById('videoChosenName');
    if (!file) { if (box) box.style.display = 'none'; return; }
    if (name) name.textContent = file.name;
    if (box) box.style.display = 'flex';
}

function handleUpload() {
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];
    if (!file) return;

    // Show filename in upload zone hint
    const hint = document.getElementById('uploadHint');
    if (hint) hint.textContent = file.name;

    const formData = new FormData();
    formData.append('video', file);

    const confirmBox = document.getElementById('videoConfirm');
    if (confirmBox) confirmBox.style.display = 'none';
    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('loader').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    const loaderText = document.getElementById('loaderText');
    loaderText.innerText = 'Extracting frames…';
    setTimeout(() => {
        if (loaderText.innerText === 'Extracting frames…') {
            loaderText.innerText = 'Running deepfake detection…';
        }
    }, 1500);

    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loader').style.display = 'none';
        if (data.error) {
            alert('Error: ' + data.error);
            resetUpload();
            return;
        }
        showResults(data);
    })
    .catch(error => {
        document.getElementById('loader').style.display = 'none';
        resetUpload();
        alert('An error occurred during analysis.');
        console.error('Error:', error);
    });
}

function handleVideoUrlAnalysis() {
    const urlInput = document.getElementById('videoUrlInput');
    const url = urlInput.value.trim();
    if (!url) return;

    const errorEl = document.getElementById('videoUrlError');
    if (errorEl) errorEl.style.display = 'none';

    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('loader').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    const loaderText = document.getElementById('loaderText');
    if (loaderText) loaderText.innerText = 'Downloading and analyzing link…';

    fetch('/analyze_video_url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
    })
    .then(response => response.json().then(data => ({ status: response.status, data })))
    .then(({ status, data }) => {
        document.getElementById('loader').style.display = 'none';
        if (data.error) {
            document.getElementById('uploadZone').style.display = 'block';
            if (errorEl) {
                errorEl.innerText = data.message || data.error;
                errorEl.style.display = 'block';
            }
            return;
        }
        showResults(data);
    })
    .catch(error => {
        document.getElementById('loader').style.display = 'none';
        document.getElementById('uploadZone').style.display = 'block';
        if (errorEl) {
            errorEl.innerText = 'An error occurred while analyzing this link.';
            errorEl.style.display = 'block';
        }
        console.error('Error:', error);
    });
}

function showResults(data, persist = true) {
    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('results').style.display = 'block';

    const badge = document.getElementById('verdictBadge');
    badge.className = `verdict-badge ${data.verdict.toLowerCase()}`;
    badge.innerText = data.verdict;

    document.getElementById('confidenceScore').innerText = `${data.score}%`;
    document.getElementById('inferenceTime').innerText = `${data.inference_time}s`;
    document.getElementById('explanationText').innerText = data.explanation;

    const filenameEl = document.getElementById('resultFilename');
    if (filenameEl) filenameEl.innerText = `File: ${data.filename}`;

    const chartContainer = document.querySelector('.chart-container');
    const frameHeading = document.getElementById('frameAnalysisHeading');
    if (data.frame_scores && data.frame_scores.length) {
        if (chartContainer) chartContainer.style.display = 'block';
        if (frameHeading) frameHeading.style.display = 'block';
        renderChart(data.frame_scores);
    } else {
        if (chartContainer) chartContainer.style.display = 'none';
        if (frameHeading) frameHeading.style.display = 'none';
    }

    if (persist) {
        sessionStorage.setItem('dg_video_state', JSON.stringify(data));

        // Pipeline mode.
        if (getPipelineMode()) {
            hideActionButtons();
            showAutoAdvanceNotice('Pipeline mode: continuing automatically to Audio Check…');
            setTimeout(() => proceedToVoicePipeline({ preventDefault: () => {} }), AUTO_ADVANCE_DELAY_MS);
        }
    }
}

function renderChart(scores) {
    const ctx = document.getElementById('scoreChart').getContext('2d');
    if (scoreChart) scoreChart.destroy();

    const THRESHOLD = 0.50;
    const labels = scores.map((_, i) => `Frame ${i + 1}`);
    
    const bgColors = scores.map(s => s >= THRESHOLD ? 'rgba(255, 71, 87, 0.25)' : 'rgba(42, 179, 142, 0.25)');
    const borderColors = scores.map(s => s >= THRESHOLD ? 'rgb(255, 71, 87)' : 'rgb(42, 179, 142)');

    scoreChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Fake Probability',
                data: scores,
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 2,
                borderRadius: 4,
                barPercentage: 0.6,
                hoverBackgroundColor: borderColors
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.6)',
                        font: { family: 'Inter', size: 11 },
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        borderDash: [5, 5]
                    }
                },
                x: {
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.6)',
                        font: { family: 'Inter', size: 11 },
                        maxRotation: 45,
                        minRotation: 0
                    },
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: '#1B263B',
                    titleColor: '#fff',
                    bodyColor: '#A9B2C3',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    padding: 10,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            let val = context.parsed.y;
                            let status = val >= THRESHOLD ? 'Suspicious' : 'Authentic';
                            return `Score: ${(val * 100).toFixed(1)}% (${status})`;
                        }
                    }
                },
                annotation: {
                    annotations: {
                        thresholdLine: {
                            type: 'line',
                            yMin: 0.50,
                            yMax: 0.50,
                            borderColor: 'rgba(255, 255, 255, 0.3)',
                            borderWidth: 1.5,
                            borderDash: [4, 4],
                            label: {
                                content: '50% Threshold',
                                enabled: true,
                                position: 'start',
                                backgroundColor: 'transparent',
                                color: 'rgba(255, 255, 255, 0.6)',
                                font: { family: 'Inter', size: 10, style: 'italic' },
                                yAdjust: -10
                            }
                        }
                    }
                }
            }
        }
    });
}

let currentWaveformPeaks = null;

function renderWaveform(peaks, progress = 0) {
    // Drawn directly on canvas rather than via Chart.js.
    const canvas = document.getElementById('waveformCanvas');
    if (!canvas) return;
    currentWaveformPeaks = peaks;

    // Match the canvas backing resolution to its displayed CSS size via devicePixelRatio, so bars
    // stay crisp rather than blurry.
    const dpr = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth || 600;
    const cssHeight = canvas.clientHeight || 120;
    canvas.width = cssWidth * dpr;
    canvas.height = cssHeight * dpr;

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const midY = cssHeight / 2;
    const barGap = 2;
    const barWidth = Math.max(1, (cssWidth / peaks.length) - barGap);
    const minBarHeight = 2; // keep near-silent segments visibly present, not invisible
    const playedBars = Math.floor(progress * peaks.length);

    peaks.forEach((peak, i) => {
        const barHeight = Math.max(minBarHeight, peak * (cssHeight * 0.9));
        const x = i * (barWidth + barGap);
        // Bars already played light up brighter/cyan so the waveform itself shows where playback
        // currently is, not just a separate progress bar.
        ctx.fillStyle = i < playedBars ? 'rgb(0, 255, 255)' : 'rgb(42, 179, 142)';
        ctx.fillRect(x, midY - barHeight / 2, barWidth, barHeight);
    });
}

function toggleWaveformPlayback() {
    const audio = document.getElementById('voiceAudioPlayer');
    const icon = document.getElementById('waveformPlayIcon');
    if (!audio || !audio.src) return;

    if (audio.paused) {
        audio.play();
        icon.className = 'fa-solid fa-pause';
    } else {
        audio.pause();
        icon.className = 'fa-solid fa-play';
    }
}

function seekWaveform(event) {
    const audio = document.getElementById('voiceAudioPlayer');
    const canvas = document.getElementById('waveformCanvas');
    if (!audio || !audio.src || !audio.duration || !canvas) return;

    const rect = canvas.getBoundingClientRect();
    const fraction = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    audio.currentTime = fraction * audio.duration;
    if (currentWaveformPeaks) renderWaveform(currentWaveformPeaks, fraction);
}

function setupWaveformPlayer(audioUrl, peaks) {
    const audio = document.getElementById('voiceAudioPlayer');
    const btn = document.getElementById('waveformPlayBtn');
    const icon = document.getElementById('waveformPlayIcon');
    const hint = document.getElementById('waveformPlayHint');
    if (!audio || !btn || !icon || !hint) return;

    // Clear listeners and state from a previous result before wiring a new one, so old handlers do
    // not stack up.
    const freshAudio = audio.cloneNode(false);
    audio.replaceWith(freshAudio);

    if (!audioUrl) {
        btn.disabled = true;
        icon.className = 'fa-solid fa-play';
        hint.innerText = 'Playback unavailable for this clip (too long to embed).';
        if (peaks) renderWaveform(peaks, 0);
        return;
    }

    freshAudio.id = 'voiceAudioPlayer';
    freshAudio.src = audioUrl;
    btn.disabled = false;
    icon.className = 'fa-solid fa-play';
    hint.innerText = 'Click play, or click the waveform to jump to a moment.';

    freshAudio.addEventListener('timeupdate', () => {
        if (currentWaveformPeaks && freshAudio.duration) {
            renderWaveform(currentWaveformPeaks, freshAudio.currentTime / freshAudio.duration);
        }
    });
    freshAudio.addEventListener('ended', () => {
        icon.className = 'fa-solid fa-play';
        if (currentWaveformPeaks) renderWaveform(currentWaveformPeaks, 0);
    });
}

function resetUpload() {
    if(document.getElementById('uploadZone')) document.getElementById('uploadZone').style.display = 'block';
    if(document.getElementById('results')) document.getElementById('results').style.display = 'none';
    if(document.getElementById('loader')) document.getElementById('loader').style.display = 'none';
    if(document.getElementById('noAudioError')) document.getElementById('noAudioError').style.display = 'none';
    
    if(document.getElementById('videoConfirm')) document.getElementById('videoConfirm').style.display = 'none';
    if(document.getElementById('videoFile')) document.getElementById('videoFile').value = '';
    if(document.getElementById('imageFile')) document.getElementById('imageFile').value = '';
    if(document.getElementById('audioFile')) document.getElementById('audioFile').value = '';
    if(document.getElementById('videoUrlInput')) document.getElementById('videoUrlInput').value = '';
    if(document.getElementById('voiceUrlInput')) document.getElementById('voiceUrlInput').value = '';

    // Clear whichever page's stored result this reset applies to - only
    // "Analyse Another" should clear these, not navigation.
    if(document.getElementById('videoFile')) sessionStorage.removeItem('dg_video_state');
    if(document.getElementById('imageFile')) sessionStorage.removeItem('dg_image_state');
    if(document.getElementById('audioFile')) {
        sessionStorage.removeItem('dg_voice_state');
        sessionStorage.removeItem('dg_caption_state');
    }
    // "Analyse Another" always cancels any in-progress pipeline mode too.
    sessionStorage.removeItem('dg_pipeline_mode');

    const hint = document.getElementById('uploadHint');
    if (hint) hint.textContent = 'Select a file to begin analysis';

    // "Analyse Another" should hand back a genuinely fresh page, not just an in-place DOM reset.
    window.location.reload();
}

function chipDotClassForVerdict(verdict) {
    // Mirrors the .verdict-badge CSS classes' color grouping (style.css) so a chip's dot color
    // always matches what the same verdict looks like as a full badge elsewhere in the app.
    const green = ['real', 'coherent', 'non-manipulative', 'authentic'];
    const amber = ['artificial', 'suspicious', 'uncertain'];
    if (!verdict) return 'chip-amber';
    const v = verdict.toLowerCase();
    if (green.includes(v)) return 'chip-green';
    if (amber.includes(v)) return 'chip-amber';
    return 'chip-red'; // deepfake, incoherent, manipulative, cloned, likely-cloned
}

function showPipelineStepBadge(labelText, chips) {
    // Renders the "Step N of M" badge + prior-step summary chips at the top of the voice/caption
    // pages.
    const badge = document.getElementById('pipelineStepBadge');
    const label = document.getElementById('pipelineStepLabel');
    const chipsEl = document.getElementById('pipelineStepChips');
    if (!badge || !label || !chipsEl) return;

    label.innerText = labelText;
    chipsEl.innerHTML = '';
    chips.forEach(({ name, verdict, score }) => {
        const chip = document.createElement('span');
        chip.className = 'pipeline-step-chip';
        const dot = document.createElement('span');
        dot.className = `chip-dot ${chipDotClassForVerdict(verdict)}`;
        chip.appendChild(dot);
        const text = document.createElement('span');
        text.innerText = score != null ? `${name}: ${verdict} (${score}%)` : `${name}: ${verdict}`;
        chip.appendChild(text);
        chipsEl.appendChild(chip);
    });
    badge.style.display = 'block';
}

const AUTO_ADVANCE_DELAY_MS = 2500;

function getPipelineMode() {
    // '1' = Video -> Audio -> Dashboard, '2' = Video -> Audio -> Caption -> Dashboard, null =
    // standalone (no auto-advance).
    const mode = sessionStorage.getItem('dg_pipeline_mode');
    return mode === '1' || mode === '2' ? mode : null;
}

function showAutoAdvanceNotice(message, anchorSelector = '.action-btn-row') {
    // A visible "what's happening and why" line for pipeline mode.
    const row = document.querySelector(anchorSelector);
    if (!row) return;
    let notice = document.getElementById('autoAdvanceNotice');
    if (!notice) {
        notice = document.createElement('div');
        notice.id = 'autoAdvanceNotice';
        notice.style.cssText = 'width:100%;text-align:center;margin-top:0.75rem;color:var(--text-muted);font-size:0.9rem;';
        row.insertAdjacentElement('afterend', notice);
    }
    notice.innerText = message;
}

function hideActionButtons() {
    // In pipeline mode the next step runs automatically, so the manual action buttons are hidden.
    const row = document.querySelector('.action-btn-row');
    if (row) row.style.display = 'none';
}

function proceedToVoicePipeline(e) {
    e.preventDefault();
    // The server already holds the just-analysed video in the session, so the voice page only
    // needs to call the chain endpoint.
    sessionStorage.setItem('dg_pending_audio_check', 'true');
    window.location.href = '/voice';
}

function handleImageUpload() {
    const fileInput = document.getElementById('imageFile');
    const file = fileInput.files[0];
    if (!file) return;

    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('loader').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    const formData = new FormData();
    formData.append('image', file);

    fetch('/analyze_image', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loader').style.display = 'none';
        if (data.error) {
            alert('Error: ' + data.error);
            resetUpload();
            return;
        }
        showImageResults(data);
    })
    .catch(error => {
        document.getElementById('loader').style.display = 'none';
        resetUpload();
        alert('An error occurred during image analysis.');
        console.error('Error:', error);
    });
}

function showImageResults(data, persist = true) {
    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('results').style.display = 'block';

    // Signal 1b returns one of 3 verdicts now (Real / Artificial / Deepfake), not a collapsed
    // binary Real/Deepfake.
    const badge = document.getElementById('verdictBadge');
    badge.className = `verdict-badge ${data.verdict.toLowerCase()}`;
    badge.innerText = data.verdict;

    document.getElementById('confidenceScore').innerText = `${data.score}%`;
    document.getElementById('inferenceTime').innerText = `${data.inference_time}s`;
    document.getElementById('explanationText').innerText = data.explanation;

    const filenameEl = document.getElementById('resultFilename');
    if (filenameEl) filenameEl.innerText = `File: ${data.filename}`;

    if (persist) {
        sessionStorage.setItem('dg_image_state', JSON.stringify(data));
    }
}

function handleVoiceUpload() {
    const fileInput = document.getElementById('audioFile');
    const file = fileInput.files[0];
    if (!file) return;

    const hint = document.getElementById('uploadHint');
    if (hint) hint.textContent = file.name;

    const formData = new FormData();
    formData.append('audio', file);

    runVoiceAnalysisRequest('/analyze_voice', { method: 'POST', body: formData });
}

function runVoiceAnalysisRequest(url, fetchOptions) {
    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('loader').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    if (document.getElementById('noAudioError')) document.getElementById('noAudioError').style.display = 'none';
    const urlErrorEl = document.getElementById('voiceUrlError');
    if (urlErrorEl) urlErrorEl.style.display = 'none';

    const loaderText = document.getElementById('loaderText');
    if (loaderText) loaderText.innerText = 'Transcribing and analyzing voice…';

    fetch(url, fetchOptions)
        .then(response => response.json().then(data => ({ status: response.status, data })))
        .then(({ status, data }) => {
            document.getElementById('loader').style.display = 'none';
            if (data.error === 'NO_AUDIO_TRACK') {
                document.getElementById('noAudioError').style.display = 'block';

                // In pipeline mode, a silent video shouldn't dead-end the whole run.
                const mode = getPipelineMode();
                if (mode) {
                    const tryAnotherBtn = document.getElementById('noAudioTryAnotherBtn');
                    if (tryAnotherBtn) tryAnotherBtn.style.display = 'none';
                    stashCaptionState(null);
                    if (mode === '2') {
                        showAutoAdvanceNotice('Pipeline mode: no audio track found — continuing to Caption Check…', '#noAudioError');
                        setTimeout(() => proceedToCaptionCheck({ preventDefault: () => {} }), AUTO_ADVANCE_DELAY_MS);
                    } else {
                        showAutoAdvanceNotice('Pipeline mode: no audio track found — continuing to the Dashboard…', '#noAudioError');
                        setTimeout(() => {
                            sessionStorage.removeItem('dg_pipeline_mode');
                            window.location.href = '/verdict';
                        }, AUTO_ADVANCE_DELAY_MS);
                    }
                }
                return;
            }
            if (data.error) {
                document.getElementById('uploadZone').style.display = 'block';
                const message = data.message || data.error;
                if (urlErrorEl) {
                    urlErrorEl.innerText = message;
                    urlErrorEl.style.display = 'block';
                } else {
                    alert('Error: ' + message);
                }
                return;
            }
            showVoiceResults(data);
        })
        .catch(error => {
            document.getElementById('loader').style.display = 'none';
            document.getElementById('uploadZone').style.display = 'block';
            const message = 'An error occurred during voice analysis.';
            if (urlErrorEl) {
                urlErrorEl.innerText = message;
                urlErrorEl.style.display = 'block';
            } else {
                alert(message);
            }
            console.error('Error:', error);
        });
}

function handleVoiceUrlAnalysis() {
    const urlInput = document.getElementById('voiceUrlInput');
    const url = urlInput.value.trim();
    if (!url) return;
    runVoiceAnalysisRequest('/analyze_voice_url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
    });
}

function runPendingAudioCheck() {
    runVoiceAnalysisRequest('/analyze_voice_chain', { method: 'POST' });
}

function proceedToCaptionCheck(e) {
    e.preventDefault();
    // dg_caption_state was already stashed by showVoiceResults() (or, for a no-audio video,
    // stashCaptionState(null) directly) when this page's voice step completed.
    window.location.href = '/caption';
}

function stashCaptionState(data) {
    // Stash everything Caption Check needs to hydrate itself without a second network call.
    let videoInfo = {};
    try {
        const videoState = sessionStorage.getItem('dg_video_state');
        if (videoState) videoInfo = JSON.parse(videoState);
    } catch (e) { /* ignore malformed stored state */ }

    const meta = (data && data.chain_meta) || {};
    sessionStorage.setItem('dg_caption_state', JSON.stringify({
        title: meta.title || videoInfo.title || (data && data.filename) || '(untitled)',
        platform: meta.platform || videoInfo.platform || 'Local upload',
        caption: meta.caption || videoInfo.caption || '',
        transcript: (data && data.transcript) || '',
        video_verdict: videoInfo.verdict || null,
        video_score: videoInfo.score || null,
        voice_clone_verdict: (data && data.verdict) || null,
        voice_clone_score: (data && data.score) || null,
        manipulation_verdict: (data && data.manipulation_verdict) || null,
        manipulation_score: (data && data.manipulation_score) || null,
    }));
}

function resetDashboard(e) {
    e.preventDefault();
    // The dashboard's own "Analyse Another".
    ['dg_video_state', 'dg_image_state', 'dg_voice_state', 'dg_caption_state', 'dg_pending_audio_check', 'dg_pipeline_mode']
        .forEach(key => sessionStorage.removeItem(key));
    window.location.href = '/reset_dashboard';
}

function showVoiceResults(data, persist = true) {
    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('results').style.display = 'block';

    const badge = document.getElementById('verdictBadge');
    if (badge) {
        badge.className = `verdict-badge ${data.verdict.toLowerCase()}`;
        badge.innerText = data.verdict;
    }

    const scoreEl = document.getElementById('confidenceScore');
    if (scoreEl) scoreEl.innerText = `${data.score}%`;

    const timeEl = document.getElementById('inferenceTime');
    if (timeEl) timeEl.innerText = `${data.inference_time}s`;

    const expEl = document.getElementById('explanationText');
    if (expEl) expEl.innerText = data.explanation;

    const fileEl = document.getElementById('resultFilename');
    if (fileEl) fileEl.innerText = `File: ${data.filename}`;

    // Voice clone verdict badge in the left panel
    const cloneNoteEl = document.getElementById('voiceCloneNote');
    if (cloneNoteEl) {
        cloneNoteEl.innerText = `Model verdict: ${data.verdict} (${data.score}% confidence)`;
    }

    // Manipulation score (Signal 3)
    const manipScoreEl = document.getElementById('manipulationScore');
    if (manipScoreEl) manipScoreEl.innerText = `${data.manipulation_score}%`;

    const manipVerdictEl = document.getElementById('manipulationVerdict');
    if (manipVerdictEl) {
        manipVerdictEl.innerText = data.manipulation_verdict;
        manipVerdictEl.style.color = data.manipulation_verdict === 'Manipulative' ? 'rgb(255, 71, 87)' : 'var(--text-muted)';
    }

    // Transcript
    const transcriptEl = document.getElementById('voiceTranscript');
    if (transcriptEl) {
        transcriptEl.innerText = data.transcript || '(no speech detected)';
        transcriptEl.style.color = 'var(--text-main)';
    }

    // Waveform.
    if (data.waveform && data.waveform.length) {
        renderWaveform(data.waveform);
        setupWaveformPlayer(data.audio_url, data.waveform);
    }

    // Emotion breakdown (top 3, from the backend)
    const emotionListEl = document.getElementById('emotionBreakdown');
    if (emotionListEl) {
        emotionListEl.innerHTML = '';
        if (data.emotions && data.emotions.length > 0) {
            data.emotions.forEach(e => {
                const li = document.createElement('li');
                li.innerText = `${e.label} — ${e.score}%`;
                emotionListEl.appendChild(li);
            });
        } else {
            emotionListEl.innerHTML = data.transcript
                ? '<li>Emotion breakdown not available for this model</li>'
                : '<li>No speech detected</li>';
        }
    }

    if (persist) {
        // audio_url can be a multi-MB base64 data URI.
        try {
            sessionStorage.setItem('dg_voice_state', JSON.stringify(data));
        } catch (e) {
            try {
                const { audio_url, ...withoutAudio } = data;
                sessionStorage.setItem('dg_voice_state', JSON.stringify(withoutAudio));
            } catch (e2) { /* still too big or storage unavailable - skip persisting */ }
        }

        stashCaptionState(data);

        // This is where the two pipelines fork: Pipeline 1 (Video -> Audio -> Dashboard) ends here
        // and goes straight to the dashboard.
        const mode = getPipelineMode();
        if (mode === '2') {
            hideActionButtons();
            showAutoAdvanceNotice('Pipeline mode: continuing automatically to Caption Check…');
            setTimeout(() => proceedToCaptionCheck({ preventDefault: () => {} }), AUTO_ADVANCE_DELAY_MS);
        } else if (mode === '1') {
            hideActionButtons();
            showAutoAdvanceNotice('Pipeline mode: continuing automatically to the Dashboard…');
            setTimeout(() => {
                sessionStorage.removeItem('dg_pipeline_mode');
                window.location.href = '/verdict';
            }, AUTO_ADVANCE_DELAY_MS);
        }
    }
}

function handleUrlAnalysis() {
    const urlInput = document.getElementById('socialUrlInput');
    const url = urlInput.value.trim();
    if (!url) return;

    const btn = document.getElementById('analyzeBtn');
    btn.style.display = 'none';
    document.getElementById('loader').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    document.getElementById('urlError').style.display = 'none';

    fetch('/analyze_url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
    })
    .then(response => response.json().then(data => ({ status: response.status, data })))
    .then(({ status, data }) => {
        document.getElementById('loader').style.display = 'none';
        btn.style.display = 'block';
        btn.innerText = 'Analyze Link';
        if (data.error) {
            document.getElementById('urlErrorText').innerText = data.message || data.error;
            document.getElementById('urlError').style.display = 'block';
            return;
        }
        showUrlResults(data);
    })
    .catch(error => {
        document.getElementById('loader').style.display = 'none';
        btn.style.display = 'block';
        btn.innerText = 'Analyze Link';
        document.getElementById('urlErrorText').innerText = 'An error occurred while analyzing this link.';
        document.getElementById('urlError').style.display = 'block';
        console.error('Error:', error);
    });
}

function showUrlResults(data) {
    document.getElementById('results').style.display = 'block';

    document.getElementById('resultTitle').innerText = data.title || '(untitled)';
    document.getElementById('resultPlatform').innerText = `Platform: ${data.platform}`;

    // Video/Voice-Clone badges and the transcript box are optional.
    const videoBadge = document.getElementById('videoVerdictBadge');
    if (videoBadge) {
        if (data.video_verdict) {
            videoBadge.className = `verdict-badge ${data.video_verdict.toLowerCase()}`;
            videoBadge.innerText = `${data.video_verdict} (${data.video_score}%)`;
        } else {
            videoBadge.className = 'verdict-badge';
            videoBadge.innerText = 'Not analyzed';
        }
    }

    const cloneBadge = document.getElementById('cloneVerdictBadge');
    if (cloneBadge) {
        if (data.voice_clone_verdict) {
            cloneBadge.className = `verdict-badge ${data.voice_clone_verdict.toLowerCase()}`;
            cloneBadge.innerText = `${data.voice_clone_verdict} (${data.voice_clone_score}%)`;
        } else {
            cloneBadge.className = 'verdict-badge';
            cloneBadge.innerText = 'No audio track';
        }
    }

    const manipBadge = document.getElementById('manipVerdictBadge');
    if (manipBadge) {
        if (data.manipulation_verdict) {
            manipBadge.className = `verdict-badge ${data.manipulation_verdict === 'Manipulative' ? 'deepfake' : 'real'}`;
            manipBadge.innerText = `${data.manipulation_verdict} (${data.manipulation_score}%)`;
        } else {
            manipBadge.className = 'verdict-badge';
            manipBadge.innerText = 'No audio track';
        }
    }

    const transcriptEl = document.getElementById('videoTranscript');
    if (transcriptEl) transcriptEl.innerText = data.transcript || '(no speech detected)';

    const captionEl = document.getElementById('extractedCaption');
    if (captionEl) captionEl.value = data.caption || '(no caption/description found)';

    // Only render a coherence result if this data actually came from a server response that
    // computed one (/analyze_url or /score_coherence).
    if ('coherence_verdict' in data) {
        showCoherenceResult(data.coherence_verdict, data.coherence_score, data.coherence_message);
    }
}

function showCoherenceResult(verdict, score, message) {
    const loadingEl = document.getElementById('coherenceLoading');
    const badgeEl = document.getElementById('coherenceVerdictBadge');
    const unavailableEl = document.getElementById('coherenceUnavailable');
    if (!loadingEl || !badgeEl || !unavailableEl) return; // page not hydrated with coherence UI

    loadingEl.style.display = 'none';
    if (verdict) {
        badgeEl.style.display = 'inline-block';
        badgeEl.className = `verdict-badge ${verdict.toLowerCase()}`;
        badgeEl.innerText = `${verdict} (${score}%)`;
        unavailableEl.style.display = 'none';
    } else {
        badgeEl.style.display = 'none';
        unavailableEl.style.display = 'block';
        document.getElementById('coherenceUnavailableText').innerText =
            message || 'No transcript or caption available for this clip, so coherence cannot be scored.';
    }
}

function scoreCoherenceForChainedData(transcript, caption) {
    fetch('/score_coherence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript, caption }),
    })
    .then(response => response.json())
    .then(data => {
        showCoherenceResult(data.coherence_verdict, data.coherence_score, data.coherence_message);

        // Caption Check is Pipeline 2's last stop.
        if (getPipelineMode() === '2') {
            hideActionButtons();
            showAutoAdvanceNotice('Pipeline mode: continuing automatically to the Dashboard…');
            setTimeout(() => {
                sessionStorage.removeItem('dg_pipeline_mode');
                window.location.href = '/verdict';
            }, AUTO_ADVANCE_DELAY_MS);
        }
    })
    .catch(error => {
        console.error('Error scoring coherence:', error);
        showCoherenceResult(null, null, 'An error occurred while scoring coherence.');
    });
}

function isGenuinePageReload() {
    // Distinguishes an actual browser refresh (F5 / reload button) from a normal navigation.
    const entries = performance.getEntriesByType('navigation');
    return entries.length > 0 && entries[0].type === 'reload';
}

function restoreStoredState(key, showFn) {
    // "Navigating back and forth shouldn't lose earlier results until Analyse Another is clicked".
    if (isGenuinePageReload()) {
        sessionStorage.removeItem(key);
        return false;
    }
    const stored = sessionStorage.getItem(key);
    if (!stored) return false;
    try {
        const data = JSON.parse(stored);
        showFn(data, false); // persist=false: don't re-write what we just read
        return true;
    } catch (e) {
        sessionStorage.removeItem(key);
        return false;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const videoInput = document.getElementById('videoFile');
    if (videoInput) {
        videoInput.addEventListener('change', showVideoConfirm);
        // A pipeline Start button on the home page links here with ?pipeline=1 or ?pipeline=2.
        const requestedPipeline = new URLSearchParams(window.location.search).get('pipeline');
        if (requestedPipeline === '1' || requestedPipeline === '2') {
            sessionStorage.setItem('dg_pipeline_mode', requestedPipeline);
        }
        restoreStoredState('dg_video_state', showResults);
    }

    const imageInput = document.getElementById('imageFile');
    if (imageInput) {
        imageInput.addEventListener('change', handleImageUpload);
        restoreStoredState('dg_image_state', showImageResults);
    }

    const audioInput = document.getElementById('audioFile');
    if (audioInput) {
        audioInput.addEventListener('change', () => handleVoiceUpload());

        // A pending Audio Check takes priority over restoring an older dg_voice_state.
        if (sessionStorage.getItem('dg_pending_audio_check') === 'true') {
            sessionStorage.removeItem('dg_pending_audio_check');
            sessionStorage.removeItem('dg_voice_state');
            sessionStorage.removeItem('dg_caption_state');

            try {
                const videoState = JSON.parse(sessionStorage.getItem('dg_video_state') || 'null');
                if (videoState) {
                    showPipelineStepBadge('Step 2 of 3 — Voice & Manipulation Check', [
                        { name: 'Video', verdict: videoState.verdict, score: videoState.score },
                    ]);
                }
            } catch (e) { /* ignore malformed stored state */ }

            setTimeout(() => runPendingAudioCheck(), 300);
        } else {
            restoreStoredState('dg_voice_state', showVoiceResults);
        }
    }

    // On the dashboard, a genuine refresh resets the verdict section to "Not run yet"; ordinary
    // navigation restores it.
    if (document.getElementById('dashOverallVerdict') && isGenuinePageReload()) {
        fetch('/reset_verdict_signals', { method: 'POST' })
            .then(() => window.location.replace(window.location.pathname))
            .catch(() => { /* reset failed - leave the page showing whatever it already has rather than risk a loop */ });
    }
});

// Exposes a handful of pure/DOM-testable functions to Jest under Node's CommonJS loader, for
// tests/unit testing/frontend/.
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        chipDotClassForVerdict,
        isGenuinePageReload,
        restoreStoredState,
        renderWaveform,
        seekWaveform,
        showCoherenceResult,
        resetDashboard,
    };
}
