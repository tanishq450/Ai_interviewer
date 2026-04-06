/* ============================================================
   app.js — AI Interviewer Voice-First Frontend
============================================================ */

const API = "http://localhost:8000";

/* ──────────────────────────────────────────
   STATE
────────────────────────────────────────── */
let appState = {
  userId: '',
  interviewState: null,
  questionCount: 0,
  maxQuestions: 10,
  answerMode: 'voice',   // 'voice' | 'text'
  mediaRecorder: null,
  audioChunks: [],
  isRecording: false,
  analyserNode: null,
  audioCtx: null,
  waveAnimId: null,
  history: [],
};

/* ──────────────────────────────────────────
   ELEMENT REFS
────────────────────────────────────────── */
const $ = id => document.getElementById(id);

const screens = {
  landing: $('screen-landing'),
  setup: $('screen-setup'),
  interview: $('screen-interview'),
  results: $('screen-results'),
};

/* ──────────────────────────────────────────
   SCREEN NAVIGATION
────────────────────────────────────────── */
function showScreen(name) {
  Object.entries(screens).forEach(([k, el]) => {
    if (k === name) {
      el.classList.remove('slide-out');
      el.classList.add('active');
    } else {
      el.classList.remove('active');
    }
  });
}

/* ──────────────────────────────────────────
   LOADER
────────────────────────────────────────── */
function showLoader(msg = 'Loading…') {
  $('loader-msg').textContent = msg;
  $('global-loader').classList.remove('hidden');
}
function hideLoader() {
  $('global-loader').classList.add('hidden');
}

/* ──────────────────────────────────────────
   LANDING → SETUP
────────────────────────────────────────── */
$('btn-get-started').addEventListener('click', () => showScreen('setup'));
$('back-setup').addEventListener('click', () => showScreen('landing'));

/* ──────────────────────────────────────────
   FILE DROP ZONE
────────────────────────────────────────── */
const dropZone = $('drop-zone');
const fileInput = $('resume-file');

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => handleFileSelect(fileInput.files[0]));

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  handleFileSelect(e.dataTransfer.files[0]);
});

function handleFileSelect(file) {
  if (!file) return;
  $('drop-content').innerHTML = `<span class="drop-icon">📄</span><span>${file.name}</span>`;
  fileInput._selectedFile = file;
}

/* ──────────────────────────────────────────
   UPLOAD RESUME & START
────────────────────────────────────────── */
$('btn-upload').addEventListener('click', async () => {
  const userId = $('user-id-input').value.trim();
  const file = fileInput._selectedFile || fileInput.files[0];

  if (!userId) return showStatus('Please enter a User ID.', 'error');
  if (!file) return showStatus('Please select a PDF resume.', 'error');

  appState.userId = userId;

  showLoader('Uploading resume…');
  try {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('user_id_form', userId);   // matches FastAPI param name in /resume/upload

    const res = await fetch(`${API}/resume/upload?user_id=${encodeURIComponent(userId)}`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');

    showStatus(`✅ Detected domain: ${data.detected_domain} | Difficulty: ${data.detected_difficulty}`, 'success');
    await sleep(900);
    await startInterview();
  } catch (err) {
    hideLoader();
    showStatus(`❌ ${err.message}`, 'error');
  }
});

function showStatus(msg, type) {
  const el = $('upload-status');
  el.textContent = msg;
  el.className = `status-msg ${type}`;
  el.classList.remove('hidden');
}

/* ──────────────────────────────────────────
   START INTERVIEW
────────────────────────────────────────── */
async function startInterview() {
  showLoader('Starting interview…');
  try {
    const res = await fetch(`${API}/interview/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: appState.userId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Could not start interview');

    appState.interviewState = data.state;
    appState.questionCount = data.state.question_count || 1;
    appState.history = [];

    // Clear chat
    $('chat-panel').innerHTML = '<div class="chat-empty" id="chat-empty">Conversation will appear here…</div>';

    showScreen('interview');
    hideLoader();

    displayQuestion(data.question, data.audio_id);
  } catch (err) {
    hideLoader();
    showStatus(`❌ ${err.message}`, 'error');
    showScreen('setup');
  }
}

/* ──────────────────────────────────────────
   DISPLAY QUESTION + PLAY AUDIO
────────────────────────────────────────── */
function displayQuestion(question, audioId) {
  // Update text
  $('question-text').textContent = question;

  // Update progress
  const qNum = appState.questionCount;
  $('q-counter').textContent = `Question ${qNum} / ${appState.maxQuestions}`;
  $('progress-fill').style.width = `${(qNum / appState.maxQuestions) * 100}%`;

  // Audio status: generating
  setAudioStatus('spinning', 'Playing audio…');
  $('btn-replay').hidden = true;

  // Push to chat
  removeChatEmpty();
  addChatMsg('ai', '🤖 AI Interviewer', question);

  // Scroll chat
  scrollChat();

  // Play audio
  if (audioId) {
    playQuestionAudio(audioId);
  } else {
    setAudioStatus('error', 'No audio available');
  }

  // Unlock answer controls
  setAnswerEnabled(true);
}

function playQuestionAudio(audioId) {
  const audio = $('question-audio');
  audio.src = `${API}/audio/${audioId}`;
  audio.hidden = false;

  audio.oncanplay = () => {
    setAudioStatus('playing', 'Playing…');
    audio.play().catch(() => setAudioStatus('error', 'Click Replay to hear the question'));
  };
  audio.onended = () => {
    setAudioStatus('done', 'Audio done');
    $('btn-replay').hidden = false;
  };
  audio.onerror = () => {
    setAudioStatus('error', 'Audio failed — read the question above');
    $('btn-replay').hidden = false;
  };
}

$('btn-replay').addEventListener('click', () => {
  const audio = $('question-audio');
  if (audio.src) {
    audio.currentTime = 0;
    setAudioStatus('playing', 'Playing…');
    audio.play().catch(() => { });
  }
});

function setAudioStatus(dotClass, text) {
  const el = $('audio-status');
  el.innerHTML = `<span class="audio-dot ${dotClass}"></span> ${text}`;
}

/* ──────────────────────────────────────────
   ANSWER MODE TOGGLE
────────────────────────────────────────── */
$('mode-voice-btn').addEventListener('click', () => setMode('voice'));
$('mode-text-btn').addEventListener('click', () => setMode('text'));

function setMode(mode) {
  appState.answerMode = mode;
  $('mode-voice-btn').classList.toggle('active', mode === 'voice');
  $('mode-text-btn').classList.toggle('active', mode === 'text');
  $('voice-panel').classList.toggle('hidden', mode !== 'voice');
  $('text-panel').classList.toggle('hidden', mode !== 'text');
}

/* ──────────────────────────────────────────
   VOICE RECORDING (MediaRecorder)
────────────────────────────────────────── */
const micBtn = $('mic-btn');

micBtn.addEventListener('mousedown', startRecording);
micBtn.addEventListener('touchstart', startRecording, { passive: true });
micBtn.addEventListener('mouseup', stopRecording);
micBtn.addEventListener('mouseleave', stopRecording);
micBtn.addEventListener('touchend', stopRecording);

function stopRecording(e) {
  if (e) e.preventDefault();
  if (!appState.isRecording || !appState.mediaRecorder) return;
  if (appState.mediaRecorder.state !== 'inactive') {
    appState.mediaRecorder.stop();
  }
}

async function startRecording(e) {
  e.preventDefault();
  if (appState.isRecording) return;

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    alert('Microphone access denied. Please allow mic access or use text mode.');
    return;
  }

  appState.isRecording = true;
  appState.audioChunks = [];
  micBtn.classList.add('recording');
  $('mic-hint').textContent = '🔴 Recording… release to submit';

  // Waveform visualiser
  setupWaveform(stream);

  const mimeType = getSupportedMime();
  appState.mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
  appState.mediaRecorder.ondataavailable = e => { if (e.data.size > 0) appState.audioChunks.push(e.data); };
  appState.mediaRecorder.onstop = handleRecordingStop;
  appState.mediaRecorder.start();
}

/* ──────────────────────────────────────────
   HANDLE ANSWER RESPONSE
────────────────────────────────────────── */
function handleAnswerResponse(data, answerText) {
  hideLoader();

  // Update state
  appState.interviewState = data.state;
  appState.questionCount = data.state.question_count || appState.questionCount + 1;

  // Show answer in chat
  addChatMsg('user', '🧑 You', answerText);

  // Transcript preview (voice mode)
  if (data.transcript) {
    $('transcript-preview').classList.remove('hidden');
    $('transcript-text').textContent = data.transcript;
  }

  // Show feedback
  if (data.feedback) {
    $('feedback-card').classList.remove('hidden');
    $('feedback-text').textContent = data.feedback;

    // Also add feedback to chat
    addChatMsg('ai', '💡 Feedback', data.feedback);
  } else {
    $('feedback-card').classList.add('hidden');
  }

  scrollChat();

  // Done?
  if (data.done) {
    setTimeout(() => showResults(), 1500);
    return;
  }

  // Next question
  if (data.question) {
    displayQuestion(data.question, data.audio_id);
  }
}

/* ──────────────────────────────────────────
   RESULTS SCREEN
────────────────────────────────────────── */
function showResults() {
  const state = appState.interviewState || {};
  const scores = state.scores || [];
  const feedback = state.feedback || [];
  const weak = state.weak_areas || [];

  // Average score
  const avg = scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1) : '—';

  $('score-grid').innerHTML = `
    <div class="score-item">
      <div class="score-val">${scores.length}</div>
      <div class="score-lbl">Questions Answered</div>
    </div>
    <div class="score-item">
      <div class="score-val">${avg}</div>
      <div class="score-lbl">Avg Score</div>
    </div>
    <div class="score-item">
      <div class="score-val">${weak.length}</div>
      <div class="score-lbl">Weak Areas</div>
    </div>
  `;

  // Weak areas
  if (weak.length) {
    $('weak-areas-wrap').style.display = '';
    $('weak-tags').innerHTML = weak.map(w => `<span>${w}</span>`).join('');
  } else {
    $('weak-areas-wrap').style.display = 'none';
  }

  // Final feedback
  $('final-feedback-list').innerHTML = feedback.map((f, i) => `
    <div class="fb-item">
      <div class="fb-q">Question ${i + 1}</div>
      <div class="fb-f">${f}</div>
    </div>
  `).join('');

  showScreen('results');
}

$('btn-restart').addEventListener('click', () => {
  appState = { ...appState, interviewState: null, questionCount: 0, history: [], answerMode: 'voice' };
  $('user-id-input').value = '';
  $('upload-status').classList.add('hidden');
  $('drop-content').innerHTML = '<span class="drop-icon">📂</span><span>Drop your PDF here or <u>browse</u></span>';
  fileInput._selectedFile = null;
  showScreen('setup');
});

/* ──────────────────────────────────────────
   CHAT HELPERS
────────────────────────────────────────── */
function removeChatEmpty() {
  const el = $('chat-empty');
  if (el) el.remove();
}

function addChatMsg(role, sender, text) {
  const panel = $('chat-panel');
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  div.innerHTML = `<div class="sender">${sender}</div><div>${escHtml(text)}</div>`;
  panel.appendChild(div);
}

function scrollChat() {
  const p = $('chat-panel');
  p.scrollTop = p.scrollHeight;
}

/* ──────────────────────────────────────────
   WAVEFORM VISUALISER
────────────────────────────────────────── */
function setupWaveform(stream) {
  appState.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = appState.audioCtx.createMediaStreamSource(stream);
  appState.analyserNode = appState.audioCtx.createAnalyser();
  appState.analyserNode.fftSize = 256;
  source.connect(appState.analyserNode);
  drawWaveform();
}

function drawWaveform() {
  const canvas = $('waveform-canvas');
  const ctx = canvas.getContext('2d');
  const analyser = appState.analyserNode;
  if (!analyser) return;

  const buf = new Uint8Array(analyser.frequencyBinCount);

  function frame() {
    appState.waveAnimId = requestAnimationFrame(frame);
    analyser.getByteFrequencyData(buf);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const barW = (canvas.width / buf.length) * 2.2;
    let x = 0;

    buf.forEach(val => {
      const h = (val / 255) * canvas.height;
      const grad = ctx.createLinearGradient(0, canvas.height - h, 0, canvas.height);
      grad.addColorStop(0, '#7c3aed');
      grad.addColorStop(1, '#22d3ee');
      ctx.fillStyle = grad;
      ctx.fillRect(x, canvas.height - h, barW - 1, h);
      x += barW;
    });
  }
  frame();
}

function clearWaveform() {
  const canvas = $('waveform-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (appState.audioCtx) { appState.audioCtx.close(); appState.audioCtx = null; }
}

/* ──────────────────────────────────────────
   UTILITIES
────────────────────────────────────────── */
function setAnswerEnabled(enabled) {
  micBtn.disabled = !enabled;
  $('btn-submit-text').disabled = !enabled;
  $('text-answer').disabled = !enabled;
  $('mode-voice-btn').disabled = !enabled;
  $('mode-text-btn').disabled = !enabled;
}

function getSupportedMime() {
  const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];
  return types.find(t => MediaRecorder.isTypeSupported(t)) || '';
}

function escHtml(str) {
  return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

/* ──────────────────────────────────────────
   HANDLE RECORDING STOP (VOICE SUBMIT)
────────────────────────────────────────── */
async function handleRecordingStop() {
  appState.isRecording = false;
  micBtn.classList.remove('recording');
  $('mic-hint').textContent = 'Processing your answer...';

  clearWaveform();

  const blob = new Blob(appState.audioChunks, { type: appState.mediaRecorder.mimeType || 'audio/webm' });
  appState.audioChunks = [];

  if (blob.size === 0) return;

  showLoader('Analysing your answer…');
  setAnswerEnabled(false);

  try {
    const fd = new FormData();
    fd.append('file', blob, 'answer.webm');
    fd.append('user_id', appState.userId);
    fd.append('state', JSON.stringify(appState.interviewState));

    const res = await fetch(`${API}/interview/answer-voice`, {
      method: 'POST',
      body: fd
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || 'Voice answer failed');
    }

    const data = await res.json();
    handleAnswerResponse(data, "Audio Answer");

  } catch (err) {
    hideLoader();
    alert(`Voice Submit Error: ${err.message}`);
    setAnswerEnabled(true);
    $('mic-hint').textContent = 'Hold to record your answer';
  }
}

/* ──────────────────────────────────────────
   TEXT ANSWER SUBMISSION
────────────────────────────────────────── */
$('btn-submit-text').addEventListener('click', submitTextAnswer);

$('text-answer').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitTextAnswer();
  }
});

async function submitTextAnswer() {
  const answerInput = $('text-answer');
  const answer = answerInput.value.trim();

  if (!answer) {
    alert('Please type an answer before submitting.');
    return;
  }

  showLoader('Submitting answer…');
  setAnswerEnabled(false);
  answerInput.value = '';

  try {
    const res = await fetch(`${API}/interview/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: appState.userId,
        answer: answer,
        state: appState.interviewState
      })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || 'Text answer failed');
    }

    const data = await res.json();
    handleAnswerResponse(data, answer);

  } catch (err) {
    hideLoader();
    alert(`Text Submit Error: ${err.message}`);
    setAnswerEnabled(true);
  }
}

/* ──────────────────────────────────────────
   INIT
────────────────────────────────────────── */
showScreen('landing');
setMode('voice');
