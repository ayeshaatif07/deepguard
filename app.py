import os

# Load models from the local HF cache only; avoids a network check on every
# from_pretrained call. Must be set before transformers is imported.
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import time
import uuid
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
from modules.frame_extractor import extract_frames
from modules.usage_log import log_usage, get_usage_stats

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
# Anchored to this file, so the path is the same whatever the working directory.
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300MB

ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
ALLOWED_VOICE_EXTENSIONS = {'mp3', 'wav', 'm4a', 'mp4', 'flac'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def unique_upload_path(filename):
    """Returns a collision-free upload path, prefixed per request so concurrent
    uploads of the same filename cannot overwrite each other.
    """
    return os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}_{filename}")


def save_upload_with_retry(file, filepath, attempts=4, base_delay=0.3):
    """Saves an uploaded file, retrying on transient filesystem errors from iCloud sync."""
    last_error = None
    for attempt in range(attempts):
        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.stream.seek(0)
            file.save(filepath)
            return
        except (FileNotFoundError, OSError) as e:
            last_error = e
            if attempt < attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_error


def remove_upload_with_retry(filepath, attempts=3, base_delay=0.2):
    """Best-effort delete of a processed upload. Never raises."""
    for attempt in range(attempts):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return
        except OSError:
            if attempt < attempts - 1:
                time.sleep(base_delay * (2 ** attempt))


def _clear_pending_chain_video():
    """Deletes any un-consumed video left over from a previous analysis, so at most
    one chainable file exists at a time.
    """
    old_path = session.pop('chainable_video_path', None)
    session.pop('chainable_video_meta', None)
    if old_path:
        remove_upload_with_retry(old_path)


# Set DEEPGUARD_MOCK=1 to run the app with dummy scores and no model weights.
MOCK_MODE = os.getenv('DEEPGUARD_MOCK', '').strip().lower() in ('1', 'true', 'yes', 'on')

# Each getter loads its model once and caches it for the process lifetime.
_scorer = None
_image_scorer = None
_voice_clone_scorer = None
_voice_manipulation_scorer = None
_coherence_scorer = None


def get_scorer():
    global _scorer
    if _scorer is None:
        # Signal 1: CViT2 (Wodajo & Atnafu).
        from modules.video_detection.cvit_scorer import CViTScorer
        print("Initializing CViTScorer...")
        _scorer = CViTScorer()
    return _scorer


def get_image_scorer():
    global _image_scorer
    if _image_scorer is None:
        from modules.image_detection.image_scorer import ImageScorer
        print("Initializing ImageScorer...")
        _image_scorer = ImageScorer()
    return _image_scorer


def get_voice_clone_scorer():
    global _voice_clone_scorer
    if _voice_clone_scorer is None:
        from modules.voice_detection.voice_clone_scorer import VoiceCloneScorer
        print("Initializing VoiceCloneScorer...")
        _voice_clone_scorer = VoiceCloneScorer()
    return _voice_clone_scorer


def get_voice_manipulation_scorer():
    global _voice_manipulation_scorer
    if _voice_manipulation_scorer is None:
        from modules.voice_detection.voice_manipulation_scorer import VoiceManipulationScorer
        print("Initializing VoiceManipulationScorer...")
        _voice_manipulation_scorer = VoiceManipulationScorer()
    return _voice_manipulation_scorer


def get_coherence_scorer():
    global _coherence_scorer
    if _coherence_scorer is None:
        from modules.caption_coherence.coherence_scorer import CaptionCoherenceScorer
        print("Initializing CaptionCoherenceScorer...")
        _coherence_scorer = CaptionCoherenceScorer()
    return _coherence_scorer


# Load all 5 models at startup so the first request doesn't pay the cost.
if not MOCK_MODE:
    get_scorer()
    get_image_scorer()
    get_voice_clone_scorer()
    get_voice_manipulation_scorer()
    get_coherence_scorer()


@app.route('/')
def home():
    return render_template('home.html', active_page='home')

@app.route('/video')
def video():
    return render_template('index.html', active_page='video')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        start_time = time.time()
        filename = secure_filename(file.filename)
        filepath = unique_upload_path(filename)
        _clear_pending_chain_video()
        save_upload_with_retry(file, filepath)

        try:
            # MOCK_MODE returns dummy scores for testing without running the model
            if MOCK_MODE:
                time.sleep(2)
                avg_score = 0.85
                frame_scores = [0.1, 0.2, 0.9, 0.95, 0.88, 0.92, 0.89, 0.91, 0.94, 0.86]
                verdict = "Deepfake"
                explanation = "High probability of AI generation detected."
            else:
                # Extract 15 frames evenly spaced across the video - CViT2's
                # native/benchmarked frame count (modules/cvit_scorer.py)
                frames = extract_frames(filepath, num_frames=15)
                video_scorer = get_scorer()
                avg_score, frame_scores = video_scorer.score_video_frames(frames)
                verdict = video_scorer.get_verdict(avg_score)
                
                if verdict == "Real":
                    explanation = "No signs of deepfake manipulation were detected across the analyzed frames."
                else:
                    explanation = "High probability of AI-generated face manipulation detected across multiple frames."
            
            inference_time = time.time() - start_time
            # Kept on disk so "Audio Check" can reuse it; deleted when consumed
            # or when the next video is analysed.
            session['chainable_video_path'] = filepath
            session['chainable_video_meta'] = {'mode': 'file', 'filename': filename}

            result = {
                'verdict': verdict,
                'score': round(avg_score * 100),
                'frame_scores': frame_scores,
                'inference_time': round(inference_time, 2),
                'explanation': explanation,
                'filename': filename
            }

            # Persist Signal 1's real result server-side so the dashboard can display it
            session['signal1_result'] = result
            log_usage('Video', verdict)

            return jsonify(result)
            
        except Exception as e:
            if os.path.exists(filepath):
                remove_upload_with_retry(filepath)
            return jsonify({'error': str(e)}), 500

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({'error': f'Unsupported file type: .{ext}'}), 400

    start_time = time.time()
    filename = secure_filename(file.filename)
    filepath = unique_upload_path(filename)
    save_upload_with_retry(file, filepath)

    try:
        if MOCK_MODE:
            time.sleep(1)
            confidence = 0.88
            verdict = "Real"
        else:
            img = Image.open(filepath)
            # Signal 1b is a 3-class model, so the verdict is its predicted class.
            confidence, verdict = get_image_scorer().score_image(img)

        explanations = {
            "Real": "No signs of manipulation or AI generation were detected. Noise patterns and structural consistency are typical of natural photography.",
            "Artificial": "This image appears to be fully AI-generated (e.g. by a diffusion or GAN model) rather than a photograph of a real scene — frequency-domain artifacts and structural inconsistencies typical of synthetic image generation were found.",
            "Deepfake": "This image appears to be a manipulated deepfake — real content that has been altered, most likely via face-swap or similar manipulation techniques.",
        }
        explanation = explanations.get(verdict, explanations["Deepfake"])

        inference_time = time.time() - start_time
        remove_upload_with_retry(filepath)

        result = {
            'verdict': verdict,
            'score': round(confidence * 100),
            'model_label': verdict,
            'inference_time': round(inference_time, 2),
            'explanation': explanation,
            'filename': filename
        }

        # Persist Signal 1b's real result server-side so the dashboard can display it
        session['signal1b_result'] = result
        log_usage('Image', verdict)

        return jsonify(result)

    except UnidentifiedImageError:
        if os.path.exists(filepath):
            remove_upload_with_retry(filepath)
        return jsonify({'error': 'File is not a valid image'}), 400
    except Exception as e:
        if os.path.exists(filepath):
            remove_upload_with_retry(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/voice')
def voice():
    return render_template('voice.html', active_page='voice')

@app.route('/analyze_voice', methods=['POST'])
def analyze_voice():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio/video file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_VOICE_EXTENSIONS:
        return jsonify({'error': f'Unsupported file type: .{ext}'}), 400

    start_time = time.time()
    filename = secure_filename(file.filename)
    filepath = unique_upload_path(filename)
    save_upload_with_retry(file, filepath)

    try:
        if MOCK_MODE:
            time.sleep(1)
            clone_score = 0.94
            clone_verdict = "Cloned"
            manip_transcript = "Your account has been frozen. Call us back immediately or you will lose all your funds today."
            manip_score = 0.35
            manip_verdict = "Manipulative"
            manip_emotions = {"fear": 0.5, "neutral": 0.2}
            waveform = [0.1 + 0.8 * abs((i % 20) - 10) / 10 for i in range(100)]
            audio_url = None
        else:
            from modules.voice_detection.audio_utils import extract_audio_array, compute_waveform_peaks, encode_playable_audio
            try:
                audio_array = extract_audio_array(filepath)
            except RuntimeError as e:
                remove_upload_with_retry(filepath)
                if str(e) == 'NO_AUDIO_TRACK':
                    return jsonify({'error': 'NO_AUDIO_TRACK',
                                     'message': 'This file contains no audio track. Voice analysis cannot proceed.'}), 400
                raise

            # Decode the audio once; Signals 2 and 3, the waveform and playback all reuse it.
            waveform = compute_waveform_peaks(audio_array)
            audio_url = encode_playable_audio(audio_array)
            clone_score, clone_verdict = get_voice_clone_scorer().score_array(audio_array)
            manip_result = get_voice_manipulation_scorer().analyze_array(audio_array)
            manip_transcript = manip_result['transcript']
            manip_score = manip_result['manipulation_score']
            manip_verdict = manip_result['verdict']
            manip_emotions = manip_result['emotions']

        clone_explanations = {
            "Cloned": "Acoustic artifacts consistent with AI voice synthesis/cloning were detected - unnatural spectral patterns and prosody atypical of genuine human speech.",
            "Real": "No signs of AI voice synthesis were detected. Spectral and prosodic characteristics are typical of genuine human speech.",
        }

        inference_time = time.time() - start_time
        remove_upload_with_retry(filepath)

        # Top 3 emotions, sorted, for a compact UI breakdown
        top_emotions = sorted(manip_emotions.items(), key=lambda kv: kv[1], reverse=True)[:3]

        result = {
            'verdict': clone_verdict,
            'score': round(clone_score * 100),
            'inference_time': round(inference_time, 2),
            'explanation': clone_explanations.get(clone_verdict, clone_explanations["Cloned"]),
            'filename': filename,
            'transcript': manip_transcript,
            'manipulation_verdict': manip_verdict,
            'manipulation_score': round(manip_score * 100),
            'emotions': [{'label': label, 'score': round(score * 100)} for label, score in top_emotions],
            'waveform': waveform,
            'audio_url': audio_url,
        }

        # audio_url is excluded; the session is cookie-based and cannot hold a multi-MB clip.
        session['signal2_result'] = {k: v for k, v in result.items() if k != 'audio_url'}
        log_usage('Voice Clone', clone_verdict)
        log_usage('Voice Manipulation', manip_verdict)

        return jsonify(result)

    except Exception as e:
        if os.path.exists(filepath):
            remove_upload_with_retry(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/caption')
def caption():
    return render_template('caption.html', active_page='caption')

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    """Social media pipeline: downloads a pasted URL and runs Signals 1, 2, 3 and 4 in one call."""
    data = request.get_json(silent=True) or request.form
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    from modules.social_media.url_extractor import download_video, InvalidURLError, UnsupportedPlatformError

    start_time = time.time()
    video_path = None
    try:
        try:
            extracted = download_video(url, app.config['UPLOAD_FOLDER'])
        except InvalidURLError as e:
            return jsonify({'error': 'INVALID_URL', 'message': str(e)}), 400
        except UnsupportedPlatformError as e:
            return jsonify({'error': 'UNSUPPORTED_PLATFORM', 'message': str(e)}), 400

        video_path = extracted['video_path']

        if MOCK_MODE:
            time.sleep(1)
            video_result = {'verdict': 'Real', 'score': 12}
            clone_verdict, clone_score = 'Real', 3
            manip_verdict, manip_score, transcript = 'Non-manipulative', 2, 'Sample transcript.'
        else:
            # Signal 1 (video)
            frames = extract_frames(video_path, num_frames=15)
            video_scorer = get_scorer()
            avg_score, _ = video_scorer.score_video_frames(frames)
            video_verdict = video_scorer.get_verdict(avg_score)
            video_result = {'verdict': video_verdict, 'score': round(avg_score * 100)}

            # Signal 2 + 3 (voice clone + manipulation) - decode audio once,
            # score twice, same pattern as /analyze_voice.
            from modules.voice_detection.audio_utils import extract_audio_array
            try:
                audio_array = extract_audio_array(video_path)
                raw_clone_score, clone_verdict = get_voice_clone_scorer().score_array(audio_array)
                clone_score = round(raw_clone_score * 100)
                manip_result = get_voice_manipulation_scorer().analyze_array(audio_array)
                manip_verdict = manip_result['verdict']
                manip_score = round(manip_result['manipulation_score'] * 100)
                transcript = manip_result['transcript']
            except RuntimeError as e:
                # No audio track: Signals 2/3 are skipped, Signal 1 and the caption still stand.
                if str(e) == 'NO_AUDIO_TRACK':
                    clone_verdict, clone_score = None, None
                    manip_verdict, manip_score, transcript = None, None, ''
                else:
                    raise

        if MOCK_MODE:
            coherence_score, coherence_verdict = 0.81, "Coherent"
        else:
            coherence_score, coherence_verdict = get_coherence_scorer().score(transcript, extracted['caption'])

        inference_time = time.time() - start_time

        result = {
            'platform': extracted['platform'],
            'title': extracted['title'],
            'caption': extracted['caption'],
            'video_verdict': video_result['verdict'],
            'video_score': video_result['score'],
            'voice_clone_verdict': clone_verdict,
            'voice_clone_score': clone_score,
            'manipulation_verdict': manip_verdict,
            'manipulation_score': manip_score,
            'transcript': transcript,
            'inference_time': round(inference_time, 2),
            'coherence_implemented': True,
            'coherence_verdict': coherence_verdict,
            'coherence_score': round(coherence_score * 100) if coherence_score is not None else None,
        }
        if coherence_verdict is None:
            result['coherence_message'] = 'No transcript or caption available for this clip, so coherence cannot be scored (e.g. a no-audio-track video, or an empty caption).'

        session['social_media_result'] = result
        if coherence_verdict is not None:
            session['signal4_result'] = {
                'verdict': coherence_verdict, 'score': result['coherence_score'],
                'inference_time': result['inference_time'], 'title': extracted['title'],
            }

        # Log each signal's own row plus one for the pipeline itself.
        log_usage('Video', video_result['verdict'])
        log_usage('Voice Clone', clone_verdict)
        log_usage('Voice Manipulation', manip_verdict)
        log_usage('Caption Check', coherence_verdict)
        log_usage('Social Media Pipeline', coherence_verdict or video_result['verdict'])
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if video_path and os.path.exists(video_path):
            remove_upload_with_retry(video_path)

# Duplicate video route removed – original definition retained earlier


@app.route('/analyze_video_url', methods=['POST'])
def analyze_video_url():
    """URL-based Signal 1 only. Keeps the downloaded file so /analyze_voice_chain
    can reuse it without downloading again.
    """
    data = request.get_json(silent=True) or request.form
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    from modules.social_media.url_extractor import download_video, InvalidURLError, UnsupportedPlatformError

    start_time = time.time()
    try:
        extracted = download_video(url, app.config['UPLOAD_FOLDER'])
    except InvalidURLError as e:
        return jsonify({'error': 'INVALID_URL', 'message': str(e)}), 400
    except UnsupportedPlatformError as e:
        return jsonify({'error': 'UNSUPPORTED_PLATFORM', 'message': str(e)}), 400

    video_path = extracted['video_path']

    try:
        if MOCK_MODE:
            time.sleep(1)
            avg_score, verdict = 0.12, 'Real'
            frame_scores = [0.1, 0.15, 0.1, 0.12, 0.08, 0.14, 0.11, 0.09, 0.13, 0.1]
            explanation = 'No signs of deepfake manipulation were detected across the analyzed frames.'
        else:
            frames = extract_frames(video_path, num_frames=15)
            video_scorer = get_scorer()
            avg_score, frame_scores = video_scorer.score_video_frames(frames)
            verdict = video_scorer.get_verdict(avg_score)
            if verdict == "Real":
                explanation = "No signs of deepfake manipulation were detected across the analyzed frames."
            else:
                explanation = "High probability of AI-generated face manipulation detected across multiple frames."

        inference_time = time.time() - start_time

        # Keep the file for a possible "Audio Check" chain step - do NOT
        # delete it here (the key difference from /analyze_url).
        _clear_pending_chain_video()
        session['chainable_video_path'] = video_path
        session['chainable_video_meta'] = {
            'mode': 'url', 'url': url, 'platform': extracted['platform'],
            'title': extracted['title'], 'caption': extracted['caption'],
        }

        result = {
            'verdict': verdict,
            'score': round(avg_score * 100),
            'frame_scores': frame_scores,
            'inference_time': round(inference_time, 2),
            'explanation': explanation,
            'filename': extracted['title'],
            'platform': extracted['platform'],
            'title': extracted['title'],
            'caption': extracted['caption'],
        }

        session['signal1_result'] = {
            'verdict': verdict, 'score': result['score'],
            'inference_time': result['inference_time'], 'explanation': explanation,
            'filename': extracted['title'],
        }
        log_usage('URL', verdict)

        return jsonify(result)

    except Exception as e:
        # Analysis itself failed (not the download) - nothing was stashed
        # for chaining, so clean the file up now rather than leaking it.
        if os.path.exists(video_path):
            remove_upload_with_retry(video_path)
        return jsonify({'error': str(e)}), 500


def _run_voice_signals(audio_path):
    """Shared Signal 2 + 3 logic: decode the audio once, score it twice."""
    from modules.voice_detection.audio_utils import extract_audio_array, compute_waveform_peaks, encode_playable_audio
    try:
        audio_array = extract_audio_array(audio_path)
    except RuntimeError as e:
        if str(e) == 'NO_AUDIO_TRACK':
            return None
        raise

    if MOCK_MODE:
        time.sleep(1)
        return {
            'clone_score': 0.94, 'clone_verdict': 'Cloned',
            'manip_verdict': 'Manipulative', 'manip_score': 0.35,
            'transcript': 'Your account has been frozen. Call us back immediately or you will lose all your funds today.',
            'manip_emotions': {'fear': 0.5, 'neutral': 0.2},
            'waveform': [0.1 + 0.8 * abs((i % 20) - 10) / 10 for i in range(100)],
            'audio_url': None,
        }

    waveform = compute_waveform_peaks(audio_array)
    audio_url = encode_playable_audio(audio_array)
    raw_clone_score, clone_verdict = get_voice_clone_scorer().score_array(audio_array)
    manip_result = get_voice_manipulation_scorer().analyze_array(audio_array)
    return {
        'clone_score': raw_clone_score, 'clone_verdict': clone_verdict,
        'manip_verdict': manip_result['verdict'],
        'manip_score': manip_result['manipulation_score'],
        'transcript': manip_result['transcript'],
        'manip_emotions': manip_result['emotions'],
        'waveform': waveform,
        'audio_url': audio_url,
    }


def _build_voice_response(start_time, voice_signals, filename, chain_meta=None, pipeline1=False):
    inference_time = time.time() - start_time
    if voice_signals is None:
        result = {
            'error': 'NO_AUDIO_TRACK',
            'message': 'This file contains no audio track. Voice analysis cannot proceed.',
        }
        return result, True

    top_emotions = sorted(voice_signals['manip_emotions'].items(), key=lambda kv: kv[1], reverse=True)[:3]
    clone_explanations = {
        "Cloned": "Acoustic artifacts consistent with AI voice synthesis/cloning were detected - unnatural spectral patterns and prosody atypical of genuine human speech.",
        "Real": "No signs of AI voice synthesis were detected. Spectral and prosodic characteristics are typical of genuine human speech.",
    }
    result = {
        'verdict': voice_signals['clone_verdict'],
        'score': round(voice_signals['clone_score'] * 100),
        'inference_time': round(inference_time, 2),
        'explanation': clone_explanations.get(voice_signals['clone_verdict'], clone_explanations["Cloned"]),
        'filename': filename,
        'transcript': voice_signals['transcript'],
        'manipulation_verdict': voice_signals['manip_verdict'],
        'manipulation_score': round(voice_signals['manip_score'] * 100),
        'emotions': [{'label': label, 'score': round(score * 100)} for label, score in top_emotions],
        'waveform': voice_signals['waveform'],
        'audio_url': voice_signals['audio_url'],
    }
    if chain_meta:
        result['chain_meta'] = chain_meta

    # audio_url excluded: too large for a cookie-based session.
    session['signal2_result'] = {k: v for k, v in result.items() if k not in ('chain_meta', 'audio_url')}
    log_usage('Voice Clone', voice_signals['clone_verdict'])
    log_usage('Voice Manipulation', voice_signals['manip_verdict'])
    if pipeline1:
        log_usage('Pipeline 1: Video -> Audio', voice_signals['clone_verdict'])
    return result, False


@app.route('/analyze_voice_chain', methods=['POST'])
def analyze_voice_chain():
    """Runs Signals 2 and 3 on the video stashed by /analyze or /analyze_video_url."""
    video_path = session.get('chainable_video_path')
    meta = session.get('chainable_video_meta') or {}
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'NO_PENDING_VIDEO',
                         'message': 'No pending video found - analyze a video first, then use Audio Check.'}), 400

    start_time = time.time()
    try:
        voice_signals = _run_voice_signals(video_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        remove_upload_with_retry(video_path)
        session.pop('chainable_video_path', None)
        session.pop('chainable_video_meta', None)

    filename = meta.get('title') or meta.get('filename') or 'video'
    chain_meta = {
        'mode': meta.get('mode'), 'url': meta.get('url'),
        'platform': meta.get('platform'), 'title': meta.get('title'),
        'caption': meta.get('caption'),
    }
    result, is_error = _build_voice_response(start_time, voice_signals, filename, chain_meta, pipeline1=True)
    return jsonify(result), (400 if is_error else 200)


@app.route('/analyze_voice_url', methods=['POST'])
def analyze_voice_url():
    """Standalone URL-based Signals 2 and 3 for the voice page. Deletes the file afterwards."""
    data = request.get_json(silent=True) or request.form
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    from modules.social_media.url_extractor import download_video, InvalidURLError, UnsupportedPlatformError

    try:
        extracted = download_video(url, app.config['UPLOAD_FOLDER'])
    except InvalidURLError as e:
        return jsonify({'error': 'INVALID_URL', 'message': str(e)}), 400
    except UnsupportedPlatformError as e:
        return jsonify({'error': 'UNSUPPORTED_PLATFORM', 'message': str(e)}), 400

    video_path = extracted['video_path']
    start_time = time.time()
    try:
        voice_signals = _run_voice_signals(video_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        remove_upload_with_retry(video_path)

    chain_meta = {
        'mode': 'url', 'url': url, 'platform': extracted['platform'],
        'title': extracted['title'], 'caption': extracted['caption'],
    }
    result, is_error = _build_voice_response(start_time, voice_signals, extracted['title'], chain_meta)
    return jsonify(result), (400 if is_error else 200)


@app.route('/score_coherence', methods=['POST'])
def score_coherence():
    """Signal 4 for the caption page, scoring a transcript/caption pair already held client-side."""
    data = request.get_json(silent=True) or request.form
    transcript = data.get('transcript') or ''
    caption = data.get('caption') or ''

    if MOCK_MODE:
        score, verdict = 0.81, "Coherent"
    else:
        score, verdict = get_coherence_scorer().score(transcript, caption)

    if verdict is None:
        return jsonify({
            'coherence_implemented': True, 'coherence_verdict': None, 'coherence_score': None,
            'coherence_message': 'No transcript or caption available for this clip, so coherence cannot be scored.',
        })

    result = {
        'coherence_implemented': True,
        'coherence_verdict': verdict,
        'coherence_score': round(score * 100),
    }
    session['signal4_result'] = {
        'verdict': verdict, 'score': result['coherence_score'],
    }
    # Completion point of the Video->Voice->Caption journey, so it logs to the
    # same Social Media Pipeline row as /analyze_url.
    log_usage('Caption Check', verdict)
    log_usage('Social Media Pipeline', verdict)
    return jsonify(result)


@app.route('/image')
def image():
    return render_template('image.html', active_page='image')

# Alias for dashboard (the verdict page)
@app.route('/dashboard')
def dashboard():
    return render_template(
        'verdict.html', active_page='dashboard',
        signal1=session.get('signal1_result'),
        signal1b=session.get('signal1b_result'),
        signal2=session.get('signal2_result'),
        signal4=session.get('signal4_result'),
        usage_stats=get_usage_stats()
    )

@app.route('/verdict')
def verdict():
    return render_template(
        'verdict.html', active_page='verdict',
        signal1=session.get('signal1_result'),
        signal1b=session.get('signal1b_result'),
        signal2=session.get('signal2_result'),
        signal4=session.get('signal4_result'),
        usage_stats=get_usage_stats()
    )

@app.route('/download_report')
def download_report():
    """Generates a PDF of whichever signals are currently in the session."""
    from modules.report_generator import generate_report_pdf
    pdf_buf = generate_report_pdf(
        signal1=session.get('signal1_result'),
        signal1b=session.get('signal1b_result'),
        signal2=session.get('signal2_result'),
        signal4=session.get('signal4_result'),
    )
    filename = f"deepguard_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(pdf_buf, mimetype='application/pdf', as_attachment=True, download_name=filename)

def _clear_all_signals():
    """Pops every signal result and any pending chain video from the session.
    Does not touch usage_log.db.
    """
    _clear_pending_chain_video()
    for key in ('signal1_result', 'signal1b_result', 'signal2_result',
                'signal4_result', 'social_media_result'):
        session.pop(key, None)


@app.route('/reset_dashboard')
def reset_dashboard():
    """Clears every signal at once for the dashboard's "Analyse Another" button."""
    _clear_all_signals()
    return redirect(url_for('video'))


@app.route('/reset_verdict_signals', methods=['POST'])
def reset_verdict_signals():
    """Clears the session signals on a genuine browser refresh of the dashboard.
    Returns JSON and stays on the page rather than redirecting.
    """
    _clear_all_signals()
    return jsonify({'ok': True})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
