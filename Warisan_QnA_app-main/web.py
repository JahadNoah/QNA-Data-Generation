# QnA Pair Generator Web App (Flask)
# Browser-based GUI for uploading text files and generating CSV output

from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context, session, redirect, url_for
import os
import csv
import json
import tempfile
import threading
import queue
import time
import uuid
import re
import traceback
import core
import functools
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # 256MB for batch uploads
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.secret_key = 'qna-generator-secret-key-2024'

# --- Hardcoded Credentials ---
LOGIN_EMAIL    = "a200363@siswa.ukm.edu.my"
LOGIN_PASSWORD = "#LLMUKM@aiBM"


def parse_max_pairs(raw_value):
    """Parse optional max_pairs from form input.

    Accepts empty/0/auto-like values as adaptive mode (None).
    """
    cleaned = (raw_value or "").strip()
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in {"0", "auto", "adaptive", "none", "null"}:
        return None

    try:
        value = int(cleaned)
    except ValueError as exc:
        raise ValueError("Nilai had pasangan mesti nombor bulat (contoh: 50) atau kosong/Auto.") from exc

    if value <= 0:
        return None
    return value

# --- Auth decorator ---
def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# --- Login / Logout Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        if email == LOGIN_EMAIL and password == LOGIN_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            error = 'E-mel atau kata laluan tidak sah.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Per-request progress queues and cancellation flags
_batch_queues = {}      # batch_id -> queue.Queue
_batch_cancel = {}      # batch_id -> bool
_batch_lock = threading.Lock()


def _create_batch(batch_id):
    with _batch_lock:
        _batch_queues[batch_id] = queue.Queue()
        _batch_cancel[batch_id] = False


def _cleanup_batch(batch_id):
    with _batch_lock:
        _batch_queues.pop(batch_id, None)
        _batch_cancel.pop(batch_id, None)


def _is_cancelled(batch_id):
    with _batch_lock:
        return _batch_cancel.get(batch_id, False)


@app.route('/api/extract', methods=['POST'])
@login_required
def extract_clean_text():
    """Run prefilter to extract CLEAN_TEXT blocks for preview (TITLE/ABSTRACT/BODY)."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not file.filename.endswith('.txt'):
            return jsonify({'error': 'Only .txt files are supported'}), 400

        full_text = file.read().decode('utf-8', errors='replace')
        src_name = secure_filename(file.filename)

        # Fallback: Check for wrapper tags
        import re
        # Try regular opening/closing tags first
        content_match = re.search(r'<Content>(.*?)</Content>', full_text, flags=re.DOTALL | re.IGNORECASE)
        
        # If not found, try self-closing tag format
        if not content_match:
            content_match = re.search(r'<Content>(.*?)<Content\s*/>', full_text, flags=re.DOTALL | re.IGNORECASE)
        
        if content_match:
            print(f"[DEBUG] Wrapper tags FOUND in {src_name}")
            # Use Content wrapper as BODY_BLOCK
            wrapped_content = content_match.group(1).strip()
            
            # Try to extract Title from <Title> wrapper
            title = ""
            title_wrapper = re.search(r'<Title>(.*?)</Title>', full_text, flags=re.DOTALL | re.IGNORECASE)
            if not title_wrapper:
                title_wrapper = re.search(r'<Title>(.*?)<Title\s*/>', full_text, flags=re.DOTALL | re.IGNORECASE)
            if title_wrapper:
                title = title_wrapper.group(1).strip()
            else:
                # Fallback to regex search for title
                title_match = re.search(r'(?:Tajuk|TITLE)\s*:\s*(.*?)(?:\n|$)', full_text, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()
            
            # Try to extract Abstract from <Abstract> wrapper
            abstract = ""
            abstract_wrapper = re.search(r'<Abstract>(.*?)</Abstract>', full_text, flags=re.DOTALL | re.IGNORECASE)
            if not abstract_wrapper:
                abstract_wrapper = re.search(r'<Abstract>(.*?)<Abstract\s*/>', full_text, flags=re.DOTALL | re.IGNORECASE)
            if abstract_wrapper:
                abstract = abstract_wrapper.group(1).strip()
            else:
                # Fallback to regex search for abstract
                abstract_match = re.search(r'(?:Abstrak|ABSTRACT)\s*:\s*(.*?)(?:\n|$)', full_text, re.IGNORECASE)
                if abstract_match:
                    abstract = abstract_match.group(1).strip()
            
            # Extract source if available
            source = ""
            source_match = re.search(r'(?:Sumber|SOURCE)\s*:\s*(.*?)(?:\n|$)', full_text, re.IGNORECASE)
            if source_match:
                source = source_match.group(1).strip()
            
            body = wrapped_content
        else:
            print(f"[DEBUG] <Content> wrapper NOT found in {src_name}, using AI extraction on first 2000 words")
            # Only send the first 2000 words to stay within the model's token limit.
            # Title/abstract are always near the start of a thesis, so this is sufficient
            # for metadata extraction. The full text is used as body for generation.
            words = full_text.split()
            excerpt = " ".join(words[:2000])
            user_prompt = f"FULL TEXT:\n{excerpt}\n\nReturn CLEAN_TEXT blocks as specified."
            raw = core.chat(core.MODEL_GEN, core.PREFILTER_SYSTEM, user_prompt, temperature=0.0)
            # Simple parse of blocks
            title = ""; abstract = ""; source = ""; body = ""
            def extract_block(label: str, text: str) -> str:
                m = re.search(rf"{label}:\s*(.*?)(?:\n\s*\n[A-Z_ ]+:|\Z)", text, flags=re.DOTALL)
                return (m.group(1).strip() if m else "")
            if raw:
                title = extract_block("TITLE", raw)
                abstract = extract_block("ABSTRACT_BLOCK", raw)
                source = extract_block("SOURCE", raw)
            # The AI only looked at the top 2000 words, so whatever BODY_BLOCK
            # it returns is truncated. Always use the full file text as the body!
            body = full_text

        return jsonify({
            'source_name': src_name,
            'title': title,
            'abstract': abstract,
            'source': source,
            'body': body
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/preview-chunks', methods=['POST'])
@login_required
def preview_chunks():
    """Preview chunks from extracted clean text (for display only)"""
    try:
        # Get extracted data from the frontend
        abstract = request.form.get('abstract', '')
        body = request.form.get('body', '')
        
        # Combine for chunking
        combined_text = f"{abstract}\n\n{body}".strip()
        
        # Use core's chunking function
        chunks = core.chunk_words(combined_text, core.CHUNK_WORDS, core.CHUNK_OVERLAP)
        
        # Format for display
        chunks_display = []
        for idx, (chunk_text, start_word, end_word) in enumerate(chunks, 1):
            # Preview first 200 characters of each chunk
            preview = chunk_text[:200] + '...' if len(chunk_text) > 200 else chunk_text
            chunks_display.append({
                'index': idx,
                'preview': preview,
                'full_text': chunk_text,
                'word_range': f"{start_word}-{end_word}",
                'word_count': len(chunk_text.split())
            })
        
        return jsonify({
            'total_chunks': len(chunks),
            'chunks': chunks_display
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
@login_required
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
@login_required
def generate_qa():
    """Process uploaded file and generate Q&A pairs with live progress"""
    try:
        # Accept file or CLEAN_TEXT blocks
        file = request.files.get('file')
        title_field = request.form.get('title')
        abstract_field = request.form.get('abstract')
        source_field = request.form.get('source')
        body_field = request.form.get('body')
        if not file and not (title_field or abstract_field or body_field):
            return jsonify({'error': 'No input provided'}), 400

        # Get all settings from request (capture before background thread)
        max_pairs_raw = request.form.get('max_pairs', '')
        try:
            max_pairs = parse_max_pairs(max_pairs_raw)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        skip_review = request.form.get('skip_review', 'false').lower() == 'true'

        # Read content
        if title_field is not None or abstract_field is not None or body_field is not None:
            body = body_field or ''
            abstract = abstract_field or ''
            source = source_field or ''
            file_content = (abstract + "\n\n" + body).strip()
            source_name = secure_filename(request.form.get('source_name') or 'uploaded.txt')
            original_filename = source_name
            doc_title = (title_field or '').strip() or source_name
        else:
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            if not file.filename.endswith('.txt'):
                return jsonify({'error': 'Only .txt files are supported'}), 400
            file_content = file.read().decode('utf-8', errors='replace')
            source_name = secure_filename(file.filename)
            original_filename = file.filename
            doc_title = source_name
            abstract = ''
            source = ''

        # Create per-request queue
        batch_id = str(uuid.uuid4())
        _create_batch(batch_id)
        q = _batch_queues[batch_id]

        def generate_with_progress():
            pairs = []

            def progress_callback(message):
                q.put({'type': 'progress', 'message': message})

            try:
                pairs = core.process_text_file(
                    file_content,
                    source_name,
                    max_pairs=max_pairs,
                    progress_callback=progress_callback,
                    skip_review=skip_review,
                    max_workers=2,
                    doc_title=doc_title
                )

                q.put({
                    'type': 'complete',
                    'pairs': pairs,
                    'count': len(pairs),
                    'original_filename': original_filename,
                    'file_size': len(file_content),
                    'word_count': len(file_content.split()),
                    'abstract': abstract,
                    'source': source,
                    'source_name': source_name
                })
            except ValueError as e:
                error_msg = str(e)
                q.put({
                    'type': 'error',
                    'error': error_msg,
                    'is_rate_limit': 'rate limit' in error_msg.lower(),
                    'is_auth_error': 'invalid api key' in error_msg.lower()
                })
            except Exception as e:
                error_details = traceback.format_exc()
                print(f"Error in generation: {error_details}")
                q.put({
                    'type': 'error',
                    'error': f"{str(e)}\n\nCheck server logs for details."
                })

        thread = threading.Thread(target=generate_with_progress)
        thread.daemon = True
        thread.start()

        def event_stream():
            try:
                while True:
                    try:
                        data = q.get(timeout=1)
                        yield f"data: {json.dumps(data)}\n\n"
                        if data['type'] in ('complete', 'error'):
                            break
                    except queue.Empty:
                        yield ": heartbeat\n\n"
                        continue
            finally:
                _cleanup_batch(batch_id)

        return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-batch', methods=['POST'])
@login_required
def generate_batch():
    """Process multiple uploaded files and generate Q&A pairs with live progress"""
    try:
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'Tiada fail dipilih'}), 400

        max_pairs_raw = request.form.get('max_pairs', '')
        try:
            max_pairs = parse_max_pairs(max_pairs_raw)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        skip_review = request.form.get('skip_review', 'false').lower() == 'true'

        # Read all files upfront (before background thread)
        files_data = []
        for f in files:
            if f.filename and f.filename.endswith('.txt'):
                content = f.read().decode('utf-8', errors='replace')
                filename = secure_filename(f.filename)
                files_data.append((content, filename, f.filename))

        if not files_data:
            return jsonify({'error': 'Tiada fail .txt yang sah'}), 400

        batch_id = str(uuid.uuid4())
        _create_batch(batch_id)
        q = _batch_queues[batch_id]

        def batch_worker():
            all_pairs = []
            total = len(files_data)

            for i, (content, safe_name, original_name) in enumerate(files_data, 1):
                if _is_cancelled(batch_id):
                    q.put({
                        'type': 'batch_cancelled',
                        'pairs': all_pairs,
                        'count': len(all_pairs)
                    })
                    return

                q.put({
                    'type': 'file_start',
                    'file_index': i,
                    'total_files': total,
                    'filename': original_name
                })

                try:
                    # Extract metadata using the same logic as /api/extract
                    title, abstract, source, body = _extract_metadata(content, safe_name)
                    file_content = (abstract + "\n\n" + body).strip() if abstract else body
                    doc_title = title or safe_name

                    def progress_cb(message, _i=i, _total=total, _name=original_name):
                        if _is_cancelled(batch_id):
                            raise InterruptedError("Batch cancelled")
                        q.put({
                            'type': 'progress',
                            'message': message,
                            'batch_progress': f'Fail {_i}/{_total}: {_name}',
                            'file_index': _i,
                            'total_files': _total
                        })

                    pairs = core.process_text_file(
                        file_content,
                        safe_name,
                        max_pairs=max_pairs,
                        progress_callback=progress_cb,
                        skip_review=skip_review,
                        max_workers=2,
                        doc_title=doc_title
                    )

                    # Tag each pair with source file info
                    for p in pairs:
                        p['source_file'] = original_name

                    all_pairs.extend(pairs)

                    q.put({
                        'type': 'file_complete',
                        'file_index': i,
                        'total_files': total,
                        'filename': original_name,
                        'file_pairs_count': len(pairs),
                        'total_pairs_count': len(all_pairs),
                        'pairs': pairs,
                        'title': doc_title,
                        'abstract': abstract,
                        'source': source
                    })

                except InterruptedError:
                    q.put({
                        'type': 'batch_cancelled',
                        'pairs': all_pairs,
                        'count': len(all_pairs)
                    })
                    return
                except Exception as e:
                    print(f"Error processing {original_name}: {traceback.format_exc()}")
                    q.put({
                        'type': 'file_error',
                        'file_index': i,
                        'total_files': total,
                        'filename': original_name,
                        'error': str(e)
                    })
                    # Continue with next file

            q.put({
                'type': 'batch_complete',
                'pairs': all_pairs,
                'count': len(all_pairs)
            })

        thread = threading.Thread(target=batch_worker)
        thread.daemon = True
        thread.start()

        def event_stream():
            try:
                # Send batch_id so frontend can cancel
                yield f"data: {json.dumps({'type': 'batch_id', 'batch_id': batch_id})}\n\n"
                while True:
                    try:
                        data = q.get(timeout=1)
                        yield f"data: {json.dumps(data)}\n\n"
                        if data['type'] in ('batch_complete', 'batch_cancelled'):
                            break
                    except queue.Empty:
                        yield ": heartbeat\n\n"
                        continue
            finally:
                _cleanup_batch(batch_id)

        return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/cancel-batch', methods=['POST'])
@login_required
def cancel_batch():
    """Cancel an in-progress batch generation"""
    data = request.json or {}
    batch_id = data.get('batch_id')
    if batch_id:
        with _batch_lock:
            if batch_id in _batch_cancel:
                _batch_cancel[batch_id] = True
                return jsonify({'cancelled': True})
    return jsonify({'cancelled': False, 'error': 'ID kelompok tidak sah'}), 400


def _extract_metadata(full_text, src_name):
    """Extract title, abstract, source, body from text (shared logic)."""
    content_match = re.search(r'<Content>(.*?)</Content>', full_text, flags=re.DOTALL | re.IGNORECASE)
    if not content_match:
        content_match = re.search(r'<Content>(.*?)<Content\s*/>', full_text, flags=re.DOTALL | re.IGNORECASE)

    if content_match:
        body = content_match.group(1).strip()
        title = ""
        tw = re.search(r'<Title>(.*?)</Title>', full_text, flags=re.DOTALL | re.IGNORECASE)
        if not tw:
            tw = re.search(r'<Title>(.*?)<Title\s*/>', full_text, flags=re.DOTALL | re.IGNORECASE)
        if tw:
            title = tw.group(1).strip()

        abstract = ""
        aw = re.search(r'<Abstract>(.*?)</Abstract>', full_text, flags=re.DOTALL | re.IGNORECASE)
        if not aw:
            aw = re.search(r'<Abstract>(.*?)<Abstract\s*/>', full_text, flags=re.DOTALL | re.IGNORECASE)
        if aw:
            abstract = aw.group(1).strip()

        source = ""
        sm = re.search(r'(?:Sumber|SOURCE)\s*:\s*(.*?)(?:\n|$)', full_text, re.IGNORECASE)
        if sm:
            source = sm.group(1).strip()

        return title, abstract, source, body
    else:
        # AI extraction fallback using first 2000 words
        words = full_text.split()
        excerpt = " ".join(words[:2000])
        user_prompt = f"FULL TEXT:\n{excerpt}\n\nReturn CLEAN_TEXT blocks as specified."
        raw = core.chat(core.MODEL_GEN, core.PREFILTER_SYSTEM, user_prompt, temperature=0.0)
        title = ""; abstract = ""; source = ""
        if raw:
            def extract_block(label, text):
                m = re.search(rf"{label}:\s*(.*?)(?:\n\s*\n[A-Z_ ]+:|\Z)", text, flags=re.DOTALL)
                return m.group(1).strip() if m else ""
            title = extract_block("TITLE", raw)
            abstract = extract_block("ABSTRACT_BLOCK", raw)
            source = extract_block("SOURCE", raw)
        return title, abstract, source, full_text


@app.route('/api/download-csv', methods=['POST'])
@login_required
def download_csv():
    """Generate and download CSV file"""
    try:
        data = request.json
        pairs = data.get('pairs', [])
        original_filename = data.get('original_filename', 'qa_bm_pairs')
        title = (data.get('title') or '').strip()
        domain = data.get('domain', 'Sejarah').strip()
        abstract = (data.get('abstract') or '').strip()
        source = (data.get('source') or '').strip()
        source_name = (data.get('source_name') or original_filename).strip()
        
        if not pairs:
            return jsonify({'error': 'No data to export'}), 400
        
        # Generate CSV filename based on extracted title if present
        def slugify(s: str) -> str:
            import re
            s = re.sub(r'[^\w\-\s]', '', s)
            s = re.sub(r'\s+', '_', s).strip('_')
            return s or 'qa_bm_pairs'
        base_name = slugify(title) if title else (original_filename.replace('.txt','') if original_filename.endswith('.txt') else original_filename)
        csv_filename = f"{base_name}.csv"
        
        # Create temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='', encoding='utf-8')
        
        writer = csv.writer(temp_file)
        # Check if batch mode (pairs have source_file field)
        is_batch = any(pair.get('source_file') for pair in pairs)
        if is_batch:
            writer.writerow(['Fail_Asal', 'Soalan', 'Jawapan', 'Abstract', 'Domain', 'Sumber', 'Potongan_teks'])
        else:
            writer.writerow(['Soalan', 'Jawapan', 'Abstract', 'Domain', 'Sumber', 'Potongan_teks'])
        # Write data
        for pair in pairs:
            sumber_value = source if source else source_name
            chunk_text = pair.get('chunk_text', '')
            if is_batch:
                writer.writerow([
                    pair.get('source_file', ''),
                    pair.get('question', ''),
                    pair.get('answer', ''),
                    abstract,
                    domain,
                    sumber_value,
                    chunk_text
                ])
            else:
                writer.writerow([
                    pair.get('question', ''),
                    pair.get('answer', ''),
                    abstract,
                    domain,
                    sumber_value,
                    chunk_text
                ])
        
        temp_file.close()
        
        return send_file(
            temp_file.name,
            mimetype='text/csv',
            as_attachment=True,
            download_name=csv_filename
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
@login_required
def health_check():
    """Check if API is configured"""
    has_config = bool(core.API_KEY and core.BASE_URL)
    return jsonify({
        'configured': has_config,
        'model_gen': core.MODEL_GEN,
        'model_review': core.MODEL_REVIEW
    })

@app.route('/api/verify-connection', methods=['GET'])
@login_required
def verify_connection():
    """Verify AI API connection by making a test call"""
    try:
        if not core.API_KEY or not core.BASE_URL:
            return jsonify({
                'connected': False,
                'error': 'API credentials not configured'
            })
        
        # Make a simple test call
        test_response = core.chat(
            core.MODEL_GEN,
            "You are a helpful assistant.",
            "Say 'OK' if you can read this.",
            temperature=0.1
        )
        
        if test_response and len(test_response) > 0:
            return jsonify({
                'connected': True,
                'model': core.MODEL_GEN,
                'message': 'Successfully connected to AI API'
            })
        else:
            return jsonify({
                'connected': False,
                'error': 'No response from API'
            })
    except ValueError as e:
        # Handle specific API errors
        error_msg = str(e)
        return jsonify({
            'connected': False,
            'error': error_msg,
            'is_rate_limit': 'rate limit' in error_msg.lower(),
            'is_auth_error': 'invalid api key' in error_msg.lower()
        }), 200
    except Exception as e:
        return jsonify({
            'connected': False,
            'error': str(e)
        }), 200

if __name__ == '__main__':
    # Check if API is configured
    if not core.API_KEY or not core.BASE_URL:
        print("\n" + "="*60)
        print("WARNING: API credentials not configured!")
        print("Please set OPENAI_API_KEY and OPENAI_BASE_URL in your .env file")
        print("="*60 + "\n")
    
    # Try port 8080 first, fallback to 5001 if needed
    port = 8080
    print("Starting QnA Pair Generator Web App...")
    print(f"Open your browser and navigate to: http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
