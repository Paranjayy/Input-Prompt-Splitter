from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import random
import redis
import string
import math
import re

try:
    import tiktoken
except ImportError:  # pragma: no cover - handled by requirements
    tiktoken = None

# Load environment variables from .env file
load_dotenv()

app = Flask(
    __name__,
    template_folder='templates',
    static_folder=os.path.join(os.path.dirname(__file__), '../static')
)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Set up Redis client (optional)
upstash_redis_url = os.environ.get('UPSTASH_REDIS_URL')
redis_client = None
if upstash_redis_url:
    try:
        redis_client = redis.from_url(upstash_redis_url)
    except Exception:
        redis_client = None

@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    split_length = ""
    file_data = []
    mode = request.form.get("mode", "chars") if request.method == "POST" else "chars"
    encoding_name = request.form.get("encoding", "cl100k_base") if request.method == "POST" else "cl100k_base"

    # Visit counter (safe without Redis)
    try:
        if redis_client is not None:
            redis_client.incr("visit_counter")
            visit_count = int(redis_client.get("visit_counter"))
        else:
            visit_count = 0
    except Exception:
        visit_count = 0

    if request.method == "POST":
        prompt = request.form["prompt"]
        split_length = int(request.form["split_length"])

        if mode == "tokens":
            file_data = split_prompt_by_tokens(prompt, split_length, encoding_name)
        else:
            file_data = split_prompt(prompt, split_length)

    hash_value = generate_random_hash(8)

    return render_template(
        "index.html",
        prompt=prompt,
        split_length=split_length,
        file_data=file_data,
        hash=hash_value,
        visit_count=visit_count,
        mode=mode,
        encoding_name=encoding_name,
    )
            
def split_prompt(text, split_length, overlap=0):
    if split_length <= 0:
        raise ValueError("Max length must be greater than 0.")
    if overlap >= split_length:
        raise ValueError("Overlap must be less than max length.")

    if not text:
        return []

    # Calculate effective chunk size (accounting for overlap)
    effective_size = split_length - overlap
    if effective_size <= 0:
        effective_size = split_length

    file_data = []
    start = 0
    n = len(text)
    i = 0

    while start < n:
        # For non-first chunks, back up by overlap amount
        if i > 0 and overlap > 0:
            overlap_start = max(0, start - overlap)
            chunk_start = overlap_start
        else:
            chunk_start = start

        end = min(chunk_start + split_length, n)
        chunk = text[chunk_start:end]

        if i == 0:
            content = f'[START PART {i + 1}]\n' + chunk + f'\n[END PART {i + 1}]'
        else:
            content = f'Do not answer yet. This is just another part of the text I want to send you. Just receive and acknowledge as "Part {i + 1} received" and wait for the next part.\n[START PART {i + 1}]\n' + chunk + f'\n[END PART {i + 1}]'
            content += f'\nRemember not answering yet. Just acknowledge you received this part with the message "Part {i + 1} received" and wait for the next part.'

        file_data.append({
            'name': f'split_{str(i + 1).zfill(3)}_of_{str(len(file_data) + 1).zfill(3)}.txt',
            'content': content
        })

        start += effective_size
        i += 1

        # If we've reached the end, break
        if start >= n:
            break

    # Update part numbers in content and filenames
    num_parts = len(file_data)
    for idx, item in enumerate(file_data):
        is_last = idx == num_parts - 1
        old_content = item['content']

        if is_last:
            new_content = old_content.replace(f'[START PART {idx + 1}]', f'[START PART {idx + 1}/{num_parts}]')
            new_content = new_content.replace(f'[END PART {idx + 1}]', f'[END PART {idx + 1}/{num_parts}]')
            new_content = new_content.replace('[START PART', f'[START PART {idx + 1}/{num_parts}]', 1)
            new_content = new_content.replace('[END PART', f'[END PART {idx + 1}/{num_parts}]', 1)
            if 'ALL PARTS SENT' not in new_content:
                new_content += '\nALL PARTS SENT. Now you can continue processing the request.'
        else:
            new_content = old_content.replace(f'[START PART {idx + 1}]', f'[START PART {idx + 1}/{num_parts}]')
            new_content = new_content.replace(f'[END PART {idx + 1}]', f'[END PART {idx + 1}/{num_parts}]')
            new_content = new_content.replace('[START PART', f'[START PART {idx + 1}/{num_parts}]', 1)
            new_content = new_content.replace('[END PART', f'[END PART {idx + 1}/{num_parts}]', 1)

        item['content'] = new_content
        item['name'] = f'split_{str(idx + 1).zfill(3)}_of_{str(num_parts).zfill(3)}.txt'

    return file_data


def _get_encoding(encoding_or_model_name: str):
    if tiktoken is None:
        raise RuntimeError("tiktoken is not installed. Please install it to use token-based features.")
    try:
        # Try model first
        return tiktoken.encoding_for_model(encoding_or_model_name)
    except Exception:
        try:
            return tiktoken.get_encoding(encoding_or_model_name)
        except Exception:
            return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoding_or_model_name: str = "cl100k_base") -> int:
    if not text:
        return 0
    enc = _get_encoding(encoding_or_model_name)
    return len(enc.encode(text))


def split_prompt_by_tokens(text: str, max_tokens: int, encoding_or_model_name: str = "cl100k_base", overlap: int = 0):
    if max_tokens <= 0:
        raise ValueError("Max tokens must be greater than 0.")
    if overlap >= max_tokens:
        raise ValueError("Overlap must be less than max tokens.")
    if not text:
        return []

    enc = _get_encoding(encoding_or_model_name)
    token_ids = enc.encode(text)
    total_tokens = len(token_ids)

    # Calculate effective chunk size (accounting for overlap)
    effective_size = max_tokens - overlap
    if effective_size <= 0:
        effective_size = max_tokens

    file_data = []
    start = 0
    i = 0

    while start < total_tokens:
        # For non-first chunks, back up by overlap amount
        if i > 0 and overlap > 0:
            overlap_start = max(0, start - overlap)
            chunk_start = overlap_start
        else:
            chunk_start = start

        end = min(chunk_start + max_tokens, total_tokens)
        chunk_ids = token_ids[chunk_start:end]
        chunk_text = enc.decode(chunk_ids)

        if i == 0:
            content = f'[START PART {i + 1}]\n' + chunk_text + f'\n[END PART {i + 1}]'
        else:
            content = f'Do not answer yet. This is just another part of the text I want to send you. Just receive and acknowledge as "Part {i + 1} received" and wait for the next part.\n[START PART {i + 1}]\n' + chunk_text + f'\n[END PART {i + 1}]'
            content += f'\nRemember not answering yet. Just acknowledge you received this part with the message "Part {i + 1} received" and wait for the next part.'

        file_data.append({
            'name': f'split_{str(i + 1).zfill(3)}_of_{str(len(file_data) + 1).zfill(3)}.txt',
            'content': content
        })

        start += effective_size
        i += 1

        # If we've reached the end, break
        if start >= total_tokens:
            break

    # Update part numbers in content and filenames
    num_parts = len(file_data)
    for idx, item in enumerate(file_data):
        is_last = idx == num_parts - 1
        old_content = item['content']

        if is_last:
            new_content = old_content.replace(f'[START PART {idx + 1}]', f'[START PART {idx + 1}/{num_parts}]')
            new_content = new_content.replace(f'[END PART {idx + 1}]', f'[END PART {idx + 1}/{num_parts}]')
            new_content = new_content.replace('[START PART', f'[START PART {idx + 1}/{num_parts}]', 1)
            new_content = new_content.replace('[END PART', f'[END PART {idx + 1}/{num_parts}]', 1)
            if 'ALL PARTS SENT' not in new_content:
                new_content += '\nALL PARTS SENT. Now you can continue processing the request.'
        else:
            new_content = old_content.replace(f'[START PART {idx + 1}]', f'[START PART {idx + 1}/{num_parts}]')
            new_content = new_content.replace(f'[END PART {idx + 1}]', f'[END PART {idx + 1}/{num_parts}]')
            new_content = new_content.replace('[START PART', f'[START PART {idx + 1}/{num_parts}]', 1)
            new_content = new_content.replace('[END PART', f'[END PART {idx + 1}/{num_parts}]', 1)

        item['content'] = new_content
        item['name'] = f'split_{str(idx + 1).zfill(3)}_of_{str(num_parts).zfill(3)}.txt'

    return file_data


@app.post("/api/count_tokens")
def api_count_tokens():
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text", "")
        encoding_or_model_name = data.get("encoding", "cl100k_base")
        token_count = count_tokens(text, encoding_or_model_name)
        return jsonify({"count": token_count})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post('/api/split')
def api_split():
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = data.get('text', '')
        unit = data.get('unit', 'tokens')  # 'tokens' | 'chars'
        size = int(data.get('size', 1000))
        encoding_or_model_name = data.get('encoding', 'cl100k_base')
        boundary = data.get('boundary', 'none')  # 'none' | 'sentence' | 'paragraph'
        overlap = int(data.get('overlap', 0))  # overlap in tokens/characters

        if unit == 'tokens':
            if boundary == 'none':
                parts = split_prompt_by_tokens(text, size, encoding_or_model_name, overlap)
            else:
                # For now, boundary splitting doesn't support overlap
                parts = split_prompt_tokens_with_boundary(text, size, encoding_or_model_name, boundary)
        else:
            if boundary == 'none':
                parts = split_prompt(text, size, overlap)
            else:
                # For now, boundary splitting doesn't support overlap
                parts = split_prompt_chars_with_boundary(text, size, boundary)
        return jsonify({
            'parts': parts,
            'count': {
                'chars': len(text),
                'tokens': count_tokens(text, encoding_or_model_name)
            }
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


_SENTENCE_BOUNDARY_REGEX = re.compile(r"(?s)(?<=[\.!?])[)\]\"']*(?=\s|\n|$)")

def _find_last_boundary_index(text: str, mode: str) -> int:
    if not text:
        return 0
    if mode == 'paragraph':
        idx = text.rfind('\n\n')
        return idx + 2 if idx != -1 else -1
    if mode == 'sentence':
        last_pos = -1
        for m in _SENTENCE_BOUNDARY_REGEX.finditer(text):
            last_pos = m.end()
        return last_pos
    return -1


def split_prompt_chars_with_boundary(text: str, max_chars: int, boundary: str):
    if max_chars <= 0:
        raise ValueError("Max length must be greater than 0.")
    if not text:
        return []

    file_data = []
    parts_text = []
    start = 0
    n = len(text)
    while start < n:
        end_hint = min(start + max_chars, n)
        window = text[start:end_hint]
        cut = _find_last_boundary_index(window, boundary)
        if cut == -1 or cut == 0:
            cut = len(window)
        chunk = window[:cut]
        parts_text.append(chunk)
        start += len(chunk)

    num_parts = len(parts_text)
    for i, chunk_text in enumerate(parts_text):
        if i == num_parts - 1:
            content = f'[START PART {i + 1}/{num_parts}]\n' + chunk_text + f'\n[END PART {i + 1}/{num_parts}]'
            content += '\nALL PARTS SENT. Now you can continue processing the request.'
        else:
            content = f'Do not answer yet. This is just another part of the text I want to send you. Just receive and acknowledge as "Part {i + 1}/{num_parts} received" and wait for the next part.\n[START PART {i + 1}/{num_parts}]\n' + chunk_text + f'\n[END PART {i + 1}/{num_parts}]'
            content += f'\nRemember not answering yet. Just acknowledge you received this part with the message "Part {i + 1}/{num_parts} received" and wait for the next part.'

        file_data.append({
            'name': f'split_{str(i + 1).zfill(3)}_of_{str(num_parts).zfill(3)}.txt',
            'content': content
        })

    return file_data


def split_prompt_tokens_with_boundary(text: str, max_tokens: int, encoding_or_model_name: str, boundary: str):
    if max_tokens <= 0:
        raise ValueError("Max tokens must be greater than 0.")
    if not text:
        return []

    enc = _get_encoding(encoding_or_model_name)
    token_ids = enc.encode(text)
    i = 0
    n = len(token_ids)
    parts = []

    while i < n:
        window_end = min(i + max_tokens, n)
        window_ids = token_ids[i:window_end]
        window_text = enc.decode(window_ids)
        cut_char_pos = _find_last_boundary_index(window_text, boundary)
        if cut_char_pos == -1 or cut_char_pos == 0:
            advance = len(window_ids)
            parts.append(window_text)
            i += advance
            continue
        advance = 0
        for j in range(1, len(window_ids) + 1):
            if len(enc.decode(window_ids[:j])) >= cut_char_pos:
                advance = max(1, j - 1)
                break
        if advance == 0:
            advance = len(window_ids)
            parts.append(window_text)
            i += advance
        else:
            chunk_text = enc.decode(window_ids[:advance])
            parts.append(chunk_text)
            i += advance

    num_parts = len(parts)
    file_data = []
    for idx, chunk_text in enumerate(parts):
        if idx == num_parts - 1:
            content = f'[START PART {idx + 1}/{num_parts}]\n' + chunk_text + f'\n[END PART {idx + 1}/{num_parts}]'
            content += '\nALL PARTS SENT. Now you can continue processing the request.'
        else:
            content = f'Do not answer yet. This is just another part of the text I want to send you. Just receive and acknowledge as "Part {idx + 1}/{num_parts} received" and wait for the next part.\n[START PART {idx + 1}/{num_parts}]\n' + chunk_text + f'\n[END PART {idx + 1}/{num_parts}]'
            content += f'\nRemember not answering yet. Just acknowledge you received this part with the message "Part {idx + 1}/{num_parts} received" and wait for the next part.'
        file_data.append({'name': f'split_{str(idx + 1).zfill(3)}_of_{str(num_parts).zfill(3)}.txt', 'content': content})

    return file_data

def generate_random_hash(length):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 3000)))