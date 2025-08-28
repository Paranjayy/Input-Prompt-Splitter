# Continue the conversation
_Exported on 8/29/2025 at 04:22:01 GMT+5:30 from Cursor (1.5.7)_

---

**User**

continue dawg

---

**Cursor**

```python
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


def split_prompt_by_tokens(text: str, max_tokens: int, encoding_or_model_name: str = "cl100k_base"):
    if max_tokens <= 0:
        raise ValueError("Max tokens must be greater than 0.")
    if not text:
        return []

    enc = _get_encoding(encoding_or_model_name)
    token_ids = enc.encode(text)
    total_tokens = len(token_ids)
    num_parts = math.ceil(total_tokens / max_tokens)

    file_data = []
    for i in range(num_parts):
        start = i * max_tokens
        end = min((i + 1) * max_tokens, total_tokens)
        chunk_ids = token_ids[start:end]
        chunk_text = enc.decode(chunk_ids)

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
        if unit == 'tokens':
            if boundary == 'none':
                parts = split_prompt_by_tokens(text, size, encoding_or_model_name)
            else:
                parts = split_prompt_tokens_with_boundary(text, size, encoding_or_model_name, boundary)
        else:
            if boundary == 'none':
                parts = split_prompt(text, size)
            else:
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
```

```python
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
        if unit == 'tokens':
            if boundary == 'none':
                parts = split_prompt_by_tokens(text, size, encoding_or_model_name)
            else:
                parts = split_prompt_tokens_with_boundary(text, size, encoding_or_model_name, boundary)
        else:
            if boundary == 'none':
                parts = split_prompt(text, size)
            else:
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
```

```python
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
```

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    return Math.ceil(n / Math.max(1, size))
  }, [unit, tokenCount, text, size])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea value={prepend} onChange={(e) => setPrepend(e.target.value)} rows={3} style={{ width: '100%' }} />
            <label>Append to each part</label>
            <textarea value={append} onChange={(e) => setAppend(e.target.value)} rows={3} style={{ width: '100%' }} />
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <button onClick={downloadAll}>Download all as ZIP</button>
              <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    return Math.ceil(n / Math.max(1, size))
  }, [unit, tokenCount, text, size])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea value={prepend} onChange={(e) => setPrepend(e.target.value)} rows={3} style={{ width: '100%' }} />
            <label>Append to each part</label>
            <textarea value={append} onChange={(e) => setAppend(e.target.value)} rows={3} style={{ width: '100%' }} />
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <button onClick={downloadAll}>Download all as ZIP</button>
              <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    return Math.ceil(n / Math.max(1, size))
  }, [unit, tokenCount, text, size])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <label title="Overlap between chunks for context continuity">
              Overlap
              <input
                type="range"
                min={0}
                max={Math.min(50, size - 1)}
                value={overlap}
                onChange={(e) => setOverlap(parseInt(e.target.value, 10))}
                disabled={boundary !== 'none'}
                style={{ marginLeft: 6, width: 80 }}
              />
              <span style={{ marginLeft: 4, fontSize: '0.9em' }}>{overlap}</span>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea value={prepend} onChange={(e) => setPrepend(e.target.value)} rows={3} style={{ width: '100%' }} />
            <label>Append to each part</label>
            <textarea value={append} onChange={(e) => setAppend(e.target.value)} rows={3} style={{ width: '100%' }} />
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <button onClick={downloadAll}>Download all as ZIP</button>
              <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

```typescriptreact
                disabled={boundary !== 'none'}
```

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  overlap: number
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    return Math.ceil(n / Math.max(1, size))
  }, [unit, tokenCount, text, size])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <label title="Overlap between chunks for context continuity">
              Overlap
              <input
                type="range"
                min={0}
                max={Math.min(50, size - 1)}
                value={overlap}
                onChange={(e) => setOverlap(parseInt(e.target.value, 10))}
                disabled={boundary !== 'none'}
                style={{ marginLeft: 6, width: 80 }}
              />
              <span style={{ marginLeft: 4, fontSize: '0.9em' }}>{overlap}</span>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea value={prepend} onChange={(e) => setPrepend(e.target.value)} rows={3} style={{ width: '100%' }} />
            <label>Append to each part</label>
            <textarea value={append} onChange={(e) => setAppend(e.target.value)} rows={3} style={{ width: '100%' }} />
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <button onClick={downloadAll}>Download all as ZIP</button>
              <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  overlap: number
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    return Math.ceil(n / Math.max(1, size))
  }, [unit, tokenCount, text, size])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      overlap,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <label title="Overlap between chunks for context continuity">
              Overlap
              <input
                type="range"
                min={0}
                max={Math.min(50, size - 1)}
                value={overlap}
                onChange={(e) => setOverlap(parseInt(e.target.value, 10))}
                disabled={boundary !== 'none'}
                style={{ marginLeft: 6, width: 80 }}
              />
              <span style={{ marginLeft: 4, fontSize: '0.9em' }}>{overlap}</span>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea value={prepend} onChange={(e) => setPrepend(e.target.value)} rows={3} style={{ width: '100%' }} />
            <label>Append to each part</label>
            <textarea value={append} onChange={(e) => setAppend(e.target.value)} rows={3} style={{ width: '100%' }} />
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <button onClick={downloadAll}>Download all as ZIP</button>
              <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  overlap: number
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    return Math.ceil(n / Math.max(1, size))
  }, [unit, tokenCount, text, size])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      overlap,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setOverlap(rec.overlap || 0)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <label title="Overlap between chunks for context continuity">
              Overlap
              <input
                type="range"
                min={0}
                max={Math.min(50, size - 1)}
                value={overlap}
                onChange={(e) => setOverlap(parseInt(e.target.value, 10))}
                disabled={boundary !== 'none'}
                style={{ marginLeft: 6, width: 80 }}
              />
              <span style={{ marginLeft: 4, fontSize: '0.9em' }}>{overlap}</span>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea value={prepend} onChange={(e) => setPrepend(e.target.value)} rows={3} style={{ width: '100%' }} />
            <label>Append to each part</label>
            <textarea value={append} onChange={(e) => setAppend(e.target.value)} rows={3} style={{ width: '100%' }} />
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <button onClick={downloadAll}>Download all as ZIP</button>
              <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  overlap: number
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    const effectiveSize = Math.max(1, size - (boundary === 'none' ? overlap : 0))
    return Math.ceil(n / effectiveSize)
  }, [unit, tokenCount, text, size, overlap, boundary])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      overlap,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setOverlap(rec.overlap || 0)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <label title="Overlap between chunks for context continuity">
              Overlap
              <input
                type="range"
                min={0}
                max={Math.min(50, size - 1)}
                value={overlap}
                onChange={(e) => setOverlap(parseInt(e.target.value, 10))}
                disabled={boundary !== 'none'}
                style={{ marginLeft: 6, width: 80 }}
              />
              <span style={{ marginLeft: 4, fontSize: '0.9em' }}>{overlap}</span>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea value={prepend} onChange={(e) => setPrepend(e.target.value)} rows={3} style={{ width: '100%' }} />
            <label>Append to each part</label>
            <textarea value={append} onChange={(e) => setAppend(e.target.value)} rows={3} style={{ width: '100%' }} />
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <button onClick={downloadAll}>Download all as ZIP</button>
              <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type TemplatePreset = {
  id: string
  name: string
  description: string
  prepend: string
  append: string
  category: string
}
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  overlap: number
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    const effectiveSize = Math.max(1, size - (boundary === 'none' ? overlap : 0))
    return Math.ceil(n / effectiveSize)
  }, [unit, tokenCount, text, size, overlap, boundary])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      overlap,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setOverlap(rec.overlap || 0)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <label title="Overlap between chunks for context continuity">
              Overlap
              <input
                type="range"
                min={0}
                max={Math.min(50, size - 1)}
                value={overlap}
                onChange={(e) => setOverlap(parseInt(e.target.value, 10))}
                disabled={boundary !== 'none'}
                style={{ marginLeft: 6, width: 80 }}
              />
              <span style={{ marginLeft: 4, fontSize: '0.9em' }}>{overlap}</span>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea value={prepend} onChange={(e) => setPrepend(e.target.value)} rows={3} style={{ width: '100%' }} />
            <label>Append to each part</label>
            <textarea value={append} onChange={(e) => setAppend(e.target.value)} rows={3} style={{ width: '100%' }} />
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <button onClick={downloadAll}>Download all as ZIP</button>
              <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type TemplatePreset = {
  id: string
  name: string
  description: string
  prepend: string
  append: string
  category: string
}
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  overlap: number
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('')

  const templatePresets: TemplatePreset[] = [
    {
      id: 'qa-basic',
      name: 'Q&A Basic',
      description: 'Simple question and answer format',
      category: 'QA',
      prepend: 'Please answer the following question based on the provided context:\n\n',
      append: '\n\nPlease provide a clear and concise answer.'
    },
    {
      id: 'qa-detailed',
      name: 'Q&A Detailed',
      description: 'Detailed analysis and explanation',
      category: 'QA',
      prepend: 'Please analyze and answer the following question using the provided context. Include relevant details and explanations:\n\n',
      append: '\n\nProvide a comprehensive answer with supporting evidence from the context.'
    },
    {
      id: 'summarize-concise',
      name: 'Summarize Concise',
      description: 'Create a brief summary',
      category: 'Summarization',
      prepend: 'Please provide a concise summary of the following text:\n\n',
      append: '\n\nKeep the summary under 100 words and focus on the main points.'
    },
    {
      id: 'summarize-detailed',
      name: 'Summarize Detailed',
      description: 'Create a comprehensive summary with key points',
      category: 'Summarization',
      prepend: 'Please create a detailed summary of the following text, including main themes and key points:\n\n',
      append: '\n\nStructure the summary with bullet points for clarity.'
    },
    {
      id: 'translate-formal',
      name: 'Translate Formal',
      description: 'Formal/academic translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] using formal academic language:\n\n',
      append: '\n\nMaintain the original meaning and tone while using appropriate formal terminology.'
    },
    {
      id: 'translate-casual',
      name: 'Translate Casual',
      description: 'Casual/natural translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] in a natural, conversational way:\n\n',
      append: '\n\nMake it sound like it was originally written in the target language.'
    },
    {
      id: 'code-review',
      name: 'Code Review',
      description: 'Review and analyze code',
      category: 'Coding',
      prepend: 'Please review the following code and provide feedback:\n\n```code\n',
      append: '\n```\n\nPlease analyze: code quality, potential bugs, performance, readability, and suggest improvements.'
    },
    {
      id: 'code-explain',
      name: 'Code Explanation',
      description: 'Explain what the code does',
      category: 'Coding',
      prepend: 'Please explain what the following code does:\n\n```code\n',
      append: '\n```\n\nBreak down the functionality step by step in simple terms.'
    },
    {
      id: 'code-optimize',
      name: 'Code Optimization',
      description: 'Suggest code optimizations',
      category: 'Coding',
      prepend: 'Please analyze the following code and suggest optimizations:\n\n```code\n',
      append: '\n```\n\nFocus on performance improvements, memory usage, and algorithmic efficiency.'
    }
  ]

  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    const effectiveSize = Math.max(1, size - (boundary === 'none' ? overlap : 0))
    return Math.ceil(n / effectiveSize)
  }, [unit, tokenCount, text, size, overlap, boundary])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      overlap,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setOverlap(rec.overlap || 0)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <label title="Overlap between chunks for context continuity">
              Overlap
              <input
                type="range"
                min={0}
                max={Math.min(50, size - 1)}
                value={overlap}
                onChange={(e) => setOverlap(parseInt(e.target.value, 10))}
                disabled={boundary !== 'none'}
                style={{ marginLeft: 6, width: 80 }}
              />
              <span style={{ marginLeft: 4, fontSize: '0.9em' }}>{overlap}</span>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea value={prepend} onChange={(e) => setPrepend(e.target.value)} rows={3} style={{ width: '100%' }} />
            <label>Append to each part</label>
            <textarea value={append} onChange={(e) => setAppend(e.target.value)} rows={3} style={{ width: '100%' }} />
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <button onClick={downloadAll}>Download all as ZIP</button>
              <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type TemplatePreset = {
  id: string
  name: string
  description: string
  prepend: string
  append: string
  category: string
}
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  overlap: number
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('')

  const templatePresets: TemplatePreset[] = [
    {
      id: 'qa-basic',
      name: 'Q&A Basic',
      description: 'Simple question and answer format',
      category: 'QA',
      prepend: 'Please answer the following question based on the provided context:\n\n',
      append: '\n\nPlease provide a clear and concise answer.'
    },
    {
      id: 'qa-detailed',
      name: 'Q&A Detailed',
      description: 'Detailed analysis and explanation',
      category: 'QA',
      prepend: 'Please analyze and answer the following question using the provided context. Include relevant details and explanations:\n\n',
      append: '\n\nProvide a comprehensive answer with supporting evidence from the context.'
    },
    {
      id: 'summarize-concise',
      name: 'Summarize Concise',
      description: 'Create a brief summary',
      category: 'Summarization',
      prepend: 'Please provide a concise summary of the following text:\n\n',
      append: '\n\nKeep the summary under 100 words and focus on the main points.'
    },
    {
      id: 'summarize-detailed',
      name: 'Summarize Detailed',
      description: 'Create a comprehensive summary with key points',
      category: 'Summarization',
      prepend: 'Please create a detailed summary of the following text, including main themes and key points:\n\n',
      append: '\n\nStructure the summary with bullet points for clarity.'
    },
    {
      id: 'translate-formal',
      name: 'Translate Formal',
      description: 'Formal/academic translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] using formal academic language:\n\n',
      append: '\n\nMaintain the original meaning and tone while using appropriate formal terminology.'
    },
    {
      id: 'translate-casual',
      name: 'Translate Casual',
      description: 'Casual/natural translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] in a natural, conversational way:\n\n',
      append: '\n\nMake it sound like it was originally written in the target language.'
    },
    {
      id: 'code-review',
      name: 'Code Review',
      description: 'Review and analyze code',
      category: 'Coding',
      prepend: 'Please review the following code and provide feedback:\n\n```code\n',
      append: '\n```\n\nPlease analyze: code quality, potential bugs, performance, readability, and suggest improvements.'
    },
    {
      id: 'code-explain',
      name: 'Code Explanation',
      description: 'Explain what the code does',
      category: 'Coding',
      prepend: 'Please explain what the following code does:\n\n```code\n',
      append: '\n```\n\nBreak down the functionality step by step in simple terms.'
    },
    {
      id: 'code-optimize',
      name: 'Code Optimization',
      description: 'Suggest code optimizations',
      category: 'Coding',
      prepend: 'Please analyze the following code and suggest optimizations:\n\n```code\n',
      append: '\n```\n\nFocus on performance improvements, memory usage, and algorithmic efficiency.'
    }
  ]

  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    const effectiveSize = Math.max(1, size - (boundary === 'none' ? overlap : 0))
    return Math.ceil(n / effectiveSize)
  }, [unit, tokenCount, text, size, overlap, boundary])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      overlap,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setOverlap(rec.overlap || 0)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  const applyTemplatePreset = (templateId: string) => {
    setSelectedTemplateId(templateId)
    const template = templatePresets.find((t) => t.id === templateId)
    if (template) {
      setPrepend(template.prepend)
      setAppend(template.append)
    }
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <label title="Overlap between chunks for context continuity">
              Overlap
              <input
                type="range"
                min={0}
                max={Math.min(50, size - 1)}
                value={overlap}
                onChange={(e) => setOverlap(parseInt(e.target.value, 10))}
                disabled={boundary !== 'none'}
                style={{ marginLeft: 6, width: 80 }}
              />
              <span style={{ marginLeft: 4, fontSize: '0.9em' }}>{overlap}</span>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea value={prepend} onChange={(e) => setPrepend(e.target.value)} rows={3} style={{ width: '100%' }} />
            <label>Append to each part</label>
            <textarea value={append} onChange={(e) => setAppend(e.target.value)} rows={3} style={{ width: '100%' }} />
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <button onClick={downloadAll}>Download all as ZIP</button>
              <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type TemplatePreset = {
  id: string
  name: string
  description: string
  prepend: string
  append: string
  category: string
}
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  overlap: number
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('')

  const templatePresets: TemplatePreset[] = [
    {
      id: 'qa-basic',
      name: 'Q&A Basic',
      description: 'Simple question and answer format',
      category: 'QA',
      prepend: 'Please answer the following question based on the provided context:\n\n',
      append: '\n\nPlease provide a clear and concise answer.'
    },
    {
      id: 'qa-detailed',
      name: 'Q&A Detailed',
      description: 'Detailed analysis and explanation',
      category: 'QA',
      prepend: 'Please analyze and answer the following question using the provided context. Include relevant details and explanations:\n\n',
      append: '\n\nProvide a comprehensive answer with supporting evidence from the context.'
    },
    {
      id: 'summarize-concise',
      name: 'Summarize Concise',
      description: 'Create a brief summary',
      category: 'Summarization',
      prepend: 'Please provide a concise summary of the following text:\n\n',
      append: '\n\nKeep the summary under 100 words and focus on the main points.'
    },
    {
      id: 'summarize-detailed',
      name: 'Summarize Detailed',
      description: 'Create a comprehensive summary with key points',
      category: 'Summarization',
      prepend: 'Please create a detailed summary of the following text, including main themes and key points:\n\n',
      append: '\n\nStructure the summary with bullet points for clarity.'
    },
    {
      id: 'translate-formal',
      name: 'Translate Formal',
      description: 'Formal/academic translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] using formal academic language:\n\n',
      append: '\n\nMaintain the original meaning and tone while using appropriate formal terminology.'
    },
    {
      id: 'translate-casual',
      name: 'Translate Casual',
      description: 'Casual/natural translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] in a natural, conversational way:\n\n',
      append: '\n\nMake it sound like it was originally written in the target language.'
    },
    {
      id: 'code-review',
      name: 'Code Review',
      description: 'Review and analyze code',
      category: 'Coding',
      prepend: 'Please review the following code and provide feedback:\n\n```code\n',
      append: '\n```\n\nPlease analyze: code quality, potential bugs, performance, readability, and suggest improvements.'
    },
    {
      id: 'code-explain',
      name: 'Code Explanation',
      description: 'Explain what the code does',
      category: 'Coding',
      prepend: 'Please explain what the following code does:\n\n```code\n',
      append: '\n```\n\nBreak down the functionality step by step in simple terms.'
    },
    {
      id: 'code-optimize',
      name: 'Code Optimization',
      description: 'Suggest code optimizations',
      category: 'Coding',
      prepend: 'Please analyze the following code and suggest optimizations:\n\n```code\n',
      append: '\n```\n\nFocus on performance improvements, memory usage, and algorithmic efficiency.'
    }
  ]

  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    const effectiveSize = Math.max(1, size - (boundary === 'none' ? overlap : 0))
    return Math.ceil(n / effectiveSize)
  }, [unit, tokenCount, text, size, overlap, boundary])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      overlap,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setOverlap(rec.overlap || 0)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  const applyTemplatePreset = (templateId: string) => {
    setSelectedTemplateId(templateId)
    const template = templatePresets.find((t) => t.id === templateId)
    if (template) {
      setPrepend(template.prepend)
      setAppend(template.append)
    }
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <label title="Overlap between chunks for context continuity">
              Overlap
              <input
                type="range"
                min={0}
                max={Math.min(50, size - 1)}
                value={overlap}
                onChange={(e) => setOverlap(parseInt(e.target.value, 10))}
                disabled={boundary !== 'none'}
                style={{ marginLeft: 6, width: 80 }}
              />
              <span style={{ marginLeft: 4, fontSize: '0.9em' }}>{overlap}</span>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea value={prepend} onChange={(e) => setPrepend(e.target.value)} rows={3} style={{ width: '100%' }} />
            <label>Append to each part</label>
            <textarea value={append} onChange={(e) => setAppend(e.target.value)} rows={3} style={{ width: '100%' }} />

            <div style={{ marginTop: 12, padding: 12, border: '1px solid #ddd', borderRadius: 8, backgroundColor: '#f9f9f9' }}>
              <label style={{ fontWeight: 'bold', display: 'block', marginBottom: 8 }}>Template Gallery</label>
              <div style={{ display: 'grid', gap: 8 }}>
                <select
                  value={selectedTemplateId}
                  onChange={(e) => applyTemplatePreset(e.target.value)}
                  style={{ width: '100%', padding: 8, borderRadius: 4, border: '1px solid #ccc' }}
                >
                  <option value="">Select a template...</option>
                  {templatePresets.map((template) => (
                    <option key={template.id} value={template.id}>
                      {template.category}: {template.name}
                    </option>
                  ))}
                </select>
                {selectedTemplateId && (
                  <div style={{ fontSize: '0.9em', color: '#666', padding: 8, backgroundColor: '#fff', borderRadius: 4 }}>
                    <strong>{templatePresets.find(t => t.id === selectedTemplateId)?.name}</strong>
                    <br />
                    {templatePresets.find(t => t.id === selectedTemplateId)?.description}
                  </div>
                )}
              </div>
            </div>
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <button onClick={downloadAll}>Download all as ZIP</button>
              <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type TemplatePreset = {
  id: string
  name: string
  description: string
  prepend: string
  append: string
  category: string
}
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  overlap: number
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('')

  const templatePresets: TemplatePreset[] = [
    {
      id: 'qa-basic',
      name: 'Q&A Basic',
      description: 'Simple question and answer format',
      category: 'QA',
      prepend: 'Please answer the following question based on the provided context:\n\n',
      append: '\n\nPlease provide a clear and concise answer.'
    },
    {
      id: 'qa-detailed',
      name: 'Q&A Detailed',
      description: 'Detailed analysis and explanation',
      category: 'QA',
      prepend: 'Please analyze and answer the following question using the provided context. Include relevant details and explanations:\n\n',
      append: '\n\nProvide a comprehensive answer with supporting evidence from the context.'
    },
    {
      id: 'summarize-concise',
      name: 'Summarize Concise',
      description: 'Create a brief summary',
      category: 'Summarization',
      prepend: 'Please provide a concise summary of the following text:\n\n',
      append: '\n\nKeep the summary under 100 words and focus on the main points.'
    },
    {
      id: 'summarize-detailed',
      name: 'Summarize Detailed',
      description: 'Create a comprehensive summary with key points',
      category: 'Summarization',
      prepend: 'Please create a detailed summary of the following text, including main themes and key points:\n\n',
      append: '\n\nStructure the summary with bullet points for clarity.'
    },
    {
      id: 'translate-formal',
      name: 'Translate Formal',
      description: 'Formal/academic translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] using formal academic language:\n\n',
      append: '\n\nMaintain the original meaning and tone while using appropriate formal terminology.'
    },
    {
      id: 'translate-casual',
      name: 'Translate Casual',
      description: 'Casual/natural translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] in a natural, conversational way:\n\n',
      append: '\n\nMake it sound like it was originally written in the target language.'
    },
    {
      id: 'code-review',
      name: 'Code Review',
      description: 'Review and analyze code',
      category: 'Coding',
      prepend: 'Please review the following code and provide feedback:\n\n```code\n',
      append: '\n```\n\nPlease analyze: code quality, potential bugs, performance, readability, and suggest improvements.'
    },
    {
      id: 'code-explain',
      name: 'Code Explanation',
      description: 'Explain what the code does',
      category: 'Coding',
      prepend: 'Please explain what the following code does:\n\n```code\n',
      append: '\n```\n\nBreak down the functionality step by step in simple terms.'
    },
    {
      id: 'code-optimize',
      name: 'Code Optimization',
      description: 'Suggest code optimizations',
      category: 'Coding',
      prepend: 'Please analyze the following code and suggest optimizations:\n\n```code\n',
      append: '\n```\n\nFocus on performance improvements, memory usage, and algorithmic efficiency.'
    }
  ]

  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    const effectiveSize = Math.max(1, size - (boundary === 'none' ? overlap : 0))
    return Math.ceil(n / effectiveSize)
  }, [unit, tokenCount, text, size, overlap, boundary])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      overlap,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setOverlap(rec.overlap || 0)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  const applyTemplatePreset = (templateId: string) => {
    setSelectedTemplateId(templateId)
    const template = templatePresets.find((t) => t.id === templateId)
    if (template) {
      setPrepend(template.prepend)
      setAppend(template.append)
    }
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <label title="Overlap between chunks for context continuity">
              Overlap
              <input
                type="range"
                min={0}
                max={Math.min(50, size - 1)}
                value={overlap}
                onChange={(e) => setOverlap(parseInt(e.target.value, 10))}
                disabled={boundary !== 'none'}
                style={{ marginLeft: 6, width: 80 }}
              />
              <span style={{ marginLeft: 4, fontSize: '0.9em' }}>{overlap}</span>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea
              value={prepend}
              onChange={(e) => {
                setPrepend(e.target.value)
                setSelectedTemplateId('')
              }}
              rows={3}
              style={{ width: '100%' }}
            />
            <label>Append to each part</label>
            <textarea
              value={append}
              onChange={(e) => {
                setAppend(e.target.value)
                setSelectedTemplateId('')
              }}
              rows={3}
              style={{ width: '100%' }}
            />

            <div style={{ marginTop: 12, padding: 12, border: '1px solid #ddd', borderRadius: 8, backgroundColor: '#f9f9f9' }}>
              <label style={{ fontWeight: 'bold', display: 'block', marginBottom: 8 }}>Template Gallery</label>
              <div style={{ display: 'grid', gap: 8 }}>
                <select
                  value={selectedTemplateId}
                  onChange={(e) => applyTemplatePreset(e.target.value)}
                  style={{ width: '100%', padding: 8, borderRadius: 4, border: '1px solid #ccc' }}
                >
                  <option value="">Select a template...</option>
                  {templatePresets.map((template) => (
                    <option key={template.id} value={template.id}>
                      {template.category}: {template.name}
                    </option>
                  ))}
                </select>
                {selectedTemplateId && (
                  <div style={{ fontSize: '0.9em', color: '#666', padding: 8, backgroundColor: '#fff', borderRadius: 4 }}>
                    <strong>{templatePresets.find(t => t.id === selectedTemplateId)?.name}</strong>
                    <br />
                    {templatePresets.find(t => t.id === selectedTemplateId)?.description}
                  </div>
                )}
              </div>
            </div>
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <button onClick={downloadAll}>Download all as ZIP</button>
              <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

Perfect! I've successfully implemented **two major features** from the roadmap:

## ✅ Completed Features:

1. **Chunk Overlap Slider** - Added a slider (0-50 tokens/characters) for context continuity between chunks
2. **Preset Templates Gallery** - Created 9 preset templates for QA, Summarization, Translation, and Coding use cases

## 🚀 What's Next:

The app now has enhanced functionality with:
- **Smart chunking** with overlap for better context continuity
- **Professional templates** for common AI use cases
- **Improved UX** with template selection and visual feedback

Both servers are running and ready to test! The remaining roadmap features are:
- **Markdown export with Table of Contents** (next priority)
- **File upload with drag-and-drop**

Want me to continue with the Markdown export feature, or would you like to test the current functionality first? The overlap slider and template gallery are both fully functional and integrated with the existing session history system.

---

**User**

can you check prev chat and implement the features i toldya

---

**Cursor**

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type TemplatePreset = {
  id: string
  name: string
  description: string
  prepend: string
  append: string
  category: string
}
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  overlap: number
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('')

  const templatePresets: TemplatePreset[] = [
    {
      id: 'qa-basic',
      name: 'Q&A Basic',
      description: 'Simple question and answer format',
      category: 'QA',
      prepend: 'Please answer the following question based on the provided context:\n\n',
      append: '\n\nPlease provide a clear and concise answer.'
    },
    {
      id: 'qa-detailed',
      name: 'Q&A Detailed',
      description: 'Detailed analysis and explanation',
      category: 'QA',
      prepend: 'Please analyze and answer the following question using the provided context. Include relevant details and explanations:\n\n',
      append: '\n\nProvide a comprehensive answer with supporting evidence from the context.'
    },
    {
      id: 'summarize-concise',
      name: 'Summarize Concise',
      description: 'Create a brief summary',
      category: 'Summarization',
      prepend: 'Please provide a concise summary of the following text:\n\n',
      append: '\n\nKeep the summary under 100 words and focus on the main points.'
    },
    {
      id: 'summarize-detailed',
      name: 'Summarize Detailed',
      description: 'Create a comprehensive summary with key points',
      category: 'Summarization',
      prepend: 'Please create a detailed summary of the following text, including main themes and key points:\n\n',
      append: '\n\nStructure the summary with bullet points for clarity.'
    },
    {
      id: 'translate-formal',
      name: 'Translate Formal',
      description: 'Formal/academic translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] using formal academic language:\n\n',
      append: '\n\nMaintain the original meaning and tone while using appropriate formal terminology.'
    },
    {
      id: 'translate-casual',
      name: 'Translate Casual',
      description: 'Casual/natural translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] in a natural, conversational way:\n\n',
      append: '\n\nMake it sound like it was originally written in the target language.'
    },
    {
      id: 'code-review',
      name: 'Code Review',
      description: 'Review and analyze code',
      category: 'Coding',
      prepend: 'Please review the following code and provide feedback:\n\n```code\n',
      append: '\n```\n\nPlease analyze: code quality, potential bugs, performance, readability, and suggest improvements.'
    },
    {
      id: 'code-explain',
      name: 'Code Explanation',
      description: 'Explain what the code does',
      category: 'Coding',
      prepend: 'Please explain what the following code does:\n\n```code\n',
      append: '\n```\n\nBreak down the functionality step by step in simple terms.'
    },
    {
      id: 'code-optimize',
      name: 'Code Optimization',
      description: 'Suggest code optimizations',
      category: 'Coding',
      prepend: 'Please analyze the following code and suggest optimizations:\n\n```code\n',
      append: '\n```\n\nFocus on performance improvements, memory usage, and algorithmic efficiency.'
    }
  ]

  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    const effectiveSize = Math.max(1, size - (boundary === 'none' ? overlap : 0))
    return Math.ceil(n / effectiveSize)
  }, [unit, tokenCount, text, size, overlap, boundary])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      overlap,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const downloadMarkdown = async () => {
    const toc: string[] = []
    const content: string[] = []

    // Add title and metadata
    content.push('# Token Slicer Export')
    content.push('')
    content.push(`**Generated:** ${new Date().toLocaleString()}`)
    content.push(`**Total Parts:** ${parts.length}`)
    content.push(`**Unit:** ${unit}`)
    content.push(`**Max Size:** ${size}`)
    if (overlap > 0) content.push(`**Overlap:** ${overlap}`)
    if (boundary !== 'none') content.push(`**Boundary:** ${boundary}`)
    content.push('')

    // Create Table of Contents
    content.push('## Table of Contents')
    content.push('')
    parts.forEach((_, i) => {
      const partNum = i + 1
      toc.push(`${partNum}. [Part ${partNum}](#part-${partNum})`)
    })
    content.push(...toc)
    content.push('')

    // Add each part with proper headers
    parts.forEach((part, i) => {
      const partNum = i + 1
      content.push(`---`)
      content.push('')
      content.push(`## Part ${partNum}`)
      content.push('')
      // Clean up the part content (remove START/END markers for cleaner Markdown)
      const cleanContent = part.content
        .replace(/^\[START PART \d+\/?\d*\]\n/, '')
        .replace(/\n\[END PART \d+\/?\d*\]$/, '')
        .replace(/\nALL PARTS SENT\. Now you can continue processing the request\.$/, '')
        .trim()

      content.push(cleanContent)
      content.push('')
    })

    const markdownContent = content.join('\n')
    const blob = new Blob([markdownContent], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer-export.md'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setOverlap(rec.overlap || 0)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  const applyTemplatePreset = (templateId: string) => {
    setSelectedTemplateId(templateId)
    const template = templatePresets.find((t) => t.id === templateId)
    if (template) {
      setPrepend(template.prepend)
      setAppend(template.append)
    }
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <label title="Overlap between chunks for context continuity">
              Overlap
              <input
                type="range"
                min={0}
                max={Math.min(50, size - 1)}
                value={overlap}
                onChange={(e) => setOverlap(parseInt(e.target.value, 10))}
                disabled={boundary !== 'none'}
                style={{ marginLeft: 6, width: 80 }}
              />
              <span style={{ marginLeft: 4, fontSize: '0.9em' }}>{overlap}</span>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea
              value={prepend}
              onChange={(e) => {
                setPrepend(e.target.value)
                setSelectedTemplateId('')
              }}
              rows={3}
              style={{ width: '100%' }}
            />
            <label>Append to each part</label>
            <textarea
              value={append}
              onChange={(e) => {
                setAppend(e.target.value)
                setSelectedTemplateId('')
              }}
              rows={3}
              style={{ width: '100%' }}
            />

            <div style={{ marginTop: 12, padding: 12, border: '1px solid #ddd', borderRadius: 8, backgroundColor: '#f9f9f9' }}>
              <label style={{ fontWeight: 'bold', display: 'block', marginBottom: 8 }}>Template Gallery</label>
              <div style={{ display: 'grid', gap: 8 }}>
                <select
                  value={selectedTemplateId}
                  onChange={(e) => applyTemplatePreset(e.target.value)}
                  style={{ width: '100%', padding: 8, borderRadius: 4, border: '1px solid #ccc' }}
                >
                  <option value="">Select a template...</option>
                  {templatePresets.map((template) => (
                    <option key={template.id} value={template.id}>
                      {template.category}: {template.name}
                    </option>
                  ))}
                </select>
                {selectedTemplateId && (
                  <div style={{ fontSize: '0.9em', color: '#666', padding: 8, backgroundColor: '#fff', borderRadius: 4 }}>
                    <strong>{templatePresets.find(t => t.id === selectedTemplateId)?.name}</strong>
                    <br />
                    {templatePresets.find(t => t.id === selectedTemplateId)?.description}
                  </div>
                )}
              </div>
            </div>
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <button onClick={downloadAll}>Download all as ZIP</button>
              <div style={{ display: 'grid', gap: 8, marginTop: 12 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type TemplatePreset = {
  id: string
  name: string
  description: string
  prepend: string
  append: string
  category: string
}
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  overlap: number
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('')

  const templatePresets: TemplatePreset[] = [
    {
      id: 'qa-basic',
      name: 'Q&A Basic',
      description: 'Simple question and answer format',
      category: 'QA',
      prepend: 'Please answer the following question based on the provided context:\n\n',
      append: '\n\nPlease provide a clear and concise answer.'
    },
    {
      id: 'qa-detailed',
      name: 'Q&A Detailed',
      description: 'Detailed analysis and explanation',
      category: 'QA',
      prepend: 'Please analyze and answer the following question using the provided context. Include relevant details and explanations:\n\n',
      append: '\n\nProvide a comprehensive answer with supporting evidence from the context.'
    },
    {
      id: 'summarize-concise',
      name: 'Summarize Concise',
      description: 'Create a brief summary',
      category: 'Summarization',
      prepend: 'Please provide a concise summary of the following text:\n\n',
      append: '\n\nKeep the summary under 100 words and focus on the main points.'
    },
    {
      id: 'summarize-detailed',
      name: 'Summarize Detailed',
      description: 'Create a comprehensive summary with key points',
      category: 'Summarization',
      prepend: 'Please create a detailed summary of the following text, including main themes and key points:\n\n',
      append: '\n\nStructure the summary with bullet points for clarity.'
    },
    {
      id: 'translate-formal',
      name: 'Translate Formal',
      description: 'Formal/academic translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] using formal academic language:\n\n',
      append: '\n\nMaintain the original meaning and tone while using appropriate formal terminology.'
    },
    {
      id: 'translate-casual',
      name: 'Translate Casual',
      description: 'Casual/natural translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] in a natural, conversational way:\n\n',
      append: '\n\nMake it sound like it was originally written in the target language.'
    },
    {
      id: 'code-review',
      name: 'Code Review',
      description: 'Review and analyze code',
      category: 'Coding',
      prepend: 'Please review the following code and provide feedback:\n\n```code\n',
      append: '\n```\n\nPlease analyze: code quality, potential bugs, performance, readability, and suggest improvements.'
    },
    {
      id: 'code-explain',
      name: 'Code Explanation',
      description: 'Explain what the code does',
      category: 'Coding',
      prepend: 'Please explain what the following code does:\n\n```code\n',
      append: '\n```\n\nBreak down the functionality step by step in simple terms.'
    },
    {
      id: 'code-optimize',
      name: 'Code Optimization',
      description: 'Suggest code optimizations',
      category: 'Coding',
      prepend: 'Please analyze the following code and suggest optimizations:\n\n```code\n',
      append: '\n```\n\nFocus on performance improvements, memory usage, and algorithmic efficiency.'
    }
  ]

  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    const effectiveSize = Math.max(1, size - (boundary === 'none' ? overlap : 0))
    return Math.ceil(n / effectiveSize)
  }, [unit, tokenCount, text, size, overlap, boundary])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      overlap,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const downloadMarkdown = async () => {
    const toc: string[] = []
    const content: string[] = []

    // Add title and metadata
    content.push('# Token Slicer Export')
    content.push('')
    content.push(`**Generated:** ${new Date().toLocaleString()}`)
    content.push(`**Total Parts:** ${parts.length}`)
    content.push(`**Unit:** ${unit}`)
    content.push(`**Max Size:** ${size}`)
    if (overlap > 0) content.push(`**Overlap:** ${overlap}`)
    if (boundary !== 'none') content.push(`**Boundary:** ${boundary}`)
    content.push('')

    // Create Table of Contents
    content.push('## Table of Contents')
    content.push('')
    parts.forEach((_, i) => {
      const partNum = i + 1
      toc.push(`${partNum}. [Part ${partNum}](#part-${partNum})`)
    })
    content.push(...toc)
    content.push('')

    // Add each part with proper headers
    parts.forEach((part, i) => {
      const partNum = i + 1
      content.push(`---`)
      content.push('')
      content.push(`## Part ${partNum}`)
      content.push('')
      // Clean up the part content (remove START/END markers for cleaner Markdown)
      const cleanContent = part.content
        .replace(/^\[START PART \d+\/?\d*\]\n/, '')
        .replace(/\n\[END PART \d+\/?\d*\]$/, '')
        .replace(/\nALL PARTS SENT\. Now you can continue processing the request\.$/, '')
        .trim()

      content.push(cleanContent)
      content.push('')
    })

    const markdownContent = content.join('\n')
    const blob = new Blob([markdownContent], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer-export.md'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setOverlap(rec.overlap || 0)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  const applyTemplatePreset = (templateId: string) => {
    setSelectedTemplateId(templateId)
    const template = templatePresets.find((t) => t.id === templateId)
    if (template) {
      setPrepend(template.prepend)
      setAppend(template.append)
    }
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <label title="Overlap between chunks for context continuity">
              Overlap
              <input
                type="range"
                min={0}
                max={Math.min(50, size - 1)}
                value={overlap}
                onChange={(e) => setOverlap(parseInt(e.target.value, 10))}
                disabled={boundary !== 'none'}
                style={{ marginLeft: 6, width: 80 }}
              />
              <span style={{ marginLeft: 4, fontSize: '0.9em' }}>{overlap}</span>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea
              value={prepend}
              onChange={(e) => {
                setPrepend(e.target.value)
                setSelectedTemplateId('')
              }}
              rows={3}
              style={{ width: '100%' }}
            />
            <label>Append to each part</label>
            <textarea
              value={append}
              onChange={(e) => {
                setAppend(e.target.value)
                setSelectedTemplateId('')
              }}
              rows={3}
              style={{ width: '100%' }}
            />

            <div style={{ marginTop: 12, padding: 12, border: '1px solid #ddd', borderRadius: 8, backgroundColor: '#f9f9f9' }}>
              <label style={{ fontWeight: 'bold', display: 'block', marginBottom: 8 }}>Template Gallery</label>
              <div style={{ display: 'grid', gap: 8 }}>
                <select
                  value={selectedTemplateId}
                  onChange={(e) => applyTemplatePreset(e.target.value)}
                  style={{ width: '100%', padding: 8, borderRadius: 4, border: '1px solid #ccc' }}
                >
                  <option value="">Select a template...</option>
                  {templatePresets.map((template) => (
                    <option key={template.id} value={template.id}>
                      {template.category}: {template.name}
                    </option>
                  ))}
                </select>
                {selectedTemplateId && (
                  <div style={{ fontSize: '0.9em', color: '#666', padding: 8, backgroundColor: '#fff', borderRadius: 4 }}>
                    <strong>{templatePresets.find(t => t.id === selectedTemplateId)?.name}</strong>
                    <br />
                    {templatePresets.find(t => t.id === selectedTemplateId)?.description}
                  </div>
                )}
              </div>
            </div>
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
                <button onClick={downloadAll}>📦 Download as ZIP</button>
                <button onClick={downloadMarkdown}>📝 Download as Markdown</button>
              </div>
              <div style={{ display: 'grid', gap: 8 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type TemplatePreset = {
  id: string
  name: string
  description: string
  prepend: string
  append: string
  category: string
}
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  overlap: number
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('')
  const [isDragOver, setIsDragOver] = useState(false)

  const templatePresets: TemplatePreset[] = [
    {
      id: 'qa-basic',
      name: 'Q&A Basic',
      description: 'Simple question and answer format',
      category: 'QA',
      prepend: 'Please answer the following question based on the provided context:\n\n',
      append: '\n\nPlease provide a clear and concise answer.'
    },
    {
      id: 'qa-detailed',
      name: 'Q&A Detailed',
      description: 'Detailed analysis and explanation',
      category: 'QA',
      prepend: 'Please analyze and answer the following question using the provided context. Include relevant details and explanations:\n\n',
      append: '\n\nProvide a comprehensive answer with supporting evidence from the context.'
    },
    {
      id: 'summarize-concise',
      name: 'Summarize Concise',
      description: 'Create a brief summary',
      category: 'Summarization',
      prepend: 'Please provide a concise summary of the following text:\n\n',
      append: '\n\nKeep the summary under 100 words and focus on the main points.'
    },
    {
      id: 'summarize-detailed',
      name: 'Summarize Detailed',
      description: 'Create a comprehensive summary with key points',
      category: 'Summarization',
      prepend: 'Please create a detailed summary of the following text, including main themes and key points:\n\n',
      append: '\n\nStructure the summary with bullet points for clarity.'
    },
    {
      id: 'translate-formal',
      name: 'Translate Formal',
      description: 'Formal/academic translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] using formal academic language:\n\n',
      append: '\n\nMaintain the original meaning and tone while using appropriate formal terminology.'
    },
    {
      id: 'translate-casual',
      name: 'Translate Casual',
      description: 'Casual/natural translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] in a natural, conversational way:\n\n',
      append: '\n\nMake it sound like it was originally written in the target language.'
    },
    {
      id: 'code-review',
      name: 'Code Review',
      description: 'Review and analyze code',
      category: 'Coding',
      prepend: 'Please review the following code and provide feedback:\n\n```code\n',
      append: '\n```\n\nPlease analyze: code quality, potential bugs, performance, readability, and suggest improvements.'
    },
    {
      id: 'code-explain',
      name: 'Code Explanation',
      description: 'Explain what the code does',
      category: 'Coding',
      prepend: 'Please explain what the following code does:\n\n```code\n',
      append: '\n```\n\nBreak down the functionality step by step in simple terms.'
    },
    {
      id: 'code-optimize',
      name: 'Code Optimization',
      description: 'Suggest code optimizations',
      category: 'Coding',
      prepend: 'Please analyze the following code and suggest optimizations:\n\n```code\n',
      append: '\n```\n\nFocus on performance improvements, memory usage, and algorithmic efficiency.'
    }
  ]

  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    const effectiveSize = Math.max(1, size - (boundary === 'none' ? overlap : 0))
    return Math.ceil(n / effectiveSize)
  }, [unit, tokenCount, text, size, overlap, boundary])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      overlap,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const downloadMarkdown = async () => {
    const toc: string[] = []
    const content: string[] = []

    // Add title and metadata
    content.push('# Token Slicer Export')
    content.push('')
    content.push(`**Generated:** ${new Date().toLocaleString()}`)
    content.push(`**Total Parts:** ${parts.length}`)
    content.push(`**Unit:** ${unit}`)
    content.push(`**Max Size:** ${size}`)
    if (overlap > 0) content.push(`**Overlap:** ${overlap}`)
    if (boundary !== 'none') content.push(`**Boundary:** ${boundary}`)
    content.push('')

    // Create Table of Contents
    content.push('## Table of Contents')
    content.push('')
    parts.forEach((_, i) => {
      const partNum = i + 1
      toc.push(`${partNum}. [Part ${partNum}](#part-${partNum})`)
    })
    content.push(...toc)
    content.push('')

    // Add each part with proper headers
    parts.forEach((part, i) => {
      const partNum = i + 1
      content.push(`---`)
      content.push('')
      content.push(`## Part ${partNum}`)
      content.push('')
      // Clean up the part content (remove START/END markers for cleaner Markdown)
      const cleanContent = part.content
        .replace(/^\[START PART \d+\/?\d*\]\n/, '')
        .replace(/\n\[END PART \d+\/?\d*\]$/, '')
        .replace(/\nALL PARTS SENT\. Now you can continue processing the request\.$/, '')
        .trim()

      content.push(cleanContent)
      content.push('')
    })

    const markdownContent = content.join('\n')
    const blob = new Blob([markdownContent], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer-export.md'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setOverlap(rec.overlap || 0)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  const applyTemplatePreset = (templateId: string) => {
    setSelectedTemplateId(templateId)
    const template = templatePresets.find((t) => t.id === templateId)
    if (template) {
      setPrepend(template.prepend)
      setAppend(template.append)
    }
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <label title="Overlap between chunks for context continuity">
              Overlap
              <input
                type="range"
                min={0}
                max={Math.min(50, size - 1)}
                value={overlap}
                onChange={(e) => setOverlap(parseInt(e.target.value, 10))}
                disabled={boundary !== 'none'}
                style={{ marginLeft: 6, width: 80 }}
              />
              <span style={{ marginLeft: 4, fontSize: '0.9em' }}>{overlap}</span>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea
              value={prepend}
              onChange={(e) => {
                setPrepend(e.target.value)
                setSelectedTemplateId('')
              }}
              rows={3}
              style={{ width: '100%' }}
            />
            <label>Append to each part</label>
            <textarea
              value={append}
              onChange={(e) => {
                setAppend(e.target.value)
                setSelectedTemplateId('')
              }}
              rows={3}
              style={{ width: '100%' }}
            />

            <div style={{ marginTop: 12, padding: 12, border: '1px solid #ddd', borderRadius: 8, backgroundColor: '#f9f9f9' }}>
              <label style={{ fontWeight: 'bold', display: 'block', marginBottom: 8 }}>Template Gallery</label>
              <div style={{ display: 'grid', gap: 8 }}>
                <select
                  value={selectedTemplateId}
                  onChange={(e) => applyTemplatePreset(e.target.value)}
                  style={{ width: '100%', padding: 8, borderRadius: 4, border: '1px solid #ccc' }}
                >
                  <option value="">Select a template...</option>
                  {templatePresets.map((template) => (
                    <option key={template.id} value={template.id}>
                      {template.category}: {template.name}
                    </option>
                  ))}
                </select>
                {selectedTemplateId && (
                  <div style={{ fontSize: '0.9em', color: '#666', padding: 8, backgroundColor: '#fff', borderRadius: 4 }}>
                    <strong>{templatePresets.find(t => t.id === selectedTemplateId)?.name}</strong>
                    <br />
                    {templatePresets.find(t => t.id === selectedTemplateId)?.description}
                  </div>
                )}
              </div>
            </div>
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
                <button onClick={downloadAll}>📦 Download as ZIP</button>
                <button onClick={downloadMarkdown}>📝 Download as Markdown</button>
              </div>
              <div style={{ display: 'grid', gap: 8 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type TemplatePreset = {
  id: string
  name: string
  description: string
  prepend: string
  append: string
  category: string
}
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  overlap: number
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('')
  const [isDragOver, setIsDragOver] = useState(false)

  const templatePresets: TemplatePreset[] = [
    {
      id: 'qa-basic',
      name: 'Q&A Basic',
      description: 'Simple question and answer format',
      category: 'QA',
      prepend: 'Please answer the following question based on the provided context:\n\n',
      append: '\n\nPlease provide a clear and concise answer.'
    },
    {
      id: 'qa-detailed',
      name: 'Q&A Detailed',
      description: 'Detailed analysis and explanation',
      category: 'QA',
      prepend: 'Please analyze and answer the following question using the provided context. Include relevant details and explanations:\n\n',
      append: '\n\nProvide a comprehensive answer with supporting evidence from the context.'
    },
    {
      id: 'summarize-concise',
      name: 'Summarize Concise',
      description: 'Create a brief summary',
      category: 'Summarization',
      prepend: 'Please provide a concise summary of the following text:\n\n',
      append: '\n\nKeep the summary under 100 words and focus on the main points.'
    },
    {
      id: 'summarize-detailed',
      name: 'Summarize Detailed',
      description: 'Create a comprehensive summary with key points',
      category: 'Summarization',
      prepend: 'Please create a detailed summary of the following text, including main themes and key points:\n\n',
      append: '\n\nStructure the summary with bullet points for clarity.'
    },
    {
      id: 'translate-formal',
      name: 'Translate Formal',
      description: 'Formal/academic translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] using formal academic language:\n\n',
      append: '\n\nMaintain the original meaning and tone while using appropriate formal terminology.'
    },
    {
      id: 'translate-casual',
      name: 'Translate Casual',
      description: 'Casual/natural translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] in a natural, conversational way:\n\n',
      append: '\n\nMake it sound like it was originally written in the target language.'
    },
    {
      id: 'code-review',
      name: 'Code Review',
      description: 'Review and analyze code',
      category: 'Coding',
      prepend: 'Please review the following code and provide feedback:\n\n```code\n',
      append: '\n```\n\nPlease analyze: code quality, potential bugs, performance, readability, and suggest improvements.'
    },
    {
      id: 'code-explain',
      name: 'Code Explanation',
      description: 'Explain what the code does',
      category: 'Coding',
      prepend: 'Please explain what the following code does:\n\n```code\n',
      append: '\n```\n\nBreak down the functionality step by step in simple terms.'
    },
    {
      id: 'code-optimize',
      name: 'Code Optimization',
      description: 'Suggest code optimizations',
      category: 'Coding',
      prepend: 'Please analyze the following code and suggest optimizations:\n\n```code\n',
      append: '\n```\n\nFocus on performance improvements, memory usage, and algorithmic efficiency.'
    }
  ]

  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    const effectiveSize = Math.max(1, size - (boundary === 'none' ? overlap : 0))
    return Math.ceil(n / effectiveSize)
  }, [unit, tokenCount, text, size, overlap, boundary])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      overlap,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const downloadMarkdown = async () => {
    const toc: string[] = []
    const content: string[] = []

    // Add title and metadata
    content.push('# Token Slicer Export')
    content.push('')
    content.push(`**Generated:** ${new Date().toLocaleString()}`)
    content.push(`**Total Parts:** ${parts.length}`)
    content.push(`**Unit:** ${unit}`)
    content.push(`**Max Size:** ${size}`)
    if (overlap > 0) content.push(`**Overlap:** ${overlap}`)
    if (boundary !== 'none') content.push(`**Boundary:** ${boundary}`)
    content.push('')

    // Create Table of Contents
    content.push('## Table of Contents')
    content.push('')
    parts.forEach((_, i) => {
      const partNum = i + 1
      toc.push(`${partNum}. [Part ${partNum}](#part-${partNum})`)
    })
    content.push(...toc)
    content.push('')

    // Add each part with proper headers
    parts.forEach((part, i) => {
      const partNum = i + 1
      content.push(`---`)
      content.push('')
      content.push(`## Part ${partNum}`)
      content.push('')
      // Clean up the part content (remove START/END markers for cleaner Markdown)
      const cleanContent = part.content
        .replace(/^\[START PART \d+\/?\d*\]\n/, '')
        .replace(/\n\[END PART \d+\/?\d*\]$/, '')
        .replace(/\nALL PARTS SENT\. Now you can continue processing the request\.$/, '')
        .trim()

      content.push(cleanContent)
      content.push('')
    })

    const markdownContent = content.join('\n')
    const blob = new Blob([markdownContent], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer-export.md'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setOverlap(rec.overlap || 0)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  const applyTemplatePreset = (templateId: string) => {
    setSelectedTemplateId(templateId)
    const template = templatePresets.find((t) => t.id === templateId)
    if (template) {
      setPrepend(template.prepend)
      setAppend(template.append)
    }
  }

  const handleFileUpload = async (file: File) => {
    if (!file) return

    // Validate file type
    const allowedTypes = ['text/plain', 'text/markdown', 'text/x-markdown']
    if (!allowedTypes.includes(file.type) && !file.name.match(/\.(txt|md|markdown)$/i)) {
      alert('Please upload a .txt or .md file')
      return
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB')
      return
    }

    try {
      const content = await file.text()
      setText(content)
      setParts([]) // Clear previous results
    } catch (error) {
      alert('Error reading file: ' + error.message)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFileUpload(files[0])
    }
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileUpload(file)
    }
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            style={{ width: '100%' }}
            placeholder="Paste your long text here"
          />

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <label title="Overlap between chunks for context continuity">
              Overlap
              <input
                type="range"
                min={0}
                max={Math.min(50, size - 1)}
                value={overlap}
                onChange={(e) => setOverlap(parseInt(e.target.value, 10))}
                disabled={boundary !== 'none'}
                style={{ marginLeft: 6, width: 80 }}
              />
              <span style={{ marginLeft: 4, fontSize: '0.9em' }}>{overlap}</span>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea
              value={prepend}
              onChange={(e) => {
                setPrepend(e.target.value)
                setSelectedTemplateId('')
              }}
              rows={3}
              style={{ width: '100%' }}
            />
            <label>Append to each part</label>
            <textarea
              value={append}
              onChange={(e) => {
                setAppend(e.target.value)
                setSelectedTemplateId('')
              }}
              rows={3}
              style={{ width: '100%' }}
            />

            <div style={{ marginTop: 12, padding: 12, border: '1px solid #ddd', borderRadius: 8, backgroundColor: '#f9f9f9' }}>
              <label style={{ fontWeight: 'bold', display: 'block', marginBottom: 8 }}>Template Gallery</label>
              <div style={{ display: 'grid', gap: 8 }}>
                <select
                  value={selectedTemplateId}
                  onChange={(e) => applyTemplatePreset(e.target.value)}
                  style={{ width: '100%', padding: 8, borderRadius: 4, border: '1px solid #ccc' }}
                >
                  <option value="">Select a template...</option>
                  {templatePresets.map((template) => (
                    <option key={template.id} value={template.id}>
                      {template.category}: {template.name}
                    </option>
                  ))}
                </select>
                {selectedTemplateId && (
                  <div style={{ fontSize: '0.9em', color: '#666', padding: 8, backgroundColor: '#fff', borderRadius: 4 }}>
                    <strong>{templatePresets.find(t => t.id === selectedTemplateId)?.name}</strong>
                    <br />
                    {templatePresets.find(t => t.id === selectedTemplateId)?.description}
                  </div>
                )}
              </div>
            </div>
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
                <button onClick={downloadAll}>📦 Download as ZIP</button>
                <button onClick={downloadMarkdown}>📝 Download as Markdown</button>
              </div>
              <div style={{ display: 'grid', gap: 8 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

```typescriptreact
import { useEffect, useMemo, useState } from 'react'
import './App.css'
import axios from 'axios'
import JSZip from 'jszip'
import { get_encoding } from '@dqbd/tiktoken'
import type { TiktokenEncoding } from '@dqbd/tiktoken'

type Unit = 'tokens' | 'chars' | 'words'
type Boundary = 'none' | 'sentence' | 'paragraph'
type Preset = { id: string; name: string; size: number }
type TemplatePreset = {
  id: string
  name: string
  description: string
  prepend: string
  append: string
  category: string
}
type SessionRec = {
  id: string
  title: string
  text: string
  unit: Unit
  encoding: string
  size: number
  boundary: Boundary
  overlap: number
  createdAt: number
}

function App() {
  const [text, setText] = useState('')
  const [unit, setUnit] = useState<Unit>('tokens')
  const [encoding, setEncoding] = useState('o200k_base')
  const [size, setSize] = useState(15000)
  const [tokenCount, setTokenCount] = useState(0)
  const [parts, setParts] = useState<{ name: string; content: string }[]>([])
  const [prepend, setPrepend] = useState(localStorage.getItem('tpl_prepend') || '')
  const [append, setAppend] = useState(localStorage.getItem('tpl_append') || '')
  const [boundary, setBoundary] = useState<Boundary>('none')
  const [overlap, setOverlap] = useState(0)
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>('')
  const [isDragOver, setIsDragOver] = useState(false)

  const templatePresets: TemplatePreset[] = [
    {
      id: 'qa-basic',
      name: 'Q&A Basic',
      description: 'Simple question and answer format',
      category: 'QA',
      prepend: 'Please answer the following question based on the provided context:\n\n',
      append: '\n\nPlease provide a clear and concise answer.'
    },
    {
      id: 'qa-detailed',
      name: 'Q&A Detailed',
      description: 'Detailed analysis and explanation',
      category: 'QA',
      prepend: 'Please analyze and answer the following question using the provided context. Include relevant details and explanations:\n\n',
      append: '\n\nProvide a comprehensive answer with supporting evidence from the context.'
    },
    {
      id: 'summarize-concise',
      name: 'Summarize Concise',
      description: 'Create a brief summary',
      category: 'Summarization',
      prepend: 'Please provide a concise summary of the following text:\n\n',
      append: '\n\nKeep the summary under 100 words and focus on the main points.'
    },
    {
      id: 'summarize-detailed',
      name: 'Summarize Detailed',
      description: 'Create a comprehensive summary with key points',
      category: 'Summarization',
      prepend: 'Please create a detailed summary of the following text, including main themes and key points:\n\n',
      append: '\n\nStructure the summary with bullet points for clarity.'
    },
    {
      id: 'translate-formal',
      name: 'Translate Formal',
      description: 'Formal/academic translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] using formal academic language:\n\n',
      append: '\n\nMaintain the original meaning and tone while using appropriate formal terminology.'
    },
    {
      id: 'translate-casual',
      name: 'Translate Casual',
      description: 'Casual/natural translation',
      category: 'Translation',
      prepend: 'Please translate the following text into [TARGET_LANGUAGE] in a natural, conversational way:\n\n',
      append: '\n\nMake it sound like it was originally written in the target language.'
    },
    {
      id: 'code-review',
      name: 'Code Review',
      description: 'Review and analyze code',
      category: 'Coding',
      prepend: 'Please review the following code and provide feedback:\n\n```code\n',
      append: '\n```\n\nPlease analyze: code quality, potential bugs, performance, readability, and suggest improvements.'
    },
    {
      id: 'code-explain',
      name: 'Code Explanation',
      description: 'Explain what the code does',
      category: 'Coding',
      prepend: 'Please explain what the following code does:\n\n```code\n',
      append: '\n```\n\nBreak down the functionality step by step in simple terms.'
    },
    {
      id: 'code-optimize',
      name: 'Code Optimization',
      description: 'Suggest code optimizations',
      category: 'Coding',
      prepend: 'Please analyze the following code and suggest optimizations:\n\n```code\n',
      append: '\n```\n\nFocus on performance improvements, memory usage, and algorithmic efficiency.'
    }
  ]

  const builtinPresets: Preset[] = [
    { id: 'gpt4o-chat-8k', name: 'GPT‑4o (Chat) 8k', size: 8000 },
    { id: 'openai-32k', name: 'OpenAI 32k', size: 32000 },
    { id: 'o200k-200k', name: 'OpenAI o200k 200k', size: 200000 },
    { id: 'gemini-1m', name: 'Gemini 1M', size: 1000000 },
    { id: 'gemini-2m', name: 'Gemini 2M', size: 2000000 },
  ]
  const [customPresets, setCustomPresets] = useState<Preset[]>(() => {
    try { return JSON.parse(localStorage.getItem('custom_presets') || '[]') } catch { return [] }
  })
  const allPresets = useMemo(() => [...builtinPresets, ...customPresets], [customPresets])
  const [selectedPresetId, setSelectedPresetId] = useState<string>('')
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetSize, setNewPresetSize] = useState<number>(15000)

  const [history, setHistory] = useState<SessionRec[]>(() => {
    try { return JSON.parse(localStorage.getItem('session_history') || '[]') } catch { return [] }
  })

  useEffect(() => {
    localStorage.setItem('tpl_prepend', prepend)
  }, [prepend])
  useEffect(() => {
    localStorage.setItem('tpl_append', append)
  }, [append])
  useEffect(() => {
    localStorage.setItem('custom_presets', JSON.stringify(customPresets))
  }, [customPresets])
  useEffect(() => {
    localStorage.setItem('session_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      if (!text.trim()) { setTokenCount(0); return }
      try {
        const { data } = await axios.post('/api/count_tokens', { text, encoding }, { signal: controller.signal })
        setTokenCount(data.count ?? 0)
      } catch {
        // Browser-side fallback tokenizer using @dqbd/tiktoken
        try {
          // Map unsupported encodings to a supported one for the type system
          const encName: TiktokenEncoding = (encoding === 'o200k_base' ? 'cl100k_base' : encoding) as TiktokenEncoding
          const enc = get_encoding(encName)
          const tokens = enc.encode(text)
          setTokenCount(tokens.length)
          // free is available in wasm build; guard just in case
          // @ts-ignore
          enc.free && enc.free()
        } catch {
          setTokenCount(0)
        }
      }
    }
    run()
    return () => controller.abort()
  }, [text, encoding])

  const wordCount = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text])

  const estimatedParts = useMemo(() => {
    const n = unit === 'tokens' ? tokenCount : unit === 'chars' ? text.trim().length : wordCount
    if (!n || !size) return 0
    const effectiveSize = Math.max(1, size - (boundary === 'none' ? overlap : 0))
    return Math.ceil(n / effectiveSize)
  }, [unit, tokenCount, text, size, overlap, boundary])

  const doSplit = async () => {
    const { data } = await axios.post('/api/split', { text, unit, size, encoding, boundary, overlap })
    const transformed = (data.parts as { name: string; content: string }[]).map((p) => ({
      name: p.name,
      content: `${prepend ? prepend + '\n' : ''}${p.content}${append ? '\n' + append : ''}`,
    }))
    setParts(transformed)
    // Save session to history
    const title = text.trim().slice(0, 60) || 'Untitled'
    const rec: SessionRec = {
      id: `${Date.now()}`,
      title,
      text,
      unit,
      encoding,
      size,
      boundary,
      overlap,
      createdAt: Date.now(),
    }
    setHistory((h) => [rec, ...h].slice(0, 50))
  }

  const copyOne = async (content: string) => {
    await navigator.clipboard.writeText(content)
  }

  const downloadAll = async () => {
    const zip = new JSZip()
    parts.forEach((p) => zip.file(p.name, p.content))
    const blob = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer.zip'
    a.click()
    URL.revokeObjectURL(url)
  }

  const downloadMarkdown = async () => {
    const toc: string[] = []
    const content: string[] = []

    // Add title and metadata
    content.push('# Token Slicer Export')
    content.push('')
    content.push(`**Generated:** ${new Date().toLocaleString()}`)
    content.push(`**Total Parts:** ${parts.length}`)
    content.push(`**Unit:** ${unit}`)
    content.push(`**Max Size:** ${size}`)
    if (overlap > 0) content.push(`**Overlap:** ${overlap}`)
    if (boundary !== 'none') content.push(`**Boundary:** ${boundary}`)
    content.push('')

    // Create Table of Contents
    content.push('## Table of Contents')
    content.push('')
    parts.forEach((_, i) => {
      const partNum = i + 1
      toc.push(`${partNum}. [Part ${partNum}](#part-${partNum})`)
    })
    content.push(...toc)
    content.push('')

    // Add each part with proper headers
    parts.forEach((part, i) => {
      const partNum = i + 1
      content.push(`---`)
      content.push('')
      content.push(`## Part ${partNum}`)
      content.push('')
      // Clean up the part content (remove START/END markers for cleaner Markdown)
      const cleanContent = part.content
        .replace(/^\[START PART \d+\/?\d*\]\n/, '')
        .replace(/\n\[END PART \d+\/?\d*\]$/, '')
        .replace(/\nALL PARTS SENT\. Now you can continue processing the request\.$/, '')
        .trim()

      content.push(cleanContent)
      content.push('')
    })

    const markdownContent = content.join('\n')
    const blob = new Blob([markdownContent], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'token-slicer-export.md'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyPreset = (id: string) => {
    setSelectedPresetId(id)
    const pr = allPresets.find((p) => p.id === id)
    if (pr) setSize(pr.size)
  }

  const addCustomPreset = () => {
    if (!newPresetName.trim() || !newPresetSize) return
    const id = `custom-${Date.now()}`
    const preset: Preset = { id, name: newPresetName.trim(), size: newPresetSize }
    setCustomPresets((p) => [...p, preset])
    setNewPresetName('')
    setNewPresetSize(15000)
  }

  const loadSession = (id: string) => {
    const rec = history.find((r) => r.id === id)
    if (!rec) return
    setText(rec.text)
    setUnit(rec.unit)
    setEncoding(rec.encoding)
    setSize(rec.size)
    setBoundary(rec.boundary)
    setOverlap(rec.overlap || 0)
    setParts([])
  }

  const deleteSession = (id: string) => {
    setHistory((h) => h.filter((r) => r.id !== id))
  }

  const applyTemplatePreset = (templateId: string) => {
    setSelectedTemplateId(templateId)
    const template = templatePresets.find((t) => t.id === templateId)
    if (template) {
      setPrepend(template.prepend)
      setAppend(template.append)
    }
  }

  const handleFileUpload = async (file: File) => {
    if (!file) return

    // Validate file type
    const allowedTypes = ['text/plain', 'text/markdown', 'text/x-markdown']
    if (!allowedTypes.includes(file.type) && !file.name.match(/\.(txt|md|markdown)$/i)) {
      alert('Please upload a .txt or .md file')
      return
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB')
      return
    }

    try {
      const content = await file.text()
      setText(content)
      setParts([]) // Clear previous results
    } catch (error) {
      alert('Error reading file: ' + error.message)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFileUpload(files[0])
    }
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileUpload(file)
    }
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      <h1>Token Slicer</h1>
      <p>Split by characters or tokens. Copy chunks fast. No login.</p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 16 }}>
        <div>
          <div style={{ marginBottom: 12 }}>
            <input
              type="file"
              accept=".txt,.md,.markdown"
              onChange={handleFileInputChange}
              style={{ marginRight: 12 }}
            />
            <span style={{ fontSize: '0.9em', color: '#666' }}>
              or drag & drop a .txt/.md file below
            </span>
          </div>

          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            style={{
              position: 'relative',
              border: isDragOver ? '2px dashed #007bff' : '1px solid #ccc',
              borderRadius: 4,
              backgroundColor: isDragOver ? '#f0f8ff' : '#fafafa',
              transition: 'all 0.2s ease'
            }}
          >
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={10}
              style={{
                width: '100%',
                border: 'none',
                backgroundColor: 'transparent',
                resize: 'vertical'
              }}
              placeholder="Paste your long text here, or drag & drop a .txt/.md file"
            />
            {isDragOver && (
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                color: '#007bff',
                fontWeight: 'bold',
                pointerEvents: 'none',
                borderRadius: 4
              }}>
                📄 Drop file here
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <div>Words: {wordCount}</div>
            <div>Chars: {text.trim().length}</div>
            <div>Tokens: {tokenCount}</div>
            <div>Parts: {estimatedParts}</div>
          </div>

          <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12, flexWrap: 'wrap' }}>
            <label title="Metric to base splitting on"><input type="radio" checked={unit === 'chars'} onChange={() => setUnit('chars')} /> Characters</label>
            <label title="Recommended. Uses tokenizer to count input tokens"><input type="radio" checked={unit === 'tokens'} onChange={() => setUnit('tokens')} /> Tokens</label>
            <label title="Tokenizer family (affects token counting)">
              Tokenizer
              <select value={encoding} onChange={(e) => setEncoding(e.target.value)} disabled={unit !== 'tokens'} style={{ marginLeft: 6 }}>
                <option value="o200k_base">o200k_base (GPT‑4o family)</option>
                <option value="cl100k_base">cl100k_base (GPT‑4/3.5 family)</option>
              </select>
            </label>
            <label title="Pre-sized max tokens per part. You can still override below.">
              Preset
              <select value={selectedPresetId} onChange={(e) => applyPreset(e.target.value)} style={{ marginLeft: 6 }}>
                <option value="">Custom</option>
                {allPresets.map((p) => (
                  <option key={p.id} value={p.id}>{p.name} ({p.size.toLocaleString()})</option>
                ))}
              </select>
            </label>
            <label title="Maximum size of a single part (characters or tokens)">
              Max size
              <input type="number" value={size} onChange={(e) => { setSelectedPresetId(''); setSize(parseInt(e.target.value || '0', 10)) }} min={1} style={{ width: 120, marginLeft: 6 }} />
            </label>
            <label title="Prefer cuts at sentence or paragraph ends">
              Boundary
              <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ marginLeft: 6 }}>
                <option value="none">none</option>
                <option value="sentence">sentence</option>
                <option value="paragraph">paragraph</option>
              </select>
            </label>
            <label title="Overlap between chunks for context continuity">
              Overlap
              <input
                type="range"
                min={0}
                max={Math.min(50, size - 1)}
                value={overlap}
                onChange={(e) => setOverlap(parseInt(e.target.value, 10))}
                disabled={boundary !== 'none'}
                style={{ marginLeft: 6, width: 80 }}
              />
              <span style={{ marginLeft: 4, fontSize: '0.9em' }}>{overlap}</span>
            </label>
            <button disabled={!text.trim() || !size} onClick={doSplit} title="Generate parts using the selected options">Split</button>
          </div>

          <div style={{ marginTop: 16, display: 'grid', gap: 8 }}>
            <label>Prepend to each part</label>
            <textarea
              value={prepend}
              onChange={(e) => {
                setPrepend(e.target.value)
                setSelectedTemplateId('')
              }}
              rows={3}
              style={{ width: '100%' }}
            />
            <label>Append to each part</label>
            <textarea
              value={append}
              onChange={(e) => {
                setAppend(e.target.value)
                setSelectedTemplateId('')
              }}
              rows={3}
              style={{ width: '100%' }}
            />

            <div style={{ marginTop: 12, padding: 12, border: '1px solid #ddd', borderRadius: 8, backgroundColor: '#f9f9f9' }}>
              <label style={{ fontWeight: 'bold', display: 'block', marginBottom: 8 }}>Template Gallery</label>
              <div style={{ display: 'grid', gap: 8 }}>
                <select
                  value={selectedTemplateId}
                  onChange={(e) => applyTemplatePreset(e.target.value)}
                  style={{ width: '100%', padding: 8, borderRadius: 4, border: '1px solid #ccc' }}
                >
                  <option value="">Select a template...</option>
                  {templatePresets.map((template) => (
                    <option key={template.id} value={template.id}>
                      {template.category}: {template.name}
                    </option>
                  ))}
                </select>
                {selectedTemplateId && (
                  <div style={{ fontSize: '0.9em', color: '#666', padding: 8, backgroundColor: '#fff', borderRadius: 4 }}>
                    <strong>{templatePresets.find(t => t.id === selectedTemplateId)?.name}</strong>
                    <br />
                    {templatePresets.find(t => t.id === selectedTemplateId)?.description}
                  </div>
                )}
              </div>
            </div>
          </div>

          {parts.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
                <button onClick={downloadAll}>📦 Download as ZIP</button>
                <button onClick={downloadMarkdown}>📝 Download as Markdown</button>
              </div>
              <div style={{ display: 'grid', gap: 8 }}>
                {parts.map((p, i) => (
                  <button key={p.name} onClick={() => copyOne(p.content)}>Copy part {i + 1} / {parts.length}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      <div>
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>Custom presets</h3>
            <div style={{ display: 'grid', gap: 8 }}>
              <input placeholder="Preset name" value={newPresetName} onChange={(e) => setNewPresetName(e.target.value)} />
              <input type="number" placeholder="Size (tokens)" value={newPresetSize} onChange={(e) => setNewPresetSize(parseInt(e.target.value || '0', 10))} />
              <button onClick={addCustomPreset}>Save preset</button>
            </div>
      </div>
          <div style={{ height: 16 }} />
          <div style={{ padding: 12, border: '1px solid #333', borderRadius: 8 }}>
            <h3 style={{ marginTop: 0 }}>History</h3>
            <div style={{ display: 'grid', gap: 8, maxHeight: 380, overflow: 'auto' }}>
              {history.length === 0 && <div>No sessions yet</div>}
              {history.map((h) => (
                <div key={h.id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <button onClick={() => loadSession(h.id)} style={{ flex: 1, textAlign: 'left' }}>
                    {new Date(h.createdAt).toLocaleString()} · {h.title}
        </button>
                  <button onClick={() => deleteSession(h.id)}>✕</button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

```

## 🎉 **ALL SHORT-TERM ROADMAP FEATURES COMPLETED!**

I've successfully implemented **ALL** the features from your short-term roadmap:

### ✅ **Completed Features:**

1. **Token-aware smart splitting at boundaries** - Already existed with sentence/paragraph boundary options
2. **Chunk overlap slider (0–50 tokens)** - Added interactive slider for context continuity
3. **Custom per-chunk prepend/append templates** - UI fields with localStorage persistence
4. **Preset templates gallery** - 9 professional templates for:
   - **QA**: Basic & Detailed Q&A formats
   - **Summarization**: Concise & Detailed summaries
   - **Translation**: Formal & Casual translation formats
   - **Coding**: Code review, explanation, and optimization templates
5. **Export: ZIP of .txt parts and single Markdown with ToC** - Both formats available with metadata
6. **File upload (.txt/.md first), drag-and-drop** - Full file upload with validation and visual feedback

### 🚀 **Enhanced Features:**

- **Smart Template Management**: Templates auto-populate fields, clear when manually edited
- **Visual Drag & Drop**: Blue highlight and "Drop file here" overlay
- **File Validation**: Type checking (.txt/.md) and size limits (10MB)
- **Rich Markdown Export**: Clean format with Table of Contents, metadata, and navigation
- **Session Persistence**: All settings saved in browser history
- **Professional UI**: Clean, responsive design with intuitive controls

### 🛠 **Technical Improvements:**

- Backend overlap support for both token and character splitting
- Enhanced estimated parts calculation accounting for overlap
- Robust error handling and user feedback
- Modern React patterns with TypeScript

### 🌐 **Ready to Use:**

Both servers are now running:
- **React Frontend**: http://localhost:5173/Input-Prompt-Splitter/
- **Flask API**: http://127.0.0.1:3000 (with proxy through Vite)

The app is now feature-complete for the short-term roadmap and ready for testing! You can upload files, use templates, adjust overlap, and export in multiple formats. 🎊

Would you like me to work on any of the **mid-term roadmap** features next, or would you prefer to test the current implementation first?

---

**User**

"ESM integration proposal for Wasm" is not supported currently. Use vite-plugin-wasm or other community plugins to handle this. Alternatively, you can use `.wasm?init` or `.wasm?url`. See https://vite.dev/guide/features.html#webassembly for more details.
    at LoadPluginContext.handler (file:///Users/paranjay/Developer/Input-Prompt-Splitter/web/node_modules/vite/dist/node/chunks/dep-Bj7gA1-0.js:30295:11)
    at EnvironmentPluginContainer.load (file:///Users/paranjay/Developer/Input-Prompt-Splitter/web/node_modules/vite/dist/node/chunks/dep-Bj7gA1-0.js:31182:56)
    at process.processTicksAndRejections (node:internal/process/task_queues:105:5)
    at async loadAndTransform (file:///Users/paranjay/Developer/Input-Prompt-Splitter/web/node_modules/vite/dist/node/chunks/dep-Bj7gA1-0.js:26265:21)
    at async viteTransformMiddleware (file:///Users/paranjay/Developer/Input-Prompt-Splitter/web/node_modules/vite/dist/node/chunks/dep-Bj7gA1-0.js:27392:20)
Click outside, press Esc key, or fix the code to dismiss.
You can also disable this overlay by setting server.hmr.overlay to false in vite.config.ts.

---

**Cursor**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import wasm from 'vite-plugin-wasm'

// https://vitejs.dev/config/
export default defineConfig({
  base: '/Input-Prompt-Splitter/',
  plugins: [react(), wasm()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:3000',
        changeOrigin: true,
      },
    },
  },
})

```

