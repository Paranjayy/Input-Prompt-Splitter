from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import random
import redis
import string
import math

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
            
def split_prompt(text, split_length):
    if split_length <= 0:
        raise ValueError("Max length must be greater than 0.")

    num_parts = -(-len(text) // split_length)
    file_data = []

    for i in range(num_parts):
        start = i * split_length
        end = min((i + 1) * split_length, len(text))

        if i == num_parts - 1:
            content = f'[START PART {i + 1}/{num_parts}]\n' + text[start:end] + f'\n[END PART {i + 1}/{num_parts}]'
            content += '\nALL PARTS SENT. Now you can continue processing the request.'
        else:
            content = f'Do not answer yet. This is just another part of the text I want to send you. Just receive and acknowledge as "Part {i + 1}/{num_parts} received" and wait for the next part.\n[START PART {i + 1}/{num_parts}]\n' + text[start:end] + f'\n[END PART {i + 1}/{num_parts}]'
            content += f'\nRemember not answering yet. Just acknowledge you received this part with the message "Part {i + 1}/{num_parts} received" and wait for the next part.'

        file_data.append({
            'name': f'split_{str(i + 1).zfill(3)}_of_{str(num_parts).zfill(3)}.txt',
            'content': content
        })

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

def generate_random_hash(length):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 3000)))