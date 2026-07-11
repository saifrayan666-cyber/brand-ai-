import os
import json
import time
from flask import Flask, render_template, request, Response, stream_with_context
from openai import OpenAI
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# OpenAI client initialized with API key from environment
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    messages = data.get('messages', [])
    model = data.get('model', 'gpt-3.5-turbo')
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 1000)

    if not messages:
        return {"error": "No messages"}, 400

    def generate():
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'content': content})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
