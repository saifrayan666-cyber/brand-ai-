import os
import json
import logging
from flask import Flask, render_template, request, Response, jsonify
from openai import OpenAI
from flask_cors import CORS

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

api_key = os.environ.get("OPENAI_API_KEY")
logger.info(f"API Key found: {bool(api_key)}")

client = OpenAI(api_key=api_key) if api_key else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return jsonify({
        "status": "ok",
        "api_key_set": bool(api_key)
    })

@app.route('/chat', methods=['POST'])
def chat():
    if not client:
        return jsonify({"error": "API key not configured"}), 500
    
    try:
        data = request.get_json()
        logger.info(f"📨 Received data: {data}")
        
        messages = data.get('messages', [])
        if not messages:
            return jsonify({"error": "No messages"}), 400
        
        logger.info(f"📝 Processing {len(messages)} messages")
        
        def generate():
            try:
                stream = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=500
                )
                
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        yield f"data: {json.dumps({'content': content})}\n\n"
                
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ OpenAI Error: {error_msg}")
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        logger.error(f"❌ Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
