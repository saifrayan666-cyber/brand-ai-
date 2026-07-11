import os
import json
import logging
from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from openai import OpenAI
from flask_cors import CORS
from dotenv import load_dotenv

# লগিং সেটআপ
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

# API Key চেক
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.error("❌ OPENAI_API_KEY not found in environment variables!")
else:
    logger.info(f"✅ API Key found: {api_key[:10]}...")

# OpenAI ক্লায়েন্ট
client = OpenAI(api_key=api_key) if api_key else None

@app.route('/')
def index():
    logger.info("📄 Serving index page")
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test():
    """হেলথ চেক এন্ডপয়েন্ট"""
    return jsonify({
        "status": "ok",
        "api_key_set": bool(api_key),
        "api_key_preview": api_key[:10] + "..." if api_key else "None"
    })

@app.route('/chat', methods=['POST'])
def chat():
    logger.info("📨 Chat request received")
    
    if not client:
        logger.error("❌ No OpenAI client available")
        return jsonify({"error": "OpenAI API key not configured"}), 500
    
    try:
        data = request.get_json()
        logger.debug(f"📦 Request data: {data}")
        
        messages = data.get('messages', [])
        if not messages:
            logger.warning("⚠️ No messages in request")
            return jsonify({"error": "No messages"}), 400
        
        logger.info(f"📝 Processing {len(messages)} messages")
        logger.debug(f"Messages: {json.dumps(messages, indent=2)}")
        
        def generate():
            try:
                logger.info("🔄 Calling OpenAI API...")
                stream = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=500
                )
                
                logger.info("✅ Stream started")
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        logger.debug(f"📤 Sending chunk: {content[:20]}...")
                        yield f"data: {json.dumps({'content': content})}\n\n"
                
                logger.info("✅ Stream complete")
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                logger.error(f"❌ OpenAI API Error: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(stream_with_context(generate()), 
                       mimetype='text/event-stream',
                       headers={'Cache-Control': 'no-cache'})
        
    except Exception as e:
        logger.error(f"❌ Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    if not client:
        return jsonify({"error": "API key not configured"}), 500
    
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({"error": "No image provided"}), 400
        
        logger.info("🖼️ Analyzing image...")
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail in Bangla and English."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            max_tokens=300
        )
        
        analysis = response.choices[0].message.content
        logger.info("✅ Image analysis complete")
        return jsonify({"analysis": analysis})
        
    except Exception as e:
        logger.error(f"❌ Image analysis error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting server on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)
