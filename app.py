from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load AI-Thalli (LLaMA model from Hugging Face)
MODEL_NAME = "Ai-Thalli/Ai-Thalli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('user_message')
def handle_message(data):
    user_input = data['message']
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, top_p=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    emit('bot_response', {'message': response})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)