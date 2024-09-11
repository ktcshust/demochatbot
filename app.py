from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from accelerate import Accelerator

app = Flask(__name__)

# Initialize the accelerator
accelerator = Accelerator()

# Load the fine-tuned model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("./tests")
tokenizer = AutoTokenizer.from_pretrained("./tests")

# Move model to device
device = accelerator.device
model.to(device)

# Prepare model with accelerate
model = accelerator.prepare(model)

def generate_answer(question):
    inputs = tokenizer(question, return_tensors="pt", max_length=512, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=512, num_beams=4, early_stopping=True)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    if question:
        answer = generate_answer(question)
        return jsonify({'answer': answer})
    return jsonify({'error': 'No question provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
