import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI
from pydantic import BaseModel

# Load the fine-tuned model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("./tests")
tokenizer = AutoTokenizer.from_pretrained("./tests")

# Ensure the model is in evaluation mode
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define FastAPI app
app = FastAPI()

# Define request body structure
class QuestionRequest(BaseModel):
    question: str

# Function to generate the answer from the question
def generate_answer(question: str) -> str:
    inputs = tokenizer(question, return_tensors="pt", max_length=512, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=512, num_beams=4, early_stopping=True)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Q&A API"}

# Define the question-answering endpoint
@app.post("/generate-answer/")
def get_answer(request: QuestionRequest):
    answer = generate_answer(request.question)
    return {"question": request.question, "answer": answer}
