import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("./tests")
tokenizer = AutoTokenizer.from_pretrained("./tests")

# Ensure the model is in evaluation mode
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def generate_answer(question):
    inputs = tokenizer(question, return_tensors="pt", max_length=512, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=512, num_beams=4, early_stopping=True)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


