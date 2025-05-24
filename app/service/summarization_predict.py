from transformers import BartTokenizer, BartForConditionalGeneration
from pathlib    import Path

artifacts = Path(__file__).resolve().parents[1] / 'artifacts' / 'summarization' / 'model'
tokenizer = BartTokenizer.from_pretrained(artifacts)
model = BartForConditionalGeneration.from_pretrained(artifacts)

def predict_tickets(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    summary_ids = model.generate(
        **inputs,
        num_beams=4,
        length_penalty=2.0,
        max_length=150,
        min_length=40,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)