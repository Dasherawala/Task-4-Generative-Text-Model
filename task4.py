!pip install -q torch transformers gradio
# Import libraries
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

def generate_paragraph(topic: str, length: int = 150) -> str:
    """Generates a coherent paragraph about the given topic"""
    prompt = f"Write a detailed paragraph about {topic}:\n\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=length,
        num_beams=5,
        no_repeat_ngram_size=3,
        early_stopping=True,
        temperature=0.7,
        do_sample=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text.strip()

# Create and launch interface
gr.Interface(
    fn=generate_paragraph,
    inputs=[
        gr.Textbox(label="Topic", placeholder="e.g. quantum computing"),
        gr.Slider(50, 300, value=150, label="Output Length (tokens)")
    ],
    outputs=gr.Textbox(label="Generated Paragraph 📝"),
    title="GENERATIVE TEXT MODEL (GPT🤖)",
    description="Enter any topic to generate a coherent paragraph 🖊️",
    examples=[["Neural networks"], ["Climate Change"], ["Ancient Roman History"]]
).launch()
