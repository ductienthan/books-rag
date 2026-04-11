from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "mlx-community/gemma-4-27B-A4B-it-4bit"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)
