# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# Use a pipeline as a high-level helper
from transformers import pipeline,BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

pipe = pipeline("text-generation", model="tiiuae/falcon-7b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b",offload_folder="offload", quantization_config=quantization_config)

print(tokenizer.default_chat_template);

pipeline = transformers.pipeline(
    "text-generation",
    model="tiiuae/falcon-7b",
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
   "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    truncation=True
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")