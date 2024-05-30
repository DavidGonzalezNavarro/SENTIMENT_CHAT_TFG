from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    GenerationConfig,
    TextIteratorStreamer,
    StoppingCriteria, 
    StoppingCriteriaList
)
import torch
import time





new_model= "NickyNicky/Mixtral-4x1.1B-TinyDolphin-2.8-1.1b_oasst2_chatML_Cluster"
model = AutoModelForCausalLM.from_pretrained(new_model,
                                             device_map="auto",
                                             trust_remote_code=True,
                                             torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage= True,
                                            #  use_flash_attention_2=False,
                                             )


tokenizer = AutoTokenizer.from_pretrained(new_model,
                                          max_length=1024,
                                          trust_remote_code=True,
                                          use_fast = True,
                                          )

tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = 'left'
#tokenizer.padding_side = 'right'

prompt = """<|im_start|>system
You are a person that really like to chat with people.<|im_end|>"""



def get_prompt():
    global prompt 
    return prompt

def set_prompt(newPrompt):
    global prompt 
    prompt = newPrompt


def generate_response(message):
    start_time=time.time()
    final_prompt = prompt + message
    inputs = tokenizer.encode(final_prompt,
                            return_tensors="pt",
                            add_special_tokens=True).cuda()#.to("cuda") # False # True
    generation_config = GenerationConfig(
                max_new_tokens=50,
                temperature=0.5,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1, #1.1, # 1.0 means no penalty, > 1.0 means penalty, 1.2 from CTRL paper
                do_sample=True,
                pad_token_id=tokenizer.unk_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
    outputs = model.generate(
                            generation_config=generation_config,
                            input_ids=inputs,
                            )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    end_time=time.time()
    print('-------------------RESPONSE GENERATED-------------------')
    print(response)
    print('-------------------RESPONSE GENERATED-------------------')
    print('-------------------TIME SPENT-------------------')    
    print(end_time-start_time)
    print('-------------------TIME SPENT-------------------')
    return response
