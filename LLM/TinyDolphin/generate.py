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

from gstop import GenerationStopper


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
      super().__init__()
      self.stops = stops
      self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
       stop_count = 0
       for stop in self.stops:
              stop_count = (stop == input_ids[0]).sum().item()
       if stop_count >= self.ENCOUNTERS:
          return True
       return False



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
                                          padding_side="left"
                                          )

tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = 'left'
#tokenizer.padding_side = 'right'

stop_words_ids = []
stop_word_1 = tokenizer.encode('.',return_tensors='pt')
stop_word_2 = tokenizer.encode('\n',return_tensors='pt')
stop_words_ids.append(stop_word_1)
stop_words_ids.append(stop_word_2)
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids,encounters=1)])




prompt = """<|im_start|>system
You are a person that really like to chat with people.<|im_end|>"""
message = "Hola, que tal?"
real_message ="""<|im_start|>user\n"""+message+"""<|im_end|>\n<|im_start|assistant\n"""

generate_response(real_message)

def get_prompt():
    global prompt 
    return prompt

def set_prompt(newPrompt):
    global prompt 
    prompt = newPrompt

def generate_response_test(message):
    print('-------------------PROMT GENERATED----------------------')
    print (message)
    print('-------------------PROMT GENERATED----------------------')
    global prompt
    prompt = prompt + message
    return prompt


