# Use a pipeline as a high-level helper
# Load model directly
import torch
from transformers import BloomForCausalLM,BloomTokenizerFast,StoppingCriteria, StoppingCriteriaList
import time

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
      super().__init__()
      self.stops = stops
      self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,stops = []):
       stop_count = 0
       for stop in self.stops:
              stop_count = (stop == input_ids[0]).sum().item()
       if stop_count >= self.ENCOUNTERS:
          return True
       return False




tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m",padding_side="left")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")
stop_words_ids = []

stop_word_1 = tokenizer.encode('.',return_tensors='pt')
stop_word_2 = tokenizer.encode('.\n',return_tensors='pt')
stop_words_ids.append(stop_word_1)
stop_words_ids.append(stop_word_2)
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids,encounters=1)])


def generateText(inputText):
    start_time=time.time()
    input_ids = tokenizer.encode(inputText, return_tensors='pt')
    outputs = model.generate(input_ids,max_new_tokens=50,stopping_criteria=stopping_criteria, do_sample = True,num_beams=3)
    response = tokenizer.decode(outputs[0],skip_special_tokens=True)
    end_time=time.time()
    print('-------------------RESPONSE GENERATED-------------------')
    print(response) 
    print('-------------------RESPONSE GENERATED-------------------')
    print('-------------------TIME SPENT-------------------')    
    print(end_time-start_time)
    print('-------------------TIME SPENT-------------------')
    return response
    

