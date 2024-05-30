config_Bloom = False;
config_Dolphing = True;
full_text = True;

if(config_Bloom):from LLM.BLOOM import BLOOM_LLM
if(config_Dolphing):from LLM.TinyDolphin import TINY_DOLPHIN_LLM


full_chat_dolphin = ''
full_chat_bloom = ''
chat = ''

def update_chat(message):
    global chat
    chat = chat + 'user:\n' + message + '\n'
    if(config_Bloom):
        if(full_text):
            chat_message = chatting_with_bloom_full_chat(message)
        else:
            chat_message = chatting_with_bloom_message(message)
    else:
        if(full_text):
            chat_message = chatting_with_dolphin_full_chat(message)
        else:
            chat_message = chatting_with_dolphin(message)
        
        
    chat = chat + 'assistant:\n' + chat_message + '\n'

    print("-------------------ACTUAL CHAT-------------------")
    print(chat)
    print("-------------------FINAL  CHAT-------------------")
    return chat_message


def chatting_with_bloom_full_chat(message):
    global full_chat_bloom
    
    if(full_chat_bloom == ''):
        full_chat_bloom = message + '\n'
    else:
        full_chat_bloom = full_chat_bloom + message + '\n'
    response = get_response_from_BLOOM(full_chat_bloom)
    response = response[len(full_chat_bloom):]
    full_chat_bloom = full_chat_bloom + response + '\n'
    print("-------------------BLOOM CHAT-------------------")
    print(full_chat_bloom)
    print("-------------------BLOOM CHAT-------------------")
    
    return response


def chatting_with_bloom_message(message):
    response = get_response_from_BLOOM(message)
    response = response[len(message):]
    return response


def get_response_from_BLOOM(message):
    return BLOOM_LLM.generateText(message)


def chatting_with_dolphin_full_chat(message):
    global full_chat_dolphin
    user_prompt = """<|im_start|>user\n"""+message+"""<|im_end|>\n<|im_start|assistant\n"""
    full_chat_dolphin = full_chat_dolphin + user_prompt
    user_promt_length = len('<|im_start|assistant\n')
    #PRIMERA INSTANCIA
    #system_prompt + user_promtp + response generated
    response = get_response_from_Dolphin(full_chat_dolphin)
    index = response.rfind('<|im_start|assistant\n')
    #response = response[system_prompt_length + user_prompt_length + 3:]
    response = response[index + user_promt_length:]
    index = response.rfind('.')
    if(index > 0):
        response = response[:index+1]
    else:
        index = response.rfind('\n')
        if(index > 0):
            response = response[:index+1]
        else:
            index = response.rfin('?')
            if(index > 0):
                response = response[:index+1]
        
    full_chat_dolphin = full_chat_dolphin + response
    print("-------------------DOLPHIN CHAT-------------------")
    print(full_chat_dolphin)
    print("-------------------DOLPHIN CHAT-------------------")
    return response


def chatting_with_dolphin(message):
    return message
def get_response_from_Dolphin(message):
    return TINY_DOLPHIN_LLM.generate_response(message)