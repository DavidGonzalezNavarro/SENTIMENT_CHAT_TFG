from transformers import pipeline

model_path = "daveni/twitter-xlm-roberta-emotion-es"
emotion_analysis = pipeline("text-classification", framework="pt", model=model_path, tokenizer=model_path)

mode = '0'
match mode:
    case 'joy':
        #Joy
        texto1 = 'Que alegre estoy!' #FACIL
        texto2 = 'Es increible que me hayan dado la beca!' #Intermedio
        texto3 = 'Por fin se acabó.' #dificl
        joy = emotion_analysis(texto1)
        joy2 = emotion_analysis(texto2)
        joy3 = emotion_analysis(texto3)
        print('----------JOY----------')
        print('----------EASY----------')
        print(joy)
        print('----------EASY----------')
        print('----------MID----------')
        print(joy2)
        print('----------MID----------')
        print('----------HARD----------')
        print(joy3)
        print('----------HARD----------')
    case 'sadness':
        #Sadness
        texto1 = 'Estoy triste.'
        texto2 = 'Llorar lava el alma, lo difícil es secar el corazón'
        texto3 = 'La vida es placentera. La muerte es pacífica. Es la transición la que es problemática'
        sadness = emotion_analysis(texto1)
        sadness2 = emotion_analysis(texto2)
        sadness3 = emotion_analysis(texto3)
        print('----------SADNESS----------')
        print('----------EASY----------')
        print(sadness)
        print('----------EASY----------')
        print('----------MID----------')
        print(sadness2)
        print('----------MID----------')
        print('----------HARD----------')
        print(sadness3)
        print('----------HARD----------')
    case 'anger':
        #Anger
        texto1 = '¡Estoy furioso!'
        texto2 = 'Me sacas de quicio'
        texto3 = 'Algún día se arrepentirán de lo que han hecho, mientras tanto dejaré que se rían'
        anger = emotion_analysis(texto1)
        anger2 = emotion_analysis(texto2)
        anger3 = emotion_analysis(texto3)
        print('----------ANGER----------')
        print('----------EASY----------')
        print(anger)
        print('----------EASY----------')
        print('----------MID----------')
        print(anger2)
        print('----------MID----------')
        print('----------HARD----------')
        print(anger3)
        print('----------HARD----------')
    case 'surprise':
        #Surprise
        texto1 = 'No me lo esperaba.'
        texto2 = '¿Pero qué me cuentas?'
        texto3 = '¿Hola?'
        surprise = emotion_analysis(texto1)
        surprise2 = emotion_analysis(texto2)
        surprise3 = emotion_analysis(texto3)
        print('----------SURPRISE----------')
        print('----------EASY----------')
        print(surprise)
        print('----------EASY----------')
        print('----------MID----------')
        print(surprise2)
        print('----------MID----------')
        print('----------HARD----------')
        print(surprise3)
        print('----------HARD----------')
    case 'disgust':
        #Disgust
        texto1 = '¡Que asco!'
        texto2 = 'Deja de comer eso, me da grima.'
        texto3 = 'Cuando veo una cucaracha me da repelús y no puedo estar cerca'
        disgust = emotion_analysis(texto1)
        disgust2 = emotion_analysis(texto2)
        disgust3 = emotion_analysis(texto3)
        print('----------DISGUST----------')
        print('----------EASY----------')
        print(disgust)
        print('----------EASY----------')
        print('----------MID----------')
        print(disgust2)
        print('----------MID----------')
        print('----------HARD----------')
        print(disgust3)
        print('----------HARD----------')
    case 'fear':
        #Fear
        texto1 = 'Tengo miedo'
        texto2 = 'He visto moverse algo en la oscuridad.'
        texto3 = 'No quiero hablar en público, me quedo paralizado.'
        fear = emotion_analysis(texto1)
        fear1 = emotion_analysis(texto2)
        fear2 = emotion_analysis(texto3)
        print('----------FEAR----------')
        print('----------EASY----------')
        print(fear)
        print('----------EASY----------')
        print('----------MID----------')
        print(fear1)
        print('----------MID----------')
        print('----------HARD----------')
        print(fear2)
        print('----------HARD----------')

