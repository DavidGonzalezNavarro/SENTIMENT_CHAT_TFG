from pysentimiento import create_analyzer
analyzer = create_analyzer(task="emotion", lang="es")



mode = 'fear'
match mode:
    case 'joy':
        #Joy
        texto1 = 'Que alegre estoy!' #FACIL
        texto2 = 'Es increible que me hayan dado la beca!' #Intermedio
        texto3 = 'Por fin se acabó.' #dificl
        joy = analyzer.predict(texto1)
        joy2 = analyzer.predict(texto2)
        joy3 = analyzer.predict(texto3)
        print('----------JOY----------')
        print('----------EASY----------')
        print(joy.output)
        print('----------EASY----------')
        print('----------MID----------')
        print(joy2.output)
        print('----------MID----------')
        print('----------HARD----------')
        print(joy3.output)
        print('----------HARD----------')
    case 'sadness':
        #Sadness
        texto1 = 'Estoy triste.'
        texto2 = 'Llorar lava el alma, lo difícil es secar el corazón'
        texto3 = 'La vida es placentera. La muerte es pacífica. Es la transición la que es problemática'
        sadness = analyzer.predict(texto1)
        sadness2 = analyzer.predict(texto2)
        sadness3 = analyzer.predict(texto3)
        print('----------SADNESS----------')
        print('----------EASY----------')
        print(sadness.output)
        print('----------EASY----------')
        print('----------MID----------')
        print(sadness2.output)
        print('----------MID----------')
        print('----------HARD----------')
        print(sadness3.output)
        print('----------HARD----------')
    case 'anger':
        #Anger
        texto1 = '¡Estoy furioso!'
        texto2 = 'Me sacas de quicio'
        texto3 = 'Algún día se arrepentirán de lo que han hecho, mientras tanto dejaré que se rían'
        anger = analyzer.predict(texto1)
        anger2 = analyzer.predict(texto2)
        anger3 = analyzer.predict(texto3)
        print('----------ANGER----------')
        print('----------EASY----------')
        print(anger.output)
        print('----------EASY----------')
        print('----------MID----------')
        print(anger2.output)
        print('----------MID----------')
        print('----------HARD----------')
        print(anger3.output)
        print('----------HARD----------')
    case 'surprise':
        #Surprise
        texto1 = 'No me lo esperaba.'
        texto2 = '¿Pero qué me cuentas?'
        texto3 = '¿Hola?'
        surprise = analyzer.predict(texto1)
        surprise2 = analyzer.predict(texto2)
        surprise3 = analyzer.predict(texto3)
        print('----------SURPRISE----------')
        print('----------EASY----------')
        print(surprise.output)
        print('----------EASY----------')
        print('----------MID----------')
        print(surprise2.output)
        print('----------MID----------')
        print('----------HARD----------')
        print(surprise3.output)
        print('----------HARD----------')
    case 'disgust':
        #Disgust
        texto1 = '¡Que asco!'
        texto2 = 'Deja de comer eso, me da grima.'
        texto3 = 'Cuando veo una cucaracha me da repelús y no puedo estar cerca'
        disgust = analyzer.predict(texto1)
        disgust2 = analyzer.predict(texto2)
        disgust3 = analyzer.predict(texto3)
        print('----------DISGUST----------')
        print('----------EASY----------')
        print(disgust.output)
        print('----------EASY----------')
        print('----------MID----------')
        print(disgust2.output)
        print('----------MID----------')
        print('----------HARD----------')
        print(disgust3.output)
        print('----------HARD----------')
    case 'fear':
        #Fear
        texto1 = 'Tengo miedo'
        texto2 = 'He visto moverse algo en la oscuridad.'
        texto3 = 'No quiero hablar en público, me quedo paralizado.'
        fear = analyzer.predict(texto1)
        fear1 = analyzer.predict(texto2)
        fear2 = analyzer.predict(texto3)
        print('----------FEAR----------')
        print('----------EASY----------')
        print(fear.output)
        print('----------EASY----------')
        print('----------MID----------')
        print(fear1.output)
        print('----------MID----------')
        print('----------HARD----------')
        print(fear2.output)
        print('----------HARD----------')