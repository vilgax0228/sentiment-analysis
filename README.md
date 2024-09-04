# Análise de sentimentos

A classificação de texto é o problema de atribuir um rótulo predefinido a um texto. Por exemplo, se alguém escreveu uma resenha de um filme dizendo "Eu gostei muito do filme, é fantástico!", queremos rotular a resenha como *positiva* ou *negativa*. Este é um problema de classificação de texto (text classification) chamado análise de sentimentos.

'''
!pip install praw nltk matplotlib

import praw
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt

# Baixar dados necessários para o VADER
nltk.download('vader_lexicon')

# Configuração da API do Reddit
reddit = praw.Reddit(client_id='1a2b3c',  #coloque aqui o seu id
                     client_secret='r1s2t3',  # coloque aqui o seu secret
                     user_agent='carmen')  # coloque aqui o seu user

# Escolha um subreddit e obtenha os posts mais populares da semana
subreddit = reddit.subreddit('TheBoys')
posts = subreddit.top('week', limit=1)

# Inicialize o analisador de sentimentos VADER
sia = SentimentIntensityAnalyzer()

# Contadores para os sentimentos
positivos = 0
neutros = 0
negativos = 0

for post in posts:
    print(f"\nTítulo do Post: {post.title}")
    post.comments.replace_more(limit=0)
    for comment in post.comments.list():
        sentiment = sia.polarity_scores(comment.body)
        print(f"Comentário: {comment.body}")
        print(f"Sentimento: {sentiment}\n")
        # Categoriza os sentimentos
        if sentiment['compound'] >= 0.05:
            positivos += 1
        elif sentiment['compound'] <= -0.05:
            negativos += 1
        else:
            neutros += 1

# Dados para o gráfico de pizza
labels = ['Positivos', 'Neutros', 'Negativos']
sizes = [positivos, neutros, negativos]
colors = ['green', 'gray', 'red']
explode = (0.1, 0, 0)  # Destacar a fatia "Positivos"

# Criando o gráfico de pizza
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Distribuição dos Sentimentos nos Comentários')
plt.show()
'''

Pegando um post qualquer do r/TheBoys e olhando o gráfico

![rTheBoys](https://github.com/user-attachments/assets/190908c1-b453-4636-8374-caf4c3cea092)


## Português vs Inglês

O Vader é treinado para lidar com inglês(?)

'''
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
text = "I hate it"
classifier(text)
'''
'''
[{'label': 'NEGATIVE', 'score': 0.9996398687362671}]
'''
Como não fornecemos um modelo para pipeline(), ele usará um modelo padrão. Nesse caso, é o distilbert-base-uncased-finetuned-sst-2-english.

'''
classifier = pipeline("sentiment-analysis", model="neuralmind/bert-base-portuguese-cased")
text = "Eu odeio isso"
classifier(text)
'''
'''
[{'label': 'LABEL_0', 'score': 0.5079694390296936}]
'''
Aqui, especificamos o modelo. LABEL_0 é negativo e LABEL_1 é positivo.

Podemos ver que apesar de ser uma sentença bem simples e sem ambiguidade, o modelo dá um score de 0.5, o que indica, numa primeira análise, que ele simplesmente não é muito bom.

