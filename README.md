# O que é uma API
Uma **API** (Application Programming Interface, ou Interface de Programação de Aplicações) é um conjunto de definições e protocolos que permitem que diferentes sistemas ou aplicativos se comuniquem entre si. Em outras palavras, uma API define as regras e métodos pelos quais um software pode solicitar serviços de outro, facilitando a integração entre sistemas.

1. **Intermediário de Comunicação:** A API age como um intermediário, permitindo que um software interaja com outro sem que eles precisem conhecer os detalhes internos de como o outro funciona.
2. **Padronização:** APIs estabelecem um conjunto de regras e funções que os desenvolvedores devem seguir ao integrar seus sistemas. Isso garante que a comunicação seja eficiente e previsível.
3. **Uso Comum:** Elas são amplamente usadas para conectar aplicativos a serviços externos, como ao fazer uma solicitação para um servidor, acessar um banco de dados ou se comunicar com um serviço online (como APIs de redes sociais, APIs de pagamento, etc.).

**Exemplo de uso:** Se um aplicativo quer exibir o clima atual de uma cidade, ele pode usar a API de um serviço meteorológico, que fornece os dados em tempo real para serem exibidos no aplicativo.

# Visão Geral

A classificação de texto é o problema de atribuir um rótulo predefinido a um texto. Por exemplo, se alguém escreveu uma resenha de um filme dizendo "Eu gostei muito do filme, é fantástico!", queremos rotular a resenha como *positiva* ou *negativa*. Este é um problema de classificação de texto (text classification) chamado análise de sentimentos.

```
!pip install praw nltk matplotlib

import praw
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt

# Baixar dados necessários para o VADER
nltk.download('vader_lexicon')

# Configuração da API do Reddit/Conectando-se à API do Reddit
reddit = praw.Reddit(client_id='abc123',  #coloque aqui o seu id
                     client_secret='1234567890abcdef1234567890abcdef',  # coloque aqui o seu secret
                     user_agent='my_user_agent')  # coloque aqui o seu user (nickname)

# Obter um post específico por ID
post_id = 'xyz123'  # Substitua pelo ID do post desejado
post = reddit.submission(id=post_id)

# Inicialize o analisador de sentimentos VADER
sia = SentimentIntensityAnalyzer()

# Contadores para os sentimentos
positivos = 0
neutros = 0
negativos = 0

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
plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Distribuição dos Sentimentos nos Comentários')
plt.show()
```

Visualizando a análise de sentimentos do post, id=1fc0h1q, 'AITA for refusing to buy my daughter another phone and "ruining her life"' graficamente:

![aita_img](https://github.com/user-attachments/assets/d1e45952-48e1-4192-9fb5-182694d9480c)

# Português vs Inglês

O **VADER** foi originalmente projetado para análise de sentimentos em textos em inglês e, portanto, não é adequado para outras línguas.

```
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
text = "I hate it"
classifier(text)
```
```
[{'label': 'NEGATIVE', 'score': 0.9996398687362671}]
```
Como não fornecemos um modelo para pipeline(), ele usará um modelo padrão. Nesse caso, é o *distilbert-base-uncased-finetuned-sst-2-english*.

```
classifier = pipeline("sentiment-analysis", model="neuralmind/bert-base-portuguese-cased")
text = "Eu odeio isso"
classifier(text)
```
```
[{'label': 'LABEL_0', 'score': 0.5079694390296936}]
```
Aqui, especificamos o modelo. LABEL_0 é para negativo e LABEL_1 é para positivo.

Podemos ver que apesar de ser uma sentença bem simples e sem ambiguidade, o modelo dá um score de 0.5, o que indica, numa primeira análise, que ele simplesmente não é muito bom.

## Coleta de Dados
* **API do Reddit:** Utilize a API do Reddit para coletar dados, como comentários e posts. Você pode usar o pacote *PRAW* (Python Reddit API Wrapper) para acessar a API do Reddit.
* **Subreddits e Posts**: Escolha os subreddits ou posts específicos dos quais você deseja extrair os dados. Use o PRAW para filtrar por tópicos, palavras-chave ou período de tempo.
```
# Escolha um subreddit e obtenha os posts mais populares da semana
subreddit = reddit.subreddit('relacionamentos')
posts = subreddit.top('week', limit=1)
```

## Pré-Processamento dos Dados
* **Limpeza de Texto:** Remova emojis, URLs, menções, e caracteres especiais dos comentários.
* **Tokenização:** Divida o texto em palavras ou frases (tokens).

## Análise de Sentimentos
* **Modelos Pré-Treinados:** Utilize modelos pré-treinados como *VADER* (do pacote *nltk*) para análise de sentimentos ou bibliotecas como *TextBlob* ou *Transformers* da *Hugging Face*.
* **Classificação:** O modelo classificará o sentimento como positivo, negativo ou neutro com base no texto processado.

## Análise e Visualização
* **Agregação dos Resultados:** Depois de obter os scores de sentimento para cada comentário, você pode agregá-los para entender o sentimento geral de um tópico ou subreddit.
* **Visualização:** Use bibliotecas como matplotlib ou seaborn para criar gráficos que mostram a distribuição dos sentimentos.

## Interpretação dos Resultados
* **Insights:** Analise os resultados para identificar padrões e tirar conclusões sobre o sentimento predominante em relação ao tópico estudado.
