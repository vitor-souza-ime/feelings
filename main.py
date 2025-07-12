# Instalar dependências
!pip install nltk spacy pandas numpy matplotlib wordcloud seaborn
!python -m spacy download pt_core_news_sm

import nltk
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from collections import Counter

# Baixar recursos NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Carregar modelo spaCy para português
nlp = spacy.load('pt_core_news_sm')

# Listas expandidas de palavras positivas e negativas
palavras_positivas = {
    "adorado", "excelente", "bom", "ótimo", "maravilhoso", "positivo", "feliz", "alegre",
    "encantador", "brilhante", "incrível", "emocionante", "agradável", "fantástico",
    "satisfatório", "confiável", "amável", "sucesso", "útil", "divertido", "generoso",
    "simpático", "amigável", "gentil", "inspirador", "excelente", "espetacular", "perfeito",
    "admiração", "entusiasmo", "orgulho", "grato", "promissor", "encorajador", "vibrante",
    "apaixonante", "harmonioso", "motivador", "proveitoso", "reconfortante"
}

palavras_negativas = {
    "ruim", "péssimo", "horrível", "terrível", "triste", "decepcionante", "negativo",
    "aborrecido", "chato", "irritante", "insuportável", "desagradável", "fracasso",
    "insucesso", "medo", "ódio", "inseguro", "problemático", "injusto", "lamentável",
    "pessimista", "deprimido", "desanimado", "frustrado", "desapontado", "cansado",
    "confuso", "desastroso", "inaceitável", "vergonhoso", "desesperador", "irritado",
    "repugnante", "desmotivador", "insatisfatório", "tedioso", "deplorável", "hostil"
}

# Stopwords e pontuação para filtro
stop_words = set(stopwords.words('portuguese'))

def analisar_sentimento(texto):
    """
    Analisa o sentimento de um texto com base em um léxico de palavras positivas e negativas.
    Usa lematização com spaCy para normalizar palavras e calcula uma pontuação de sentimento.
    Retorna dicionário com sentimento, contagens, pontuação e tokens.
    """
    # Processar texto com spaCy
    doc = nlp(texto.lower())
    # Lematizar e filtrar stopwords e pontuação
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.text not in punctuation]
    
    # Contagem de palavras positivas e negativas
    positivas = sum(token in palavras_positivas for token in tokens)
    negativas = sum(token in palavras_negativas for token in tokens)
    
    # Calcular pontuação de sentimento (positivas - negativas)
    pontuacao = positivas - negativas
    
    # Determinar sentimento
    if pontuacao > 0:
        sentimento = "Positivo"
    elif pontuacao < 0:
        sentimento = "Negativo"
    else:
        sentimento = "Neutro"
    
    return {
        'sentimento': sentimento,
        'positivas': positivas,
        'negativas': negativas,
        'pontuacao': pontuacao,
        'tokens': tokens  # Incluindo tokens para nuvem de palavras
    }

# Lista de frases para teste
frases = [
    "Eu adorei o filme, foi uma experiência incrível e emocionante!",
    "O atendimento foi ruim e a espera foi horrível.",
    "A comida estava boa, mas o serviço poderia ser melhor.",
    "Não gostei do final, foi decepcionante e triste.",
    "Foi um dia neutro, nada de especial aconteceu.",
    "O produto é ótimo e muito útil para o dia a dia.",
    "Estou frustrado com o resultado, esperava algo melhor.",
    "Que lugar agradável e divertido para passar o tempo!",
    "O evento foi simplesmente fantástico, superou todas as expectativas!",
    "O software é problemático, cheio de erros e muito confuso."
]

# Analisar frases e armazenar resultados
resultados = []
resultados_completos = []  # Lista para armazenar todos os dados incluindo tokens

for frase in frases:
    resultado = analisar_sentimento(frase)
    
    # Para o DataFrame (sem tokens)
    resultados.append({
        'Frase': frase,
        'Sentimento': resultado['sentimento'],
        'Palavras Positivas': resultado['positivas'],
        'Palavras Negativas': resultado['negativas'],
        'Pontuação': resultado['pontuacao']
    })
    
    # Para a nuvem de palavras (com tokens)
    resultados_completos.append(resultado)

# Criar DataFrame com resultados
df_resultados = pd.DataFrame(resultados)

# Visualizações
# 1. Gráfico de barras: Distribuição de sentimentos
plt.figure(figsize=(8, 5))
sns.countplot(data=df_resultados, x='Sentimento', hue='Sentimento', palette='viridis', legend=False)
plt.title('Distribuição de Sentimentos nas Frases Analisadas')
plt.xlabel('Sentimento')
plt.ylabel('Número de Frases')
plt.tight_layout()
plt.show()

# 2. Nuvem de palavras: Tokens mais frequentes
all_tokens = [token for result in resultados_completos for token in result['tokens']]
token_counts = Counter(all_tokens)
wordcloud = WordCloud(width=800, height=400, background_color='white', min_font_size=10).generate_from_frequencies(token_counts)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuvem de Palavras dos Tokens nas Frases')
plt.show()

# 3. Tabela de resultados
print("\nTabela de Resultados:")
print(df_resultados[['Frase', 'Sentimento', 'Palavras Positivas', 'Palavras Negativas', 'Pontuação']].to_string(index=False))

# Salvar resultados em CSV para artigo
df_resultados.to_csv('sentiment_analysis_results.csv', index=False)

# Resumo estatístico
print("\nResumo da Análise de Sentimento:")
print(f"Total de frases analisadas: {len(frases)}")
print(f"Distribuição de sentimentos: {df_resultados['Sentimento'].value_counts().to_dict()}")
print(f"Palavras-chave mais frequentes: {', '.join([word for word, count in token_counts.most_common(5)])}")
