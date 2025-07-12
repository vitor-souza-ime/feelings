
# 🧠 Feelings – Lexicon-Based Sentiment Analysis in Portuguese

This project performs **lexicon-based sentiment analysis** on a set of Portuguese sentences using Python. It employs `spaCy`, `NLTK`, and `pandas` to preprocess text via **lemmatization**, **stopword removal**, and classification based on predefined **positive and negative word lists**.

---

## 🎯 Objective

This repository serves as a research-friendly implementation for analyzing sentiment in Portuguese using rule-based natural language processing (NLP). It is designed to support academic articles, NLP experiments, and explainable AI demonstrations.

---

## ✅ Features

- Lemmatization using `spaCy` for accurate word normalization
- Stopword filtering via `nltk`
- Custom positive/negative lexicon in Brazilian Portuguese
- Token visualization through word clouds
- Sentiment distribution chart
- Export of structured results to `.csv`

---

## 🧩 Requirements

Install the required dependencies with:

```bash
pip install nltk spacy pandas numpy matplotlib seaborn wordcloud
python -m spacy download pt_core_news_sm
````

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/feelings.git
cd feelings
```

2. Run the analysis script:

```bash
python main.py
```

It will output:

* A table with classified sentences
* A sentiment distribution chart
* A word cloud
* A CSV file with detailed metrics

---

## 📝 Sample Sentences

```text
"Eu adorei o filme, foi uma experiência incrível e emocionante!" → Positive  
"O atendimento foi ruim e a espera foi horrível." → Negative  
"O produto é ótimo e muito útil para o dia a dia." → Positive  
"Não gostei do final, foi decepcionante e triste." → Negative  
```

---

## 📊 Outputs

* `sentiment_analysis_results.csv`: Contains each sentence, sentiment score, positive/negative word counts, and classification.
* Sentiment distribution chart (via seaborn/matplotlib)
* Word cloud of the most frequent tokens in the dataset

---

## 📚 Applications

* Fake news detection
* Opinion mining in reviews
* Baseline for supervised classifiers
* Linguistic analysis of texts in Portuguese

---

## 🔭 Future Improvements

* Context-aware sentiment detection using deep learning (BERTimbau, XLM-R)
* Irony and sarcasm detection
* Integration with real-world datasets (Twitter, news, reviews)
* Expansion to multilingual support

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

```


