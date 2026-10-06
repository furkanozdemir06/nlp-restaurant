🍽️ Restaurant Review Sentiment Analysis with NLP

Can a model tell if a restaurant review is a rave or a rant just by reading the text? This project builds a Natural Language Processing (NLP) pipeline and a neural network that classifies Las Vegas buffet reviews as positive or negative.

🎯 Result
Metric	Score
Test accuracy	96.2%
Macro F1-score	0.94
F1 (Positive)	0.98
F1 (Negative)	0.89

The model exceeded the 95% accuracy target. Its main weakness is recall on negative reviews (0.84): 32 of 202 negative reviews were predicted as positive.

📦 Dataset

10,417 restaurant reviews with star ratings, review text and metadata (date, useful/funny/cool votes). For a clear sentiment signal, only the 1-star (negative) and 5-star (positive) reviews were kept, giving about 5,300 reviews. The data is imbalanced, with roughly 81% positive reviews. The dataset is not included in this repository.

🔧 How It Works
Text cleaning: lowercase, remove URLs, punctuation, numbers, line breaks and extra spaces.
NLP preprocessing: tokenization, stop word removal, rare word removal and lemmatization with NLTK.
Exploration: star rating distribution, most common words, and word clouds for all and for 1-star reviews.
Vectorization: bag-of-words with unigrams and bigrams, limited to the 5,000 most frequent features (CountVectorizer).
Model: a neural network with two hidden layers (128 and 64 units), Dropout (0.2) and a sigmoid output.
Training: Adam optimizer (learning rate 0.0001), binary cross-entropy and Early Stopping with best weights restored.
Evaluation: accuracy, classification report and confusion matrix on a held-out 20% test set.
🛠️ Tech Stack

Python, NLTK, scikit-learn, TensorFlow/Keras, pandas, NumPy, WordCloud, Matplotlib, Seaborn

📁 Files
NLPRestaurant.ipynb: the full notebook, from data cleaning to evaluation

🚀 How to Run
Install the dependencies:
bash
   pip install pandas numpy scikit-learn tensorflow nltk wordcloud matplotlib seaborn
Place restaurant.csv next to the notebook.
Run the notebook. The required NLTK data is downloaded in the first cell.
