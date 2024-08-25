from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Cargar datos
data = pd.read_csv('data/items.csv')

# Vectorización TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['description'])

# Calcular similitudes de coseno
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Función de recomendación
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = data.index[data['title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Obtener las 3 más similares
    item_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[item_indices]

# Ruta principal
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        title = request.form['title']
        recommendations = get_recommendations(title)
        return render_template('index.html', title=title, recommendations=recommendations)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
