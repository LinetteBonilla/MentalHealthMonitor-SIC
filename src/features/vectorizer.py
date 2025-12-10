from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Ruta absoluta al archivo final
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
VECTORIZER_PATH = os.path.join(ROOT_DIR, "src", "models", "saved", "tfidf_vectorizer.pkl")

def create_vectorizer(X_train):
    # Asegurar carpeta
    os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)

    vectorizer = TfidfVectorizer(max_features=5000)

    # ENTRENARLO ANTES DE GUARDARLO
    vectorizer.fit(X_train)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[OK] Vectorizador guardado en: {VECTORIZER_PATH}")

    return vectorizer

def load_vectorizer():
    print(">>> DEBUG: Cargando vectorizador desde:", VECTORIZER_PATH)

    try:
        vec = joblib.load(VECTORIZER_PATH)
        print(">>> DEBUG: Vectorizador cargado correctamente:", type(vec))
        print(">>> DEBUG: Tiene atributo idf_?:", hasattr(vec, "idf_"))
        return vec
    except Exception as e:
        print(">>> ERROR AL CARGAR VECTORIZADOR:", e)
        raise e
