from sklearn.feature_extraction.text import CountVectorizer


class SklClassifier:

    def __init__(self, model, vocabulary):
        self.model = model
        self.vocabulary = vocabulary

    def predict(self, tipo: str, uvas: str, regiao: str, vinicola: str, amadurecimento: str,
                classificacao: str, visual: str, aroma: str):
        text = ''
        for n in [tipo, uvas, regiao, vinicola, amadurecimento, classificacao, visual, aroma]:
            text = text + ' ' + str(n)

        vetorizar = CountVectorizer(lowercase=False, vocabulary=self.vocabulary)
        bag_of_words = vetorizar.transform([text])

        return self.model.predict(bag_of_words)

    def params(self):
        return {"accuracy": 0.44813278008298757,
                "confusion_matrix": [[0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 0],
                                     [1, 7, 27, 36, 14],
                                     [3, 7, 32, 74, 18],
                                     [0, 1, 3, 10, 7]],
                "precision": {"1": 0, "2": 0, "3": 0.44, "4": 0.61, "5": 0.18},
                "recall": {"1": 0, "2": 0, "3": 0.32, "4": 0.55, "5": 0.33},
                "f1-score": {"1": 0, "2": 0, "3": 0.37, "4": 0.58, "5": 0.23},
                "model_params": self.model.get_params(deep=True)}
