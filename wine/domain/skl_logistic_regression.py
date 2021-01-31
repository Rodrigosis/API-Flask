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
        return {"accuracy": 0.44813278008298757, "model_params": self.model.get_params(deep=True)}
