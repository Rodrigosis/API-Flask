#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump, load

#%%
data = pd.read_csv('wine.csv')
data.head(5)

#%%
def classificar(tipo, uvas, regiao, vinicola, amadurecimento, classificacao, visual, aroma, nota):
    data = []
    for i in range(len(nota)):
        text = ''
        for n in [tipo, uvas, regiao, vinicola, amadurecimento, classificacao, visual, aroma]:
            text = text + ' ' + str(n[i])
        data.append(text)

    vetorizar = CountVectorizer(lowercase=False, max_features=10000)
    bag_of_words = vetorizar.fit_transform(data)
    treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words, nota, random_state = 42, test_size=0.4)
    
    regressao_logistica = LogisticRegression(solver='lbfgs')
    regressao_logistica.fit(treino.astype('int'), classe_treino.astype('int'))
    modelo = regressao_logistica
    dump(modelo, 'modeloSKL2.joblib')
    dump(vetorizar.vocabulary_, 'VOC_modeloSKL2.joblib')
    return regressao_logistica.score(teste.astype('int'), classe_teste.astype('int'))

#%%
classificar(data['tipo'], data['uvas'], data['regiao'], data['vinicola'], data['amadurecimento'], data['classificacao'], data['visual'], data['aroma'], data['nota'])

#%%
def predict(tipo, uvas, regiao, vinicola, amadurecimento, classificacao, visual, aroma, model, vocabulary): 
    text = ''
    for n in [tipo, uvas, regiao, vinicola, amadurecimento, classificacao, visual, aroma]:
        text = text + ' ' + str(n)
    
    vetorizar = CountVectorizer(lowercase=False, vocabulary=vocabulary)
    bag_of_words = vetorizar.transform([text])
    # isso printa as classes
    #print(model.classes_)

    return model.predict(bag_of_words)

SKLdescription = load('modeloSKL2.joblib') #MODELO
VOC_SKLdescription = load('VOC_modeloSKL2.joblib') #VOCABULARIO

#%%
tipo = "Licoroso"
uvas = "Castas tradicionais no Douro"
regiao = "Douro"
vinicola = "Burmester"
amadurecimento = "40 anos em barricas de carvalho"
classificacao = "Suave/Doce" 
visual = "Acastanhado"
aroma = "Intenso, frutas secas, especiarias,mel"

result = predict(tipo, uvas, regiao, vinicola, amadurecimento, classificacao, visual, aroma, SKLdescription, VOC_SKLdescription)
print(str(result[0]))

#%%
