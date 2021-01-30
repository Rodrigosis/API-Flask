## Setup

- Crie a imagem docker:
  ```bash
  docker-compose build
  ```
- Suba os containers:
  ```bash
  docker-compose up
  ```
  
## Testing
- rodar tests e verificação de tipagem e mau cheiro de código:
    ```bash
        docker-compose exec app bash ./bin/ci.sh
    ``` 
- rodar somente verificação de tipagem e mau cheiro de código:
    ```bash
        docker-compose exec app bash ./bin/lint.sh
    ``` 
- rodar somente tests:
    ```bash
        docker-compose exec app bash ./bin/test.sh
    ``` 


## Arquitetura
```console
wine/
├── application/  # Camada que recebe uma entrada do mundo externo, manipula o domain e retorna algo para o mundo externo.
├── domain/  # Camada onde os dados recebidos do mundo externo são processados (aqui é onde os modelos de ML estão localizados).
└── infra/  # Camada onde ficam elementos infraestrutura que ajudam o serviço a funcionar (banco de dados, cache, framework web, etc).
    └── models/  # Camada onde são armazenados arquivos de treino para serem carregados em memória.
```

## Modelo

    Todas as informações mostradas abaixo estão no arquivo analyze.py dentro da pasta notebooks

- Para compor a informação que o modelo utilizara para predição foram escolhidas 9 colunas contêm palavras chave, estas são: tipo, uvas, regiao, vinicola, amadurecimento, classificacao, visual e aroma.

  #### 1º Etapa:

    - Concatenar as sentenças:
    ```bash
        data = []
        for i in range(len(nota)):
            text = ''
            for n in [tipo, uvas, regiao, vinicola, amadurecimento, classificacao, visual, aroma]:
                text = text + ' ' + str(n[i])
            data.append(text)
    ```

  #### 2º Etapa:

    - Vetorizar o texto:
    ```bash
        vetorizar = CountVectorizer(lowercase=False, max_features=10000)
        bag_of_words = vetorizar.fit_transform(data)
    ```

  #### 3º Etapa:

    - Separa os dados em dados de treino e teste:
    ```bash
        treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words, nota, random_state = 42, test_size=0.4)
    ```

  #### 4º Etapa:

    - Treinar o modelo:
    ```bash
        regressao_logistica = LogisticRegression(solver='lbfgs')
        regressao_logistica.fit(treino.astype('int'), classe_treino.astype('int'))
    ```

  #### 5º Etapa:

    - Salvar o modelo treinado para ser usado posteriormente:
    ```bash
        modelo = regressao_logistica
        dump(modelo, 'modeloSKL2.joblib')
        dump(vetorizar.vocabulary_, 'VOC_modeloSKL2.joblib')
    ```

  #### 6º Etapa:

    - retornar a acurácia do modelo:
    ```bash
        return regressao_logistica.score(teste.astype('int'), classe_teste.astype('int'))
  
        >>> 0.44813278008298757
    ```

  #### 7º Etapa:

    - Carregando os modelos salvos:
    ```bash
        SKLdescription = load('modeloSKL2.joblib') #MODELO
        VOC_SKLdescription = load('VOC_modeloSKL2.joblib') #VOCABULARIO
    ```

  #### 8º Etapa:

    - Criando função para predição com os modelos salvos:
    ```bash
        def predict(tipo, uvas, regiao, vinicola, amadurecimento, classificacao, visual, aroma, model, vocabulary): 
            text = ''
            for n in [tipo, uvas, regiao, vinicola, amadurecimento, classificacao, visual, aroma]:
                text = text + ' ' + str(n)
            
            vetorizar = CountVectorizer(lowercase=False, vocabulary=vocabulary)
            bag_of_words = vetorizar.transform([text])
        
            return model.predict(bag_of_words)
    ```

  #### 9º Etapa:

    - Realizando predições:
    ```bash
        tipo = "Licoroso"
        uvas = "Castas tradicionais no Douro"
        regiao = "Douro"
        vinicola = "Burmester"
        amadurecimento = "40 anos em barricas de carvalho"
        classificacao = "Suave/Doce"
        visual = "Acastanhado"
        aroma = "Intenso, frutas secas, especiarias,mel"
        
        predict(tipo, uvas, regiao, vinicola, amadurecimento, classificacao, visual, aroma, SKLdescription, VOC_SKLdescription)

        >>> array([4])
    ```

  #### 10º Etapa:

    - Consultando parâmetros do modelo:
    ```bash
        SKLdescription.get_params(deep=True)
  
        >>> {'C': 1.0,
             'class_weight': None,
             'dual': False,
             'fit_intercept': True,
             'intercept_scaling': 1,
             'l1_ratio': None,
             'max_iter': 100,
             'multi_class': 'auto',
             'n_jobs': None,
             'penalty': 'l2',
             'random_state': None,
             'solver': 'lbfgs',
             'tol': 0.0001,
             'verbose': 0,
             'warm_start': False}
    ```