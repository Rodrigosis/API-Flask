# Desafio Machine Learning Engineer

Este desafio é uma parte do processo de seleção da [Birdie](http://birdie.ai).

Seu objetivo é implementar um modelo de Machine Learning simples, considerando etapas básicas para um modelo.

## Modelo de Machine Learning

Desenvolver um modelo de regressão que seja capaz de predizer **a nota de um vinho** a partir de algumas caracteríticas do mesmo.
Para tanto, deve-se utilizar o dataset **wine.csv** como base. Você terá liberdade para escolher as features que achar
mais relevantes, bastando informá-las no README que for desenvolvido.


## O que esperamos?

Itens de 1 a 5 são *obrigatórios*.

1. **Etapas de um modelo de Machine Learning**

  Construir um modelo simples a partir do dataset escolhido. O foco principal **não é gerar o melhor modelo**, e sim criar as etapas básicas do mesmo e gravar um arquivo .pickle no final. Esse arquivo deverá ser usado para os passos posteriores.

2. **Um endpoint /predict respondendo a um verbo HTTP GET**

  Receber como payload as features que julgue relevantes para obtenção da predição, retornando o valor predito pelo algoritmo.

3. **Um endpoint /metrics respondendo a um verbo HTTP GET**

  Retornar uma lista chave/valor das métricas do modelo gerado.

4. **README.md**

  README com todo tipo de detalhe sobre a sua solução: diagrama técnico, lista de bibliotecas, referências, explicação de funcionamento, etc).

5. **Outros**

  - Criar um swagger com a descrição da API
  - Testes na API
  - Ambiente dockerizado
 
6. **Bônus(Desejável mas não mandatório)**

  - Criar mais de um modelo pra realizar a predição
  - Modificar o endpoint /predict pra incluir no request o modelo, e retornar o resultado correspondente.


## Como participar?

- Dê um fork neste repositório.
- Clone o fork na sua máquina.
- Crie em seu repositório um README.md descrevendo os passos para treinar seu dataset, descrevendo de forma sucinta as etapas até o modelo final.
Assim que concluir, Abra uma issue neste repositório com o título '[DESAFIO Machine Learning Engineer] {{Seu nome}}'.
No conteúdo da issue faça qualquer comentário sobre como foi sua experiência na execução do teste(sugestões, elogios, críticas, etc).
Assim que sua issue for aberta, alguém de nosso time técnico da Birdie irá analisar seu desafio, e eventualmente esteja preparado para defender a solução que construiu.

Quanto mais informações tivermos no README.md, melhor conseguiremos te avaliar.

Aguardamos seu desafio, e boa sorte!
