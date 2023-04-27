# ia_atvs

# Predição de Ataque Cardíaco
### Descrição
Este projeto tem como objetivo prever a ocorrência de ataques cardíacos em pacientes com base em informações clínicas. Para isso, utiliza-se um conjunto de dados contendo informações demográficas e médicas dos pacientes e aplicam-se modelos de aprendizado de máquina para classificar os pacientes como tendo alta ou baixa chance de sofrer um ataque cardíaco.

### Dependências
As bibliotecas necessárias para este projeto são:

- pandas
- numpy
- matplotlib
- sklearn
### Conjunto de dados
O conjunto de dados utilizado é o arquivo saude.csv, que contém informações sobre pacientes e seus respectivos riscos de ataque cardíaco. As colunas do conjunto de dados são:

- idade
- tipoDorPeito
- pressaoRepouso
- colesterol
- acucarSangue
- eletrocardiograma
- freqCardiacaMax
- ataqueCardiaco (variável resposta)
### Análise exploratória
Uma análise exploratória dos dados é realizada para entender a distribuição da variável resposta e a relação entre idade e frequência cardíaca dos pacientes.

### Modelos de aprendizado de máquina
Três modelos de aprendizado de máquina são utilizados neste projeto:

1. Suporte de Máquinas Vetoriais (SVM)
2. Floresta Aleatória (Random Forest)
3. K Vizinhos mais próximos (KNN)
Os modelos são treinados e avaliados usando as métricas de acurácia e precisão. Para cada modelo, 50 iterações são realizadas, com 30% dos dados sendo utilizados para treinamento e 70% para teste.

### Resultados
Os resultados da acurácia e precisão média para cada modelo são apresentados, além de boxplots que mostram a distribuição dos resultados.

### Predição para novo paciente
Ao final do projeto, o usuário pode inserir informações de um novo paciente, e o modelo de Floresta Aleatória é usado para prever a chance de o paciente sofrer um ataque cardíaco. A resposta é apresentada como "Muita chance de sofrer ataque cardíaco" ou "Pouca chance de sofrer ataque cardíaco".


![image](https://user-images.githubusercontent.com/80500801/234926025-8cf6d174-ab21-4a14-8534-d8e8ea86702b.png)
![image](https://user-images.githubusercontent.com/80500801/234926062-ececd94a-b8f4-4208-9352-12df638f668c.png)
![image](https://user-images.githubusercontent.com/80500801/234926095-6e18f4f0-570d-4d95-8a08-5d1a571a9d8e.png)


