import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.metrics import accuracy_score  
from sklearn.metrics import precision_score 
from sklearn.model_selection import train_test_split 

from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier


base_saude = pd.read_csv('saude.csv', sep=";")
base_saude.head(10)


# Distribuição da variável resposta
num_categorias = base_saude['ataqueCardiaco'].value_counts()
print(num_categorias)

# Relação entre idade e frequencia cardiaca pela variável resposta
coluna_idade = 'idade'
coluna_freq_cardiaca = 'freqCardiacaMax'
coluna_variavel_resposta = 'ataqueCardiaco'

colors = ListedColormap(['orange', 'purple'])
scatter = plt.scatter(x=base_saude[coluna_idade], y=base_saude[coluna_freq_cardiaca],
                      c=base_saude[coluna_variavel_resposta], cmap=colors, alpha=0.8)

plt.title("Relação entre idade e frequência cardíaca dos pacientes")
plt.xlabel("Idade")
plt.ylabel("Frequência Cardíaca")
plt.legend(*scatter.legend_elements())
plt.show()

X = base_saude[['idade','tipoDorPeito', 'pressaoRepouso', 'colesterol', 'acucarSangue', 'eletrocardiograma', 'freqCardiacaMax']] #variaveis explicativas
Y = base_saude['ataqueCardiaco'] #variáveis resposta

acuracia_modelo1 = []
acuracia_modelo2 = []
acuracia_modelo3 = []

precisao_modelo1 = []
precisao_modelo2 = []
precisao_modelo3 = []

np.random.seed(14)

iteracoes = 50 
train_size = 0.3

for i in range(iteracoes):
    
    X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, train_size = train_size)

    # Modelo 1 =  SVM
    model1 = SVC(kernel='rbf', random_state=42)
    model1.fit(X_treino, Y_treino)

    # Modelo 2 =  Random Forest
    model2 = RandomForestClassifier(n_estimators=1000, random_state=42)
    model2.fit(X_treino, Y_treino)

    # Modelo 3 =  KNN
    model3 = KNeighborsClassifier(n_neighbors=7)
    model3.fit(X_treino, Y_treino)

    predictions1 = model1.predict(X_teste)
    acuracia_modelo1.append(accuracy_score(Y_teste, predictions1))
    precisao_modelo1.append(precision_score(Y_teste, predictions1))

    predictions2 = model2.predict(X_teste)
    acuracia_modelo2.append(accuracy_score(Y_teste, predictions2))
    precisao_modelo2.append(precision_score(Y_teste, predictions2))

    predictions3 = model3.predict(X_teste)
    acuracia_modelo3.append(accuracy_score(Y_teste, predictions3))
    precisao_modelo3.append(precision_score(Y_teste, predictions3))



print('Média acurácia modelo 1:')
print(np.round(np.mean(acuracia_modelo1),2))
print('Média acurácia modelo 2:')
print(np.round(np.mean(acuracia_modelo2),2))
print('Média acurácia modelo 3:')
print(np.round(np.mean(acuracia_modelo3),2))

print('Média precisão modelo 1:')
print(np.round(np.mean(precisao_modelo1),2))
print('Média precisão modelo 2:')
print(np.round(np.mean(precisao_modelo2),2))
print('Média precisão modelo 3:')
print(np.round(np.mean(precisao_modelo3),2))


dados_resultados_acuracia =[acuracia_modelo1, acuracia_modelo2, acuracia_modelo3]

plt.boxplot(dados_resultados_acuracia)
plt.title("Boxplot da amostra de acerto dos modelos")
plt.show()


dados_resultados_precisao =[precisao_modelo1, precisao_modelo2, precisao_modelo3]

plt.boxplot(dados_resultados_precisao)
plt.title("Boxplot da amostra de precisão dos modelos para a classe 1")
plt.show()


idade = int(input('Informe a idade do paciente: '))
dor = int(input('Informe a categoria de dor no peito do paciente: '))
pressao = int(input('Informe a pressão arterial observada do paciente: '))
colesterol = int(input('Informe a taxa de colesterol observada: '))
eletro = int(input('Informe o resultafo do exame eletrocardiograma do paciente: '))
frequencia = int(input('Indiue a frequência cardíaca máxima observada do paciente: '))
acucar = int(input('Foi identificado açucar no sangue do paciente? '))

novo_paciente = [[idade, dor, pressao, colesterol, eletro, frequencia, acucar]]
variaveis = ['idade', 'tipoDorPeito', 'pressaoRepouso', 'colesterol',	'acucarSangue',	'eletrocardiograma',	'freqCardiacaMax']
novo_paciente =  pd.DataFrame(novo_paciente, columns = variaveis)
resposta = model2.predict(novo_paciente) 

for i in range(len(resposta)):
  if resposta[i] == 1.0 :
    print('Muita chance de sofrer ataque cardíaco')
  else:
    print('Pouca chance de sofrer ataque cardíaco')
