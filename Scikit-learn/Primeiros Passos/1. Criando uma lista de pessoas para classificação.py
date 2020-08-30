#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Para efeitos de sorteio de valores médios neste tutorial, vamos utilizar uma 
# função de distribuição normal. Isso é uma distribuição em formato de sino (bell curve,
# ou curva normal), bem centrada em volta do valor médio (mais prováveis), mas ainda 
# com ocorrência dos valores mais extremos.
#
# Vamos criar uma função, que pode gerar um número variável de amostras, e testar.
# Vamos passar o desvio padrão (um valor que determina quanto os valores se espalham
# fora do centro, o quanto a curva é alta  íngreme) também como argumento, para 
# deixar a função mais versátil e reutilizável.
#
# Essa função gira em torno da truncnorm (normal truncada) da biblioteca scipy.
import numpy as np
import scipy.stats

def randomizaNormal(limiteInferior, limiteSuperior, desvioPadrao, numeroAmostras):
    limiteInferior = 1
    limiteSuperior = 10
    media = (limiteInferior+limiteSuperior)/2
    amostras = scipy.stats.truncnorm.rvs((limiteInferior-media)/desvioPadrao,(limiteSuperior-media)/desvioPadrao,loc=media,scale=desvioPadrao,size=numeroAmostras)
    amostras = np.around(amostras,0).astype(int)
    return amostras


# In[2]:


# Vamos testar: limite mínimo 1, limite máximo 1, desvio padrão 3, 10.000 amostras:
amostragemTeste = randomizaNormal(1, 10, 3, 10000)

# No matplotlib, vemos que funciona maravilhosamente.
import matplotlib.pyplot as plt

n, bins, patches = plt.hist(amostragemTeste, 10, density=True, facecolor='g', alpha=1)
plt.xlabel('Número')
plt.ylabel('Probabilidade')
plt.title('Distribuição Normal')
plt.xlim(0, 11)
plt.ylim(0, 0.2)
plt.grid(True)
plt.show()


# In[3]:


# Vamos criar uma variação dessa função. Desta vez, ela recebe uma lista e um desvio padrão
# como parâmetro - e simplesmente escolhe um dos valores da lista, de novo, com uma distribuição
# normal, centrada nos valoes intermediarios. Faremos isso com uma lista e um índice, em vez
# de usar um modo mais direto, para que isso possa ser feito inclusive com listas contendo
# valores não-numéricos, como textos, etc. Ela assumirá que os valores do miolo são "médios".
# Ela receberá também um valor "deslocamento", para deslocar a média inicial para a esquerda
# ou para a direita - isso auxiliará a criar populações com comportamentos diferentes.

def randomizaNormalLista(lista, desvioPadrao, deslocamento):
    limiteInferior = 1
    limiteSuperior = len(lista)-1
    media = (limiteInferior+limiteSuperior)/2 + deslocamento
    indice = scipy.stats.truncnorm.rvs((limiteInferior-media)/desvioPadrao,(limiteSuperior-media)/desvioPadrao,loc=media,scale=desvioPadrao,size=1)
    indice = np.around(indice,0).astype(int)
    indice = indice[0]
    return lista[indice]
    


# In[4]:


# Vamos testar, com desvio padrão e e deslocamento zero. 
universoTeste = ["zero","um","dois","três","quatro","cinco","seis","sete","oito","nove","dez"]
amostraTeste = randomizaNormalLista(universoTeste, 3, 0)
print(amostraTeste)


# In[5]:


#vamos seguir em frente, com o seguinte:
#
# 1 - Criaremos algumas tabelas com características gerais comuns
# 2 - Criaremos algumas tabelas com características "mais brasileiras" a "mais americanas"
# 3 - Criaremos uma pessoa hipotética, e sortearemos um gênero e nacionalidade 
# 5 - A partir do gênero e nacionalidade iremos gerar algumas características, com base na distribuição
#     normal, com pequeno deslocamento para cima ou para baixo na média, dependendo da nacionalidade
# 5 - Iremos então adicionar esta pessoa em uma lista de pessoas, que crescerá até as 10.000 pessoas.
#
# Inicializando o contador de pessoas:
contadorPessoas = 1
# Inicializando a tabela de pessoas:
pessoas = []
# Tabela nacionalidade
tabelaNacionalidade = ["Brasil", "EUA"]
# Tabela gênero
tabelaGenero = ["M","F"]


# In[6]:


# Primeiro, as tabelas com características em comum: altura e peso.
# 
# Tabela altura em centímetros, vamos usar uma função range, com o operador * 
# (que descomprime na hora o resultado). O range começa no 160 e vai até o 200,
# subindo de 2 em 2. Ela terá, então, 20 itens.
tabelaAltura = [*range(160, 210, 4)]

# Mesma coisa para a tabela peso
tabelaPeso = [*range(50, 130, 5)]


# In[7]:


# Agora as tabelas "mais ou menos" brasileiras/americanas. Vamos começar com uma tabela
# de comidas favoritas, que começa em comidas tipicamente brasileiras, passa por comidas comuns,
# e termina em comidas tipicamente americanas. Embora a função normal gere números de 1 a 10,
# vamos precisar de 2 números adicionais de cada lado, pois faremos um pequeno deslocamento para
# um lado ou para o outro, dependendo se o sujeito for brasileiro ou americano.
tabelaComida = ["Maniçoba", "Vatapá", "Acarajé", "Arroz Trop.", "Feijoada", "Churrasco", "Pizza", "Hamburguer", "Hummus", "Bagel", "Clam Chowder", "Tacos", "Torta de Maçã"]

# O mesmo para a tabela programa favorito
tabelaPrograma = ["A Praça é Nossa", "A Grande Família", "Chaves", "Jaspion", "A Usurpadora", "A Casa de Papel", "Narcos", "Friends", "Law and Order", "Chicago Fire", "WWE Smackdown", "Jeopardy", "Family Feud", "Cops"]

# e o mesmo para a tabela esporte favorito
tabelaEsporte = ["Vôlei de Praia", "Futebol de Salão", "Vôlei", "Futebol", "MMA", "Natação", "Atletismo", "Boxe", "Basquete", "Boliche", "Futebol Am.", "Luta Livre", "Basebol", "Esqui"]


# In[8]:


# E estamos prontos para gerar pessoas. Vamos lá.
# para nacionalidade e sexo temos valores binários, então só a função random serve (gera números
# entre 0 e 1). Para as demais, vamos usar a função randomizaNormal com um shift de -2 a +2
# o que, somando, dá 4 passos de diferença de média entre brasileiros e americanos.
# Vamos complicar o trabalho do nosso preditor: colocar uma informação que não quer dizer nada
# (a altura, que será a mesma, sem shift), perfeitamente aleatória,  e supor que 
# nos dois países homens sejam em média 2 centímetros maiores que mulheres, 
# e 5 quilos mais pesados. (shift de um passo para cima).
# um pequeno ajuste traz o número randomizado para o centro da tabela, no final.
# Vamos criar uma função para isso, que devolve, separada, a nacionalidade e a pessoa:
def criaPessoa():
    nacionalidade = tabelaNacionalidade[round(np.random.random())]
    if nacionalidade == "Brasil":
        shiftNac = -1
    elif nacionalidade == "EUA":
        shiftNac = 1            
    genero = tabelaGenero[round(np.random.random())]
    if genero == "M":
        shiftGen = 1
    elif genero == "F":
        shiftGen = 0
    altura = randomizaNormalLista(tabelaAltura, 3, shiftGen)
    peso = randomizaNormalLista(tabelaPeso, 3, shiftNac + shiftGen)
    comida = randomizaNormalLista(tabelaComida, 3, shiftNac)
    programa = randomizaNormalLista(tabelaPrograma, 3, shiftNac)
    esporte = randomizaNormalLista(tabelaEsporte, 3, shiftNac)
    pessoa = [nacionalidade, genero, altura, peso, comida, programa, esporte]
    return pessoa


# In[9]:


#Vamos mostrar 10 pessoas aleatórias:
for i in range(10):
    pessoa = criaPessoa()
    print(pessoa)


# In[10]:


# Com isso, vamos gerar uma lista de 10.000 pessoas, sem informação 
# de nacionalidade (que o nosso sistema tentará deduzir). Vamos iniciar
# com um registro que será o título das colunas:
pessoas=[["país", "gênero", "altura", "peso", "comida fav.", "programa fav.", "esporte fav."]]
for i in range(10000):
    pessoa = criaPessoa()
    pessoas.append(pessoa)    


# In[11]:


# O melhor jeito de trabalhar com esse tipo de dados é com a biblioteca pandas,
# que tem um objeto próprio, o dataframe, com métodos muito úteis. Vamos
# importar a biblioteca e transformar essa lista em um dataframe.
# Após, vamos pegar o registro 0 como título das colunas e descartá-lo da lista.

import pandas as pd
pessoas = pd.DataFrame(data=pessoas)
pessoas.columns = pessoas.iloc[0]
pessoas = pessoas.drop(0)


# In[13]:


# Vejamos que o dataframe é bem mais organizado e apresentável.
print(pessoas)


# In[15]:


# com a biblioteca pandas é facílimo salvar o dataset em csv (comma separated values):
pessoas.to_csv("pessoas.csv")


# In[18]:


# A biblioteca pandas tem, inclusive, um método que permite facilmente salvar em Excel
# (desde que o módulo openpyxl esteja instalado no ambiente, claro:)
pessoas.to_excel("pessoas.xlsx", sheet_name="pessoas")


# In[ ]:




