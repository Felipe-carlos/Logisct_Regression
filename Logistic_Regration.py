# Título: Regressão logistica usando o descenso do gradiente
# Autor: Felipe C. dos Santos
#

import random

import numpy as np
import matplotlib . pyplot as plt
import math


plt.rc('axes', grid=True)

 # conjunto de dados {(x,y)}
alfa = float                                     #valor de alfa
epocas = 2000                                  #total de epocas
n=1                                             #grau dopolinomio em theta a ser aproximado
m = 500                                         #numero total de pontos
mean0 , std0 = -0.4 , 0.5                       #média e desvio padrão da classe positiva
mean1 , std1 = 0.9 , 0.3                        #média e desvio padrão da classe positiva

x1s = np.random.randn( m //2) * std1 + mean1
x0s = np.random.randn( m //2) * std0 + mean0
xs = np.hstack((x1s,x0s ))

ys = np.hstack(( np.ones (m //2), np . zeros( m //2) ) )




def sigmoid(z):     #recebe um vetor de zs e retorna um vetor da sigmoid destes pontos
    return [1/(1+math.exp(-d))for d in z ]

#hipotese:
def hip(thetas,xs,n):  #recebe uma lista com theta0 e theta1 onde h(x)= theta0 + theta1 * x in xs e n o grau do modelo
  resultado= []
  for x in xs:
      y=0
      for _ in range(n+1):              #função generalizada para regressão polinomial
          y += thetas[_]* (x **_)
      resultado.append(y)
  return sigmoid(resultado)


#Função de custo: entropia cruzada:
def cost(xs,ys,thetas):         #recebe os vetores com os valores de xs,ys e thetas e retorna um vetor com os valores de custo para cara ponto
    result = []
    for pos, a in enumerate(hip(thetas,xs,n)):
        custo = (ys[pos] * math.log10(a)) +((1-ys[pos]) * math.log10(1-a))
        result.append(custo)
    return result

def j(xs,ys,thetas):       #recebe os vetores com os valores de xs,ys e thetas e retorna o J(theta) total
    return -(1/len(xs))*sum(cost(xs, ys, thetas))

def predictios(thetas,xs,n):
    pred =[]
    for i in hip(thetas,xs,n):
        if i>=0.5: pred.append(1)
        else: pred.append(0)
    return pred


def accuracy(ys,xs,thetas,n):
    num = sum(ys == predictios(thetas,xs,n))
    return num/len(ys)




def gradiente_step(thetas,xs,ys,alfa,n):   #recebe os thetas, xs, ys e alfa e retorna os novos valores dos thetas
    new_theta=[]
    for pos, theta in enumerate(thetas):
        new_theta.append(theta - alfa * (1 / len(xs) * sum((hip(thetas, xs, n) - ys) * xs ** pos)))  # função generalizada para regressão polinomial

    return new_theta

thetas_in= [random.uniform(-5, 5) for i in range(n + 1)]  # chute inicial dos valres de theta como numeros aleatorios entre -5 e 5
#fluxo principal
for alfa in [0.9, 0.1, 0.001]:

    thetas = thetas_in
    accuracy_over_epochs = []
    for x in range(epocas):
        thetas = gradiente_step(thetas,xs,ys,alfa,n)
        accuracy_over_epochs.append(accuracy(ys, xs, thetas, n))

    print(f'Acurácia para alfa= {alfa}: ',accuracy(ys, xs, thetas, n))

    plt.figure()
    plt.plot(xs[: m // 2], ys[: m // 2], '.', label='dataset')
    plt.plot(xs[m // 2:], ys[m // 2:], '.', label='dataset')
    plt.plot(xs,predictios(thetas, xs, n)+ np.full(len(xs),0.05),'.',color ='red',label='Previsões')
    plt.axvline(x=max(np.roots(thetas[::-1])), color='k', linestyle='--', label='Fronteira')
    plt.title(f'Predições para alfa = {alfa}')
    plt.legend()
    plt.savefig(f'Prediçoes_alfa={alfa}.png')
    #------------------------------
    plt.figure()
    plt.plot(range(epocas),accuracy_over_epochs)
    plt.xlabel('Épocas')
    plt.ylabel(f'Acuracia')
    plt.title(f'Acuracia vs Epocas para alfa = {alfa}')
    plt.ylim(top=1)
    plt.savefig(f'Acuracia_alfa={alfa}.png')

plt.show()
