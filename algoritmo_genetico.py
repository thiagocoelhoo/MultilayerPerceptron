import math
import numpy as np
import matplotlib.pyplot as plt
import random


def funcao_objetivo(x, y):
    return math.sqrt(x**3 + 2*y**4)


def para_binario(numero, digitos=6):
    return f'{numero:0{digitos}b}'


def cruzamento(individuo1, individuo2):
    ponto_corte = random.randint(0, 5)
    binario1 = para_binario(individuo1)
    binario2 = para_binario(individuo2)
    novo_individuo = int(binario1[:ponto_corte] + binario2[ponto_corte:], 2)
    return novo_individuo


def mutacao(individuo):
    ponto_mutacao = random.randint(0, 5)
    binario_individuo = para_binario(individuo)
    novo_binario = binario_individuo[:ponto_mutacao] + ('0' if binario_individuo[ponto_mutacao] == '1' else '1') + binario_individuo[ponto_mutacao + 1:]
    return int(novo_binario, 2)


def main():
    tamanho_populacao = 12
    tamanho_selecao = 4
    taxa_mutacao = 0.05

    # Gerar populacao inicial
    populacao = [random.randint(50, 63) for _ in range(tamanho_populacao)]
    
    epocas = 1000
    melhores_individuos = []
    fitness = []

    for epoca in range(epocas):
        # Avaliação da população

        valores_fitness = []
        
        for indviduo in populacao:
            a = indviduo >> 3
            b = indviduo & 0b000111
            valor = funcao_objetivo(a, b)
            valores_fitness.append(valor)
        
        # Adiciona melhor individuo da população na lista para geração de gráfico

        indice_melhor_ind = valores_fitness.index(min(valores_fitness))
        melhores_individuos.append(populacao[indice_melhor_ind])
        fitness.append(min(valores_fitness))

        print(f'Epoca {epoca + 1:>2}\nFitness: {min(valores_fitness):>6.2f}\nIndividuo: {melhores_individuos[-1]}\n')
        
        if min(valores_fitness) == 0.0:
            break
        
        # Seleção dos indivíduos para a próxima geração

        fitness_inverso = [1 / valor for valor in valores_fitness]
        pesos = [(1 / valor) / sum(fitness_inverso) for valor in valores_fitness]

        selecao = np.random.choice(
            populacao,
            size=tamanho_selecao,
            replace=False, 
            p=pesos
        ).tolist()
        
        # Gerar nova população

        nova_populacao = []
        while len(nova_populacao) < tamanho_populacao:
            pai1 = random.choice(selecao)
            pai2 = random.choice(selecao)
            
            filho = cruzamento(pai1, pai2)
            if random.random() <= taxa_mutacao:
                filho = mutacao(filho)
            
            nova_populacao.append(filho)
        
        populacao = nova_populacao

    # Plotar informações de treino
    
    plt.plot(range(epoca + 1), fitness)
    plt.plot(range(epoca + 1), melhores_individuos)
    plt.show()


if __name__ == '__main__':
    main()
