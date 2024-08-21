import numpy as np
from mlp import MultilayerPerceptron, Layer


def main():
    mlp = MultilayerPerceptron(
        hidden_layers=[
            Layer(2, 4),
            Layer(4, 1)
        ]
    )
    
    x_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])

    y_train = np.array([
        0,
        1,
        1,
        0
    ])

    mlp.train(x_train, y_train, epochs=3000, alpha=0.5)
    print("Treinamento concluído!\n")

    print("Teste:")
    for x in x_train:
        y_pred = mlp.predict(x)
        print(f'{x} -> {y_pred}')


if __name__ == '__main__':
    # Inciar programa
    main()
