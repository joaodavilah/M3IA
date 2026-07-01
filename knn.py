from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("SISTEMA DE PREVISÃO DE APROVAÇÃO DE ALUNOS - KNN")
print("=" * 60)

dados = [
    [9.5, 98, 20],
    [8.7, 95, 16],
    [7.8, 90, 12],
    [7.0, 85, 10],
    [6.5, 80, 8],
    [8.0, 88, 14],
    [9.0, 92, 18],
    [6.8, 82, 9],
    [5.5, 75, 5],
    [4.8, 70, 4],
    [3.5, 60, 2],
    [5.0, 65, 3],
    [6.0, 68, 4],
    [4.2, 58, 2],
    [2.8, 50, 1],
    [5.8, 72, 5],
]

classes = [
    1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0
]

nomes_classes = ["Reprovado", "Aprovado"]

print("\n1) Base de dados carregada.")
print("Cada aluno possui: média, frequência e horas de estudo por semana.")

dados_treino, dados_teste, classe_treino, classe_teste = train_test_split(
    dados,
    classes,
    test_size=0.30,
    random_state=42
)

print("\n2) Dados separados em treino e teste.")
print(f"Registros de treino: {len(dados_treino)}")
print(f"Registros de teste: {len(dados_teste)}")

modelo = KNeighborsClassifier(n_neighbors=3)

print("\n3) Treinando o modelo KNN...")

modelo.fit(dados_treino, classe_treino)

print("Treinamento concluído.")

print("\n4) Realizando previsões com os dados de teste...")

previsoes = modelo.predict(dados_teste)

acuracia = accuracy_score(classe_teste, previsoes)

print("\n================ RESULTADOS ================\n")
print(f"Acurácia do modelo: {acuracia:.2%}")

print("\nRelatório de classificação:\n")
print(
    classification_report(
        classe_teste,
        previsoes,
        target_names=nomes_classes
    )
)

print("\n================ TESTE COM NOVO ALUNO ================\n")

media = float(input("Digite a média do aluno: "))
frequencia = float(input("Digite a frequência do aluno (%): "))
horas_estudo = float(input("Digite as horas de estudo por semana: "))

novo_aluno = [[media, frequencia, horas_estudo]]

resultado = modelo.predict(novo_aluno)

print("\nDados informados:")
print(f"Média: {media}")
print(f"Frequência: {frequencia}%")
print(f"Horas de estudo por semana: {horas_estudo}")

print("\nClassificação prevista pelo sistema:")

print(nomes_classes[resultado[0]].upper())

print("\n============================================")
print("Fim da execução.")
print("============================================")