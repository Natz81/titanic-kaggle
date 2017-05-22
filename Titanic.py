import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#Lendo conjuntos de teste e treino
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Retirando colunas com nome, ingresso e cabine dos conjuntos
train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#Criação de novo DataFrame a partir do One-hot encoding
new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(test)

#Quantidade de valores nulos no conjunto de treino
new_data_train.isnull().sum().sort_values(ascending=False).head(10)

#Preenchendo valores nulos
new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace=True)
new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace=True)

#Quantidade de valores nulos no conjunto de teste
new_data_test.isnull().sum().sort_values(ascending=False).head(10)

#Preenchendo valores nulos
new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace=True)

#Separando features e target para criação do modelo
X = new_data_train.drop('Survived', axis=1)
y = new_data_train['Survived']

#Criação do modelo
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X, y)

#Verificando score no conjunto de treino
tree.score(X, y)

submission = pd.DataFrame()
submission['PassengerId'] = new_data_test['PassengerId']
submission['Survived'] = tree.predict(new_data_test)

submission.to_csv('submission.csv', index=False)

