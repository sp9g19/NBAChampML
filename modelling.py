import pandas as pd
import numpy as np
import tensorflow.keras as ke
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

if __name__ == "__main__":
    allDataDf = pd.read_csv('C:\\Users\\Sol Parker\\OneDrive\\Documents\\NBAChampML\\dataframes\\Training.csv')
    allDataDf = allDataDf.iloc[:, 1:]
    useful_stats = ['FG%', '3P%', '2P%', 'DRB', 'OFG%', 'O2P%', 'ODRB', 'OBLK', 'RSW', 'PyW', 'PyL', 'MOV', 'SRS', 'ORtg', 'DRtg', 'eFG%', 'OeFG%', 'PlW']
    allData = allDataDf[useful_stats].to_numpy()

    X = allData[:, :-1]
    y = allData[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    reg = LinearRegression().fit(X_train, y_train)
    y_pred = np.rint(reg.predict(X_test))
    print("Linear Regression Acc: ", accuracy_score(y_test, y_pred))

    SGD = SGDClassifier('hinge').fit(X_train, y_train)
    y_pred = SGD.predict(X_test)
    print("Stochastic Gradient Descent SVM Acc: ", accuracy_score(y_test, y_pred))

    SVC = make_pipeline(StandardScaler(), SVC(gamma='auto')).fit(X_train, y_train)
    y_pred = SVC.predict(X_test)
    print("SVC Acc: ", accuracy_score(y_test, y_pred))

    neigh = KNeighborsClassifier(n_neighbors=60).fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    print("KNC Acc: ", accuracy_score(y_test, y_pred))

    forest = RandomForestClassifier().fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    print("RFC Acc: ", accuracy_score(y_test, y_pred))

    DTree = DecisionTreeClassifier(criterion="entropy", max_depth=4).fit(X_train, y_train)
    y_pred = DTree.predict(X_test)
    print("DTree Acc: ", accuracy_score(y_test, y_pred))

    def baseline_model():
        model = ke.models.Sequential()
        model.add(ke.layers.Dense(56, input_dim=17, activation='relu'))
        model.add(ke.layers.Dense(32, activation='relu'))
        model.add(ke.layers.Dense(17, activation='softplus'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


    NN = baseline_model()
    y_train_binary = ke.utils.to_categorical(y_train)
    NN.fit(X_train, y_train_binary)
    y_pred = np.argmax(NN.predict(X_test), axis=1)
    print("NN Acc: ", accuracy_score(y_test, y_pred))


def train_and_predict_year(stats_by_year, year_to_pred, stat_idxs_to_use):
    train_list = []
    test_list = []
    team_idxs = []
    for y_offset in range(3, 23):
        year = str(2000 + y_offset)
        for team, stats in stats_by_year[year].items():
            if year == year_to_pred:
                test_list.append([stats_by_year[year][team][i] for i in stat_idxs_to_use])
                team_idxs.append(team)
            else:
                train_list.append([stats_by_year[year][team][i] for i in stat_idxs_to_use])
    train = np.array(train_list)
    test = np.array(test_list)
    train_X = train[:, :-1]
    train_y = train[:, -1]
    test_X = test[:, :-1]
    test_y = test[:, -1]
    # print("Train_X: ", train_X, "Train_y: ", train_y, "Test_X: ", test_X, "Test_y: ", test_y)
    DTree = DecisionTreeClassifier(criterion="entropy", max_depth=4).fit(train_X, train_y)
    pred_y = DTree.predict(test_X)
    pred_y = [max(v, 0) for v in pred_y]
    pred_y = [min(v, 16) for v in pred_y]
    pred_tups = sorted(list(zip(team_idxs, pred_y)), key=lambda x: x[1])
    target_tups = sorted(list(zip(team_idxs, test_y)), key=lambda x: x[1])
    print("LR Acc: ", accuracy_score(test_y, pred_y), "Preds:", pred_tups, "Targets:", target_tups)
    plt.bar(*zip(*pred_tups))
    plt.xticks(rotation=90)
    plt.title(year_to_pred)
    plt.ylabel("Predicted Number of Playoff Wins")
    plt.show()
