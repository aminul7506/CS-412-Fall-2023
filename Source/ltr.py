import numpy as np
import tensorflow as tf
import lightgbm as lgb
import warnings

from sklearn import svm
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearnex import patch_sklearn

from Source.utilities import read_data, ndcg_score_custom, mrr_score_custom


LTR_LAYERS_DIMENSIONS = [132, 64, 32, 16, 8, 4, 2, 1]
EPOCHS = 20
BATCH_SIZE = 8


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_ltr_model_lr(X, Y, solver, C, max_iter=1000):
    return LogisticRegression(solver=solver, C=C, max_iter=max_iter).fit(X, Y)


def predict_from_model_lr(model, test_df_features_without_qid):
    return sigmoid(np.dot(test_df_features_without_qid, model.coef_[0]) + model.intercept_)


def get_ltr_model_svm(X, Y, kernel, C):
    ksvm = svm.SVR(kernel=kernel, C=C, tol=.1, degree=3)
    ksvm.fit(X, Y)
    return ksvm


def predict_from_model_svm(model, test_df_features_without_qid):
    return model.predict(test_df_features_without_qid)


def get_ltr_model_nn(X, Y, layers_dimensions):
    model = tf.keras.Sequential()

    for i in range(len(layers_dimensions)):
        dimension = layers_dimensions[i]
        if i == 0:
            model.add(tf.keras.layers.Dense(dimension, input_shape=(X.shape[1],), activation='relu'))
        elif i == len(layers_dimensions) - 1:
            model.add(tf.keras.layers.Dense(dimension, activation='sigmoid'))
        else:
            model.add(tf.keras.layers.Dense(dimension, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    return model


def predict_from_model_nn(model, test_df_features_without_qid):
    return model.predict(test_df_features_without_qid)


def get_ltr_model_lin_r(X, Y, fit_intercept):
    lin_model = LinearRegression(fit_intercept=fit_intercept)
    lin_model.fit(X, Y)
    return lin_model


def predict_from_model_lin_r(model, test_df_features_without_qid):
    return model.predict(test_df_features_without_qid)


def get_ltr_model_lambda_mart(feature_columns_with_qid, Y, num_trees, learning_rate):
    warnings.filterwarnings("ignore")

    qids = feature_columns_with_qid.groupby(["qid"])["qid"].count().to_numpy()
    feature_columns = feature_columns_with_qid[feature_columns_with_qid.columns.drop(['qid'])]

    param = {
        "task": "train",
        "objective": "lambdarank",
        "learning_rate": learning_rate,
        "num_leaves": 255,
        "num_trees": num_trees,
        "num_threads": 16,
        "min_data_in_leaf" : 0,
        "min_sum_hessian_in_leaf": 100,
        # "verbose": 10,
        "metric": "ndcg"
    }

    train_data = lgb.Dataset(feature_columns, label=Y, group=qids)
    train_data.set_group(qids)

    model = lgb.train(
        param, train_data,
        # valid_sets=[valid_data], valid_names=["valid"],
        num_boost_round=50)

    return model


def predict_from_model_lambda_mart(model, test_data_features_with_qid):
    predictions = []
    for name, group in test_data_features_with_qid.groupby(['qid']):
        qid = group['qid'].min()
        df_qid = test_data_features_with_qid[test_data_features_with_qid['qid'] == qid]
        df_without_qid = df_qid[df_qid.columns.drop(['qid'])]
        predictions.extend(model.predict(df_without_qid))

    return predictions


def ndcg_evaluation(test_df, predictions):
    print("NDCG evaluations: ")
    ndcg_score_custom(test_df, predictions, 10)
    ndcg_score_custom(test_df, predictions, 5)
    ndcg_score_custom(test_df, predictions, 3)
    ndcg_score_custom(test_df, predictions, 1)
    print()


def evaluations(test_df, predictions):
    ndcg_evaluation(test_df, predictions)


def run_logistic_regression():
    train_df_binary = read_data("../Data/set2.train.binary.csv")
    test_valid_df_binary = read_data("../Data/set2.test.binary.csv")

    train_df_features_without_qid_binary = train_df_binary[train_df_binary.columns.drop(['qid', 'C'])]

    split_index = int(len(test_valid_df_binary) * .67)
    test_df_binary = test_valid_df_binary.iloc[:split_index]
    valid_df_binary = test_valid_df_binary.iloc[split_index:]
    test_df_features_without_qid_binary = test_df_binary[test_df_binary.columns.drop(['qid', 'C'])]
    valid_df_features_without_qid_binary = valid_df_binary[valid_df_binary.columns.drop(['qid', 'C'])]

    lr_solvers = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
    C_list = [0.5, 1, 1.5, 2]

    best_lr_model = None
    best_score = 0.0
    best_solver = "lbfgs"
    best_C = 0.5
    for solver in lr_solvers:
        for C in C_list:
            model_lr = get_ltr_model_lr(train_df_features_without_qid_binary, train_df_binary["C"], solver, C)
            predictions_valid = predict_from_model_lr(model_lr, valid_df_features_without_qid_binary)
            print("solver name: " + solver + ", C: " + str(C))
            score = ndcg_score_custom(valid_df_binary, predictions_valid, 10)
            if score > best_score:
                best_lr_model = model_lr
                best_score = score
                best_solver = solver
                best_C = C

    print("best solver: " + best_solver + ", best C: " + str(best_C))
    predictions_test = predict_from_model_lr(best_lr_model, test_df_features_without_qid_binary)
    evaluations(test_df_binary, predictions_test)


def run_linear_regression(train_df, test_df,
                          train_df_features_without_qid,
                          valid_df_features_without_qid,
                          test_df_features_without_qid):
    intercepts = [True, False]
    best_lin_model = None
    best_score = 0.0
    best_intercept = True

    for intercept in intercepts:
        model_lr = get_ltr_model_lin_r(train_df_features_without_qid, train_df["C"], intercept)
        predictions_valid = predict_from_model_lin_r(model_lr, valid_df_features_without_qid)
        print("Fit intercept: " + str(intercept))
        score = ndcg_score_custom(valid_df, predictions_valid, 10)
        if score > best_score:
            best_lin_model = model_lr
            best_score = score
            best_intercept = intercept

    print("Best fit intercept: " + str(best_intercept))
    predictions = predict_from_model_lin_r(best_lin_model, test_df_features_without_qid)
    evaluations(test_df, predictions)


def run_svm(train_df, test_df,
                          train_df_features_without_qid,
                          valid_df_features_without_qid,
                          test_df_features_without_qid):
    patch_sklearn()

    kernels = ["linear", "poly", "rbf", "sigmoid"]
    C_list = [0.5, 1, 1.5, 2]
    best_svm_model = None
    best_score = 0.0
    best_kernel = "linear"
    best_C = 0.5

    for kernel in kernels:
        for C in C_list:
            model_svm = get_ltr_model_svm(train_df_features_without_qid, train_df["C"], kernel, C)
            predictions_valid = predict_from_model_lin_r(model_svm, valid_df_features_without_qid)
            print("Kernel: " + kernel + ", C: " + str(C))
            score = ndcg_score_custom(valid_df, predictions_valid, 10)
            if score > best_score:
                best_svm_model = model_svm
                best_score = score
                best_kernel = kernel
                best_C = C

    print("best kernel: " + best_kernel + ", best C: " + str(best_C))
    predictions = predict_from_model_svm(best_svm_model, test_df_features_without_qid)
    evaluations(test_df, predictions)


def run_lambda_mart(train_df, test_df,
                          train_df_features_with_qid,
                          valid_df_features_with_qid,
                          test_df_features_with_qid):

    num_trees = [100, 200, 300, 400]
    learning_rates = [0.01, 0.1, 0.5, 1]
    best_lm_model = None
    best_score = 0.0
    best_num_tree = 300
    # best_num_boost_round = 25
    best_learning_rate = 0.01

    for num_tree in num_trees:
        # for num_boost_round in num_boost_rounds:
        for learning_rate in learning_rates:
            model_lm = get_ltr_model_lambda_mart(train_df_features_with_qid, train_df["C"],
                                                 num_tree, learning_rate)
            predictions_valid = predict_from_model_lambda_mart(model_lm, valid_df_features_with_qid)
            print("num_tree: " + str(num_tree) + ", learning_rate: " + str(learning_rate))
            score = ndcg_score_custom(valid_df, predictions_valid, 10)
            if score > best_score:
                best_lm_model = model_lm
                best_score = score
                best_num_tree = num_tree
                best_learning_rate = learning_rate

    print("best_num_tree: " + str(best_num_tree) + ", best_learning_rate: " + str(best_learning_rate))
    predictions = predict_from_model_lambda_mart(best_lm_model, test_df_features_with_qid)
    evaluations(test_df, predictions)


if __name__ == "__main__":

    run_logistic_regression()

    train_df = read_data("../Data/set2.train.csv")
    train_df_features_without_qid = train_df[train_df.columns.drop(['qid', 'C'])]
    train_df_features_with_qid = train_df[train_df.columns.drop(['C'])]

    test_valid_df = read_data("../Data/set2.test.csv")

    split_index = int(len(test_valid_df) * .67)
    test_df = test_valid_df.iloc[:split_index]
    valid_df = test_valid_df.iloc[split_index:]
    test_df_features_with_qid = test_df[test_df.columns.drop(['C'])]
    test_df_features_without_qid = test_df[test_df.columns.drop(['qid', 'C'])]
    valid_df_features_with_qid = valid_df[valid_df.columns.drop(['C'])]
    valid_df_features_without_qid = valid_df[valid_df.columns.drop(['qid', 'C'])]

    run_linear_regression(train_df, test_df,
                          train_df_features_without_qid,
                          valid_df_features_without_qid,
                          test_df_features_without_qid)

    run_svm(train_df, test_df,
                          train_df_features_without_qid,
                          valid_df_features_without_qid,
                          test_df_features_without_qid)

    run_lambda_mart(train_df, test_df,
            train_df_features_with_qid,
            valid_df_features_with_qid,
            test_df_features_with_qid)