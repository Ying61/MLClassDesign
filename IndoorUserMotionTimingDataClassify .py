import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    # 读取数据
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # 分离数据集
    X_train_c = train.drop(['ID', 'CLASS'], axis=1).values  # 训练集删除ID和CLASS两列，只留下T时刻特征数据
    y_train_c = train['CLASS'].values
    X_test_c = test.drop(['ID'], axis=1).values  # 测试集删除ID列，只留下T时刻特征数据
    nfold = 5
    kf = KFold(n_splits=nfold, shuffle=True, random_state=2020)
    prediction1 = np.zeros((len(X_test_c),))
    i = 0
    for train_index, valid_index in kf.split(X_train_c, y_train_c):
        print("\nFold {}".format(i + 1))
        X_train, label_train = X_train_c[train_index], y_train_c[train_index]
        X_valid, label_valid = X_train_c[valid_index], y_train_c[valid_index]
        clf = GridSearchCV(SVR(), param_grid={"kernel": ("linear", 'rbf', 'sigmoid'), "C": np.logspace(-3, 3, 7),
                                              "gamma": np.logspace(-3, 3, 7)})
        clf.fit(X_train, label_train)
        x1 = clf.predict(X_valid)
        y1 = clf.predict(X_test_c)
        prediction1 += y1 / nfold
        i += 1
    # 为后续模型融合做准备，这里保留原始的预测结果result1
    result1 = prediction1

# 使用tsfresh对时序数据进行特征提取
if __name__ == '__main__':
    import pandas
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    # 数据加载
    dataframe = pandas.read_csv("train.csv")
    dataframe_test = pandas.read_csv("test.csv")

    X = dataframe.iloc[:, 1:241].astype(float)
    Y = dataframe.iloc[:, 241]
    X_test = dataframe_test.iloc[:, 1:241].astype(float)

    # 为了方便之后特征的提取，对数据进行重构
    data_new = list()
    for i in range(len(X)):
        data_new.append(X.loc[i])
    data_new = np.array(data_new).reshape(-1, 1)
    time_id = np.tile(np.array([i for i in range(0, 240)]), len(X)).reshape(-1, 1)
    id_index = np.array([i for i in range(0, 210)]).repeat(240).reshape(-1, 1)

    data_format = pandas.DataFrame(np.concatenate([id_index, time_id, data_new], axis=1))
    # id代表第几个训练样本，time是每一个训练样本的不同时间点，time_series是相应id的训练样本在time点的值
    data_format.columns = ['id', 'time', 'time_series']

    # 从训练数据中提取和筛选数据
    from tsfresh import extract_features

    extracted_features = extract_features(data_format, column_id="id", column_sort="time")

    # 特征筛选
    from tsfresh import select_features
    from tsfresh.utilities.dataframe_functions import impute

    impute(extracted_features)
    features_filtered = select_features(extracted_features, Y)

    from tsfresh import feature_extraction

    kind_to_fc_parameters = feature_extraction.settings.from_columns(features_filtered)

    # 对测试数据进行重构并且提取与训练数据相同的特征
    # 测试数据重构
    data_new = list()
    for i in range(len(X_test)):
        data_new.append(X_test.loc[i])
    data_new = np.array(data_new).reshape(-1, 1)
    time_id = np.tile(np.array([i for i in range(0, 240)]), len(X_test)).reshape(-1, 1)
    id_index = np.array([i for i in range(0, 104)]).repeat(240).reshape(-1, 1)
    data_format_test = pandas.DataFrame(np.concatenate([id_index, time_id, data_new], axis=1))
    data_format_test.columns = ['id', 'time', 'time_series']

    features_filtered_test = extract_features(data_format_test, column_id="id", column_sort="time",
                                              kind_to_fc_parameters=kind_to_fc_parameters)

    features_filtered_test = features_filtered_test[features_filtered.columns]
    # 查看相应的特征
    features_filtered_test.info()
    # 修改特征名称
    new_col = ['fea%s' % i for i in range(67)]
    print(new_col)
    features_filtered_test.columns = new_col
    features_filtered.columns = new_col

if __name__ == '__main__':
    # 定义10折交叉验证和lgb参数等
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    num_folds = 10
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2020)
    test_result = np.zeros(len(features_filtered_test))
    auc_score = 0

    params = {'num_leaves': int(16),
              'objective': 'regression',
              'max_depth': int(4),
              'min_data_in_leaf': int(5),
              'min_sum_hessian_in_leaf': int(0),
              'learning_rate': 0.18,
              'boosting': 'gbdt',
              'feature_fraction': 0.8,
              'bagging_freq': int(2),
              'bagging_fraction': 1,
              'bagging_seed': 8,
              'lambda_l1': 0.01,
              'lambda_l2': 0.01,
              'metric': 'auc',  # 评价函数选择
              "random_state": 2020,  # 随机数种子，可以防止每次运行的结果不一致
              }

    # lgb模型的训练和预测
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(features_filtered, Y)):
        print("Fold: ", fold_ + 1)
        X_train, y_train = features_filtered.iloc[trn_idx], Y.iloc[trn_idx]
        X_valid, y_valid = features_filtered.iloc[val_idx], Y.iloc[val_idx]
        trn_data = lgb.Dataset(X_train, y_train)
        val_data = lgb.Dataset(X_valid, y_valid, reference=trn_data)

        clf = lgb.train(params,
                        trn_data,
                        10000,
                        valid_sets=val_data,
                        verbose_eval=50,
                        early_stopping_rounds=50)
        y_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)
        auc = roc_auc_score(y_valid, y_pred)
        print(auc)
        auc_score += auc

        preds = clf.predict(features_filtered_test, num_iteration=clf.best_iteration)
        test_result += preds

    # 预测结果保存在test_result中，输出auc检查
    auc_score = auc_score / folds.n_splits
    print("AUC score: ", auc_score)
    test_result = test_result / folds.n_splits
    Y_test = np.round(test_result)

# 模型融合
if __name__ == '__main__':
    ans = [0 for i in range(104)]
    for i in range(104):
        if result1[i] > 0.5 and test_result[i] > 0.5:
            ans[i] = int(1)
        if result1[i] < 0.5 and test_result[i] < 0.5:
            ans[i] = int(0)
        if result1[i] > 0.5 and test_result[i] < 0.5:
            d1 = result1[i] - 0.5
            d2 = 0.5 - test_result[i]
            if d1 > d2 + 0.1:
                ans[i] = int(1)
            else:
                ans[i] = int(0)
        if result1[i] < 0.5 and test_result[i] > 0.5:
            d1 = 0.5 - result1[i]
            d2 = test_result[i] - 0.5
            if d2 + 0.1 > d1:
                ans[i] = int(1)
            else:
                ans[i] = int(0)

    # 最终的预测结果保存在ans中，将预测结果ans按提交格式写入csv文件
    id_ = range(210, 314)
    df = pd.DataFrame({'ID': id_, 'CLASS': ans})
    df.to_csv("result.csv", index=False)
