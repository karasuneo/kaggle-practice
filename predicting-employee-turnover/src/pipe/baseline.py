from typing import List, Tuple

import japanize_matplotlib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     cross_validate)
from sklearn.pipeline import make_pipeline


class Baseline:
    def __init__(
        self,
        train_file_path: str,
    ):
        params = {
            "learning_rate": 0.05,
            "max_depth": 10,
            "n_estimators": 100,
            "num_leaves": 31,
        }
        self.__df_train = pd.read_csv(train_file_path)
        self.__X = self.__get_explanatory_variable(df=self.__df_train)
        self.__y = self.__df_train["Attrition"]
        self.__id = self.__df_train["id"]
        self.__model = lgb.LGBMClassifier(**params)

    def run(self, n_split: int = 5) -> Tuple[pd.DataFrame, float]:
        """
        ## モデルの作成及び評価・特徴量の重要度を取得
        """
        # 交差検証の実行
        scores = cross_val_score(
            self.__model, self.__X, self.__y, cv=n_split, scoring="accuracy"
        )

        self.__model.fit(self.__X, self.__y)

        return self.__get_feature_importance(), scores.mean()

    def model_tuning(self, n_split: int = 5):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
        }

        # モデルの作成
        model = lgb.LGBMClassifier(**params)

        # グリッドサーチによるハイパーパラメータのチューニング
        param_grid = {
            "num_leaves": [31, 50, 100],
            "max_depth": [10, 20, 30],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200, 300],
        }

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(self.__X, self.__y)

        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")

    def add_feature(self, feature: pd.Series):
        """
        ## 特徴量を追加
        """
        self.__X = pd.concat([self.__X, feature], axis=1)

    def remove_feature(self, feature_name: str):
        """
        ## 特徴量を削除
        """
        self.__X = self.__X.drop([feature_name], axis=1)

    def make_submit(self, test_file_path: str, submit_file_path: str):
        """
        ## 提出用ファイルを作成
        """
        df_test = pd.read_csv(test_file_path)
        X_test = self.__get_explanatory_variable(df=df_test)
        id = df_test["id"]
        y_pred = self.__model.predict(X_test)

        # 提出用ファイルを作成
        df_submit = pd.DataFrame(
            {
                "PassengerId": id,
                "Attrition": y_pred,
            }
        )

        df_submit.to_csv(submit_file_path, index=False, header=False)

    def __get_feature_importance(self) -> pd.DataFrame:
        """
        ## 特徴量の重要度を取得
        """
        # 特徴量の重要度を取得
        importance = self.__model.feature_importances_

        # 特徴量の名前を取得
        feature_names = self.__X.columns

        df_feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        )

        return df_feature_importance

    def __get_explanatory_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.__overtime(df=df)
        df = self.__business_travel(df=df)

        X = df[
            [
                "RelationshipSatisfaction",
                "YearsInCurrentRole",
                "NumCompaniesWorked",
                "OverTime",
                "BusinessTravel_Non-Travel",
                # "YearsWithCurrManager",  # add
                # "Age",
                # "YearAtCompany",
            ]
        ]

        return X

    def __overtime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ## OverTimeの特徴量を追加の処理
        """
        df = pd.get_dummies(df, columns=["OverTime"])
        df["OverTime"] = df["OverTime_Yes"]
        df = df.drop(["OverTime_No", "OverTime_Yes"], axis=1)

        return df

    def __business_travel(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ## BusinessTravelの特徴量を追加の処理
        """
        df_train_dummy_business_travel = pd.get_dummies(df, columns=["BusinessTravel"])
        df = pd.concat(
            [
                df,
                df_train_dummy_business_travel["BusinessTravel_Non-Travel"],
            ],
            axis=1,
        )

        return df
