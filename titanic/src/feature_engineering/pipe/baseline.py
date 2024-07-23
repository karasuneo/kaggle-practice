from typing import Any, List, Tuple

import japanize_matplotlib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score


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
            "verbose": -1,
        }
        self.__df_train = pd.read_csv(train_file_path)
        age_model = self.__age_model_fit(self.__df_train)
        self.__df_train = self.__age_model_predict(
            df=self.__df_train, age_model=age_model
        )
        self.__X = self.__get_explanatory_variable(df=self.__df_train)
        self.__y = self.__df_train["Survived"]
        self.__id = self.__df_train["PassengerId"]
        self.__model = lgb.LGBMClassifier(**params)

    def run(self, n_split: int = 5, random_state=24) -> Tuple[pd.DataFrame, float]:
        """
        モデルの作成及び評価・特徴量の重要度を取得
        """
        # 層化交差検証（データ分割するにあたってクラスの分布が同じになりように分割する）
        skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=random_state)
        # skfを入れないと分布が偏る可能性がある（ただの交差検証になる）
        scores = cross_val_score(
            self.__model, self.__X, self.__y, cv=skf, scoring="accuracy"
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

        grid_search = GridSearchCV(model, param_grid, cv=n_split, scoring="accuracy")
        grid_search.fit(self.__X, self.__y)

        best_model = grid_search.best_estimator_

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

        # df_test["Sex", "Age", "TItle"] = 
        df_test = self.__age_model_predict(
            df=df_test, age_model=self.__age_model_fit(self.__df_train)
        )
        X_test = self.__get_explanatory_variable(df=df_test)
        id = df_test["PassengerId"]
        y_pred = self.__model.predict(X_test)

        # 提出用ファイルを作成
        df_submit = pd.DataFrame(
            {
                "PassengerId": id,
                "Survived": y_pred,
            }
        )

        df_submit.to_csv(submit_file_path, index=False)

    def get_X(self) -> pd.DataFrame:
        return self.__X
    
    def get_model(self) -> LGBMClassifier:
        return self.__model

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
        df = self.__title(df)
        df = self.__family_size(df)
        df = self.__cabin(df)
        df = self.__ticket_count(df)
        # df = self.__family_feature(df)

        X = df[
            [
                "Fare",
                "Pclass",
                "Title_Master",
                "Title_Miss",
                "Title_Mr",
                "Title_Mrs",
                "Title_Officer",
                "Title_Royalty",
                "Age",
                "FamilySize",
                # "Cabin_Prefix_A",
                "Cabin_Prefix_B",
                "Cabin_Prefix_C",
                "Cabin_Prefix_D",
                "Cabin_Prefix_E",
                # "Cabin_Prefix_F",
                # "Cabin_Prefix_G",
                "Cabin_Prefix_N",
                # "Fare_LastName",
                # "Ticket_LastName",
                # "LastName"
            ]
        ]

        return X

    def __get_age_x(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[
            [
                "Pclass",
                "SibSp",
                "Parch",
            ]
        ]
        return X

    def __title(self, df: pd.DataFrame) -> pd.DataFrame:
        # タイトルの抽出と置換
        df["Title"] = df["Name"].map(lambda x: x.split(", ")[1].split(". ")[0])
        df["Title"] = df["Title"].replace(
            ["Capt", "Col", "Major", "Dr", "Rev"], "Officer"
        )
        df["Title"] = df["Title"].replace(
            ["Don", "Sir", "the Countess", "Lady", "Dona"], "Royalty"
        )
        df["Title"] = df["Title"].replace(["Mme", "Ms"], "Mrs")
        df["Title"] = df["Title"].replace(["Mlle"], "Miss")
        df["Title"] = df["Title"].replace(["Jonkheer"], "Master")

        # ダミー変数の作成
        df = pd.get_dummies(df, columns=["Title"])

        return df

    def __cabin(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Cabin_Prefix"] = df["Cabin"].str[0]
        df["Cabin_Prefix"] = df["Cabin_Prefix"].fillna("N")

        df = pd.get_dummies(df, columns=["Cabin_Prefix"])

        return df

    def __family_size(self, df: pd.DataFrame) -> pd.DataFrame:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

        return df

    def __ticket_count(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Ticket_Count"] = df.groupby("Ticket")["Ticket"].transform("count")
        return df

    def __family_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df["LastName"] = df["Name"].map(lambda x: x.split(",")[0])

        df["Fare_LastName"] = df["LastName"] + "_" + df["Fare"].astype(str)
        df["Ticket_LastName"] = df["LastName"] + "_" + df["Ticket"]
        return df

    def __age_model_fit(self, df: pd.DataFrame) -> LGBMRegressor:
        df_drop_age = df.dropna(subset=["Age"])

        params = {
            "learning_rate": 0.05,
            "max_depth": 10,
            "n_estimators": 100,
            "num_leaves": 31,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)

        X = self.__get_age_x(df_drop_age)
        y = df_drop_age["Age"]

        model.fit(X, y)

        return model

    def __age_model_predict(
        self, df: pd.DataFrame, age_model: LGBMRegressor
    ) -> pd.DataFrame:
        df_drop_age = df[df["Age"].isna()]
        X = self.__get_age_x(df_drop_age)
        y_pred = age_model.predict(X)

        df.loc[df["Age"].isna(), "Age"] = y_pred

        return df
