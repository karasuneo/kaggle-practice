from typing import Tuple

import japanize_matplotlib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_score


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
        self.__X = self.__get_explanatory_variable(df=self.__df_train)
        self.__y = np.log1p(self.__df_train["SalePrice"])
        self.__model = lgb.LGBMRegressor(**params)

    def run(self, n_split: int = 5, random_state=24) -> Tuple[pd.DataFrame, float]:
        """
        モデルの作成及び評価・特徴量の重要度を取得
        """
        skf = KFold(n_splits=n_split, shuffle=True, random_state=random_state)
        scores = cross_val_score(
            self.__model,
            self.__X,
            self.__y,
            cv=skf,
            scoring="neg_root_mean_squared_error",
        )

        self.__model.fit(self.__X, self.__y)

        return self.__get_feature_importance(), scores.mean()

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
        X = df[
            [
                "OverallQual",
                "GrLivArea",
                "GarageCars",
                "TotalBsmtSF",
                "FullBath",
                "YearBuilt",
            ]
        ]

        return X
