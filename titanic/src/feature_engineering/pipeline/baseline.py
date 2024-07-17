from typing import Tuple

import japanize_matplotlib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pipeline.age import Age
from sklearn.model_selection import cross_val_score


class Baseline:
    def __init__(
        self,
        input_X: pd.DataFrame,
        input_y: pd.DataFrame,
        input_id: pd.DataFrame,
    ):
        self.__X = input_X
        self.__y = input_y
        self.__id = input_id

    def create_model(self, n_split: int = 5) -> Tuple[pd.DataFrame, float]:
        """
        ## モデルの作成及び評価・特徴量の重要度を取得
        """
        # モデルの作成
        model = lgb.LGBMClassifier(verbose=-1)

        age = Age(self.__X)
        self.__X = age.discretization()

        # 交差検証の実行
        scores = cross_val_score(
            model, self.__X, self.__y, cv=n_split, scoring="accuracy"
        )

        model.fit(self.__X, self.__y)

        return self.__get_feature_importance(model=model), scores.mean()

    def plot_feature_importance(self, df_feature_importance: pd.DataFrame):
        """
        ## 特徴量の重要度をプロット
        """

        # 特徴量の重要度をプロット
        plt.figure(figsize=(12, 6))
        plt.barh(
            df_feature_importance["feature"],
            df_feature_importance["importance"],
        )
        plt.xlabel("特徴量の重要度")
        plt.ylabel("特徴量名")
        plt.show()

    def __get_feature_importance(self, model) -> pd.DataFrame:
        """
        ## 特徴量の重要度を取得
        """
        # 特徴量の重要度を取得
        importance = model.feature_importances_

        # 特徴量の名前を取得
        feature_names = self.__X.columns

        # 特徴量の重要度をデータフレームに追加
        df_feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        )

        return df_feature_importance
