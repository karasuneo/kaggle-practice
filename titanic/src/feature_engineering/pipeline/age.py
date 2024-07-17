import pandas as pd


class Age:
    def __init__(self, df_age: pd.DataFrame):
        self.__df_age = df_age.copy()  # データフレームのコピーを作成

    def discretization(self) -> pd.DataFrame:
        """
        ## 欠損値処理をした上で年齢をビン分割
        """
        self.__missing_fill_by_average()
        self.__df_age.loc[:, "Age_bin"] = pd.cut(
            self.__df_age.loc[:, "Age"],
            bins=[0, 10, 20, 30, 40, 50, 60, 100],
            labels=["10代未満", "10代", "20代", "30代", "40代", "50代", "60代以上"],
            right=False,
            duplicates="raise",
            include_lowest=True,
        )

        self.__df_age = self.__df_age.drop("Age", axis=1)

        return self.__df_age

    def __missing_fill_by_average(self):
        """
        ## 平均値で欠損値を補完
        """
        self.__df_age.loc[:, "Age"] = self.__df_age["Age"].fillna(
            self.__df_age["Age"].mean()
        )
