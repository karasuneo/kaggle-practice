import pandas as pd


class Age:
    def __init__(self, df_age: pd.DataFrame):
        self.__df_age = df_age

    def missing_fill_by_average(self) -> pd.DataFrame:
        """
        ## 平均値で欠損値を補完
        """
        self.__df_age["Age"] = self.__df_age["Age"].fillna(self.__df_age["Age"].mean())
        return self.__df_age
