import os
import pandas as pd


def read_csv(filepath: str, dropna: bool = True, dropduplicates: bool =True, encoding: str = 'latin1') -> pd.DataFrame:
    """
    read_csv returns DataFrame read from specified CSV format file.

    :param filepath: A path to file with data.
    :param dropna: Flag specifying if missed values should be deleted or not.
    :param encoding: Encoding to use for UTF when reading.
    :returns: A Pandas DataFrame.

    Example usage:
        read_csv(filepath='./data.csv', dropna=False)
    """
    try:
        data = pd.read_csv(filepath, encoding=encoding)
        if dropna:
            data = data.dropna()
        if dropduplicates:
            data = data.drop_duplicates()
        return data
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
    
