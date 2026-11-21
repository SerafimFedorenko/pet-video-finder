import shutil
import unittest
from pathlib import Path

import pandas as pd

from dataframe_creator import DataFrameCreator

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_FOLDER = PROJECT_ROOT / "test_cats_data_files"


class TestDataFrameCreator(unittest.TestCase):
    def setUp(self):
        self.dfc = DataFrameCreator()

    def tearDown(self):
        shutil.rmtree(TEST_FOLDER, ignore_errors=True)

    def test_create_dataframe(self):
        for cat in ("cat1", "cat2"):
            directory = TEST_FOLDER / "frames" / cat / "train"
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(1, 4):
                (directory / f"{cat}_image{index}.jpg").touch()

        path = self.dfc.create_dataframe("train", TEST_FOLDER.name)

        self.assertTrue(path.exists())
        df = pd.read_excel(path)
        self.assertListEqual(list(df.columns), ["First", "Second", "Res"])
        # 2 кошки x 3 уникальные пары "своя-своя" = 6 позитивных,
        # столько же негативных после балансировки.
        self.assertEqual(len(df), 12)
        self.assertEqual(int((df["Res"] == 1).sum()), 6)
        self.assertEqual(int((df["Res"] == 0).sum()), 6)

    def test_create_dataframe_invalid_folder(self):
        with self.assertRaises(FileNotFoundError):
            self.dfc.create_dataframe("train", "invalid_folder")


if __name__ == "__main__":
    unittest.main()
