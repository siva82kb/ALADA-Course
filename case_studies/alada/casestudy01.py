"""Module containing functions, and other data structures required for the case 
study 01.

Author: Sivakuar Balasubramanian
Date: 02 August 2024
"""

import pathlib

# Data folder.
datadir = pathlib.Path("data/case_study_01/")

# Intialize case study
create_data_folder = lambda: datadir.mkdir(parents=True, exist_ok=True)

# Data check function
check_dataset = lambda: ("Success! You can run this notebook."
                         if pathlib.Path(datadir / "reports.csv").is_file()
                         else "Falied! Check the instructions. Else contact the TAs.")
