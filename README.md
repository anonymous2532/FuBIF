# Function Based Isolation Forest (FuBIF)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [License](#license)
- [Contact](#contact)

## Overview
FuBIF (Function-Based Isolation Forest) is an advanced anomaly detection framework designed to generalize the traditional Isolation Forest (IF) algorithm. Traditional IF models have inherent biases and adaptability issues when applied to diverse datasets. FuBIF addresses these limitations by introducing a more flexible branching mechanism that leverages a set of real-valued functions for dataset splitting. This allows FuBIF to adapt more effectively to complex, non-linear data structures.

Key features of FuBIF include:

- **Flexible Branching Mechanism:** FuBIF trees are built using a set of real-valued functions, allowing for more adaptable and precise data splitting.
- **FuBIF Feature Importance (FuBIFFI):** This generalizes the concept of feature importance in IF-based methods, providing comprehensive feature importance scores across all potential FuBIF models.

FuBIF is designed to be a unifying framework, integrating multiple existing IF-type models within a common mathematical structure and equipping them with a standardized interpretability algorithm. This makes FuBIF not only powerful in detecting anomalies but also valuable in providing insights into the underlying data patterns.

## Installation
Clone the repository:
    ```sh
    git clone https://github.com/your-username/FuBIF.git
    cd FuBIF
    ```

## Usage


## Directory Structure
The repository is organized as follows:

```
FuBIF-main/
├── Experiments/               # Directory for experimental scripts
├── FuBIF.py                   # Main script
├── FuBIF_optimized.py         # Optimized version of the main script
├── README.md                  # Project README file
├── data/                      # Directory for input data
├── notebooks/                 # Jupyter notebooks for analysis
├── projections.py             # Script for projections
├── results/                   # Directory for storing results
├── sinusoid_u/                # Directory related to sinusoid functions
├── split_functions.py         # Script for split functions
├── split_functions_optimized.py # Optimized version of split functions
└── utils/                     # Utility scripts
```


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact
For any questions or feedback, please contact [your-email@example.com].
