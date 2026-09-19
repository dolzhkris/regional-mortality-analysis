# regional-mortality-analysis
Python data analysis and K-Means clustering of child mortality rates across regions of Uzbekistan in 2013.

## About

The project combines several data analysis techniques:

* descriptive statistical analysis;
* data visualization;
* K-Means clustering;
* classification by mortality level;
* comparison of minimum and maximum values;
* generation of a final analytical report.

The project was developed as part of university work during the fourth year of university.

## Key Variables

* data - source dataset containing regions and mortality values;
* df - Pandas DataFrame containing the regional data;
* X - NumPy array containing mortality values;
* names - list containing region names;
* X_cluster - two-dimensional dataset used for K-Means clustering;
* n_clusters - number of clusters;
* random_state - value used to initialize the K-Means algorithm;
* kmeans - K-Means clustering model;
* y_pred - cluster assigned to each region;
* clusters - dictionary containing regions grouped by cluster;
* mortalities - mortality values belonging to the current cluster;
* max_idx - index of the region with the highest mortality;
* min_idx - index of the region with the lowest mortality;
* low_threshold - lower percentile threshold;
* high_threshold - upper percentile threshold;
* low_regions - regions classified with a low mortality level;
* medium_regions - regions classified with a medium mortality level;
* high_regions - regions classified with a high mortality level;
* bars - bar objects used for the mortality visualization;
* colors - list of colors used for cluster visualization.

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/dolzhkris/uzbekistan-regional-mortality-analysis.git
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the program:

```bash
python main.py
```

Data source:
[National Indicators of Uzbekistan](https://ns1.stat.uz/ru/goal/6)
