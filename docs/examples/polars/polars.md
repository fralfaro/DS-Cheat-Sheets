<div style="display: flex; align-items: center;">
    <a href="https://colab.research.google.com/github/fralfaro/DS-Cheat-Sheets/blob/main/docs/examples/polars/polars.ipynb" target="_parent">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
</div>

# Polars 

<img src="https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/main/docs/examples/polars/polars.png" alt="numpy logo" width = "200">

[Polars](https://pola-rs.github.io/polars-book/) is a highly performant DataFrame library for manipulating structured data. The core is written in Rust, but the library is also available in Python.

## Install and import Polars

`
$ pip install polars
`


```python
# Import Polars convention
import polars as pl
```

### Creating/reading DataFrames


```python
# Create DataFrame
df = pl.DataFrame(
    {
        "nrs": [1, 2, 3, None, 5],
        "names": ["foo", "ham", "spam", "egg", None],
        "random": [0.3, 0.7, 0.1, 0.9, 0.6],
        "groups": ["A", "A", "B", "C", "B"],
    }
)
```


```python
# Read CSV
df = pl.read_csv("https://j.mp/iriscsv", has_header=True)
```


```python
# Read parquet
df = pl.read_parquet("path.parquet", columns=["select", "columns"])
```

### Expressions
Polars expressions can be performed in sequence. This improves readability of code.


```python
df.filter(pl.col("nrs") < 4).groupby("groups").agg(pl.all().sum())
```

### Subset Observations - rows


```python
# Filter: Extract rows that meet logical criteria.
df.filter(pl.col("random") > 0.5)
df.filter((pl.col("groups") == "B") & (pl.col("random") > 0.5))
```


```python
# Sample
# Randomly select fraction of rows.
#df.sample(frac=0.5)

# Randomly select n rows.
df.sample(n=2)
```


```python
# Select first n rows
df.head(n=2)

# Select last n rows.
df.tail(n=2)
```

### Subset Variables - columns


```python
# Select multiple columns with specific names.
df.select(["nrs", "names"])
```


```python
# Select columns whose name matches regular expression regex.
df.select(pl.col("^n.*$"))
```

### Subsets - rows and columns


```python
# Select rows 2-4.
df[2:4, :]
```


```python
# Select columns in positions 1 and 3 (first column is 0).
df[:, [1, 3]]
```


```python
# Select rows meeting logical condition, and only the specific columns.
df[df["random"] > 0.5, ["names", "groups"]]
```

### Reshaping Data – Change layout, sorting, renaming


```python
df2 = pl.DataFrame(
    {
        "nrs": [6],
        "names": ["wow"],
        "random": [0.9],
        "groups": ["B"],
    }
)

df3 = pl.DataFrame(
    {
        "primes": [2, 3, 5, 7, 11],
    }
)
```


```python
# Append rows of DataFrames.
pl.concat([df, df2])
```


```python
# Append columns of DataFrames
pl.concat([df, df3], how="horizontal")
```


```python
# Gather columns into rows
df.melt(id_vars="nrs", value_vars=["names", "groups"])
```


```python
# Spread rows into columns
df.pivot(values="nrs", index="groups", columns="names")
```


```python
# Order rows by values of a column (low to high)
df.sort("random")
```


```python
# Order rows by values of a column (high to low)
df.sort("random", reverse=True)
```


```python
# Rename the columns of a DataFrame
df.rename({"nrs": "idx"})
```


```python
# Drop columns from DataFrame
df.drop(["names", "random"])
```

### Summarize Data


```python
# Count number of rows with each unique value of variable
df["groups"].value_counts()
```


```python
# # of rows in DataFrame
len(df)
# or
df.height
```


```python
# Tuple of # of rows, # of columns in DataFrame
df.shape
```


```python
# # of distinct values in a column
df["groups"].n_unique()
```


```python
# Basic descriptive and statistics for each column
df.describe()
```


```python
# Aggregation functions
df.select(
    [
        # Sum values
        pl.sum("random").alias("sum"),
        # Minimum value
        pl.min("random").alias("min"),
        # Maximum value
        pl.max("random").alias("max"),
        # or
        pl.col("random").max().alias("other_max"),
        # Standard deviation
        pl.std("random").alias("std_dev"),
        # Variance
        pl.var("random").alias("variance"),
        # Median
        pl.median("random").alias("median"),
        # Mean
        pl.mean("random").alias("mean"),
        # Quantile
        pl.quantile("random", 0.75).alias("quantile_0.75"),
        # or
        pl.col("random").quantile(0.75).alias("other_quantile_0.75"),
        # First value
        pl.first("random").alias("first"),
    ]
)
```

### Group Data


```python
# Group by values in column named "col", returning a GroupBy object
df.groupby("groups")
```


```python
# All of the aggregation functions from above can be applied to a group as well
df.groupby(by="groups").agg(
    [
        # Sum values
        pl.sum("random").alias("sum"),
        # Minimum value
        pl.min("random").alias("min"),
        # Maximum value
        pl.max("random").alias("max"),
        # or
        pl.col("random").max().alias("other_max"),
        # Standard deviation
        pl.std("random").alias("std_dev"),
        # Variance
        pl.var("random").alias("variance"),
        # Median
        pl.median("random").alias("median"),
        # Mean
        pl.mean("random").alias("mean"),
        # Quantile
        pl.quantile("random", 0.75).alias("quantile_0.75"),
        # or
        pl.col("random").quantile(0.75).alias("other_quantile_0.75"),
        # First value
        pl.first("random").alias("first"),
    ]
)
```


```python
# Additional GroupBy functions
df.groupby(by="groups").agg(
    [
        # Count the number of values in each group
        pl.count("random").alias("size"),
        # Sample one element in each group
        pl.col("names").apply(lambda group_df: group_df.sample(1)),
    ]
)
```

### Handling Missing Data


```python
# Drop rows with any column having a null value
df.drop_nulls()
```


```python
# Replace null values with given value
df.fill_null(42)
```


```python
# Replace null values using forward strategy
df.fill_null(strategy="forward")
# Other fill strategies are "backward", "min", "max", "mean", "zero" and "one"
```


```python
# Replace floating point NaN values with given value
df.fill_nan(42)
```

### Make New Columns


```python
# Add a new column to the DataFrame
df.with_column((pl.col("random") * pl.col("nrs")).alias("product"))
```


```python
# Add several new columns to the DataFrame
df.with_columns(
    [
        (pl.col("random") * pl.col("nrs")).alias("product"),
        pl.col("names").str.lengths().alias("names_lengths"),
    ]
)
```


```python
# Add a column at index 0 that counts the rows
df.with_row_count()
```

### Rolling Functions


```python
# The following rolling functions are available
import numpy as np

df.select(
    [
        pl.col("random"),
        # Rolling maximum value
        pl.col("random").rolling_max(window_size=2).alias("rolling_max"),
        # Rolling mean value
        pl.col("random").rolling_mean(window_size=2).alias("rolling_mean"),
        # Rolling median value
        pl.col("random")
        .rolling_median(window_size=2, min_periods=2)
        .alias("rolling_median"),
        # Rolling minimum value
        pl.col("random").rolling_min(window_size=2).alias("rolling_min"),
        # Rolling standard deviation
        pl.col("random").rolling_std(window_size=2).alias("rolling_std"),
        # Rolling sum values
        pl.col("random").rolling_sum(window_size=2).alias("rolling_sum"),
        # Rolling variance
        pl.col("random").rolling_var(window_size=2).alias("rolling_var"),
        # Rolling quantile
        pl.col("random")
        .rolling_quantile(quantile=0.75, window_size=2, min_periods=2)
        .alias("rolling_quantile"),
        # Rolling skew
        pl.col("random").rolling_skew(window_size=2).alias("rolling_skew"),
        # Rolling custom function
        pl.col("random")
        .rolling_apply(function=np.nanstd, window_size=2)
        .alias("rolling_apply"),
    ]
)
```

### Window functions


```python
# Window functions allow to group by several columns simultaneously
df.select(
    [
        "names",
        "groups",
        pl.col("random").sum().over("names").alias("sum_by_names"),
        pl.col("random").sum().over("groups").alias("sum_by_groups"),
    ]
)
```

### Combine Data Sets


```python
df4 = pl.DataFrame(
    {
        "nrs": [1, 2, 5, 6],
        "animals": ["cheetah", "lion", "leopard", "tiger"],
    }
)
```


```python
# Inner join
# Retains only rows with a match in the other set.
df.join(df4, on="nrs")
# or
df.join(df4, on="nrs", how="inner")
```


```python
# Left join
# Retains each row from "left" set (df).
df.join(df4, on="nrs", how="left")
```


```python
# Outer join
# Retains each row, even if no other matching row exists.
df.join(df4, on="nrs", how="outer")
```


```python
# Anti join
# Contains all rows from df that do not have a match in df4.
df.join(df4, on="nrs", how="anti")
```
