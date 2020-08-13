# Introduction

**Goal:** 

To predict the selling prices of houses in King County, WA. using Multiple Linear Regression to determine what are the important factors to consider when determining the price of a house, and provide recommendations and intuition for those aiming to sell their homes.

**Methodology:**

We are using the OSEMN Process:
- **Obtain**: Since our dataset is handed to us, we don't have to jump through any hoops and can load in the dataset.

- **Scrub**: This is where we clean the data. In this section, we will be looking for: outliers, null values, and viewing the types of values within each column.

- **Explore**: Looking at our data and the relationship between variables, along with more cleaning aspects like: normalizing and scaling, one hot encoding, and looking for multicollinearity.

- **Model**: We will then create linear regression models to determine which factors play a significant role when determining the selling price of a house.

- **Interpret**: Results and Conclusion.

# Step 1: Obtaining the Data

Our data was given to us in a nice csv.file, so we simply load it in.


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
pd.set_option('display.max_columns', 0)
plt.style.use('seaborn')

kc_df_raw = pd.read_csv('kc_house_data.csv')
```


```python
kc_df_raw.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>



# Step 2: Scrub

- We take a look at the types of values within each column. 
- We recast certain columns to different types.
- We look for and remove any null/NaN values.
- We create new columns to use for our models.
- Drop any columns we feel are not necessary for our model.
- Filter our dataframe to remove major outliers.



```python
kc_df_raw.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 21 columns):
    id               21597 non-null int64
    date             21597 non-null object
    price            21597 non-null float64
    bedrooms         21597 non-null int64
    bathrooms        21597 non-null float64
    sqft_living      21597 non-null int64
    sqft_lot         21597 non-null int64
    floors           21597 non-null float64
    waterfront       19221 non-null float64
    view             21534 non-null float64
    condition        21597 non-null int64
    grade            21597 non-null int64
    sqft_above       21597 non-null int64
    sqft_basement    21597 non-null object
    yr_built         21597 non-null int64
    yr_renovated     17755 non-null float64
    zipcode          21597 non-null int64
    lat              21597 non-null float64
    long             21597 non-null float64
    sqft_living15    21597 non-null int64
    sqft_lot15       21597 non-null int64
    dtypes: float64(8), int64(11), object(2)
    memory usage: 3.5+ MB
    

### Column Descriptions:
* **id** - unique identified for a house
* **date** - house was sold
* **price** -  is prediction target
* **bedrooms** -  of Bedrooms/House
* **bathrooms** -  of bathrooms/bedrooms
* **sqft_living** -  footage of the home
* **sqft_lot** -  footage of the lot
* **floors** -  floors (levels) in house
* **waterfront** - House which has a view to a waterfront
* **view** - Has been viewed
* **condition** - How good the condition is ( Overall )
* **grade** - overall grade given to the housing unit, based on King County grading system
* **sqft_above** - square footage of house apart from basement
* **sqft_basement** - square footage of the basement
* **yr_built** - Built Year
* **yr_renovated** - Year when house was renovated
* **zipcode** - zip
* **lat** - Latitude coordinate
* **long** - Longitude coordinate
* **sqft_living15** - The square footage of interior housing living space for the nearest 15 neighbors
* **sqft_lot15** - The square footage of the land lots of the nearest 15 neighbors

### 'bath_per_bed' column

we create this column to view the ratio of baths per bed of each house within our dataset.

### 'renovated' column
we create this column to replace 'yr_renovated' and make it a column consisting of 0's and 1's indicating if the house was renovated. We will later on use this as a categorical column.

### 'has_basement' column
We create this columnn to replace 'sqft_basement' and also make this column consisting of boolean values to determine if the house has a basement.

### 'subregion' Column

We create this column by finding the subregions for groups of zipcodes on a website (can be found here: http://www.agingkingcounty.org/wp-content/uploads/sites/185/2016/09/SubRegZipCityNeighborhood.pdf). I manually create a dictionary, and then use that dictionary to create a new column in our Dataframe that determines the subregion for each house based on it's zipcode. This column will be used later on as a categorical column.

## Setting the filters for our dataset

I created a function that pulls the 1st and 99th quantiles of every column and prints out the column name along with the number of values within the range of the quantiles and the number of outliers outside the range.

Since **'sqft_lot'** would only drop 7 rows for cutting off extreme outliers, we **filter our dataset** by cutting off those houses outside the **1st and 99th quantiles** of our **'sqft_lot'** column.

# Step 3 - Exploring Our Data


```python
#Loading in our cleaned dataframe - we used pickle to maintain the changed types of columns
with open('kc_housing.pickle', 'rb') as f:
    kc_clean = pickle.load(f)
```

### We remove any columns we don't need for our model


```python
kc_clean.drop(['sqft_basement', 'lat', 'long', 'month'], axis=1, inplace=True)
```

### Viewing all continuous columns and their relationship with Price

We begin by running a **for** loop for each chosen continuous column within our dataframe and comparing it with price in a sns.regplot() to determine which columns we will want to inspect further and which we can throw away. We also create a  dataframe of df.corr() by comparing each column to price, and then sort those values from highest to lowest. This gives us a rough idea of which values could be significant when determining the price of a house.


```python
cont_vars = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'grade', 
            'sqft_above', 'sqft_living15', 'sqft_lot15', 'bath_per_bed', 'house_age']
```

### Question 1:
**What variables are linearly related to price?**

This would give us insight on which columns to explore more and keep as predictor variables for our model


```python
for col in kc_clean[cont_vars]:
    plt.figure(figsize=(8,5))
    sns.regplot(data=kc_clean, x=col, y='price', line_kws={'color': 'red'})
```


![png](output_26_0.png)



![png](output_26_1.png)



![png](output_26_2.png)



![png](output_26_3.png)



![png](output_26_4.png)



![png](output_26_5.png)



![png](output_26_6.png)



![png](output_26_7.png)



![png](output_26_8.png)



![png](output_26_9.png)



![png](output_26_10.png)



![png](output_26_11.png)



```python
pd.DataFrame(kc_clean.corr()['price']).sort_values(by='price',ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>price</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>sqft_living</td>
      <td>0.701991</td>
    </tr>
    <tr>
      <td>grade</td>
      <td>0.666973</td>
    </tr>
    <tr>
      <td>sqft_above</td>
      <td>0.602473</td>
    </tr>
    <tr>
      <td>sqft_living15</td>
      <td>0.584590</td>
    </tr>
    <tr>
      <td>price_per_sqft</td>
      <td>0.563504</td>
    </tr>
    <tr>
      <td>bathrooms</td>
      <td>0.523608</td>
    </tr>
    <tr>
      <td>view</td>
      <td>0.357022</td>
    </tr>
    <tr>
      <td>bedrooms</td>
      <td>0.307952</td>
    </tr>
    <tr>
      <td>bath_per_bed</td>
      <td>0.283479</td>
    </tr>
    <tr>
      <td>waterfront</td>
      <td>0.267153</td>
    </tr>
    <tr>
      <td>floors</td>
      <td>0.264694</td>
    </tr>
    <tr>
      <td>has_basement</td>
      <td>0.182697</td>
    </tr>
    <tr>
      <td>renovated</td>
      <td>0.120049</td>
    </tr>
    <tr>
      <td>sqft_lot</td>
      <td>0.108235</td>
    </tr>
    <tr>
      <td>sqft_lot15</td>
      <td>0.080068</td>
    </tr>
    <tr>
      <td>condition</td>
      <td>0.035466</td>
    </tr>
    <tr>
      <td>house_age</td>
      <td>-0.052713</td>
    </tr>
  </tbody>
</table>
</div>



### Answer for Question 1:
- these scatterplots and correlation for each continuous variable when compared to price show us which are linearly related to price.
- It also gives us a good indication as to which aspects of a house we want to focus on for our models.

### Separating Our cleaned dataframe into two different dataframes: Continuous and Categorical Variables.

Within this dataframe, we have all of our **numerical** columns we want to inspect and are going to **normalize (if needed) and scale using the z-score method.** 

**The numerical columns are**: 
- 'sqft_above'
- 'sqft_living'
- sqft_living15'
- 'sqft_lot'
- 'sqft_lot15'
- 'bedrooms'
- 'bathrooms'
- 'floors'
- 'grade'
- 'condition'
- 'house_age'
- 'bath_per_bed'
- 'price'



```python
kc_continuous = ['sqft_above', 'sqft_living', 'sqft_living15', 'sqft_lot', 'sqft_lot15', 
                 'bedrooms', 'bathrooms', 'floors', 'grade', 'condition', 'house_age', 'bath_per_bed', 'price']
kc_categorical = ['waterfront', 'renovated', 'view', 'zipcode', 'has_basement']
cont_df = kc_clean[kc_continuous]
cat_df = kc_clean[kc_categorical]
```

### Normalizing (if needed) and Scaling Our Continuous Data

First, we take a look at the distribution of each column in our dataframe to determine whether or not we should normalize the data:


```python
with plt.style.context('seaborn'):
    cont_df.hist(figsize=(10,10), bins='auto')
    plt.tight_layout();
```


![png](output_34_0.png)


Selecting the columns we need to be normalized:


```python
selected_log_cols = ['price', 'sqft_above', 'sqft_living', 'sqft_living15', 'sqft_lot', 'sqft_lot15']
```

Next, we log the chosen columns, and add them to our dataframe.


```python
for col in selected_log_cols:
    cont_df[f'{col}_log'] = cont_df[col].map(lambda x: np.log(x))
```


```python
with plt.style.context('seaborn'):
    cont_df.hist(figsize=(10,10), bins='auto')
    plt.tight_layout();
```


![png](output_39_0.png)


Next, we grab the selected columns that have been normalized (and ones that were already normal) and put them into a new dataframe to use. 


```python
select_df = cont_df[['price_log', 'sqft_above_log', 'sqft_living_log', 
                     'sqft_living15_log', 'sqft_lot_log','sqft_lot15_log', 'bedrooms', 'bathrooms', 'bath_per_bed', 
                     'floors', 'grade', 'condition', 'house_age']]
                       
```

From here, we define a function to z-score the columns using **StandardScaler**:


```python
def zscore(df):
    """
    Definition:
    Uses the StandardScaler() class from sklearn.preprocessing to scale numerical data of the 
    selected dataframe by z-scoring the data.
    
    Args:
    df = chosen dataframe. Must have all numerical values.
    
    Returns:
    a dataframe containing the scaled data    
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.values)
    print(scaled_data.shape)
    select_scaled = pd.DataFrame(data=scaled_data, index=df.index, columns=df.columns)
    return select_scaled
```


```python
select_scaled_df = zscore(select_df)
```

    (20995, 13)
    

We add price back into the dataframe


```python
select_scaled_df['price'] = kc_clean['price']
```


```python
select_scaled_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_log</th>
      <th>sqft_above_log</th>
      <th>sqft_living_log</th>
      <th>sqft_living15_log</th>
      <th>sqft_lot_log</th>
      <th>sqft_lot15_log</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>bath_per_bed</th>
      <th>floors</th>
      <th>grade</th>
      <th>condition</th>
      <th>house_age</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>20995.000000</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>1.434961e-16</td>
      <td>-1.765273e-15</td>
      <td>-4.331959e-16</td>
      <td>8.447319e-16</td>
      <td>-1.451206e-15</td>
      <td>3.682165e-16</td>
      <td>-1.204826e-16</td>
      <td>-4.331959e-17</td>
      <td>8.663917e-17</td>
      <td>0.000000</td>
      <td>1.028840e-16</td>
      <td>-4.331959e-17</td>
      <td>2.165979e-17</td>
      <td>5.403582e+05</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>3.689172e+05</td>
    </tr>
    <tr>
      <td>min</td>
      <td>-3.376132e+00</td>
      <td>-3.490173e+00</td>
      <td>-3.886682e+00</td>
      <td>-4.754174e+00</td>
      <td>-2.548468e+00</td>
      <td>-3.308065e+00</td>
      <td>-2.577353e+00</td>
      <td>-2.104856e+00</td>
      <td>-2.797676e+00</td>
      <td>-0.912126</td>
      <td>-3.978142e+00</td>
      <td>-3.703696e+00</td>
      <td>-1.513924e+00</td>
      <td>7.800000e+04</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>-7.031571e-01</td>
      <td>-7.216539e-01</td>
      <td>-6.820303e-01</td>
      <td>-7.209081e-01</td>
      <td>-5.451191e-01</td>
      <td>-5.482791e-01</td>
      <td>-4.132815e-01</td>
      <td>-4.743437e-01</td>
      <td>-6.597818e-01</td>
      <td>-0.912126</td>
      <td>-5.591049e-01</td>
      <td>-6.379692e-01</td>
      <td>-8.660575e-01</td>
      <td>3.200000e+05</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>-5.759156e-02</td>
      <td>-8.926629e-02</td>
      <td>1.641963e-02</td>
      <td>-5.844825e-02</td>
      <td>-4.764336e-02</td>
      <td>-1.620418e-02</td>
      <td>-4.132815e-01</td>
      <td>1.778614e-01</td>
      <td>-6.189610e-02</td>
      <td>0.027433</td>
      <td>-5.591049e-01</td>
      <td>-6.379692e-01</td>
      <td>-1.158966e-01</td>
      <td>4.500000e+05</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>6.240959e-01</td>
      <td>7.152760e-01</td>
      <td>6.890731e-01</td>
      <td>6.998093e-01</td>
      <td>3.562867e-01</td>
      <td>3.499212e-01</td>
      <td>6.687543e-01</td>
      <td>5.039639e-01</td>
      <td>5.359896e-01</td>
      <td>0.966991</td>
      <td>2.956543e-01</td>
      <td>8.948945e-01</td>
      <td>6.683625e-01</td>
      <td>6.450000e+05</td>
    </tr>
    <tr>
      <td>max</td>
      <td>5.319609e+00</td>
      <td>3.982592e+00</td>
      <td>4.370288e+00</td>
      <td>3.648516e+00</td>
      <td>4.082217e+00</td>
      <td>5.395171e+00</td>
      <td>3.204779e+01</td>
      <td>7.678220e+00</td>
      <td>8.906389e+00</td>
      <td>3.785666</td>
      <td>4.569450e+00</td>
      <td>2.427758e+00</td>
      <td>2.407372e+00</td>
      <td>7.700000e+06</td>
    </tr>
  </tbody>
</table>
</div>



While there are outliers within this dataframe, we will one-hot-encode the categorical dataframe and join it before removing those outliers.


```python
cat_df = pd.get_dummies(cat_df, drop_first=True)
```


```python
cat_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>waterfront</th>
      <th>renovated</th>
      <th>view</th>
      <th>has_basement</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>zipcode_98028</th>
      <th>zipcode_98029</th>
      <th>zipcode_98030</th>
      <th>zipcode_98031</th>
      <th>zipcode_98032</th>
      <th>zipcode_98033</th>
      <th>zipcode_98034</th>
      <th>zipcode_98038</th>
      <th>zipcode_98039</th>
      <th>zipcode_98040</th>
      <th>zipcode_98042</th>
      <th>zipcode_98045</th>
      <th>zipcode_98052</th>
      <th>zipcode_98053</th>
      <th>zipcode_98055</th>
      <th>zipcode_98056</th>
      <th>zipcode_98058</th>
      <th>zipcode_98059</th>
      <th>zipcode_98065</th>
      <th>zipcode_98070</th>
      <th>zipcode_98072</th>
      <th>zipcode_98074</th>
      <th>zipcode_98075</th>
      <th>zipcode_98077</th>
      <th>zipcode_98092</th>
      <th>zipcode_98102</th>
      <th>zipcode_98103</th>
      <th>zipcode_98105</th>
      <th>zipcode_98106</th>
      <th>zipcode_98107</th>
      <th>zipcode_98108</th>
      <th>zipcode_98109</th>
      <th>zipcode_98112</th>
      <th>zipcode_98115</th>
      <th>zipcode_98116</th>
      <th>zipcode_98117</th>
      <th>zipcode_98118</th>
      <th>zipcode_98119</th>
      <th>zipcode_98122</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>7129300520</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>6414100192</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5631500400</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2487200875</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1954400510</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
merge_selected = pd.concat([select_scaled_df, cat_df], axis=1)
```


```python
merge_selected.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_log</th>
      <th>sqft_above_log</th>
      <th>sqft_living_log</th>
      <th>sqft_living15_log</th>
      <th>sqft_lot_log</th>
      <th>sqft_lot15_log</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>bath_per_bed</th>
      <th>floors</th>
      <th>grade</th>
      <th>condition</th>
      <th>house_age</th>
      <th>price</th>
      <th>waterfront</th>
      <th>renovated</th>
      <th>view</th>
      <th>has_basement</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>zipcode_98028</th>
      <th>zipcode_98029</th>
      <th>zipcode_98030</th>
      <th>zipcode_98031</th>
      <th>zipcode_98032</th>
      <th>zipcode_98033</th>
      <th>zipcode_98034</th>
      <th>...</th>
      <th>zipcode_98055</th>
      <th>zipcode_98056</th>
      <th>zipcode_98058</th>
      <th>zipcode_98059</th>
      <th>zipcode_98065</th>
      <th>zipcode_98070</th>
      <th>zipcode_98072</th>
      <th>zipcode_98074</th>
      <th>zipcode_98075</th>
      <th>zipcode_98077</th>
      <th>zipcode_98092</th>
      <th>zipcode_98102</th>
      <th>zipcode_98103</th>
      <th>zipcode_98105</th>
      <th>zipcode_98106</th>
      <th>zipcode_98107</th>
      <th>zipcode_98108</th>
      <th>zipcode_98109</th>
      <th>zipcode_98112</th>
      <th>zipcode_98115</th>
      <th>zipcode_98116</th>
      <th>zipcode_98117</th>
      <th>zipcode_98118</th>
      <th>zipcode_98119</th>
      <th>zipcode_98122</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>20995.000000</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>2.099500e+04</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>...</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
      <td>20995.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>1.434961e-16</td>
      <td>-1.765273e-15</td>
      <td>-4.331959e-16</td>
      <td>8.447319e-16</td>
      <td>-1.451206e-15</td>
      <td>3.682165e-16</td>
      <td>-1.204826e-16</td>
      <td>-4.331959e-17</td>
      <td>8.663917e-17</td>
      <td>0.000000</td>
      <td>1.028840e-16</td>
      <td>-4.331959e-17</td>
      <td>2.165979e-17</td>
      <td>5.403582e+05</td>
      <td>0.006906</td>
      <td>0.034865</td>
      <td>0.097738</td>
      <td>0.392522</td>
      <td>0.009383</td>
      <td>0.013146</td>
      <td>0.014956</td>
      <td>0.008002</td>
      <td>0.023339</td>
      <td>0.006621</td>
      <td>0.013479</td>
      <td>0.004382</td>
      <td>0.009240</td>
      <td>0.005096</td>
      <td>0.008383</td>
      <td>0.009859</td>
      <td>0.023434</td>
      <td>0.003144</td>
      <td>0.018052</td>
      <td>0.013432</td>
      <td>0.015051</td>
      <td>0.012003</td>
      <td>0.012908</td>
      <td>0.005859</td>
      <td>0.020529</td>
      <td>0.025816</td>
      <td>...</td>
      <td>0.012384</td>
      <td>0.019243</td>
      <td>0.021196</td>
      <td>0.022005</td>
      <td>0.014289</td>
      <td>0.004811</td>
      <td>0.012908</td>
      <td>0.020624</td>
      <td>0.016956</td>
      <td>0.008764</td>
      <td>0.015909</td>
      <td>0.004620</td>
      <td>0.027102</td>
      <td>0.010764</td>
      <td>0.015099</td>
      <td>0.012241</td>
      <td>0.008764</td>
      <td>0.004954</td>
      <td>0.012336</td>
      <td>0.027149</td>
      <td>0.014956</td>
      <td>0.025768</td>
      <td>0.023482</td>
      <td>0.008288</td>
      <td>0.013146</td>
      <td>0.018957</td>
      <td>0.016575</td>
      <td>0.022529</td>
      <td>0.011860</td>
      <td>0.015289</td>
      <td>0.013384</td>
      <td>0.002667</td>
      <td>0.021005</td>
      <td>0.011908</td>
      <td>0.012574</td>
      <td>0.012098</td>
      <td>0.012289</td>
      <td>0.006430</td>
      <td>0.013098</td>
      <td>0.014813</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>1.000024e+00</td>
      <td>3.689172e+05</td>
      <td>0.082819</td>
      <td>0.183443</td>
      <td>0.296967</td>
      <td>0.488324</td>
      <td>0.096414</td>
      <td>0.113903</td>
      <td>0.121379</td>
      <td>0.089097</td>
      <td>0.150981</td>
      <td>0.081099</td>
      <td>0.115318</td>
      <td>0.066053</td>
      <td>0.095684</td>
      <td>0.071209</td>
      <td>0.091176</td>
      <td>0.098807</td>
      <td>0.151281</td>
      <td>0.055981</td>
      <td>0.133142</td>
      <td>0.115117</td>
      <td>0.121759</td>
      <td>0.108901</td>
      <td>0.112880</td>
      <td>0.076318</td>
      <td>0.141803</td>
      <td>0.158589</td>
      <td>...</td>
      <td>0.110594</td>
      <td>0.137380</td>
      <td>0.144039</td>
      <td>0.146704</td>
      <td>0.118683</td>
      <td>0.069194</td>
      <td>0.112880</td>
      <td>0.142125</td>
      <td>0.129111</td>
      <td>0.093207</td>
      <td>0.125125</td>
      <td>0.067816</td>
      <td>0.162384</td>
      <td>0.103194</td>
      <td>0.121949</td>
      <td>0.109962</td>
      <td>0.093207</td>
      <td>0.070209</td>
      <td>0.110384</td>
      <td>0.162522</td>
      <td>0.121379</td>
      <td>0.158446</td>
      <td>0.151431</td>
      <td>0.090661</td>
      <td>0.113903</td>
      <td>0.136376</td>
      <td>0.127677</td>
      <td>0.148400</td>
      <td>0.108258</td>
      <td>0.122704</td>
      <td>0.114916</td>
      <td>0.051578</td>
      <td>0.143404</td>
      <td>0.108473</td>
      <td>0.111431</td>
      <td>0.109327</td>
      <td>0.110174</td>
      <td>0.079932</td>
      <td>0.113699</td>
      <td>0.120807</td>
    </tr>
    <tr>
      <td>min</td>
      <td>-3.376132e+00</td>
      <td>-3.490173e+00</td>
      <td>-3.886682e+00</td>
      <td>-4.754174e+00</td>
      <td>-2.548468e+00</td>
      <td>-3.308065e+00</td>
      <td>-2.577353e+00</td>
      <td>-2.104856e+00</td>
      <td>-2.797676e+00</td>
      <td>-0.912126</td>
      <td>-3.978142e+00</td>
      <td>-3.703696e+00</td>
      <td>-1.513924e+00</td>
      <td>7.800000e+04</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>-7.031571e-01</td>
      <td>-7.216539e-01</td>
      <td>-6.820303e-01</td>
      <td>-7.209081e-01</td>
      <td>-5.451191e-01</td>
      <td>-5.482791e-01</td>
      <td>-4.132815e-01</td>
      <td>-4.743437e-01</td>
      <td>-6.597818e-01</td>
      <td>-0.912126</td>
      <td>-5.591049e-01</td>
      <td>-6.379692e-01</td>
      <td>-8.660575e-01</td>
      <td>3.200000e+05</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>-5.759156e-02</td>
      <td>-8.926629e-02</td>
      <td>1.641963e-02</td>
      <td>-5.844825e-02</td>
      <td>-4.764336e-02</td>
      <td>-1.620418e-02</td>
      <td>-4.132815e-01</td>
      <td>1.778614e-01</td>
      <td>-6.189610e-02</td>
      <td>0.027433</td>
      <td>-5.591049e-01</td>
      <td>-6.379692e-01</td>
      <td>-1.158966e-01</td>
      <td>4.500000e+05</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>6.240959e-01</td>
      <td>7.152760e-01</td>
      <td>6.890731e-01</td>
      <td>6.998093e-01</td>
      <td>3.562867e-01</td>
      <td>3.499212e-01</td>
      <td>6.687543e-01</td>
      <td>5.039639e-01</td>
      <td>5.359896e-01</td>
      <td>0.966991</td>
      <td>2.956543e-01</td>
      <td>8.948945e-01</td>
      <td>6.683625e-01</td>
      <td>6.450000e+05</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>5.319609e+00</td>
      <td>3.982592e+00</td>
      <td>4.370288e+00</td>
      <td>3.648516e+00</td>
      <td>4.082217e+00</td>
      <td>5.395171e+00</td>
      <td>3.204779e+01</td>
      <td>7.678220e+00</td>
      <td>8.906389e+00</td>
      <td>3.785666</td>
      <td>4.569450e+00</td>
      <td>2.427758e+00</td>
      <td>2.407372e+00</td>
      <td>7.700000e+06</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 87 columns</p>
</div>



### Now that we have combined our zipcode dataframe, we can remove any scaled outliers by selecting the rows of numerical columns that are within 3 standard deviations of the mean for each column:


```python
num_cols = merge_selected.drop('price', axis=1).columns

for col in num_cols:
    merge_selected = merge_selected.loc[merge_selected[f'{col}'] <= 3]
    merge_selected = merge_selected.loc[merge_selected[f'{col}'] >= -3]
```


```python
zipcode_model = merge_selected
```

### We run through the same process with our Subregion Dataframe


```python
#loading in the subregion dataframe from our notebook
subregion_model = pd.read_csv('subregion_df.csv')
```


```python
subregion_model.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price_log</th>
      <th>sqft_above_log</th>
      <th>sqft_living_log</th>
      <th>sqft_living15_log</th>
      <th>sqft_lot_log</th>
      <th>sqft_lot15_log</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>bath_per_bed</th>
      <th>floors</th>
      <th>grade</th>
      <th>condition</th>
      <th>house_age</th>
      <th>price</th>
      <th>lat</th>
      <th>long</th>
      <th>waterfront</th>
      <th>renovated</th>
      <th>view</th>
      <th>has_basement</th>
      <th>subregion_east_urban</th>
      <th>subregion_north</th>
      <th>subregion_north_and_seattle</th>
      <th>subregion_seattle</th>
      <th>subregion_south_and_seattle</th>
      <th>subregion_south_rural</th>
      <th>subregion_south_urban</th>
      <th>subregion_vashon_island</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>7129300520</td>
      <td>-1.396379</td>
      <td>-0.761202</td>
      <td>-1.137533</td>
      <td>-1.045715</td>
      <td>-0.417664</td>
      <td>-0.416471</td>
      <td>-0.413282</td>
      <td>-1.452651</td>
      <td>-1.456963</td>
      <td>-0.912126</td>
      <td>-0.559105</td>
      <td>-0.637969</td>
      <td>0.531970</td>
      <td>221900.0</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>6414100192</td>
      <td>0.280619</td>
      <td>0.672297</td>
      <td>0.707592</td>
      <td>-0.335350</td>
      <td>-0.109907</td>
      <td>-0.012872</td>
      <td>-0.413282</td>
      <td>0.177861</td>
      <td>0.535990</td>
      <td>0.966991</td>
      <td>-0.559105</td>
      <td>-0.637969</td>
      <td>0.668363</td>
      <td>538000.0</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>5631500400</td>
      <td>-1.792645</td>
      <td>-1.765663</td>
      <td>-2.149421</td>
      <td>1.121457</td>
      <td>0.290145</td>
      <td>0.059247</td>
      <td>-1.495317</td>
      <td>-1.452651</td>
      <td>-0.659782</td>
      <td>-0.912126</td>
      <td>-1.413864</td>
      <td>-0.637969</td>
      <td>1.282131</td>
      <td>180000.0</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2487200875</td>
      <td>0.499734</td>
      <td>-1.035858</td>
      <td>0.065296</td>
      <td>-1.000364</td>
      <td>-0.569183</td>
      <td>-0.580015</td>
      <td>0.668754</td>
      <td>1.156169</td>
      <td>0.535990</td>
      <td>-0.912126</td>
      <td>-0.559105</td>
      <td>2.427758</td>
      <td>0.190987</td>
      <td>604000.0</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1954400510</td>
      <td>0.179413</td>
      <td>0.070077</td>
      <td>-0.300108</td>
      <td>-0.142320</td>
      <td>0.025839</td>
      <td>-0.036910</td>
      <td>-0.413282</td>
      <td>-0.148241</td>
      <td>0.137399</td>
      <td>-0.912126</td>
      <td>0.295654</td>
      <td>-0.637969</td>
      <td>-0.559174</td>
      <td>510000.0</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Our next step is to remove any columns that have a high collinearity with another column other than price_log (and price).

We do this by running our dataframes through a heat correlation map function:


```python
def heat_collinearity(corr, figsize=(8, 8)):
    """
    Definition:
    Shows the bottom triangle of a heat correlation
    
    Args:
    corr = corr() function - usually obtained from the outside function heat_corr()
    figsize = default = (12,12)
    
    Returns:
    A heat map
    """
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.zeros_like(corr, dtype=np.bool)
    idx = np.triu_indices_from(mask)
    mask[idx] = True
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, square=True, mask=mask, cmap='Reds', ax=ax)
    ax.set_ylim(len(corr.columns), 0);
    return fig, ax
```


```python
def heat_corr(df, target_cols, figsize=(8, 8)):
    """
    Definition:
    Creates a corr() of a dataframe and places the corr() within the heat_collinearity() function.
    If the amount of columns is > 50, figsize will change into (50,50)
    
    Args:
    df = selected dataframe
    target_cols = all continuous variables you wish to correlate
    figsize = default is (12,12)
    
    Returns:
    A heat map
    """
    corr = abs(df[target_cols].corr().round(2))
    if len(target_cols) > 50:
        heat_collinearity(corr, figsize=(50,50))
    else:
        heat_collinearity(corr, figsize=(12,12))
```


```python
zipcode_model.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_log</th>
      <th>sqft_above_log</th>
      <th>sqft_living_log</th>
      <th>sqft_living15_log</th>
      <th>sqft_lot_log</th>
      <th>sqft_lot15_log</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>bath_per_bed</th>
      <th>floors</th>
      <th>grade</th>
      <th>condition</th>
      <th>house_age</th>
      <th>price</th>
      <th>waterfront</th>
      <th>renovated</th>
      <th>view</th>
      <th>has_basement</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>zipcode_98028</th>
      <th>zipcode_98029</th>
      <th>zipcode_98030</th>
      <th>zipcode_98031</th>
      <th>zipcode_98032</th>
      <th>zipcode_98033</th>
      <th>zipcode_98034</th>
      <th>...</th>
      <th>zipcode_98055</th>
      <th>zipcode_98056</th>
      <th>zipcode_98058</th>
      <th>zipcode_98059</th>
      <th>zipcode_98065</th>
      <th>zipcode_98070</th>
      <th>zipcode_98072</th>
      <th>zipcode_98074</th>
      <th>zipcode_98075</th>
      <th>zipcode_98077</th>
      <th>zipcode_98092</th>
      <th>zipcode_98102</th>
      <th>zipcode_98103</th>
      <th>zipcode_98105</th>
      <th>zipcode_98106</th>
      <th>zipcode_98107</th>
      <th>zipcode_98108</th>
      <th>zipcode_98109</th>
      <th>zipcode_98112</th>
      <th>zipcode_98115</th>
      <th>zipcode_98116</th>
      <th>zipcode_98117</th>
      <th>zipcode_98118</th>
      <th>zipcode_98119</th>
      <th>zipcode_98122</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>7129300520</td>
      <td>-1.396379</td>
      <td>-0.761202</td>
      <td>-1.137533</td>
      <td>-1.045715</td>
      <td>-0.417664</td>
      <td>-0.416471</td>
      <td>-0.413282</td>
      <td>-1.452651</td>
      <td>-1.456963</td>
      <td>-0.912126</td>
      <td>-0.559105</td>
      <td>-0.637969</td>
      <td>0.53197</td>
      <td>221900.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 87 columns</p>
</div>




```python
#Making a list of all continuous columns within our dataframe to use in our heat_corr function
cont_vars = list(zipcode_model.columns[:13])
cont_vars
```




    ['price_log',
     'sqft_above_log',
     'sqft_living_log',
     'sqft_living15_log',
     'sqft_lot_log',
     'sqft_lot15_log',
     'bedrooms',
     'bathrooms',
     'bath_per_bed',
     'floors',
     'grade',
     'condition',
     'house_age']




```python
heat_corr(zipcode_model, cont_vars)
```


![png](output_65_0.png)



    <Figure size 864x864 with 0 Axes>


### Our strategy for removing columns will be:
    
- For any columns that share a high collinearity with one another, remove the one that has a weaker relationship with price_log.

For our zipcode model, we end up removing sqft_lot15_log, sqft_living15_log, 'sqft_above_log, and bathrooms


```python
zipcode_model.drop(['sqft_lot15_log', 'sqft_living15_log', 'bathrooms', 'sqft_above_log'], axis=1, inplace=True)


```


```python
for item in ['sqft_lot15_log', 'sqft_living15_log', 'bathrooms', 'sqft_above_log']:
    cont_vars.remove(item)
```


```python
heat_corr(zipcode_model, cont_vars)
```


![png](output_70_0.png)



    <Figure size 864x864 with 0 Axes>


We run through the same process above with our **subregion model.**

# Step 4 - Modeling

### Ordinary Least Squares

For this project, the linear model we are using to infer and predict housing prices comes from the statsmodels library:

**statsmodels.regression.linear_model.ols**

While not the strongest in terms of prediction, this model allows us to interpret which factors best determine our dependent variable given our independent variables. 

### Baseline Model

First we set the target **dependent variable('price_log')**, then we select the **predictors** we want to use in the model. We then feed these variables into our ols function along with our dataframe, and fit our model.


```python
target = 'price_log'
predictors = zipcode_model.drop(['price', 'price_log'], axis=1).columns
preds = '+'.join(predictors)
f = target + '~' + preds
model = ols(formula=f, data=zipcode_model).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>price_log</td>    <th>  R-squared:         </th> <td>   0.873</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.872</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1687.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 13 Aug 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>14:57:53</td>     <th>  Log-Likelihood:    </th> <td> -6490.4</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 20027</td>      <th>  AIC:               </th> <td>1.314e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 19945</td>      <th>  BIC:               </th> <td>1.379e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    81</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>       <td>   -0.9271</td> <td>    0.018</td> <td>  -50.903</td> <td> 0.000</td> <td>   -0.963</td> <td>   -0.891</td>
</tr>
<tr>
  <th>sqft_living_log</th> <td>    0.3732</td> <td>    0.006</td> <td>   67.178</td> <td> 0.000</td> <td>    0.362</td> <td>    0.384</td>
</tr>
<tr>
  <th>sqft_lot_log</th>    <td>    0.0845</td> <td>    0.004</td> <td>   20.383</td> <td> 0.000</td> <td>    0.076</td> <td>    0.093</td>
</tr>
<tr>
  <th>bedrooms</th>        <td>    0.0044</td> <td>    0.004</td> <td>    0.996</td> <td> 0.319</td> <td>   -0.004</td> <td>    0.013</td>
</tr>
<tr>
  <th>bath_per_bed</th>    <td>    0.0337</td> <td>    0.004</td> <td>    8.592</td> <td> 0.000</td> <td>    0.026</td> <td>    0.041</td>
</tr>
<tr>
  <th>floors</th>          <td>   -0.0005</td> <td>    0.004</td> <td>   -0.135</td> <td> 0.893</td> <td>   -0.008</td> <td>    0.007</td>
</tr>
<tr>
  <th>grade</th>           <td>    0.2405</td> <td>    0.005</td> <td>   53.362</td> <td> 0.000</td> <td>    0.232</td> <td>    0.249</td>
</tr>
<tr>
  <th>condition</th>       <td>    0.0572</td> <td>    0.003</td> <td>   20.889</td> <td> 0.000</td> <td>    0.052</td> <td>    0.063</td>
</tr>
<tr>
  <th>house_age</th>       <td>    0.0199</td> <td>    0.004</td> <td>    4.685</td> <td> 0.000</td> <td>    0.012</td> <td>    0.028</td>
</tr>
<tr>
  <th>waterfront</th>      <td>    1.0500</td> <td>    0.036</td> <td>   28.881</td> <td> 0.000</td> <td>    0.979</td> <td>    1.121</td>
</tr>
<tr>
  <th>renovated</th>       <td>    0.1182</td> <td>    0.014</td> <td>    8.475</td> <td> 0.000</td> <td>    0.091</td> <td>    0.146</td>
</tr>
<tr>
  <th>view</th>            <td>    0.2989</td> <td>    0.009</td> <td>   32.299</td> <td> 0.000</td> <td>    0.281</td> <td>    0.317</td>
</tr>
<tr>
  <th>has_basement</th>    <td>   -0.0868</td> <td>    0.006</td> <td>  -13.398</td> <td> 0.000</td> <td>   -0.100</td> <td>   -0.074</td>
</tr>
<tr>
  <th>zipcode_98002</th>   <td>   -0.0275</td> <td>    0.030</td> <td>   -0.912</td> <td> 0.362</td> <td>   -0.087</td> <td>    0.032</td>
</tr>
<tr>
  <th>zipcode_98003</th>   <td>    0.0197</td> <td>    0.027</td> <td>    0.727</td> <td> 0.467</td> <td>   -0.033</td> <td>    0.073</td>
</tr>
<tr>
  <th>zipcode_98004</th>   <td>    2.1252</td> <td>    0.027</td> <td>   77.815</td> <td> 0.000</td> <td>    2.072</td> <td>    2.179</td>
</tr>
<tr>
  <th>zipcode_98005</th>   <td>    1.4147</td> <td>    0.032</td> <td>   43.571</td> <td> 0.000</td> <td>    1.351</td> <td>    1.478</td>
</tr>
<tr>
  <th>zipcode_98006</th>   <td>    1.2459</td> <td>    0.024</td> <td>   51.181</td> <td> 0.000</td> <td>    1.198</td> <td>    1.294</td>
</tr>
<tr>
  <th>zipcode_98007</th>   <td>    1.2347</td> <td>    0.034</td> <td>   36.280</td> <td> 0.000</td> <td>    1.168</td> <td>    1.301</td>
</tr>
<tr>
  <th>zipcode_98008</th>   <td>    1.2333</td> <td>    0.027</td> <td>   45.354</td> <td> 0.000</td> <td>    1.180</td> <td>    1.287</td>
</tr>
<tr>
  <th>zipcode_98010</th>   <td>    0.4396</td> <td>    0.042</td> <td>   10.450</td> <td> 0.000</td> <td>    0.357</td> <td>    0.522</td>
</tr>
<tr>
  <th>zipcode_98011</th>   <td>    0.8792</td> <td>    0.030</td> <td>   29.206</td> <td> 0.000</td> <td>    0.820</td> <td>    0.938</td>
</tr>
<tr>
  <th>zipcode_98014</th>   <td>    0.5422</td> <td>    0.041</td> <td>   13.081</td> <td> 0.000</td> <td>    0.461</td> <td>    0.623</td>
</tr>
<tr>
  <th>zipcode_98019</th>   <td>    0.6245</td> <td>    0.032</td> <td>   19.539</td> <td> 0.000</td> <td>    0.562</td> <td>    0.687</td>
</tr>
<tr>
  <th>zipcode_98022</th>   <td>    0.0246</td> <td>    0.031</td> <td>    0.791</td> <td> 0.429</td> <td>   -0.036</td> <td>    0.086</td>
</tr>
<tr>
  <th>zipcode_98023</th>   <td>   -0.0423</td> <td>    0.024</td> <td>   -1.797</td> <td> 0.072</td> <td>   -0.088</td> <td>    0.004</td>
</tr>
<tr>
  <th>zipcode_98024</th>   <td>    0.7686</td> <td>    0.057</td> <td>   13.562</td> <td> 0.000</td> <td>    0.658</td> <td>    0.880</td>
</tr>
<tr>
  <th>zipcode_98027</th>   <td>    1.0577</td> <td>    0.026</td> <td>   40.764</td> <td> 0.000</td> <td>    1.007</td> <td>    1.109</td>
</tr>
<tr>
  <th>zipcode_98028</th>   <td>    0.8039</td> <td>    0.027</td> <td>   29.821</td> <td> 0.000</td> <td>    0.751</td> <td>    0.857</td>
</tr>
<tr>
  <th>zipcode_98029</th>   <td>    1.1690</td> <td>    0.026</td> <td>   44.132</td> <td> 0.000</td> <td>    1.117</td> <td>    1.221</td>
</tr>
<tr>
  <th>zipcode_98030</th>   <td>    0.1029</td> <td>    0.028</td> <td>    3.699</td> <td> 0.000</td> <td>    0.048</td> <td>    0.157</td>
</tr>
<tr>
  <th>zipcode_98031</th>   <td>    0.1420</td> <td>    0.027</td> <td>    5.182</td> <td> 0.000</td> <td>    0.088</td> <td>    0.196</td>
</tr>
<tr>
  <th>zipcode_98032</th>   <td>   -0.0356</td> <td>    0.036</td> <td>   -1.003</td> <td> 0.316</td> <td>   -0.105</td> <td>    0.034</td>
</tr>
<tr>
  <th>zipcode_98033</th>   <td>    1.4987</td> <td>    0.024</td> <td>   61.183</td> <td> 0.000</td> <td>    1.451</td> <td>    1.547</td>
</tr>
<tr>
  <th>zipcode_98034</th>   <td>    1.0382</td> <td>    0.023</td> <td>   44.809</td> <td> 0.000</td> <td>    0.993</td> <td>    1.084</td>
</tr>
<tr>
  <th>zipcode_98038</th>   <td>    0.3260</td> <td>    0.023</td> <td>   14.062</td> <td> 0.000</td> <td>    0.281</td> <td>    0.371</td>
</tr>
<tr>
  <th>zipcode_98039</th>   <td>    2.4393</td> <td>    0.064</td> <td>   38.086</td> <td> 0.000</td> <td>    2.314</td> <td>    2.565</td>
</tr>
<tr>
  <th>zipcode_98040</th>   <td>    1.6946</td> <td>    0.028</td> <td>   60.254</td> <td> 0.000</td> <td>    1.639</td> <td>    1.750</td>
</tr>
<tr>
  <th>zipcode_98042</th>   <td>    0.1215</td> <td>    0.023</td> <td>    5.236</td> <td> 0.000</td> <td>    0.076</td> <td>    0.167</td>
</tr>
<tr>
  <th>zipcode_98045</th>   <td>    0.6077</td> <td>    0.031</td> <td>   19.851</td> <td> 0.000</td> <td>    0.548</td> <td>    0.668</td>
</tr>
<tr>
  <th>zipcode_98052</th>   <td>    1.2384</td> <td>    0.023</td> <td>   53.888</td> <td> 0.000</td> <td>    1.193</td> <td>    1.283</td>
</tr>
<tr>
  <th>zipcode_98053</th>   <td>    1.1688</td> <td>    0.026</td> <td>   45.270</td> <td> 0.000</td> <td>    1.118</td> <td>    1.219</td>
</tr>
<tr>
  <th>zipcode_98055</th>   <td>    0.2699</td> <td>    0.028</td> <td>    9.675</td> <td> 0.000</td> <td>    0.215</td> <td>    0.325</td>
</tr>
<tr>
  <th>zipcode_98056</th>   <td>    0.6285</td> <td>    0.025</td> <td>   25.524</td> <td> 0.000</td> <td>    0.580</td> <td>    0.677</td>
</tr>
<tr>
  <th>zipcode_98058</th>   <td>    0.3091</td> <td>    0.024</td> <td>   12.808</td> <td> 0.000</td> <td>    0.262</td> <td>    0.356</td>
</tr>
<tr>
  <th>zipcode_98059</th>   <td>    0.6778</td> <td>    0.024</td> <td>   28.207</td> <td> 0.000</td> <td>    0.631</td> <td>    0.725</td>
</tr>
<tr>
  <th>zipcode_98065</th>   <td>    0.8510</td> <td>    0.027</td> <td>   31.641</td> <td> 0.000</td> <td>    0.798</td> <td>    0.904</td>
</tr>
<tr>
  <th>zipcode_98070</th>   <td>    0.4853</td> <td>    0.047</td> <td>   10.292</td> <td> 0.000</td> <td>    0.393</td> <td>    0.578</td>
</tr>
<tr>
  <th>zipcode_98072</th>   <td>    0.9118</td> <td>    0.028</td> <td>   32.706</td> <td> 0.000</td> <td>    0.857</td> <td>    0.966</td>
</tr>
<tr>
  <th>zipcode_98074</th>   <td>    1.0743</td> <td>    0.025</td> <td>   43.758</td> <td> 0.000</td> <td>    1.026</td> <td>    1.122</td>
</tr>
<tr>
  <th>zipcode_98075</th>   <td>    1.1228</td> <td>    0.026</td> <td>   43.275</td> <td> 0.000</td> <td>    1.072</td> <td>    1.174</td>
</tr>
<tr>
  <th>zipcode_98077</th>   <td>    0.8550</td> <td>    0.032</td> <td>   26.349</td> <td> 0.000</td> <td>    0.791</td> <td>    0.919</td>
</tr>
<tr>
  <th>zipcode_98092</th>   <td>    0.0253</td> <td>    0.026</td> <td>    0.961</td> <td> 0.337</td> <td>   -0.026</td> <td>    0.077</td>
</tr>
<tr>
  <th>zipcode_98102</th>   <td>    1.8980</td> <td>    0.041</td> <td>   46.157</td> <td> 0.000</td> <td>    1.817</td> <td>    1.979</td>
</tr>
<tr>
  <th>zipcode_98103</th>   <td>    1.6086</td> <td>    0.024</td> <td>   65.663</td> <td> 0.000</td> <td>    1.561</td> <td>    1.657</td>
</tr>
<tr>
  <th>zipcode_98105</th>   <td>    1.8547</td> <td>    0.031</td> <td>   60.539</td> <td> 0.000</td> <td>    1.795</td> <td>    1.915</td>
</tr>
<tr>
  <th>zipcode_98106</th>   <td>    0.6825</td> <td>    0.027</td> <td>   25.620</td> <td> 0.000</td> <td>    0.630</td> <td>    0.735</td>
</tr>
<tr>
  <th>zipcode_98107</th>   <td>    1.6340</td> <td>    0.029</td> <td>   56.174</td> <td> 0.000</td> <td>    1.577</td> <td>    1.691</td>
</tr>
<tr>
  <th>zipcode_98108</th>   <td>    0.7124</td> <td>    0.031</td> <td>   22.861</td> <td> 0.000</td> <td>    0.651</td> <td>    0.773</td>
</tr>
<tr>
  <th>zipcode_98109</th>   <td>    1.9274</td> <td>    0.039</td> <td>   49.248</td> <td> 0.000</td> <td>    1.851</td> <td>    2.004</td>
</tr>
<tr>
  <th>zipcode_98112</th>   <td>    2.0240</td> <td>    0.030</td> <td>   67.321</td> <td> 0.000</td> <td>    1.965</td> <td>    2.083</td>
</tr>
<tr>
  <th>zipcode_98115</th>   <td>    1.5797</td> <td>    0.024</td> <td>   66.346</td> <td> 0.000</td> <td>    1.533</td> <td>    1.626</td>
</tr>
<tr>
  <th>zipcode_98116</th>   <td>    1.4929</td> <td>    0.027</td> <td>   54.801</td> <td> 0.000</td> <td>    1.439</td> <td>    1.546</td>
</tr>
<tr>
  <th>zipcode_98117</th>   <td>    1.5695</td> <td>    0.024</td> <td>   64.763</td> <td> 0.000</td> <td>    1.522</td> <td>    1.617</td>
</tr>
<tr>
  <th>zipcode_98118</th>   <td>    0.9071</td> <td>    0.024</td> <td>   37.367</td> <td> 0.000</td> <td>    0.859</td> <td>    0.955</td>
</tr>
<tr>
  <th>zipcode_98119</th>   <td>    1.8794</td> <td>    0.033</td> <td>   56.371</td> <td> 0.000</td> <td>    1.814</td> <td>    1.945</td>
</tr>
<tr>
  <th>zipcode_98122</th>   <td>    1.5779</td> <td>    0.029</td> <td>   54.705</td> <td> 0.000</td> <td>    1.521</td> <td>    1.634</td>
</tr>
<tr>
  <th>zipcode_98125</th>   <td>    1.0854</td> <td>    0.025</td> <td>   43.468</td> <td> 0.000</td> <td>    1.036</td> <td>    1.134</td>
</tr>
<tr>
  <th>zipcode_98126</th>   <td>    1.0679</td> <td>    0.026</td> <td>   40.486</td> <td> 0.000</td> <td>    1.016</td> <td>    1.120</td>
</tr>
<tr>
  <th>zipcode_98133</th>   <td>    0.8825</td> <td>    0.024</td> <td>   36.743</td> <td> 0.000</td> <td>    0.835</td> <td>    0.930</td>
</tr>
<tr>
  <th>zipcode_98136</th>   <td>    1.3220</td> <td>    0.029</td> <td>   45.916</td> <td> 0.000</td> <td>    1.266</td> <td>    1.378</td>
</tr>
<tr>
  <th>zipcode_98144</th>   <td>    1.3243</td> <td>    0.027</td> <td>   48.439</td> <td> 0.000</td> <td>    1.271</td> <td>    1.378</td>
</tr>
<tr>
  <th>zipcode_98146</th>   <td>    0.5449</td> <td>    0.027</td> <td>   20.020</td> <td> 0.000</td> <td>    0.492</td> <td>    0.598</td>
</tr>
<tr>
  <th>zipcode_98148</th>   <td>    0.2908</td> <td>    0.049</td> <td>    5.972</td> <td> 0.000</td> <td>    0.195</td> <td>    0.386</td>
</tr>
<tr>
  <th>zipcode_98155</th>   <td>    0.8175</td> <td>    0.024</td> <td>   33.779</td> <td> 0.000</td> <td>    0.770</td> <td>    0.865</td>
</tr>
<tr>
  <th>zipcode_98166</th>   <td>    0.5794</td> <td>    0.028</td> <td>   20.553</td> <td> 0.000</td> <td>    0.524</td> <td>    0.635</td>
</tr>
<tr>
  <th>zipcode_98168</th>   <td>    0.1741</td> <td>    0.028</td> <td>    6.260</td> <td> 0.000</td> <td>    0.120</td> <td>    0.229</td>
</tr>
<tr>
  <th>zipcode_98177</th>   <td>    1.1422</td> <td>    0.028</td> <td>   40.489</td> <td> 0.000</td> <td>    1.087</td> <td>    1.198</td>
</tr>
<tr>
  <th>zipcode_98178</th>   <td>    0.2860</td> <td>    0.028</td> <td>   10.239</td> <td> 0.000</td> <td>    0.231</td> <td>    0.341</td>
</tr>
<tr>
  <th>zipcode_98188</th>   <td>    0.1778</td> <td>    0.034</td> <td>    5.199</td> <td> 0.000</td> <td>    0.111</td> <td>    0.245</td>
</tr>
<tr>
  <th>zipcode_98198</th>   <td>    0.1184</td> <td>    0.027</td> <td>    4.341</td> <td> 0.000</td> <td>    0.065</td> <td>    0.172</td>
</tr>
<tr>
  <th>zipcode_98199</th>   <td>    1.6618</td> <td>    0.027</td> <td>   60.624</td> <td> 0.000</td> <td>    1.608</td> <td>    1.715</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1270.369</td> <th>  Durbin-Watson:     </th> <td>   1.976</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>5049.532</td>
</tr>
<tr>
  <th>Skew:</th>           <td>-0.195</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 5.429</td>  <th>  Cond. No.          </th> <td>    109.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



Reading our model.summary(), we see that some of the predictor values are not significant for determining 'price_log'. Instead of manually removing the non-significant predictors, rerunning the model, and repeating the process, we create some functions that do all that for us.


```python
def make_model(target, data, x_cols):
    """
    Definition:
    makes a model by using ols from statsmodels.formula.api
    
    Args:
    target = dependent variable
    data = selected dataframe
    x_cols = independent variables
    
    Returns:
    A model
    """
    preds = '+'.join(x_cols)
    f = target + '~' + preds
    model = ols(formula=f, data=data).fit()
    return model    
```


```python
def loop_alphas(target, data, x_cols, n=5):
    """
    Definition:
    Runs through a loop n number of times, taking out predictor variables within the model.summary table
    that have p-values greater than 0.05. Returns the new model and model.summary
    
    Args:
    target = dependent variable
    data = selected Dataframe
    x_cols = independent variables
    n = number of times to run sequence
    
    Returns:
    The model and the summary
    """
    i = 1
    while i < n:
        model = make_model(target, data, x_cols)
        summary = make_model(target, data, x_cols).summary()
        p_table = summary.tables[1]
        p_table_df = pd.DataFrame(p_table.data)
        p_table_df.columns = p_table_df.iloc[0]
        p_table_df.drop(0, inplace=True)
        p_table_df.set_index(p_table_df.columns[0], inplace=True)
        p_table_df['P>|t|'] = p_table_df['P>|t|'].astype('float64')
        new_x_cols = list(p_table_df[p_table_df['P>|t|'] < 0.05].index)
        new_x_cols.remove('Intercept')
        x_cols = new_x_cols
        i += 1
#         print(len(new_x_cols), len(p_table_df))
    return model, summary
```


```python
def qq_plot(model):
    """
    Definition:
    Creates a qq plot of the model's residuals to test for normality
    
    Args:
    model = selected model
    
    Returns:
    A graphed qq plot
    """
    resids = model.resid
    with plt.style.context('seaborn'):
        sm.graphics.qqplot(resids, stats.norm, line='45', fit=True);
```


```python
def resid_scatter(model, target, data):
    """
    Definition:
    Plots a scatter plot to test model residuals for any obvious heteroscedasticity
    
    Args:
    model = selected model
    target = dependent variable
    data = selected DataFrame
    
    Returns:
    a graphed scatter plot comparing model residuals to target variable
    """
    resids = model.resid
    with plt.style.context('seaborn'):
        plt.scatter(data[target], resids);
```


```python
def dist_resid(model):
    """
    Definition:
    plots a distribution plot of the model's residuals
    
    Args:
    model = selected model
    
    Returns:
    A graphed distplot of the residuals
    """
    resids = model.resid
    plt.figure(figsize=(8,5))
    sns.distplot(resids);
```

Using our new functions, we can easily run through these functions to remove insignificant values (variables that have a p-value > 0.05), use the qqplot to determine how normally distributed the data residuals are, use a scatterplot to see how the reisduals lay, and view the distribution of the residuals as well.

### Question 2: 
**Does a model perform better when we use zipcode as our location category or subregions as our location category?**

We want to know if there is any difference in strength between the zipcodes and subregions categories. This information woud allow us to see which model may be better to use. It could help homesellers by giving them a better understanding of why their home is priced the way it is. Also, if they intend to stay in King County it could give them an idea of the next zipcode/subregion they would like to live in given the price of houses in those areas.


```python
model, summary = loop_alphas('price_log', zipcode_model, zipcode_model.drop(['price_log', 'price'], axis=1).columns)
```


```python
summary
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>price_log</td>    <th>  R-squared:         </th> <td>   0.873</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.872</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1871.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 13 Aug 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>14:58:59</td>     <th>  Log-Likelihood:    </th> <td> -6497.5</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 20027</td>      <th>  AIC:               </th> <td>1.314e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 19953</td>      <th>  BIC:               </th> <td>1.373e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    73</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>       <td>   -0.9340</td> <td>    0.008</td> <td> -113.869</td> <td> 0.000</td> <td>   -0.950</td> <td>   -0.918</td>
</tr>
<tr>
  <th>sqft_living_log</th> <td>    0.3766</td> <td>    0.004</td> <td>   90.475</td> <td> 0.000</td> <td>    0.368</td> <td>    0.385</td>
</tr>
<tr>
  <th>sqft_lot_log</th>    <td>    0.0849</td> <td>    0.004</td> <td>   21.615</td> <td> 0.000</td> <td>    0.077</td> <td>    0.093</td>
</tr>
<tr>
  <th>bath_per_bed</th>    <td>    0.0315</td> <td>    0.003</td> <td>   10.269</td> <td> 0.000</td> <td>    0.025</td> <td>    0.037</td>
</tr>
<tr>
  <th>grade</th>           <td>    0.2398</td> <td>    0.004</td> <td>   53.695</td> <td> 0.000</td> <td>    0.231</td> <td>    0.249</td>
</tr>
<tr>
  <th>condition</th>       <td>    0.0574</td> <td>    0.003</td> <td>   21.094</td> <td> 0.000</td> <td>    0.052</td> <td>    0.063</td>
</tr>
<tr>
  <th>house_age</th>       <td>    0.0187</td> <td>    0.004</td> <td>    4.581</td> <td> 0.000</td> <td>    0.011</td> <td>    0.027</td>
</tr>
<tr>
  <th>waterfront</th>      <td>    1.0474</td> <td>    0.036</td> <td>   28.839</td> <td> 0.000</td> <td>    0.976</td> <td>    1.119</td>
</tr>
<tr>
  <th>renovated</th>       <td>    0.1197</td> <td>    0.014</td> <td>    8.609</td> <td> 0.000</td> <td>    0.092</td> <td>    0.147</td>
</tr>
<tr>
  <th>view</th>            <td>    0.2994</td> <td>    0.009</td> <td>   32.425</td> <td> 0.000</td> <td>    0.281</td> <td>    0.318</td>
</tr>
<tr>
  <th>has_basement</th>    <td>   -0.0873</td> <td>    0.006</td> <td>  -15.352</td> <td> 0.000</td> <td>   -0.098</td> <td>   -0.076</td>
</tr>
<tr>
  <th>zipcode_98004</th>   <td>    2.1323</td> <td>    0.022</td> <td>   97.354</td> <td> 0.000</td> <td>    2.089</td> <td>    2.175</td>
</tr>
<tr>
  <th>zipcode_98005</th>   <td>    1.4222</td> <td>    0.028</td> <td>   50.643</td> <td> 0.000</td> <td>    1.367</td> <td>    1.477</td>
</tr>
<tr>
  <th>zipcode_98006</th>   <td>    1.2527</td> <td>    0.018</td> <td>   69.356</td> <td> 0.000</td> <td>    1.217</td> <td>    1.288</td>
</tr>
<tr>
  <th>zipcode_98007</th>   <td>    1.2426</td> <td>    0.030</td> <td>   41.600</td> <td> 0.000</td> <td>    1.184</td> <td>    1.301</td>
</tr>
<tr>
  <th>zipcode_98008</th>   <td>    1.2413</td> <td>    0.022</td> <td>   57.155</td> <td> 0.000</td> <td>    1.199</td> <td>    1.284</td>
</tr>
<tr>
  <th>zipcode_98010</th>   <td>    0.4451</td> <td>    0.039</td> <td>   11.446</td> <td> 0.000</td> <td>    0.369</td> <td>    0.521</td>
</tr>
<tr>
  <th>zipcode_98011</th>   <td>    0.8860</td> <td>    0.025</td> <td>   34.894</td> <td> 0.000</td> <td>    0.836</td> <td>    0.936</td>
</tr>
<tr>
  <th>zipcode_98014</th>   <td>    0.5474</td> <td>    0.038</td> <td>   14.311</td> <td> 0.000</td> <td>    0.472</td> <td>    0.622</td>
</tr>
<tr>
  <th>zipcode_98019</th>   <td>    0.6304</td> <td>    0.028</td> <td>   22.810</td> <td> 0.000</td> <td>    0.576</td> <td>    0.685</td>
</tr>
<tr>
  <th>zipcode_98024</th>   <td>    0.7752</td> <td>    0.054</td> <td>   14.260</td> <td> 0.000</td> <td>    0.669</td> <td>    0.882</td>
</tr>
<tr>
  <th>zipcode_98027</th>   <td>    1.0643</td> <td>    0.020</td> <td>   52.553</td> <td> 0.000</td> <td>    1.025</td> <td>    1.104</td>
</tr>
<tr>
  <th>zipcode_98028</th>   <td>    0.8107</td> <td>    0.022</td> <td>   37.591</td> <td> 0.000</td> <td>    0.768</td> <td>    0.853</td>
</tr>
<tr>
  <th>zipcode_98029</th>   <td>    1.1757</td> <td>    0.021</td> <td>   56.280</td> <td> 0.000</td> <td>    1.135</td> <td>    1.217</td>
</tr>
<tr>
  <th>zipcode_98030</th>   <td>    0.1098</td> <td>    0.023</td> <td>    4.843</td> <td> 0.000</td> <td>    0.065</td> <td>    0.154</td>
</tr>
<tr>
  <th>zipcode_98031</th>   <td>    0.1490</td> <td>    0.022</td> <td>    6.737</td> <td> 0.000</td> <td>    0.106</td> <td>    0.192</td>
</tr>
<tr>
  <th>zipcode_98033</th>   <td>    1.5054</td> <td>    0.018</td> <td>   82.178</td> <td> 0.000</td> <td>    1.470</td> <td>    1.541</td>
</tr>
<tr>
  <th>zipcode_98034</th>   <td>    1.0456</td> <td>    0.017</td> <td>   63.247</td> <td> 0.000</td> <td>    1.013</td> <td>    1.078</td>
</tr>
<tr>
  <th>zipcode_98038</th>   <td>    0.3323</td> <td>    0.017</td> <td>   19.921</td> <td> 0.000</td> <td>    0.300</td> <td>    0.365</td>
</tr>
<tr>
  <th>zipcode_98039</th>   <td>    2.4460</td> <td>    0.062</td> <td>   39.492</td> <td> 0.000</td> <td>    2.325</td> <td>    2.567</td>
</tr>
<tr>
  <th>zipcode_98040</th>   <td>    1.7020</td> <td>    0.023</td> <td>   74.435</td> <td> 0.000</td> <td>    1.657</td> <td>    1.747</td>
</tr>
<tr>
  <th>zipcode_98042</th>   <td>    0.1279</td> <td>    0.017</td> <td>    7.668</td> <td> 0.000</td> <td>    0.095</td> <td>    0.161</td>
</tr>
<tr>
  <th>zipcode_98045</th>   <td>    0.6137</td> <td>    0.026</td> <td>   23.526</td> <td> 0.000</td> <td>    0.563</td> <td>    0.665</td>
</tr>
<tr>
  <th>zipcode_98052</th>   <td>    1.2452</td> <td>    0.016</td> <td>   76.511</td> <td> 0.000</td> <td>    1.213</td> <td>    1.277</td>
</tr>
<tr>
  <th>zipcode_98053</th>   <td>    1.1738</td> <td>    0.020</td> <td>   58.255</td> <td> 0.000</td> <td>    1.134</td> <td>    1.213</td>
</tr>
<tr>
  <th>zipcode_98055</th>   <td>    0.2769</td> <td>    0.023</td> <td>   12.207</td> <td> 0.000</td> <td>    0.232</td> <td>    0.321</td>
</tr>
<tr>
  <th>zipcode_98056</th>   <td>    0.6351</td> <td>    0.019</td> <td>   34.290</td> <td> 0.000</td> <td>    0.599</td> <td>    0.671</td>
</tr>
<tr>
  <th>zipcode_98058</th>   <td>    0.3160</td> <td>    0.018</td> <td>   17.650</td> <td> 0.000</td> <td>    0.281</td> <td>    0.351</td>
</tr>
<tr>
  <th>zipcode_98059</th>   <td>    0.6842</td> <td>    0.018</td> <td>   38.434</td> <td> 0.000</td> <td>    0.649</td> <td>    0.719</td>
</tr>
<tr>
  <th>zipcode_98065</th>   <td>    0.8566</td> <td>    0.022</td> <td>   39.808</td> <td> 0.000</td> <td>    0.814</td> <td>    0.899</td>
</tr>
<tr>
  <th>zipcode_98070</th>   <td>    0.4904</td> <td>    0.044</td> <td>   11.070</td> <td> 0.000</td> <td>    0.404</td> <td>    0.577</td>
</tr>
<tr>
  <th>zipcode_98072</th>   <td>    0.9178</td> <td>    0.023</td> <td>   40.302</td> <td> 0.000</td> <td>    0.873</td> <td>    0.962</td>
</tr>
<tr>
  <th>zipcode_98074</th>   <td>    1.0805</td> <td>    0.018</td> <td>   58.573</td> <td> 0.000</td> <td>    1.044</td> <td>    1.117</td>
</tr>
<tr>
  <th>zipcode_98075</th>   <td>    1.1288</td> <td>    0.020</td> <td>   55.658</td> <td> 0.000</td> <td>    1.089</td> <td>    1.169</td>
</tr>
<tr>
  <th>zipcode_98077</th>   <td>    0.8602</td> <td>    0.028</td> <td>   30.505</td> <td> 0.000</td> <td>    0.805</td> <td>    0.916</td>
</tr>
<tr>
  <th>zipcode_98102</th>   <td>    1.9065</td> <td>    0.037</td> <td>   50.928</td> <td> 0.000</td> <td>    1.833</td> <td>    1.980</td>
</tr>
<tr>
  <th>zipcode_98103</th>   <td>    1.6165</td> <td>    0.018</td> <td>   90.466</td> <td> 0.000</td> <td>    1.581</td> <td>    1.652</td>
</tr>
<tr>
  <th>zipcode_98105</th>   <td>    1.8634</td> <td>    0.026</td> <td>   72.572</td> <td> 0.000</td> <td>    1.813</td> <td>    1.914</td>
</tr>
<tr>
  <th>zipcode_98106</th>   <td>    0.6907</td> <td>    0.021</td> <td>   32.848</td> <td> 0.000</td> <td>    0.649</td> <td>    0.732</td>
</tr>
<tr>
  <th>zipcode_98107</th>   <td>    1.6425</td> <td>    0.024</td> <td>   69.117</td> <td> 0.000</td> <td>    1.596</td> <td>    1.689</td>
</tr>
<tr>
  <th>zipcode_98108</th>   <td>    0.7206</td> <td>    0.026</td> <td>   27.198</td> <td> 0.000</td> <td>    0.669</td> <td>    0.772</td>
</tr>
<tr>
  <th>zipcode_98109</th>   <td>    1.9358</td> <td>    0.035</td> <td>   54.754</td> <td> 0.000</td> <td>    1.867</td> <td>    2.005</td>
</tr>
<tr>
  <th>zipcode_98112</th>   <td>    2.0322</td> <td>    0.025</td> <td>   81.494</td> <td> 0.000</td> <td>    1.983</td> <td>    2.081</td>
</tr>
<tr>
  <th>zipcode_98115</th>   <td>    1.5874</td> <td>    0.017</td> <td>   92.634</td> <td> 0.000</td> <td>    1.554</td> <td>    1.621</td>
</tr>
<tr>
  <th>zipcode_98116</th>   <td>    1.5003</td> <td>    0.022</td> <td>   69.422</td> <td> 0.000</td> <td>    1.458</td> <td>    1.543</td>
</tr>
<tr>
  <th>zipcode_98117</th>   <td>    1.5772</td> <td>    0.018</td> <td>   89.221</td> <td> 0.000</td> <td>    1.543</td> <td>    1.612</td>
</tr>
<tr>
  <th>zipcode_98118</th>   <td>    0.9148</td> <td>    0.018</td> <td>   51.180</td> <td> 0.000</td> <td>    0.880</td> <td>    0.950</td>
</tr>
<tr>
  <th>zipcode_98119</th>   <td>    1.8879</td> <td>    0.029</td> <td>   65.580</td> <td> 0.000</td> <td>    1.831</td> <td>    1.944</td>
</tr>
<tr>
  <th>zipcode_98122</th>   <td>    1.5866</td> <td>    0.023</td> <td>   67.661</td> <td> 0.000</td> <td>    1.541</td> <td>    1.633</td>
</tr>
<tr>
  <th>zipcode_98125</th>   <td>    1.0926</td> <td>    0.019</td> <td>   57.832</td> <td> 0.000</td> <td>    1.056</td> <td>    1.130</td>
</tr>
<tr>
  <th>zipcode_98126</th>   <td>    1.0751</td> <td>    0.021</td> <td>   52.202</td> <td> 0.000</td> <td>    1.035</td> <td>    1.115</td>
</tr>
<tr>
  <th>zipcode_98133</th>   <td>    0.8897</td> <td>    0.018</td> <td>   50.518</td> <td> 0.000</td> <td>    0.855</td> <td>    0.924</td>
</tr>
<tr>
  <th>zipcode_98136</th>   <td>    1.3293</td> <td>    0.024</td> <td>   56.358</td> <td> 0.000</td> <td>    1.283</td> <td>    1.375</td>
</tr>
<tr>
  <th>zipcode_98144</th>   <td>    1.3323</td> <td>    0.022</td> <td>   61.439</td> <td> 0.000</td> <td>    1.290</td> <td>    1.375</td>
</tr>
<tr>
  <th>zipcode_98146</th>   <td>    0.5521</td> <td>    0.022</td> <td>   25.281</td> <td> 0.000</td> <td>    0.509</td> <td>    0.595</td>
</tr>
<tr>
  <th>zipcode_98148</th>   <td>    0.2982</td> <td>    0.046</td> <td>    6.491</td> <td> 0.000</td> <td>    0.208</td> <td>    0.388</td>
</tr>
<tr>
  <th>zipcode_98155</th>   <td>    0.8248</td> <td>    0.018</td> <td>   45.955</td> <td> 0.000</td> <td>    0.790</td> <td>    0.860</td>
</tr>
<tr>
  <th>zipcode_98166</th>   <td>    0.5864</td> <td>    0.023</td> <td>   25.451</td> <td> 0.000</td> <td>    0.541</td> <td>    0.632</td>
</tr>
<tr>
  <th>zipcode_98168</th>   <td>    0.1809</td> <td>    0.023</td> <td>    7.998</td> <td> 0.000</td> <td>    0.137</td> <td>    0.225</td>
</tr>
<tr>
  <th>zipcode_98177</th>   <td>    1.1490</td> <td>    0.023</td> <td>   49.973</td> <td> 0.000</td> <td>    1.104</td> <td>    1.194</td>
</tr>
<tr>
  <th>zipcode_98178</th>   <td>    0.2936</td> <td>    0.023</td> <td>   12.932</td> <td> 0.000</td> <td>    0.249</td> <td>    0.338</td>
</tr>
<tr>
  <th>zipcode_98188</th>   <td>    0.1853</td> <td>    0.030</td> <td>    6.150</td> <td> 0.000</td> <td>    0.126</td> <td>    0.244</td>
</tr>
<tr>
  <th>zipcode_98198</th>   <td>    0.1252</td> <td>    0.022</td> <td>    5.708</td> <td> 0.000</td> <td>    0.082</td> <td>    0.168</td>
</tr>
<tr>
  <th>zipcode_98199</th>   <td>    1.6692</td> <td>    0.022</td> <td>   76.451</td> <td> 0.000</td> <td>    1.626</td> <td>    1.712</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1269.591</td> <th>  Durbin-Watson:     </th> <td>   1.976</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>5027.566</td>
</tr>
<tr>
  <th>Skew:</th>           <td>-0.197</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 5.423</td>  <th>  Cond. No.          </th> <td>    44.8</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
qq_plot(model)
```


![png](output_88_0.png)



```python
resid_scatter(model, 'price_log', zipcode_model)
```


![png](output_89_0.png)



```python
dist_resid(model)
```


![png](output_90_0.png)


Next, we need to make sure there is no multicollinearity within any of our columns, so we will create another function that uses variance inflation factor to find out if there are such columns in our model.

### VIF


```python
def vif(df, non_target_cols):
    """
    Definition:
    Returns a list with each object in the list being the name of a column and it's variance inflation factor
    
    Args:
    df = selected dataframe
    non_target_cols = columns that are not independent variables
    """
    x_targets = df.drop(non_target_cols, axis=1)
    vif = [variance_inflation_factor(x_targets.values, i) for i in range(x_targets.shape[1])]
    return list(zip(x_targets, vif))
```


```python
vif(zipcode_model, ['price', 'price_log'])
```




    [('sqft_living_log', 4.972197986574189),
     ('sqft_lot_log', 2.3667472290482294),
     ('bedrooms', 2.8957244681030447),
     ('bath_per_bed', 2.4208232571921595),
     ('floors', 2.576384759447206),
     ('grade', 3.082980390946963),
     ('condition', 1.3145809460108067),
     ('house_age', 3.1460872492180276),
     ('waterfront', 1.1052866935761168),
     ('renovated', 1.1627623987045597),
     ('view', 1.3798812577545436),
     ('has_basement', 2.903060580740776),
     ('zipcode_98002', 1.0184760399520656),
     ('zipcode_98003', 1.0204509081439295),
     ('zipcode_98004', 1.0301140549082917),
     ('zipcode_98005', 1.0230110082012023),
     ('zipcode_98006', 1.0681157607136005),
     ('zipcode_98007', 1.0081316035207502),
     ('zipcode_98008', 1.0254336851311725),
     ('zipcode_98010', 1.0209745706432425),
     ('zipcode_98011', 1.0128319209601941),
     ('zipcode_98014', 1.0235505221452745),
     ('zipcode_98019', 1.0235164739723064),
     ('zipcode_98022', 1.0234838817121872),
     ('zipcode_98023', 1.0407327157144168),
     ('zipcode_98024', 1.0086103999123788),
     ('zipcode_98027', 1.044169600008814),
     ('zipcode_98028', 1.020157252766611),
     ('zipcode_98029', 1.0320204986389645),
     ('zipcode_98030', 1.0183150909543717),
     ('zipcode_98031', 1.0207978249731613),
     ('zipcode_98032', 1.0135940926973241),
     ('zipcode_98033', 1.025257455840336),
     ('zipcode_98034', 1.040616378859402),
     ('zipcode_98038', 1.058493473465964),
     ('zipcode_98039', 1.0072106700681476),
     ('zipcode_98040', 1.0537781797284143),
     ('zipcode_98042', 1.0529623835170123),
     ('zipcode_98045', 1.0344340600433028),
     ('zipcode_98052', 1.0394492366424555),
     ('zipcode_98053', 1.0489018179091172),
     ('zipcode_98055', 1.0140364280830674),
     ('zipcode_98056', 1.0296652255509726),
     ('zipcode_98058', 1.0299643968918526),
     ('zipcode_98059', 1.034160223864461),
     ('zipcode_98065', 1.0373858514373406),
     ('zipcode_98070', 1.0637942098237754),
     ('zipcode_98072', 1.0478772203821505),
     ('zipcode_98074', 1.046683762291207),
     ('zipcode_98075', 1.0515274376706087),
     ('zipcode_98077', 1.0702497055065232),
     ('zipcode_98092', 1.0258102682059145),
     ('zipcode_98102', 1.0682050002740215),
     ('zipcode_98103', 1.2464293298445661),
     ('zipcode_98105', 1.099944966734597),
     ('zipcode_98106', 1.0667472541037266),
     ('zipcode_98107', 1.1237400615553852),
     ('zipcode_98108', 1.0386339411682943),
     ('zipcode_98109', 1.0662668574323853),
     ('zipcode_98112', 1.1355505766150846),
     ('zipcode_98115', 1.1663334817026374),
     ('zipcode_98116', 1.1088157484960957),
     ('zipcode_98117', 1.1772625822217853),
     ('zipcode_98118', 1.114605059667573),
     ('zipcode_98119', 1.1087240653645491),
     ('zipcode_98122', 1.1511411882571272),
     ('zipcode_98125', 1.0524549841829516),
     ('zipcode_98126', 1.0922609597739577),
     ('zipcode_98133', 1.0519658475017348),
     ('zipcode_98136', 1.0796921363417735),
     ('zipcode_98144', 1.1300995321806275),
     ('zipcode_98146', 1.037600615281755),
     ('zipcode_98148', 1.0040652181050846),
     ('zipcode_98155', 1.0411958383051279),
     ('zipcode_98166', 1.0352357601137474),
     ('zipcode_98168', 1.0477440390737212),
     ('zipcode_98177', 1.037185349211171),
     ('zipcode_98178', 1.047089218967789),
     ('zipcode_98188', 1.013282571962256),
     ('zipcode_98198', 1.0355511102577912),
     ('zipcode_98199', 1.1127729784134357)]



As long as we don't see anything higher than 10, we should be relatively ok with this. Our last step for testing the zipcode model will be to validate it. We want to make sure it's a stable model.

### Cross Validation


```python
linreg = LinearRegression()
X = zipcode_model.drop(['price', 'price_log'], axis=1)
y = zipcode_model[['price_log']]
```


```python
cv_5_mse = np.mean(cross_val_score(linreg, X, y, cv=5, scoring='neg_mean_squared_error'))
cv_10_mse = np.mean(cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error'))
cv_20_mse = np.mean(cross_val_score(linreg, X, y, cv=20, scoring='neg_mean_squared_error'))

cv_5_r_squared = np.mean(cross_val_score(linreg, X, y, cv=5, scoring='r2'))
cv_10_r_squared = np.mean(cross_val_score(linreg, X, y, cv=10, scoring='r2'))
cv_20_r_squared = np.mean(cross_val_score(linreg, X, y, cv=20, scoring='r2'))
```


```python
print(f"CV 5 MSE: {cv_5_mse}\nCV 10 MSE: {cv_10_mse}\nCV 20 MSE: {cv_20_mse}"
      f"\n\nCV 5 R2: {cv_5_r_squared}\nCV 10 R2: {cv_10_r_squared}\nCV 20 R2: {cv_20_r_squared}")
```

    CV 5 MSE: -0.1140227941738172
    CV 10 MSE: -0.11388728959479921
    CV 20 MSE: -0.11338189853840208
    
    CV 5 R2: 0.8698312804889257
    CV 10 R2: 0.8695018429732556
    CV 20 R2: 0.8701195339865408
    

Based on the results of the cross validation, our model seems to be stable and isn't overfitted or underfitted.

### We repeat this process with our subregion model


```python
#loading in subregion dataframe from our jupyter notebook
subregion_model = pd.read_csv('sub_model_final.csv', index_col='id')
```


```python
subregion_model.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_log</th>
      <th>sqft_living_log</th>
      <th>sqft_living15_log</th>
      <th>sqft_lot_log</th>
      <th>bedrooms</th>
      <th>bath_per_bed</th>
      <th>floors</th>
      <th>grade</th>
      <th>condition</th>
      <th>house_age</th>
      <th>price</th>
      <th>waterfront</th>
      <th>renovated</th>
      <th>view</th>
      <th>has_basement</th>
      <th>subregion_east_urban</th>
      <th>subregion_north</th>
      <th>subregion_north_and_seattle</th>
      <th>subregion_seattle</th>
      <th>subregion_south_and_seattle</th>
      <th>subregion_south_rural</th>
      <th>subregion_south_urban</th>
      <th>subregion_vashon_island</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>7129300520</td>
      <td>-1.396379</td>
      <td>-1.137533</td>
      <td>-1.045715</td>
      <td>-0.417664</td>
      <td>-0.413282</td>
      <td>-1.456963</td>
      <td>-0.912126</td>
      <td>-0.559105</td>
      <td>-0.637969</td>
      <td>0.531970</td>
      <td>221900.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>6414100192</td>
      <td>0.280619</td>
      <td>0.707592</td>
      <td>-0.335350</td>
      <td>-0.109907</td>
      <td>-0.413282</td>
      <td>0.535990</td>
      <td>0.966991</td>
      <td>-0.559105</td>
      <td>-0.637969</td>
      <td>0.668363</td>
      <td>538000.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5631500400</td>
      <td>-1.792645</td>
      <td>-2.149421</td>
      <td>1.121457</td>
      <td>0.290145</td>
      <td>-1.495317</td>
      <td>-0.659782</td>
      <td>-0.912126</td>
      <td>-1.413864</td>
      <td>-0.637969</td>
      <td>1.282131</td>
      <td>180000.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2487200875</td>
      <td>0.499734</td>
      <td>0.065296</td>
      <td>-1.000364</td>
      <td>-0.569183</td>
      <td>0.668754</td>
      <td>0.535990</td>
      <td>-0.912126</td>
      <td>-0.559105</td>
      <td>2.427758</td>
      <td>0.190987</td>
      <td>604000.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1954400510</td>
      <td>0.179413</td>
      <td>-0.300108</td>
      <td>-0.142320</td>
      <td>0.025839</td>
      <td>-0.413282</td>
      <td>0.137399</td>
      <td>-0.912126</td>
      <td>0.295654</td>
      <td>-0.637969</td>
      <td>-0.559174</td>
      <td>510000.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sub_model, sub_summary = loop_alphas('price_log', subregion_model, subregion_model.drop(['price', 'price_log'], axis=1).columns)
```


```python
sub_summary
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>price_log</td>    <th>  R-squared:         </th> <td>   0.802</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.801</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   4253.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 13 Aug 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>14:59:47</td>     <th>  Log-Likelihood:    </th> <td> -10929.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 20027</td>      <th>  AIC:               </th> <td>2.190e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 20007</td>      <th>  BIC:               </th> <td>2.206e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    19</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
               <td></td>                  <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                   <td>   -0.1494</td> <td>    0.016</td> <td>   -9.547</td> <td> 0.000</td> <td>   -0.180</td> <td>   -0.119</td>
</tr>
<tr>
  <th>sqft_living_log</th>             <td>    0.3096</td> <td>    0.006</td> <td>   53.077</td> <td> 0.000</td> <td>    0.298</td> <td>    0.321</td>
</tr>
<tr>
  <th>sqft_living15_log</th>           <td>    0.1372</td> <td>    0.005</td> <td>   26.690</td> <td> 0.000</td> <td>    0.127</td> <td>    0.147</td>
</tr>
<tr>
  <th>bath_per_bed</th>                <td>    0.0401</td> <td>    0.004</td> <td>   10.479</td> <td> 0.000</td> <td>    0.033</td> <td>    0.048</td>
</tr>
<tr>
  <th>floors</th>                      <td>    0.0215</td> <td>    0.004</td> <td>    4.971</td> <td> 0.000</td> <td>    0.013</td> <td>    0.030</td>
</tr>
<tr>
  <th>grade</th>                       <td>    0.2939</td> <td>    0.006</td> <td>   53.219</td> <td> 0.000</td> <td>    0.283</td> <td>    0.305</td>
</tr>
<tr>
  <th>condition</th>                   <td>    0.0744</td> <td>    0.003</td> <td>   22.346</td> <td> 0.000</td> <td>    0.068</td> <td>    0.081</td>
</tr>
<tr>
  <th>house_age</th>                   <td>    0.1269</td> <td>    0.005</td> <td>   26.805</td> <td> 0.000</td> <td>    0.118</td> <td>    0.136</td>
</tr>
<tr>
  <th>waterfront</th>                  <td>    0.9772</td> <td>    0.045</td> <td>   21.676</td> <td> 0.000</td> <td>    0.889</td> <td>    1.066</td>
</tr>
<tr>
  <th>renovated</th>                   <td>    0.1589</td> <td>    0.017</td> <td>    9.184</td> <td> 0.000</td> <td>    0.125</td> <td>    0.193</td>
</tr>
<tr>
  <th>view</th>                        <td>    0.2569</td> <td>    0.011</td> <td>   22.661</td> <td> 0.000</td> <td>    0.235</td> <td>    0.279</td>
</tr>
<tr>
  <th>has_basement</th>                <td>   -0.0288</td> <td>    0.008</td> <td>   -3.718</td> <td> 0.000</td> <td>   -0.044</td> <td>   -0.014</td>
</tr>
<tr>
  <th>subregion_east_urban</th>        <td>    0.4128</td> <td>    0.017</td> <td>   24.841</td> <td> 0.000</td> <td>    0.380</td> <td>    0.445</td>
</tr>
<tr>
  <th>subregion_north</th>             <td>    0.0857</td> <td>    0.020</td> <td>    4.322</td> <td> 0.000</td> <td>    0.047</td> <td>    0.125</td>
</tr>
<tr>
  <th>subregion_north_and_seattle</th> <td>    0.1742</td> <td>    0.022</td> <td>    7.786</td> <td> 0.000</td> <td>    0.130</td> <td>    0.218</td>
</tr>
<tr>
  <th>subregion_seattle</th>           <td>    0.5203</td> <td>    0.018</td> <td>   29.346</td> <td> 0.000</td> <td>    0.486</td> <td>    0.555</td>
</tr>
<tr>
  <th>subregion_south_and_seattle</th> <td>   -0.1830</td> <td>    0.030</td> <td>   -6.122</td> <td> 0.000</td> <td>   -0.242</td> <td>   -0.124</td>
</tr>
<tr>
  <th>subregion_south_rural</th>       <td>   -0.4371</td> <td>    0.021</td> <td>  -20.437</td> <td> 0.000</td> <td>   -0.479</td> <td>   -0.395</td>
</tr>
<tr>
  <th>subregion_south_urban</th>       <td>   -0.5518</td> <td>    0.017</td> <td>  -33.292</td> <td> 0.000</td> <td>   -0.584</td> <td>   -0.519</td>
</tr>
<tr>
  <th>subregion_vashon_island</th>     <td>   -0.1560</td> <td>    0.056</td> <td>   -2.769</td> <td> 0.006</td> <td>   -0.266</td> <td>   -0.046</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>633.451</td> <th>  Durbin-Watson:     </th> <td>   1.969</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1749.986</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.003</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.448</td>  <th>  Cond. No.          </th> <td>    36.2</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
qq_plot(sub_model)
```


![png](output_106_0.png)



```python
resid_scatter(sub_model, 'price_log', subregion_model)
```


![png](output_107_0.png)



```python
dist_resid(sub_model)
```


![png](output_108_0.png)



```python
vif(subregion_model, ['price', 'price_log'])
```




    [('sqft_living_log', 5.4924494622871185),
     ('sqft_living15_log', 2.8651390998984203),
     ('sqft_lot_log', 2.027555402890867),
     ('bedrooms', 2.8446848254289185),
     ('bath_per_bed', 2.3918016438820398),
     ('floors', 2.4478369319754627),
     ('grade', 2.9766043871070784),
     ('condition', 1.2594202008039639),
     ('house_age', 2.655715574107942),
     ('waterfront', 1.0966428397149965),
     ('renovated', 1.1528463146619523),
     ('view', 1.3325747755683244),
     ('has_basement', 2.784870639140419),
     ('subregion_east_urban', 1.3879021133656904),
     ('subregion_north', 1.0844176922310527),
     ('subregion_north_and_seattle', 1.0679672546483665),
     ('subregion_seattle', 2.5859599962922077),
     ('subregion_south_and_seattle', 1.037933464884607),
     ('subregion_south_rural', 1.068557309756816),
     ('subregion_south_urban', 1.318276591346026),
     ('subregion_vashon_island', 1.0603013716086513)]




```python
linreg = LinearRegression()
X = subregion_model.drop(['price', 'price_log'], axis=1)
y = subregion_model[['price_log']]
```


```python
cv_5_mse = np.mean(cross_val_score(linreg, X, y, cv=5, scoring='neg_mean_squared_error'))
cv_10_mse = np.mean(cross_val_score(linreg, X, y, cv=10, scoring='neg_mean_squared_error'))
cv_20_mse = np.mean(cross_val_score(linreg, X, y, cv=20, scoring='neg_mean_squared_error'))

cv_5_r_squared = np.mean(cross_val_score(linreg, X, y, cv=5, scoring='r2'))
cv_10_r_squared = np.mean(cross_val_score(linreg, X, y, cv=10, scoring='r2'))
cv_20_r_squared = np.mean(cross_val_score(linreg, X, y, cv=20, scoring='r2'))
```


```python
print(f"CV 5 MSE: {cv_5_mse}\nCV 10 MSE: {cv_10_mse}\nCV 20 MSE: {cv_20_mse}"
      f"\n\nCV 5 R2: {cv_5_r_squared}\nCV 10 R2: {cv_10_r_squared}\nCV 20 R2: {cv_20_r_squared}")
```

    CV 5 MSE: -0.17554311156652974
    CV 10 MSE: -0.17547332856018602
    CV 20 MSE: -0.17511437505069288
    
    CV 5 R2: 0.7994721333240489
    CV 10 R2: 0.7986499722631241
    CV 20 R2: 0.7989679967633752
    

### Answer for Question 2:
**Does a model perform better when we use zipcode as our location category or subregions as our location category?**

After refining our **Zipcode Model** and our **Subregion Model**, we see that the most important factors when determining price is the location of the house. While some locations have a stronger relationship than price, some are also negatively related to price. It seems that from our subregion model, any southern subregion has a negative correlation to price. In terms of which model has a higher R squared value, we see that our zipcode model beats our subregions model with an 0.87 compared to our subregion model's R squared value of 0.80. 

However, our subregion model contains only 19 different variables, while our zipcode model contains 73. Our zipcode model is stronger, but our subregion model is easier to digest and interpret.

# Step 5 - Interpretation

### Question 3: 
**What are the most important factors under a homeowner's control when it comes to determining the price of a house in King County, WA?**

The answer for this question could help focus homeowners on the aspects of their house they can improve in order to increase the value of the home they are selling. Not only that, but it could also give them insight on which factors they do not necessarily need to worry about as much as others.

### Factors That Don't Hold Much Weight When Determining the Price of a Home:

- If a house has a basement
- The number of floors in a house
- Bathroom to bedroom ratio
- Condition of the house

### Important Factors a Homeseller CANNOT Control:
- Location
- if the house is considered a waterfront property
- the average sqft living area of the closest 15 neighbors
- Age of the house


### Answer for Question 3:

### Important Factors a Homeseller CAN Control:

- Grade of the House
- Renovation
- Sqft Living Area
- How many people view your house


## Recommendations

For homeowners looking to sell their King County, WA house:

- All important factors a homeowner can control are tied together. For example, if a homeowner hired a well reviewed licensed contractor to increase the sqft living area of their home through renovation using high quality materials, that would increase both the grade and sqft living area, which would increase the overall value of the house.


- As far as increasing views of a house, putting their house listing on multiple websites would attract attention, and hiring a professional photographer to maximize the attractiveness of their house and using those pictures could definitely help in terms of viewers. 
