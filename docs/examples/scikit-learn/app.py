import streamlit as st
from pathlib import Path
import base64
import requests


# Initial page config
st.set_page_config(
    page_title='Scikit-Learn Cheat Sheet',
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    """
    Main function to set up the Streamlit app layout.
    """
    cs_sidebar()
    cs_body()
    return None

# Define img_to_bytes() function
def img_to_bytes(img_url):
    response = requests.get(img_url)
    img_bytes = response.content
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# Define the cs_sidebar() function
def cs_sidebar():
    """
    Populate the sidebar with various content sections related to Scikit-learn.
    """
    st.sidebar.markdown(
        '''[<img src='data:image/png;base64,{}' class='img-fluid' width=95 >](https://streamlit.io/)'''.format(
            img_to_bytes("https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/main/docs/examples/scikit-learn/scikit-learn.png")), unsafe_allow_html=True)

    st.sidebar.header('Scikit-Learn Cheat Sheet')
    st.sidebar.markdown('''
<small>[Scikit-learn](https://scikit-learn.org/) is an open source Python library that
 implements a range of
machine learning,
 preprocessing, cross-validation and visualization
algorithms using a unified interface.</small>
    ''', unsafe_allow_html=True)

    # Scikit-Learn installation and import
    st.sidebar.markdown('__Install and import Scikit-Learn__')
    st.sidebar.code('$ pip install scikit-learn')
    st.sidebar.code('''
# Import Scikit-Learn convention
>>> import sklearn
''')

    # Add the Scikit-learn example
    st.sidebar.markdown('__Scikit-learn Example__')
    st.sidebar.markdown(
        '''[<img src='data:image/png;base64,{}' class='img-fluid' width=450 >](https://streamlit.io/)'''.format(
            img_to_bytes("https://raw.githubusercontent.com/fralfaro/DS-Cheat-Sheets/main/docs/examples/scikit-learn/sk-tree.png")), unsafe_allow_html=True)

    st.sidebar.code("""
    from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()

# Split the dataset into features (X) and target (y)
X, y = iris.data[:, :2], iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)

# Standardize the features using StandardScaler
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create a K-Nearest Neighbors classifier
knn = neighbors.KNeighborsClassifier(n_neighbors=5)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Predict the target values on the test data
y_pred = knn.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
""")
    return None


# Define the cs_body() function
def cs_body():
    """
    Create content sections for the main body of the Streamlit cheat sheet with Scikit-learn examples.
    """
    col1, col2, col3 = st.columns(3)  # Create columns for layout

    #######################################
    # COLUMN 1
    #######################################

    # Loading The Data
    col1.subheader('Loading The Data')
    col1.code('''
    from sklearn import datasets

    # Load the Iris dataset
    iris = datasets.load_iris()

    # Split the dataset into features (X) and target (y)
    X, y = iris.data, iris.target

    # Print the lengths of X and y
    print("Size of X:", X.shape) #  (150, 4)
    print("Size of y:", y.shape) #  (150, )
        ''')

    # Training And Test Data
    col1.subheader('Training And Test Data')
    col1.code('''
    # Import train_test_split from sklearn
    from sklearn.model_selection import train_test_split

    # Split the data into training and test sets with test_size=0.2 (20% for test set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        ''')

    # Create instances of the models
    col1.subheader('Create instances of the models')
    col1.code('''
        # Import necessary classes from sklearn libraries
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        # Create instances of supervised learning models
        # Logistic Regression classifier (max_iter=1000)
        lr = LogisticRegression(max_iter=1000)

        # k-Nearest Neighbors classifier with 5 neighbors
        knn = KNeighborsClassifier(n_neighbors=5)

        # Support Vector Machine classifier
        svc = SVC()

        # Create instances of unsupervised learning models
        # k-Means clustering with 3 clusters and 10 initialization attempts
        k_means = KMeans(n_clusters=3, n_init=10)

        # Principal Component Analysis with 2 components
        pca = PCA(n_components=2)
        ''')


    # Model Fitting
    col1.subheader('Model Fitting')
    col1.code('''
    # Supervised learning
    lr.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    svc.fit(X_train, y_train)

    # Unsupervised Learning
    k_means.fit(X_train)
    pca.fit_transform(X_train)
        ''')

    # Prediction
    col1.subheader('Prediction')
    col1.code('''
    # Supervised Estimators
    y_pred = svc.predict(X_test) # Predict labels
    y_pred = lr.predict(X_test) # Predict labels
    y_pred = knn.predict_proba(X_test) # Estimate probability of a label

    # Unsupervised Estimators
    y_pred = k_means.predict(X_test) # Predict labels in clustering algos
        ''')





    #######################################
    # COLUMN 2
    #######################################

    # Preprocessing The Data

    # Standardization
    col2.subheader('Standardization')
    col2.code('''
    from sklearn.preprocessing import StandardScaler

    # Create an instance of the StandardScaler and fit it to training data
    scaler = StandardScaler().fit(X_train)

    # Transform the training and test data using the scaler
    standardized_X = scaler.transform(X_train)
    standardized_X_test = scaler.transform(X_test)
    ''')

    # Normalization
    col2.subheader('Normalization')
    col2.code('''
    from sklearn.preprocessing import Normalizer
    scaler = Normalizer().fit(X_train)
    normalized_X = scaler.transform(X_train)
    normalized_X_test = scaler.transform(X_test)
    ''')

    # Binarization
    col2.subheader('Binarization')
    col2.code('''
    import numpy as np
    from sklearn.preprocessing import Binarizer

    # Create a sample data array
    data = np.array([[1.5, 2.7, 0.8],
                     [0.2, 3.9, 1.2],
                     [4.1, 1.0, 2.5]])

    # Create a Binarizer instance with a threshold of 2.0
    binarizer = Binarizer(threshold=2.0)

    # Apply binarization to the data
    binarized_data = binarizer.transform(data)
    ''')

    # Encoding Categorical Features
    col2.subheader('Encoding Categorical Features')
    col2.code('''
    from sklearn.preprocessing import LabelEncoder

    # Sample data: categorical labels
    labels = ['cat', 'dog', 'dog', 'fish', 'cat', 'dog', 'fish']

    # Create a LabelEncoder instance
    label_encoder = LabelEncoder()

    # Fit and transform the labels
    encoded_labels = label_encoder.fit_transform(labels)
    ''')

    # Imputing Missing Values
    col2.subheader('Imputing Missing Values')
    col2.code('''
    import numpy as np
    from sklearn.impute import SimpleImputer

    # Sample data with missing values
    data = np.array([[1.0, 2.0, np.nan],
                     [4.0, np.nan, 6.0],
                     [7.0, 8.0, 9.0]])

    # Create a SimpleImputer instance with strategy='mean'
    imputer = SimpleImputer(strategy='mean')

    # Fit and transform the imputer on the data
    imputed_data = imputer.fit_transform(data)
    ''')

    # Generating Polynomial Features
    col2.subheader('Generating Polynomial Features')
    col2.code('''
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures

    # Sample data
    data = np.array([[1, 2],
                     [3, 4],
                     [5, 6]])

    # Create a PolynomialFeatures instance of degree 2
    poly = PolynomialFeatures(degree=2)

    # Transform the data to include polynomial features
    poly_data = poly.fit_transform(data)
    ''')



    #######################################
    # COLUMN 3
    #######################################

    # Comparison operations
    # Classification Metrics
    col3.subheader('Classification Metrics')
    col3.code('''
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # Accuracy Score
    accuracy_knn = knn.score(X_test, y_test)
    print("Accuracy Score (knn):", knn.score(X_test, y_test))

    accuracy_y_pred = accuracy_score(y_test, y_pred_lr)
    print("Accuracy Score (y_pred):", accuracy_y_pred)

    # Classification Report
    classification_rep_y_pred = classification_report(y_test, y_pred_lr)
    print("Classification Report (y_pred):", classification_rep_y_pred)

    classification_rep_y_pred_lr = classification_report(y_test, y_pred_lr)
    print("Classification Report (y_pred_lr):", classification_rep_y_pred_lr)

    # Confusion Matrix
    conf_matrix_y_pred_lr = confusion_matrix(y_test, y_pred_lr)
    print("Confusion Matrix (y_pred_lr):", conf_matrix_y_pred_lr)
        ''')

    # Regression Metrics
    col3.subheader('Regression Metrics')
    col3.code('''
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Data: True/Predicted values 
    y_true = [3, -0.5, 2]
    y_pred = [2.8, -0.3, 1.8]

    # Calculate Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    print("Mean Absolute Error:", mae)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)
    print("Mean Squared Error:", mse)

    # Calculate R² Score
    r2 = r2_score(y_true, y_pred)
    print("R² Score:", r2)
        ''')

    # Clustering Metrics
    col3.subheader('Clustering Metrics')
    col3.code('''
    from sklearn.metrics import adjusted_rand_score, homogeneity_score, v_measure_score

    # Adjusted Rand Index
    adjusted_rand_index = adjusted_rand_score(y_test, y_pred_kmeans)
    print("Adjusted Rand Index:", adjusted_rand_index)

    # Homogeneity Score
    homogeneity = homogeneity_score(y_test, y_pred_kmeans)
    print("Homogeneity Score:", homogeneity)

    # V-Measure Score
    v_measure = v_measure_score(y_test, y_pred_kmeans)
    print("V-Measure Score:", v_measure)
        ''')

    # Cross-Validation
    col3.subheader('Cross-Validation')
    col3.code('''
    # Import necessary library
    from sklearn.model_selection import cross_val_score

    # Cross-validation with KNN estimator
    knn_scores = cross_val_score(knn, X_train, y_train, cv=4)
    print(knn_scores)

    # Cross-validation with Linear Regression estimator
    lr_scores = cross_val_score(lr, X, y, cv=2)
    print(lr_scores)
        ''')

    # Grid Search
    col3.subheader('Grid Search')
    col3.code('''
    # Import necessary library
    from sklearn.model_selection import GridSearchCV

    # Define parameter grid
    params = {
        'n_neighbors': np.arange(1, 3),
        'weights': ['uniform', 'distance']
    }

    # Create GridSearchCV object
    grid = GridSearchCV(estimator=knn, param_grid=params)

    # Fit the grid to the data
    grid.fit(X_train, y_train)

    # Print the best parameters found
    print("Best parameters:", grid.best_params_)

    # Print the best cross-validation score
    print("Best cross-validation score:", grid.best_score_)

    # Print the accuracy on the test set using the best parameters
    best_knn = grid.best_estimator_
    test_accuracy = best_knn.score(X_test, y_test)
    print("Test set accuracy:", test_accuracy)
        ''')

    # Asking for Help
    col1.subheader('Asking for Help')
    col1.code('''
    import sklearn.cluster

    # Use the help() function to get information about the KMeans class
    help(sklearn.cluster.KMeans)
    ''')



# Run the main function if the script is executed directly
if __name__ == '__main__':
    main()
