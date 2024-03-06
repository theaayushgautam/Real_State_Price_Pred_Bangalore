# Real_State_Price_Pred_Bangalore

This data science project series walks through step by step process of how to build a real estate price prediction website. We will first build a model using sklearn and linear regression using banglore home prices dataset from kaggle.com. Second step would be to write a python flask server that uses the saved model to serve http requests. Third component is the website built in html, css and javascript that allows user to enter home square ft area, bedrooms etc and it will call python flask server to retrieve the predicted price. During model building we will cover almost all data science concepts such as data load and cleaning, outlier detection and removal, feature engineering, dimensionality reduction, gridsearchcv for hyperparameter tunning, k fold cross validation etc. Technology and tools wise this project covers,
1) Python
2) Numpy and Pandas for data cleaning
3) Matplotlib for data visualization
4) Sklearn for model building
5) Jupyter notebook, visual studio code and pycharm as IDE
6) Python flask for http server
7) HTML/CSS/Javascript for UI


STEPS :

1. Download and Load the Dataset:

Download the dataset from the provided link.
Load the dataset into a pandas DataFrame.

2. Initial Data Examination:

Check the structure of the dataset (columns and rows).
Understand the meaning of each column (independent variables and dependent variable).

3. Drop Unnecessary Columns:

Drop columns that are not considered important for the prediction task (e.g., 'availability', 'society', 'area_type').

4. Handle Missing Values:

Identify columns with missing values using the isnull() function.
Decide whether to drop rows with missing values or impute them (e.g., by filling with median values).

5.Feature Engineering - Create 'BHK' Column:

Examine the 'size' column to identify inconsistencies (e.g., '2 BHK', '4 Bedroom').
Create a new column 'BHK' to represent the number of bedrooms in each property.
Handle inconsistencies (e.g., convert '4 Bedroom' to '4 BHK').

6. Data Cleaning - Handle Outliers:

Explore unique values in the 'BHK' column to identify outliers.
Decide how to handle outliers (e.g., remove rows with unrealistic values).

7. Clean 'total_sqft' Column:

Examine the 'total_sqft' column to identify inconsistencies (e.g., ranges, units in square meter or porch).
Write a function to convert ranges to average values and handle other inconsistencies.
Apply the function to clean the 'total_sqft' column.

8. Create a New DataFrame:

Create a new DataFrame ('df_clean') to store the cleaned data.

9. Copy DataFrame and Create 'Price per Square Feet' Column:

Copy the existing DataFrame into a new DataFrame, say df5.
Calculate the 'Price per Square Feet' by dividing the 'price' column by the 'total_sqft' column.

10. Exploring and Handling 'Location' Column:

Explore unique values in the 'location' column to understand the distribution of locations.
Remove leading and trailing spaces from the location names using a lambda function.
Group the DataFrame by 'location' and calculate the count of data points for each location.
Sort the locations by the number of data points in descending order.
Determine a threshold for the number of data points per location (e.g., less than 10 data points).
Identify and label locations with less than the threshold as 'other'.
Transform the 'location' column by replacing locations with less than the threshold with 'other'.
Verify the unique values in the 'location' column to ensure that the transformation was successful.

11. Finalizing DataFrame for Model Building:

Print a sample of the DataFrame to verify the changes made.
Proceed with outlier detection and removal in the next steps

12. Introduction to Outlier Detection and Removal:

Explain the significance of identifying and removing outliers in datasets.
Discuss different techniques for outlier detection and removal, such as standard deviation and domain knowledge.

13. Domain Knowledge-Based Outlier Identification:

Utilize domain knowledge, such as real estate expertise, to identify potential outliers.
Establish thresholds based on typical characteristics, like square footage per bedroom, to identify outliers.

14. Threshold-Based Outlier Removal:

Apply logical criteria to filter out data points that deviate significantly from established thresholds.
Remove properties with square footage per bedroom below or above the defined threshold.

15. Statistical Outlier Detection:

Use statistical methods like mean and standard deviation to detect outliers.
Calculate mean and standard deviation for specific features, such as price per square foot, within location groups.

16. Outlier Removal Using Statistical Methods:

Filter out data points that lie beyond a certain number of standard deviations from the mean.
Remove properties with extremely low or high price per square foot values based on statistical thresholds.

17. Comparison of Property Prices:

Analyze whether the prices of 3-bedroom apartments are higher than those of 2-bedroom apartments for the same square footage.
Visualize the comparison using scatter plots to identify anomalies.

18. Outlier Removal Based on Bedroom-Bathroom Ratio:

Establish criteria for the number of bathrooms relative to the number of bedrooms.
Remove properties with an abnormal number of bathrooms relative to the number of bedrooms based on established criteria.

19. Data Cleanup and Visualization:

Clean the dataset by removing identified outliers.
Visualize the dataset to assess the effectiveness of outlier removal and verify data integrity.

20. Preparing Data for Model Building:

Drop unnecessary features, such as 'price per square foot' and 'size,' post outlier removal.
Ensure the dataset is ready for further analysis and machine learning model training.

21. Data Cleaning Recap:

Review the steps taken in the previous tutorial to clean the dataset and prepare it for model building.

22. Converting Categorical Data to Numeric:

Utilize one-hot encoding (dummy encoding) to convert categorical location data into numerical format.
Implement one-hot encoding using the pd.get_dummies() method in pandas.

23. Combining Dummy Columns with Main Dataframe:

Append the dummy columns representing location information to the main dataframe.
Drop unnecessary columns, such as the original location column, to avoid redundancy.

24. Preparing Data for Model Training:

Separate independent variables (features) and the dependent variable (target) from the dataframe.
Create variables X and y for model training, where X contains independent variables and y contains the target variable.

25. Splitting Data into Training and Test Sets:

Utilize the train_test_split method from sklearn.model_selection to split the dataset into training and test sets.

26. Building a Linear Regression Model:

Create a linear regression model using LinearRegression() from sklearn.linear_model.
Fit the model to the training data using the fit() method.
Evaluate the model's performance using the score() method.

27. K-Fold Cross-Validation:

Implement k-fold cross-validation using ShuffleSplit to evaluate the model's performance with different train-test splits.

28. Exploring Alternative Regression Algorithms:

Explore alternative regression algorithms such as Lasso regression and Decision Tree regression.
Use GridSearchCV to perform hyperparameter tuning and select the best algorithm.

29. Predicting Property Prices:

Write a function predict_price() to estimate property prices based on input features.
Test the function by predicting prices for sample properties in different locations and configurations.

30. Exporting Model and Columns Information:

Export the trained model to a pickle file using pickle.dump() for future use.
Export column information to a JSON file for reference during prediction.

31. Finalizing for Deployment:

Ensure all necessary artifacts, including the model and column information, are exported and ready for deployment in a Python Flask server.

32. Setting up the Python Flask Server Project:

Open PyCharm and create a new project folder named "Bangalore Home Prices."
Inside the project folder, create subdirectories for "client," "server," and "model."
Copy the exported artifacts (saved model and columns JSON file) into the "server/artifacts" directory.

33. Creating the Flask Server File:

Create a new file named "server.py" inside the "server" directory.
Import the Flask module and create an instance of the Flask app.
Define a simple route to return "hi" as a test response.

34. Running the Flask Server:

Configure the interpreter to use Anaconda or install Flask using pip if not using Anaconda.
Run the Flask server and verify that it's running by accessing the defined route in a web browser.

35. Implementing Endpoint to Retrieve Location Names:

Create a function named "get_location_names" to retrieve all location names from the columns JSON file.
Define a route in the Flask server to expose the location names via HTTP GET request.
Load the saved artifacts (columns JSON file) and extract location names.

36. Creating Utility Functions:

Create a separate file named "util.py" inside the "server" directory to store utility functions.
Define a function named "load_saved_artifacts" to load the saved artifacts (columns JSON file and model) into memory.

40. Implementing Endpoint to Get Estimated Price:

Create a function named "get_estimated_price" to predict home prices based on input features (location, square foot area, BHK, bathroom).
Define a route in the Flask server to expose the estimated price calculation via HTTP POST request.
Parse input parameters from the request form, call the utility function to predict the price, and return the estimated price.

41. Testing Endpoints with Postman:

Use Postman to test the implemented endpoints by sending HTTP GET and POST requests.
Verify the response for retrieving location names and predicting home prices.

42. Completing Flask Server:

Ensure that all routes and functions are working correctly by testing different input combinations.
Verify that the Flask server is running smoothly and ready to be integrated with the UI application.





