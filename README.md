# Automated Data Pre-processing

Welcome to the Automated Data Pre-processing application! This Streamlit application allows you to upload a CSV or Excel file, clean and preprocess the data with various methods, and download the processed data.

## Features

- **File Upload**: Upload CSV or Excel files.
- **Data Preview**: View a preview and description of the uploaded data.
- **Datetime Conversion**: Convert selected columns to datetime format.
- **Remove Duplicates**: Remove duplicate rows from the data.
- **Handle Missing Values**: Fill missing values in numeric columns with mean or median, and in categorical columns with mode.
- **Remove Outliers**: Remove outliers in numeric columns using the Interquartile Range (IQR) method.
- **Download Processed Data**: Download the final processed DataFrame as a CSV file.
- 
## Demo

![Working of the application](gif/one.gif)

## Installation

To run this application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/An1rud/Automated-Data-Pre-processing.git
   cd Automated-Data-Pre-processing
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload File**: Click on the "Browse files" button to upload a CSV or Excel file.

2. **Data Preview**: After uploading, a preview of the data and its description will be displayed.

3. **Datetime Conversion**: Select columns to convert to datetime format and click the "Convert Columns to Datetime" button.

4. **Remove Duplicates**: Click the "Remove Duplicates" button to remove duplicate rows from the data.

5. **Fill Missing Values**: Select numeric columns to fill missing values with mean or median, and categorical columns to fill with mode. Click the respective buttons to fill the missing values.

6. **Remove Outliers**: Select numeric columns to remove outliers using the IQR method and click the "Remove Outliers" button.

7. **Download Processed Data**: The final processed DataFrame will be displayed. Click the "Download Cleaned Data (CSV)" link to download the data as a CSV file.

## Example

Here is a simple example of how to use the application:

1. Upload your data file.
2. Select the columns for converting datetime,filling missing values with mean, median, or mode.
3. Remove outliers from the numeric columns.
4. Download the cleaned data.

## Data

Any Excel or csv file can be used.
I have provided a [Sample Data](sample_data) which you can check out

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

