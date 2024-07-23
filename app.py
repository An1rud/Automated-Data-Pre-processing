import streamlit as st
import pandas as pd
import numpy as np
import base64
import io

def load_data(file):
    if file is not None:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return None
            
            st.subheader('Preview of Data')
            st.write(df.head())
            st.write(f'**Data Shape**: {df.shape}')

             # Display data description
            st.subheader('Data Description')
            st.write(df.describe(include='all'))  # include='all' provides description for all types of columns

            # Display data info directly
            st.subheader('Data Information')
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

            # Display the number of missing values
            missing_values = df.isnull().sum().to_dict()
            st.subheader('**Total Missing Values**')
            for column, missing in missing_values.items():
                if missing > 0:
                    st.write(f'**{column}**: {missing}')

            # Display the number of duplicate rows
            duplicate_rows = df.duplicated().sum()
            st.markdown('**Number of Duplicate Rows:**')
            st.write(duplicate_rows)
            st.write('---')
            
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

def remove_duplicates(df):
    if df is not None:
        original_len = len(df)
        df_cleaned = df.drop_duplicates()
        removed_len = original_len - len(df_cleaned)
        st.write(f"Removed {removed_len} duplicate rows.")
        st.write(f"Data Shape after removing duplicates: {df_cleaned.shape}")
        return df_cleaned
    return df

def get_csv_download_link(df, filename="cleaned_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert CSV to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Cleaned Data (CSV)</a>'
    return href

# Function to handle missing values by filling with mean
def fill_missing_with_mean(df, columns):
    df_filled_mean = df.copy()
    if df_filled_mean is not None and isinstance(columns, (list, pd.Index)) and len(columns) > 0:
        columns = list(columns)  # Ensure columns is a list
        for col in columns:
            if df_filled_mean[col].dtype in ['float64', 'int64'] and df_filled_mean[col].isnull().sum() > 0:  # Only fill if there are missing values
                mean_value = df_filled_mean[col].mean()
                df_filled_mean[col].fillna(mean_value, inplace=True)
                st.write(f"Filled missing values in column '{col}' with mean: {mean_value}")
        st.write("After filling with mean:")
        st.write(df_filled_mean.head())
    return df_filled_mean

# Function to handle missing values by filling with median
def fill_missing_with_median(df, columns):
    df_filled_median = df.copy()
    if df_filled_median is not None and isinstance(columns, (list, pd.Index)) and len(columns) > 0:
        columns = list(columns)  # Ensure columns is a list
        for col in columns:
            if df_filled_median[col].dtype in ['float64', 'int64'] and df_filled_median[col].isnull().sum() > 0:  # Only fill if there are missing values
                median_value = df_filled_median[col].median()
                df_filled_median[col].fillna(median_value, inplace=True)
                st.write(f"Filled missing values in column '{col}' with median: {median_value}")
        st.write("After filling with median:")
        st.write(df_filled_median.head())
    return df_filled_median

# Function to handle missing values by filling with mode for object data types
def fill_missing_with_mode(df, columns):
    df_filled_mode = df.copy()
    if df_filled_mode is not None and isinstance(columns, (list, pd.Index)) and len(columns) > 0:
        columns = list(columns)  # Ensure columns is a list
        for col in columns:
            if df_filled_mode[col].dtype == 'object' and df_filled_mode[col].isnull().sum() > 0:  # Only fill if there are missing values
                mode_value = df_filled_mode[col].mode()[0]
                df_filled_mode[col].fillna(mode_value, inplace=True)
                st.write(f"Filled missing values in column '{col}' with mode: {mode_value}")
        st.write("After filling with mode:")
        st.write(df_filled_mode.head())
    return df_filled_mode

# Function to remove outliers using IQR
def remove_outliers_iqr(df, columns):
    df_no_outliers = df.copy()
    if df_no_outliers is not None and isinstance(columns, (list, pd.Index)) and len(columns) > 0:
        columns = list(columns)  # Ensure columns is a list
        for col in columns:
            if df_no_outliers[col].dtype in ['float64', 'int64']:
                Q1 = df_no_outliers[col].quantile(0.25)
                Q3 = df_no_outliers[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                before_shape = df_no_outliers.shape
                df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)]
                after_shape = df_no_outliers.shape
                st.write(f"Removed outliers in column '{col}' using IQR. Rows before: {before_shape[0]}, Rows after: {after_shape[0]}")
        st.write("After removing outliers:")
        st.write(df_no_outliers.head())
    return df_no_outliers

# Function to apply all preprocessing steps together
def apply_all_functions(df):
    if df is not None:
        df = remove_duplicates(df)

        # Fill missing values with mean
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df_filled_mean = fill_missing_with_mean(df, numeric_columns)

        # Fill missing values with median
        numeric_columns = df_filled_mean.select_dtypes(include=['float64', 'int64']).columns
        df_filled_median = fill_missing_with_median(df_filled_mean, numeric_columns)

        # Fill missing values with mode
        object_columns = df_filled_median.select_dtypes(include=['object']).columns
        df_filled_mode = fill_missing_with_mode(df_filled_median, object_columns)

        # Remove outliers
        numeric_columns = df_filled_mode.select_dtypes(include=['float64', 'int64']).columns
        df_no_outliers = remove_outliers_iqr(df_filled_mode, numeric_columns)

        return df_no_outliers
    return df

def main():
    st.title('Automated Data Pre-processing')
    

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV or Excel file here!", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)

        if df is not None:
            # Prepare placeholders for processed DataFrames
            df_filled_mean = df.copy()
            df_filled_median = None
            df_filled_mode = None
            df_no_outliers = None

            # Fill missing values with mean
            if df_filled_mean is not None:
                st.subheader('Fill Missing Values with Mean')
                numeric_columns = df_filled_mean.select_dtypes(include=['float64', 'int64']).columns
                selected_columns_mean = st.multiselect("Select columns to fill missing values with mean", numeric_columns)
                if st.button("Fill Missing Values with Mean"):
                    df_filled_mean = fill_missing_with_mean(df_filled_mean, selected_columns_mean)

            # Fill missing values with median
            if df_filled_mean is not None:
                df_filled_median = df_filled_mean.copy()
                st.subheader('Fill Missing Values with Median')
                numeric_columns = df_filled_mean.select_dtypes(include=['float64', 'int64']).columns
                selected_columns_median = st.multiselect("Select columns to fill missing values with median", numeric_columns)
                if st.button("Fill Missing Values with Median"):
                    df_filled_median = fill_missing_with_median(df_filled_median, selected_columns_median)

            # Fill missing values with mode
            if df_filled_median is not None:
                df_filled_mode = df_filled_median.copy()
                st.subheader('Fill Missing Values with Mode')
                object_columns = df_filled_median.select_dtypes(include=['object']).columns
                selected_columns_mode = st.multiselect("Select columns to fill missing values with mode", object_columns)
                if st.button("Fill Missing Values with Mode"):
                    df_filled_mode = fill_missing_with_mode(df_filled_mode, selected_columns_mode)

            # Remove outliers
            if df_filled_mode is not None:
                df_no_outliers = df_filled_mode.copy()
                st.subheader('Remove Outliers')
                numeric_columns = df_filled_mode.select_dtypes(include=['float64', 'int64']).columns
                if st.button("Remove Outliers"):
                    df_no_outliers = remove_outliers_iqr(df_filled_mode, numeric_columns)

            # Apply all functions together
            st.subheader('Apply All Functions Together')
            if st.button("Apply All Functions Together"):
                df_all_functions = apply_all_functions(df)
                st.write("After applying all functions together:")
                st.write(df_all_functions.head())
                st.write(f'Processed Data Shape: {df_all_functions.shape}')
                # Provide download link for final combined DataFrame
                st.markdown(get_csv_download_link(df_all_functions, filename="final_combined_data.csv"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
