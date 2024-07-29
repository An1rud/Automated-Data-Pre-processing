import streamlit as st
import pandas as pd
import numpy as np
import base64

# Function to load and display data head
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

            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

# Function to parse and handle dates
def parse_date(date_str):
    if isinstance(date_str, str):
        parts = date_str.split('/')
        
        # Ensure the date has exactly three parts
        if len(parts) == 3:
            day, month, year = parts
            
            # Check for invalid day, month, or year
            if day == '00' or month == '00' or len(year) != 4 or not year.isdigit():
                return pd.NaT
            
            # Construct a valid date string if parts are not '00'
            try:
                return pd.to_datetime(date_str, format='%d/%m/%Y', errors='coerce')
            except ValueError:
                return pd.NaT
        else:
            return pd.NaT
    else:
        return pd.NaT

# Function to convert columns to datetime
def convert_column_to_datetime(df, columns):
    if df is not None:
        for column in columns:
            if column in df.columns:
                st.write(f"Before conversion - Column: {column}")
                st.write(df[column].head(10))  # Display the first 10 values for debugging
                df[column] = df[column].apply(parse_date)
                st.write(f"After conversion - Column: {column}")
                st.write(df[column].head(10))  # Display the first 10 values after conversion for debugging
        return df
    return None

# Function to remove duplicates
def remove_duplicates(df):
    if df is not None:
        original_len = len(df)
        df_cleaned = df.drop_duplicates()
        removed_len = original_len - len(df_cleaned)
        st.write(f"Removed {removed_len} duplicate rows.")
        st.write(f"Data Shape after removing duplicates: {df_cleaned.shape}")
        return df_cleaned
    return None

# Function to convert DataFrame to CSV and generate download link
def get_csv_download_link(df, filename="cleaned_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert CSV to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Cleaned Data (CSV)</a>'
    return href

# Function to handle missing values by filling with mean
def fill_missing_with_mean(df, columns):
    if df is not None and isinstance(columns, (list, pd.Index)) and len(columns) > 0:
        df_filled_mean = df.copy()
        for col in columns:
            if df_filled_mean[col].dtype in ['float64', 'int64'] and df_filled_mean[col].isnull().sum() > 0:  # Only fill if there are missing values
                mean_value = df_filled_mean[col].mean()
                df_filled_mean[col].fillna(mean_value, inplace=True)
                st.write(f"Filled missing values in column '{col}' with mean: {mean_value}")
        st.write("After filling with mean:")
        st.write(df_filled_mean.head())
        return df_filled_mean
    return None

# Function to handle missing values by filling with median
def fill_missing_with_median(df, columns):
    if df is not None and isinstance(columns, (list, pd.Index)) and len(columns) > 0:
        df_filled_median = df.copy()
        for col in columns:
            if df_filled_median[col].dtype in ['float64', 'int64'] and df_filled_median[col].isnull().sum() > 0:  # Only fill if there are missing values
                median_value = df_filled_median[col].median()
                df_filled_median[col].fillna(median_value, inplace=True)
                st.write(f"Filled missing values in column '{col}' with median: {median_value}")
        st.write("After filling with median:")
        st.write(df_filled_median.head())
        return df_filled_median
    return None

# Function to handle missing values by filling with mode for object data types
def fill_missing_with_mode(df, columns):
    if df is not None and isinstance(columns, (list, pd.Index)) and len(columns) > 0:
        df_filled_mode = df.copy()
        for col in columns:
            if df_filled_mode[col].dtype == 'object' and df_filled_mode[col].isnull().sum() > 0:  # Only fill if there are missing values
                mode_value = df_filled_mode[col].mode()[0]
                df_filled_mode[col].fillna(mode_value, inplace=True)
                st.write(f"Filled missing values in column '{col}' with mode: {mode_value}")
        st.write("After filling with mode:")
        st.write(df_filled_mode.head())
        return df_filled_mode
    return None

# Function to remove outliers using IQR
def remove_outliers_iqr(df, columns):
    if df is not None and isinstance(columns, (list, pd.Index)) and len(columns) > 0:
        df_no_outliers = df.copy()
        any_outliers_removed = False
        
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
                if before_shape != after_shape:
                    any_outliers_removed = True
                    st.write(f"Removed outliers in column '{col}' using IQR. Rows before: {before_shape[0]}, Rows after: {after_shape[0]}")
        
        if not any_outliers_removed:
            st.write("No outliers found in the selected columns.")
        else:
            st.write("After removing outliers:")
            st.write(df_no_outliers.head())
        
        return df_no_outliers
    return None

# Function to filter numeric columns
def get_numeric_columns(df):
    return df.select_dtypes(include=['float64', 'int64']).columns

# Function to filter categorical columns
def get_categorical_columns(df):
    return df.select_dtypes(include=['object']).columns

# Main function
def main():
    st.title('Automated Pre-processing')
    # Initialize session state variables if they do not exist
    session_keys = ['df_original', 'df_datetime_converted', 'df_deduplicated', 'df_mean_filled', 'df_median_filled', 'df_mode_filled', 'df_outliers_removed']
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)

        if df is not None:
            # Store the original DataFrame in session state
            st.session_state.df_original = df

            # Convert columns to datetime
            st.subheader('Convert Columns to Datetime')
            datetime_columns_key = 'datetime_columns_select'
            
            selected_datetime_columns = st.multiselect(
                "Select columns to convert to datetime",
                df.columns,
                key=datetime_columns_key
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button('Convert Columns to Datetime'):
                    st.session_state.df_datetime_converted = convert_column_to_datetime(df, selected_datetime_columns)
                    st.write("Conversion complete. Data after conversion:")
                    st.write(st.session_state.df_datetime_converted.head())

            # Handle duplicate rows
            st.subheader('Remove Duplicate Rows')
            if st.button('Remove Duplicates'):
                st.session_state.df_deduplicated = remove_duplicates(st.session_state.df_datetime_converted if st.session_state.df_datetime_converted is not None else st.session_state.df_original)

            # Fill missing values with mean
            st.subheader('Fill Missing Values')
            numeric_columns = get_numeric_columns(st.session_state.df_deduplicated if st.session_state.df_deduplicated is not None else df)
            categorical_columns = get_categorical_columns(st.session_state.df_deduplicated if st.session_state.df_deduplicated is not None else df)

            # Fill missing values with mean
            st.subheader('Fill Missing Values with Mean')
            mean_columns_key = 'mean_columns_select'
            selected_mean_columns = st.multiselect(
                "Select columns to fill missing values with mean",
                numeric_columns,
                key=mean_columns_key
            )
            col3, col4 = st.columns(2)
            with col3:
                if st.button('Fill Missing Values with Mean'):
                    st.session_state.df_mean_filled = fill_missing_with_mean(st.session_state.df_deduplicated if st.session_state.df_deduplicated is not None else st.session_state.df_original, selected_mean_columns)

            # Fill missing values with median
            st.subheader('Fill Missing Values with Median')
            median_columns_key = 'median_columns_select'
            selected_median_columns = st.multiselect(
                "Select columns to fill missing values with median",
                numeric_columns,
                key=median_columns_key
            )
            col5, col6 = st.columns(2)
            with col5:
                if st.button('Fill Missing Values with Median'):
                    st.session_state.df_median_filled = fill_missing_with_median(st.session_state.df_mean_filled if st.session_state.df_mean_filled is not None else st.session_state.df_deduplicated if st.session_state.df_deduplicated is not None else st.session_state.df_original, selected_median_columns)

            # Fill missing values with mode
            st.subheader('Fill Missing Values with Mode')
            mode_columns_key = 'mode_columns_select'
            selected_mode_columns = st.multiselect(
                "Select columns to fill missing values with mode",
                categorical_columns,
                key=mode_columns_key
            )
            col7, col8 = st.columns(2)
            with col7:
                if st.button('Fill Missing Values with Mode'):
                    st.session_state.df_mode_filled = fill_missing_with_mode(st.session_state.df_median_filled if st.session_state.df_median_filled is not None else st.session_state.df_mean_filled if st.session_state.df_mean_filled is not None else st.session_state.df_deduplicated if st.session_state.df_deduplicated is not None else st.session_state.df_original, selected_mode_columns)

            # Remove outliers
            st.subheader('Remove Outliers')
            outlier_columns_key = 'outlier_columns_select'
            selected_outlier_columns = st.multiselect(
                "Select columns to remove outliers using IQR",
                numeric_columns,
                key=outlier_columns_key
            )
            col9, col10 = st.columns(2)
            with col9:
                if st.button('Remove Outliers'):
                    st.session_state.df_outliers_removed = remove_outliers_iqr(st.session_state.df_mode_filled if st.session_state.df_mode_filled is not None else st.session_state.df_median_filled if st.session_state.df_median_filled is not None else st.session_state.df_mean_filled if st.session_state.df_mean_filled is not None else st.session_state.df_deduplicated if st.session_state.df_deduplicated is not None else st.session_state.df_original, selected_outlier_columns)

            # Display final processed DataFrame
            st.subheader('Final Processed Data')
            final_df = st.session_state.df_outliers_removed if st.session_state.df_outliers_removed is not None else st.session_state.df_mode_filled if st.session_state.df_mode_filled is not None else st.session_state.df_median_filled if st.session_state.df_median_filled is not None else st.session_state.df_mean_filled if st.session_state.df_mean_filled is not None else st.session_state.df_deduplicated if st.session_state.df_deduplicated is not None else st.session_state.df_original
            st.write(final_df.head())

            # Provide download link for the final processed DataFrame
            if final_df is not None:
                st.markdown(get_csv_download_link(final_df, "processed_data.csv"), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
