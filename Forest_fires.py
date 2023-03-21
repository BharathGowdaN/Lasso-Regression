import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Define a function to impute null values based on the selected method
def impute_null_values (df, method): 
    if method == "mean":
        df = df.fillna (df.mean())
    elif method == 'median':
        df = df.fillna(df.median()) 
    elif method == 'mode':
        df = df.fillna(df.mode().iloc[0])
        return df

# Define a function to perform Label encoding 
def label_encode(df, column):
    le = LabelEncoder()
    df[column]= le.fit_transform(df[column])
    return df

# Define a function to perform one-hot encoding
def one_hot_encode(df, column):
    ohe = OneHotEncoder()
    encoded_array = ohe. fit_transform(df[[column]])
    encoded_df= pd.DataFrame(encoded_array.toarray(), columns=ohe.get_feature_names([column]))
    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop([column], axis=1)
    return df

# Define the Streamlit app
def app():
    st.title("Data Preprocessing")
    
    # Upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv") 
    if uploaded_file is not None:
        df= pd.read_csv(uploaded_file)
        
        # Display the DataFrame
        st.write("original DataFrame:")
        st.write(df)
        
        # Display a dropdown menu to select the imputation method 
        impute_method = st.selectbox("Select an imputation method", options=["mean" 'median', 'mode'])
        
        # Impute the null values based on the selected method
        df_imputed= impute_null_values (df, impute_method)
        
        # Display a dropdown menu to select the encoding technique
        encoding_method = st.selectbox("select an encoding technique", options=['None', 'Label Encoding', 'One-Hot Encoding'])
        
        #Perform the selected encoding technique
        if encoding_method == 'Label Encoding':
            column= st.selectbox("Select a column to label encode", options=list(df_imputed.columns)) 
            df_encoded =label_encode(df_imputed, column)
        elif encoding_method =='One-Hot Encoding': 
            column =st.selectbox("Select a column to one-hot encode", options=list (df_imputed.columns)) 
            df_encoded = one_hot_encode(df_imputed, column)
        else:
            df_encoded = df_imputed
            
        # Display the encoded Dataframe
        st.write("Encoded DataFrame:")
        st.write(df_encoded)

# Run the streamlit app
if __name__ == '__main__':
    app()  

