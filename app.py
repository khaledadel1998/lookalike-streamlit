import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gdown
import pickle
import os
st.set_page_config(page_title="Company Similarity Lookup", layout="wide")
st.title("Find Similar Companies")
# --- File ID from Google Drive ---
file_id_1 = '12spuDG-ENcujv9TizJZzGB3U_OS7anTQ'  # replace with your real file ID
file_id_2 = '1bXgBmWCU8yZOA04spPs95_TYT_V-87xr'  # second file

# --- Output file paths ---
file_path_1 = 'vectTotal2.pkl'
file_path_2 = 'total140Kset.pkl'

# --- Download files if not already downloaded ---
if not os.path.exists(file_path_1):
    gdown.download(f'https://drive.google.com/uc?id={file_id_1}', file_path_1, quiet=False)

if not os.path.exists(file_path_2):
    gdown.download(f'https://drive.google.com/uc?id={file_id_2}', file_path_2, quiet=False)

# --- Load the pickle files ---
with open(file_path_1, 'rb') as f:
    df = pickle.load(f)

with open(file_path_2, 'rb') as f:
    quick = pickle.load(f)

# --- Use the loaded objects ---
st.write("Pickle files loaded successfully!")

# Sample Data
# df = pd.read_pickle("vect.pkl")
# quick = pd.read_excel("processed.xlsx")
df = pd.read_pickle("vectTotal2.pkl")
quick = pd.read_pickle("total140Kset.pkl")

df = df[['coresignal_id','industry', 'subindustry', 'is_b2b', 'company_size_range', 'hq_country',
       'geographic_region', 'keywords', 'description', 'linguistics_region']]

def inverse_absolute_difference(val1, val2):
    """
    Compute similarity using inverse absolute difference scaling.
    Ensures values are between 0 and 1.
    """
    return 1 - abs(val1 - val2)  # Since values are normalized, no need for max_diff

def pre_filter(target, vectorized_df):
    size = target["company_size_range"].iloc[0]
    industry = target["industry"].iloc[0]
    subindustry = target["subindustry"].iloc[0]
    is_b2b = target["is_b2b"].iloc[0]
    ling = target["linguistics_region"].iloc[0]

    filtered_df = vectorized_df[
        vectorized_df["company_size_range"].apply(lambda arr: np.array_equal(arr, size)) &
        vectorized_df["industry"].apply(lambda arr: np.array_equal(arr, industry)) &
        vectorized_df["subindustry"].apply(lambda arr: np.array_equal(arr, subindustry)) &
        vectorized_df["is_b2b"].apply(lambda arr: np.array_equal(arr, is_b2b)) &
        vectorized_df["linguistics_region"].apply(lambda arr: np.array_equal(arr, ling))
    ]
    
    return filtered_df




def get_similarity_details(vectorized_df, orig_df, target, numerical_cols=None , filter = False, weights=None):
    """
    Compute similarity for each numerical/text column and return a DataFrame 
    with top similar companies, their similarity scores for each column, 
    and their original categorical values.
    """
    if filter == True : 
        vectorized_df = pre_filter(target,vectorized_df)
    if 'coresignal_id' not in vectorized_df.columns:
        raise ValueError("The dataframe must contain 'coresignal_id' column.")

    # Ensure no NaN values
    vectorized_df = vectorized_df.fillna(0)

    # Extract the row for the given ID
    target_row = target
    target_id = target_row['coresignal_id'].values[0]

    if target_row.empty:
        raise ValueError(f"ID {target_id} not found in dataset.")

    # Compute similarity
    similarity_scores = {'coresignal_id': vectorized_df['coresignal_id'].values}

    for col in vectorized_df.columns:
        if col == "coresignal_id":
            continue

        # Handle numerical columns with inverse absolute difference scaling
        if numerical_cols and col in numerical_cols:
            similarity_scores[col + "_sim"] = vectorized_df[col].apply(
                lambda x: inverse_absolute_difference(target_row[col].values[0], x)
            ).values

        else:  # Compute cosine similarity for other features (embeddings, text, etc.)
            print(col)
            col_vectors = np.vstack(vectorized_df[col].values)  # Convert to 2D NumPy array
            target_vector = np.array(target_row[col].values[0]).reshape(1, -1)  # Reshape for cosine similarity
            similarity_scores[col + "_sim"] = cosine_similarity(target_vector, col_vectors)[0]

    # Convert similarity scores to DataFrame
    similarity_df = pd.DataFrame(similarity_scores)

    # Compute average similarity across all columns
    similarity_df["avg_sim"] = similarity_df.drop(columns=['coresignal_id']).mean(axis=1)
    # Define weights for each field (Higher weight = More importance)


    # Calculate Weighted Similarity Average
    similarity_df['weighted_avg_sim'] = similarity_df[list(weights.keys())].mul(list(weights.values())).sum(axis=1) / sum(weights.values())


    # Merge with original DataFrame to get categorical values
    final_df = similarity_df.merge(orig_df, on="coresignal_id", how="inner")

    # Sort by average similarity and exclude the target company itself
    final_df = final_df[final_df["coresignal_id"] != target_id]

    return final_df

# Example usage:
numerical_cols = ['founded_year', 'employees_count', 'rank_global']  # Already normalized

pd.set_option("display.max_colwidth", 20)  # Prevent truncation of column values
pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns

# target = df[df['coresignal_id'] == 89192405]
# top_similar_companies = get_similarity_details(df, quick, target, numerical_cols, filter=True)



# Define default weights
default_weights = {      
    'hq_country_sim': 0.0,        
    'description_sim': 0.5,        
    'keywords_sim': 0.5,          
    'geographic_region_sim': 0.0,
    'company_size_range_sim' : 0.2
}

# Streamlit Page Settings


# **Predefined Coresignal IDs for Quick Selection**
col1, col2, col3 = st.columns(3)

if col1.button("Surfe"):
    st.session_state["input_id"] = 89192405  
if col2.button("Hubspot"):  
    st.session_state["input_id"] = 752371  
if col3.button("Apollo.io"):  
    st.session_state["input_id"] = 20513792  

# Ensure session state exists for input_id
if "input_id" not in st.session_state:
    st.session_state["input_id"] = 89192405  

# User Input for coresignal_id (Uses session state)
input_id = st.number_input("Enter Coresignal ID:", min_value=0, step=1, format="%d", 
                           value=st.session_state["input_id"])
st.markdown(
    f"""
    <div style="display: flex; justify-content: center; align-items: center; height: 10vh;">
        <div style="padding: 5px; border-radius: 1px; background-color: #000000; box-shadow: 2px 2px 8px rgba(0,0,0,0.1); text-align: center;">
            <h1 style="margin: 0;">Sample size : {len(df)}</h1>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# Sidebar for Weight Adjustments
st.sidebar.header("Adjust Weights")
weights = {}
for key, value in default_weights.items():
    weights[key] = st.sidebar.slider(
        f"{key.replace('_sim', '').replace('_', ' ').title()}",
        min_value=0.0, max_value=1.0, value=value, step=0.05
    )

# Normalize Weights (Ensure sum ≤ 1)
weight_sum = sum(weights.values())
if weight_sum > 1:
    weights = {k: v / weight_sum for k, v in weights.items()}  # Scale down proportionally

# Toggle Button for Strict Filter
strict_filter = st.sidebar.toggle("Strict Filter", value=True)

# Checkbox for showing details


# Apply filtering if ID is entered
target_info = quick[quick["coresignal_id"]==input_id]
if input_id:
    target = df[df['coresignal_id'] == input_id]
    
    if not target.empty:
        # Get top similar companies with user-defined weights & strict filter toggle
        top_similar_companies = get_similarity_details(df, quick, target, numerical_cols, filter=strict_filter, weights=weights)

        # **Sort the DataFrame by "weighted_avg_sim" in descending order**
        top_similar_companies = top_similar_companies.sort_values(by="weighted_avg_sim", ascending=False)

        # Display results
        st.subheader(f"Target Data (Coresignal ID: {input_id})")
        st.dataframe(target_info, use_container_width=True)
        st.subheader(f"Top Similar Companies")
        
        # Show only selected columns if show_details is False
        show_details = st.checkbox("Show Details", value=False)
        if not show_details:
            selected_columns = ["names", "weighted_avg_sim", "description" ,"keywords", "company_size_range"]
            top_similar_companies = top_similar_companies[selected_columns]

        st.dataframe(top_similar_companies, use_container_width=True)
    else:
        st.warning(f"No matching record found for Coresignal ID: {input_id}")
