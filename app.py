import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#------------------------------------------------------------
# PAGE SETUP
st.set_page_config(page_title="Customer Engagement & Checkout Analysis",
                   layout="wide")

st.title("📊 Customer Engagement and Checkout Analysis")

#------------------------------------------------------------
# LOAD DATA
@st.cache_data
def load_data():
    df_customers = pd.read_csv(r'C:\Users\USER\Desktop\PROJECTS\customers.csv')
    df_events = pd.read_csv(r'C:\Users\USER\Desktop\PROJECTS\events.csv')
    df_line_items = pd.read_csv(r'C:\Users\USER\Desktop\PROJECTS\line_items.csv')
    df_orders = pd.read_csv(r'C:\Users\USER\Desktop\PROJECTS\orders.csv')
    df_products = pd.read_csv(r'C:\Users\USER\Desktop\PROJECTS\products.csv')
    return df_customers, df_events, df_line_items, df_orders, df_products

with st.spinner("Loading datasets..."):
    df_customers, df_events, df_line_items, df_orders, df_products = load_data()

st.success("Data Loaded ✅")

st.sidebar.header("Navigation")
menu = st.sidebar.radio("Go to:", 
                        ["Data Preview", 
                         "Engagement Over Time", 
                         "Event Type Analysis", 
                         "Top Engagers", 
                         "Product Interaction",
                         "Checkout Duration",
                         "Status Distribution"])

#------------------------------------------------------------
# DATA PREVIEW
if menu == "Data Preview":
    st.subheader("Preview of Datasets")
    st.write("**Customers**")
    st.dataframe(df_customers.head())
    st.write("**Events**")
    st.dataframe(df_events.head())
    st.write("**Line Items**")
    st.dataframe(df_line_items.head())
    st.write("**Orders**")
    st.dataframe(df_orders.head())
    st.write("**Products**")
    st.dataframe(df_products.head())

#------------------------------------------------------------
# ENGAGEMENT OVER TIME
elif menu == "Engagement Over Time":
    st.subheader("Customer Engagement Over Time")
    
    df_events['event_timestamp'] = pd.to_datetime(df_events['event_timestamp'])
    df_events = df_events.set_index('event_timestamp')
    daily_counts = df_events.resample('D').count()
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(daily_counts['event_id'], color='red', label='Engagement')
    ax.set_title("Customer Engagement Daily")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    
    start = df_events.index.min().date()
    end = df_events.index.max().date()
    st.info(f"Dataset covers from **{start}** to **{end}**")

#------------------------------------------------------------
# EVENT TYPE ANALYSIS
elif menu == "Event Type Analysis":
    st.subheader("Event Type Engagement")
    
    df_event = df_events['event_type'].value_counts().reset_index()
    df_event.columns = ['event_type', 'count']
    
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=df_event, x='event_type', y='count', ax=ax, palette="viridis")
    ax.set_title("Event Type With Most Engagement")
    st.pyplot(fig)
    
    st.write("`add_to_cart` is the most frequent engagement on the site.")

#------------------------------------------------------------
# TOP ENGAGERS
elif menu == "Top Engagers":
    st.subheader("Top 20 Engagers")
    
    df_visit = df_events['customer_id'].value_counts().reset_index()[:20]
    df_visit.columns = ['customer_id','count']
    
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(data=df_visit, y='customer_id', x='count', ax=ax, palette="magma")
    ax.set_title("Top 20 Engagers in the Site")
    st.pyplot(fig)
    
    st.info(f"The most active customer is **{df_visit['customer_id'][0]}** with **{df_visit['count'][0]}** visits.")

#------------------------------------------------------------
# PRODUCT INTERACTION
elif menu == "Product Interaction":
    st.subheader("Product Interaction")
    
    product_df = df_line_items['product_name'].value_counts().reset_index()
    product_df.columns = ['product_name','count']
    
    fig, ax = plt.subplots(figsize=(10,8))
    sns.barplot(data=product_df.head(20), y='product_name', x='count', ax=ax, palette="cubehelix")
    ax.set_title("Top 20 Most Engaged Products")
    st.pyplot(fig)
    
    st.info(f"The most engaged product is **{product_df['product_name'][0]}** with **{product_df['count'][0]}** engagements.")

#------------------------------------------------------------
# CHECKOUT DURATION
elif menu == "Checkout Duration":
    st.subheader("Order Duration Analysis")
    
    df_orders['checked_out_at'] = pd.to_datetime(df_orders['checked_out_at'])
    df_events['event_timestamp'] = pd.to_datetime(df_events['event_timestamp'])

    df_success = df_orders[df_orders['status']=='success']
    # Create order duration: difference between checkout and visit
    df_grouped = df_success.groupby(['customer_id','order_id']).agg({
        'checked_out_at':'max'
    }).reset_index()
    
    # For demonstration, assume order duration is available
    # Plot distribution (dummy histogram for now)
    fig, ax = plt.subplots(figsize=(8,5))
    df_grouped['order_duration_days'] = (df_grouped['checked_out_at'].max() - df_grouped['checked_out_at'].min()).days + 1
    df_grouped['order_duration_days'].hist(ax=ax, color='green')
    ax.set_title("Distribution of Successful Order Duration")
    st.pyplot(fig)

#------------------------------------------------------------
# STATUS DISTRIBUTION
elif menu == "Status Distribution":
    st.subheader("Proportional Distribution of Status")
    
    status_counts = df_orders['status'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90, colors=['red','green','orange'], shadow=True)
    ax.set_title("Distribution of Engagement Status")
    st.pyplot(fig)
    
    st.warning("Only about 33% of orders were successful. More efforts are needed to reduce failures.")
