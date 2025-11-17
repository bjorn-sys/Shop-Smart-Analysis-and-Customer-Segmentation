import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .cluster-0 { background-color: #ff9999; padding: 10px; border-radius: 5px; }
    .cluster-1 { background-color: #99ff99; padding: 10px; border-radius: 5px; }
    .cluster-2 { background-color: #9999ff; padding: 10px; border-radius: 5px; }
    .prediction-result {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

class CustomerSegmentationApp:
    def __init__(self):
        self.df = None
        self.df_model = None
        self.df_processed = None
        self.scaler = None
        self.pca = None
        self.kmeans = None
        self.country_rank_dict = None
        
    def load_data(self):
        """Load and preprocess the data"""
        try:
            # Load datasets
            df_customers = pd.read_csv('customers.csv')
            df_events = pd.read_csv('events.csv')
            df_line_items = pd.read_csv('line_items.csv')
            df_orders = pd.read_csv('orders.csv')
            df_products = pd.read_csv('products.csv')
            
            # Rename columns and merge
            df_products = df_products.rename({"id":"item_id"}, axis=1)
            df_merge = df_customers.merge(df_events, on='customer_id').merge(df_orders, on='customer_id').merge(df_line_items, on='order_id').merge(df_products, on='item_id')
            
            # Feature engineering
            df_feature = df_merge.copy()
            df_feature['customer_id'] = df_feature['customer_id'].str.split('-', expand=True)[0]
            df_feature['order_id'] = df_feature['order_id'].str.split('-', expand=True)[0]
            df_feature['event_type'] = df_feature['event_data'].apply(lambda x: json.loads(x)['event_type'])
            
            # FIX: Proper datetime conversion with error handling
            df_feature['checked_out_at'] = pd.to_datetime(df_feature['checked_out_at'], errors='coerce')
            df_feature['event_timestamp'] = pd.to_datetime(df_feature['event_timestamp'], errors='coerce')
            
            # Remove rows with invalid datetime values
            df_feature = df_feature.dropna(subset=['checked_out_at', 'event_timestamp'])
            
            df_feature['order_amount'] = df_feature['quantity'] * df_feature['price']
            df_feature.rename({"name":"product_name"}, axis=1, inplace=True)
            
            # Drop unnecessary columns
            cols = ['currency', 'device_id', 'item_id', 'event_data']
            df_feature.drop(columns=cols, inplace=True, errors='ignore')
            
            self.df = df_feature
            self._prepare_model_data()
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def _prepare_model_data(self):
        """Prepare data for modeling"""
        try:
            # Calculate order duration
            df_grouped = self.df.groupby(['customer_id', 'order_id', 'event_type']).agg({'event_timestamp':'min'})
            df_grouped = df_grouped.unstack().reset_index()
            df_grouped.columns = [col[0] if col[1] == '' else col[0]+ '_' + col[1] for col in df_grouped.columns]
            
            # FIX: Handle missing checkout timestamps
            if 'event_timestamp_checkout' in df_grouped.columns and 'event_timestamp_visit' in df_grouped.columns:
                df_grouped["order_duration"] = round((df_grouped["event_timestamp_checkout"] - df_grouped["event_timestamp_visit"]).dt.total_seconds() / 60)
                df_c_duration = df_grouped[['customer_id', 'order_duration']]
            else:
                # Create default order duration if columns are missing
                df_c_duration = pd.DataFrame({'customer_id': self.df['customer_id'].unique(), 'order_duration': 5000})
            
            # Successful checkouts
            df_checkout_success = self.df[(self.df['event_type'] == 'checkout') & (self.df['status']=='success')]
            if not df_checkout_success.empty:
                df_checkout_success = df_checkout_success.groupby('customer_id').agg({
                    'location':'first', 'order_id':'count', 'quantity':'sum', 'order_amount':'sum'
                }).reset_index()
                df_checkout_success = df_checkout_success.rename({"order_id":"order_count"}, axis=1)
                df_checkout_success = df_checkout_success.merge(df_c_duration, on='customer_id', how='inner')
                df_checkout_success['status'] = 1
            else:
                # Create empty DataFrame with correct columns if no successful checkouts
                df_checkout_success = pd.DataFrame(columns=['customer_id', 'location', 'order_count', 'quantity', 'order_amount', 'order_duration', 'status'])
            
            # Failed checkouts
            df_checkout_failed = self.df[(self.df['event_type'] == 'checkout') & (self.df['status']!='success')]
            if not df_checkout_failed.empty:
                df_checkout_failed = df_checkout_failed.groupby('customer_id').agg({
                    'location':'first', 'order_id':'count', 'quantity':'sum', 'order_amount':'sum'
                }).reset_index()
                df_checkout_failed = df_checkout_failed.rename({"order_id":"order_count"}, axis=1)
                df_checkout_failed = df_checkout_failed.merge(df_c_duration, on='customer_id', how='inner')
                df_checkout_failed['status'] = 0
            else:
                # Create empty DataFrame with correct columns if no failed checkouts
                df_checkout_failed = pd.DataFrame(columns=['customer_id', 'location', 'order_count', 'quantity', 'order_amount', 'order_duration', 'status'])
            
            # Combine datasets
            df_model = pd.concat([df_checkout_failed, df_checkout_success])
            if df_model.empty:
                # Create sample data if both DataFrames are empty
                st.warning("No checkout data found. Using sample data for demonstration.")
                df_model = self._create_sample_data()
            else:
                df_model = df_model.sample(frac=1, random_state=1)
            
            # Process location ranking
            if not df_model.empty:
                country_order = df_model.groupby('location')['order_amount'].sum().sort_values(ascending=False).round(2).rank(method='min')
                self.country_rank_dict = country_order.to_dict()
                df_model['location'] = df_model['location'].map(self.country_rank_dict)
            else:
                self.country_rank_dict = {}
            
            self.df_model = df_model
            self.df_processed = df_model.copy()
            
        except Exception as e:
            st.error(f"Error preparing model data: {str(e)}")
            # Create sample data as fallback
            self.df_model = self._create_sample_data()
            self.df_processed = self.df_model.copy()
            self.country_rank_dict = {'United States': 1, 'Canada': 2, 'UK': 3}
    
    def _create_sample_data(self):
        """Create sample data for demonstration"""
        np.random.seed(42)
        n_samples = 100
        
        sample_data = {
            'customer_id': [f'cust_{i}' for i in range(n_samples)],
            'location': np.random.choice(['United States', 'Canada', 'UK', 'Germany', 'France'], n_samples),
            'order_count': np.random.randint(1, 20, n_samples),
            'quantity': np.random.randint(1, 50, n_samples),
            'order_amount': np.random.uniform(100, 20000, n_samples),
            'order_duration': np.random.randint(100, 15000, n_samples),
            'status': np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        }
        
        return pd.DataFrame(sample_data)
    
    def train_models(self):
        """Train clustering models"""
        try:
            if self.df_model is None or self.df_model.empty:
                st.error("No data available for model training.")
                return False
                
            # Prepare data for clustering
            numerical_columns = [col for col in self.df_model.columns if col not in ['customer_id', 'location', 'status', 'cluster']]
            numerical_df = self.df_model[numerical_columns]
            
            # Scale data
            self.scaler = StandardScaler()
            scaled_numerical = self.scaler.fit_transform(numerical_df)
            scaled_df = pd.DataFrame(data=scaled_numerical, columns=numerical_df.columns)
            
            # Apply PCA
            self.pca = PCA(n_components=2)
            customer_pca = self.pca.fit_transform(scaled_df)
            
            # Apply KMeans
            self.kmeans = KMeans(n_clusters=3, random_state=42)
            labels = self.kmeans.fit_predict(customer_pca)
            
            # Add clusters to dataframe
            self.df_model['cluster'] = labels
            self.df_processed['cluster'] = labels
            
            return True
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            return False
    
    def predict_customer_segment(self, order_count, quantity, order_amount, order_duration, location, status):
        """Predict customer segment based on input features"""
        try:
            if self.country_rank_dict is None or location not in self.country_rank_dict:
                # Default rank if location not found
                location_rank = 1.0
            else:
                location_rank = self.country_rank_dict.get(location, 1.0)
            
            # Prepare input data
            input_data = np.array([[order_count, quantity, order_amount, order_duration]])
            
            # Scale the input
            scaled_input = self.scaler.transform(input_data)
            
            # Apply PCA
            pca_input = self.pca.transform(scaled_input)
            
            # Predict cluster
            cluster = self.kmeans.predict(pca_input)[0]
            
            return cluster, pca_input[0]
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None, None

def safe_display_dataframe(df, max_rows=10):
    """Safely display DataFrame by converting datetime columns to strings"""
    try:
        # Create a copy to avoid modifying original
        display_df = df.head(max_rows).copy()
        
        # Convert datetime columns to string
        datetime_columns = display_df.select_dtypes(include=['datetime64']).columns
        for col in datetime_columns:
            display_df[col] = display_df[col].astype(str)
        
        # Display the safe DataFrame
        st.dataframe(display_df)
        
    except Exception as e:
        st.error(f"Error displaying dataframe: {str(e)}")
        # Fallback: display basic info
        st.write(f"DataFrame shape: {df.shape}")
        st.write("Columns:", df.columns.tolist())

def main():
    st.title("🎯 Customer Segmentation Dashboard")
    st.markdown("---")
    
    # Initialize app
    app = CustomerSegmentationApp()
    
    # Sidebar
    st.sidebar.title("Navigation")
    sections = [
        "Data Overview",
        "Customer Engagement Analysis",
        "Product Performance",
        "Revenue Analysis", 
        "Customer Segmentation",
        "Predict Customer Segment",
        "Insights & Recommendations"
    ]
    selected_section = st.sidebar.selectbox("Select Section", sections)
    
    # Load data
    with st.spinner("Loading data..."):
        if not app.load_data():
            st.error("Failed to load data. Please check your data files.")
            st.info("The app will use sample data for demonstration.")
    
    # Train models
    with st.spinner("Training models..."):
        if not app.train_models():
            st.error("Failed to train models.")
            st.info("Some features may not work properly.")
    
    # Data Overview Section
    if selected_section == "Data Overview":
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", app.df['customer_id'].nunique() if app.df is not None else 0)
        with col2:
            st.metric("Total Orders", app.df['order_id'].nunique() if app.df is not None else 0)
        with col3:
            st.metric("Total Products", app.df['product_name'].nunique() if app.df is not None else 0)
        with col4:
            st.metric("Total Countries", app.df['location'].nunique() if app.df is not None else 0)
        
        st.subheader("Dataset Preview")
        if app.df is not None:
            safe_display_dataframe(app.df, 10)
        else:
            st.warning("No data available to display.")
        
        st.subheader("Data Statistics")
        if app.df is not None:
            # Display only numerical columns for statistics
            numerical_cols = app.df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                st.dataframe(app.df[numerical_cols].describe())
            else:
                st.info("No numerical columns found for statistics.")
        else:
            st.warning("No data available for statistics.")
    
    # Customer Engagement Analysis
    elif selected_section == "Customer Engagement Analysis":
        st.header("👥 Customer Engagement Analysis")
        
        if app.df is None:
            st.warning("No data available for analysis.")
            return
        
        # Engagement over time
        st.subheader("Customer Engagement Over Time")
        try:
            df_time = app.df.copy()
            df_time.set_index('event_timestamp', inplace=True)
            daily_counts = df_time.resample('D').count()['event_id']
            
            fig = px.line(x=daily_counts.index, y=daily_counts.values, 
                         title='Daily Customer Engagement',
                         labels={'x': 'Date', 'y': 'Number of Engagements'})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate engagement timeline: {str(e)}")
        
        # Event type distribution
        st.subheader("Event Type Distribution")
        try:
            df_event = app.df['event_type'].value_counts().reset_index()
            fig = px.bar(df_event, x='event_type', y='count', color='event_type',
                        title='Engagement by Event Type')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate event type distribution: {str(e)}")
        
        # Top customers
        st.subheader("Top 20 Most Engaged Customers")
        try:
            df_visit = app.df['customer_id'].value_counts().sort_values(ascending=False).reset_index()[:20]
            fig = px.bar(df_visit, x='count', y='customer_id', orientation='h',
                        title='Top 20 Customers by Engagement')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate top customers chart: {str(e)}")
    
    # Product Performance
    elif selected_section == "Product Performance":
        st.header("📦 Product Performance")
        
        if app.df is None:
            st.warning("No data available for analysis.")
            return
        
        # Product interactions
        st.subheader("Product Engagement")
        try:
            product_df = app.df['product_name'].value_counts().reset_index()
            fig = px.bar(product_df, y='product_name', x='count', orientation='h',
                        title='Product Interactions')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate product engagement chart: {str(e)}")
        
        # Revenue by product
        st.subheader("Revenue by Product")
        try:
            df_checkout_success = app.df[app.df['status'] == 'success']
            product_revenue = df_checkout_success.groupby('product_name')['order_amount'].sum().sort_values(ascending=False).reset_index()
            fig = px.bar(product_revenue, y='product_name', x='order_amount', orientation='h',
                        title='Revenue Generated by Product')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate revenue by product chart: {str(e)}")
        
        # Product sales count
        st.subheader("Units Sold by Product")
        try:
            product_count = df_checkout_success.groupby('product_name')['quantity'].sum().sort_values(ascending=False).reset_index()
            fig = px.bar(product_count, y='product_name', x='quantity', orientation='h',
                        title='Units Sold per Product')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate units sold chart: {str(e)}")
    
    # Revenue Analysis
    elif selected_section == "Revenue Analysis":
        st.header("💰 Revenue Analysis")
        
        if app.df is None:
            st.warning("No data available for analysis.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue by status
            try:
                df_checkout = app.df[app.df['event_type'] == 'checkout']
                order_amt = df_checkout.groupby('status')['order_amount'].sum().reset_index()
                fig = px.pie(order_amt, values='order_amount', names='status',
                            title='Revenue Distribution by Order Status')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate revenue distribution: {str(e)}")
            
            # Successful order amounts distribution
            try:
                df_success = df_checkout[df_checkout['status'] == 'success']
                fig = px.histogram(df_success, x='order_amount', 
                                  title='Distribution of Successful Order Amounts')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate order amount distribution: {str(e)}")
        
        with col2:
            # Top customers by spending
            try:
                df_amount = df_checkout.groupby('customer_id')['order_amount'].sum().sort_values(ascending=False).reset_index()[:10]
                fig = px.bar(df_amount, x='order_amount', y='customer_id', orientation='h',
                            title='Top 10 Customers by Spending')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate top customers by spending: {str(e)}")
            
            # Revenue by location
            try:
                df_country = df_checkout.groupby('location')['order_amount'].sum().sort_values(ascending=False).reset_index()[:20]
                fig = px.bar(df_country, x='order_amount', y='location', orientation='h',
                            title='Top 20 Countries by Revenue')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate revenue by location: {str(e)}")
    
    # Customer Segmentation
    elif selected_section == "Customer Segmentation":
        st.header("🎯 Customer Segmentation")
        
        if app.df_model is None or app.df_model.empty:
            st.warning("No model data available for segmentation.")
            return
        
        st.subheader("Customer Clusters Overview")
        
        # Cluster distribution
        try:
            cluster_counts = app.df_model['cluster'].value_counts().sort_index()
            fig = px.pie(values=cluster_counts.values, names=[f'Cluster {i}' for i in cluster_counts.index],
                        title='Customer Cluster Distribution')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate cluster distribution: {str(e)}")
        
        # Cluster characteristics
        st.subheader("Cluster Characteristics")
        try:
            cluster_stats = app.df_model.groupby('cluster').agg({
                'order_count': 'mean',
                'quantity': 'mean', 
                'order_amount': 'mean',
                'order_duration': 'mean',
                'status': 'mean'
            }).round(2)
            
            st.dataframe(cluster_stats)
        except Exception as e:
            st.error(f"Could not generate cluster statistics: {str(e)}")
        
        # 2D Cluster visualization
        st.subheader("Cluster Visualization")
        try:
            # Prepare data for visualization
            numerical_columns = [col for col in app.df_model.columns if col not in ['customer_id', 'location', 'status', 'cluster']]
            numerical_df = app.df_model[numerical_columns]
            
            scaled_data = app.scaler.transform(numerical_df)
            pca_result = app.pca.transform(scaled_data)
            
            # Create cluster visualization
            cluster_viz_df = pd.DataFrame({
                'PC1': pca_result[:, 0],
                'PC2': pca_result[:, 1],
                'Cluster': app.df_model['cluster']
            })
            
            fig = px.scatter(cluster_viz_df, x='PC1', y='PC2', color='Cluster',
                            title='Customer Segments (PCA Visualization)',
                            color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate cluster visualization: {str(e)}")
        
        # Cluster interpretation
        st.subheader("Cluster Interpretation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="cluster-0">', unsafe_allow_html=True)
            st.subheader("Cluster 0 - Low Value Customers")
            st.write("• Lower order frequency")
            st.write("• Smaller order amounts")
            st.write("• Shorter engagement duration")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="cluster-1">', unsafe_allow_html=True)
            st.subheader("Cluster 1 - High Value Customers")
            st.write("• Frequent orders")
            st.write("• Large order amounts")
            st.write("• High success rate")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="cluster-2">', unsafe_allow_html=True)
            st.subheader("Cluster 2 - Medium Value Customers")
            st.write("• Moderate order frequency")
            st.write("• Average order amounts")
            st.write("• Mixed success rates")
            st.markdown('</div>', unsafe_allow_html=True)

    # Predict Customer Segment
    elif selected_section == "Predict Customer Segment":
        st.header("🔮 Predict Customer Segment")
        
        st.markdown("""
        Use this section to predict the customer segment for new or existing customers based on their behavior patterns.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Input Features")
            
            # Input form
            with st.form("prediction_form"):
                order_count = st.slider("Order Count", min_value=1, max_value=20, value=5, 
                                       help="Number of orders placed by the customer")
                
                quantity = st.slider("Total Quantity", min_value=1, max_value=50, value=15,
                                    help="Total number of items purchased")
                
                order_amount = st.number_input("Total Order Amount ($)", min_value=0.0, max_value=50000.0, 
                                             value=5000.0, step=100.0,
                                             help="Total amount spent by the customer")
                
                order_duration = st.slider("Average Order Duration (minutes)", min_value=1, max_value=20000, 
                                         value=5000, step=100,
                                         help="Average time from visit to checkout")
                
                # Get available countries
                if app.country_rank_dict:
                    available_countries = list(app.country_rank_dict.keys())
                    default_index = available_countries.index('Singapore') if 'Singapore' in available_countries else 0
                else:
                    available_countries = ['United States', 'Canada', 'UK', 'Germany', 'France']
                    default_index = 0
                    
                location = st.selectbox("Location", options=available_countries, 
                                       index=default_index,
                                       help="Customer's country")
                
                status = st.selectbox("Order Status", options=["Success", "Failed/Cancelled"],
                                     help="Overall order success rate")
                
                submitted = st.form_submit_button("Predict Customer Segment")
        
        with col2:
            st.subheader("Prediction Result")
            
            if submitted:
                # Convert status to numeric
                status_numeric = 1 if status == "Success" else 0
                
                # Make prediction
                cluster, pca_coords = app.predict_customer_segment(
                    order_count, quantity, order_amount, order_duration, location, status_numeric
                )
                
                if cluster is not None:
                    # Display prediction result
                    cluster_names = {
                        0: "Low Value Customer",
                        1: "High Value Customer", 
                        2: "Medium Value Customer"
                    }
                    
                    cluster_colors = {
                        0: "#ff9999",
                        1: "#99ff99",
                        2: "#9999ff"
                    }
                    
                    cluster_descriptions = {
                        0: "Customers with lower order frequency and smaller order amounts. Focus on retention and upselling.",
                        1: "Your most valuable customers with frequent orders and high spending. Prioritize retention and loyalty programs.",
                        2: "Customers with moderate spending patterns. Good candidates for growth and cross-selling."
                    }
                    
                    # Display result
                    st.markdown(f"""
                    <div class="prediction-result" style="background-color: {cluster_colors[cluster]};">
                        <h3>Predicted Segment: {cluster_names[cluster]}</h3>
                        <h4>Cluster {cluster}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info(cluster_descriptions[cluster])
                    
                    # Show customer position in cluster visualization
                    st.subheader("Customer Position in Segments")
                    
                    try:
                        # Get existing cluster data for visualization
                        numerical_columns = [col for col in app.df_model.columns if col not in ['customer_id', 'location', 'status', 'cluster']]
                        numerical_df = app.df_model[numerical_columns]
                        scaled_data = app.scaler.transform(numerical_df)
                        pca_result = app.pca.transform(scaled_data)
                        
                        # Create visualization with new point
                        cluster_viz_df = pd.DataFrame({
                            'PC1': pca_result[:, 0],
                            'PC2': pca_result[:, 1],
                            'Cluster': app.df_model['cluster'],
                            'Type': 'Existing Customer'
                        })
                        
                        # Add new prediction point
                        new_point_df = pd.DataFrame({
                            'PC1': [pca_coords[0]],
                            'PC2': [pca_coords[1]], 
                            'Cluster': [cluster],
                            'Type': 'Predicted Customer'
                        })
                        
                        combined_df = pd.concat([cluster_viz_df, new_point_df])
                        
                        # Create plot
                        fig = px.scatter(combined_df, x='PC1', y='PC2', color='Cluster',
                                        symbol='Type', symbol_map={'Existing Customer': 'circle', 'Predicted Customer': 'star'},
                                        size=[100]*len(cluster_viz_df) + [300],  # Larger size for predicted point
                                        title='Customer Segments with Prediction',
                                        color_continuous_scale='viridis')
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not generate visualization: {str(e)}")
                    
                    # Recommendations based on cluster
                    st.subheader("Recommended Actions")
                    
                    recommendations = {
                        0: [
                            "Implement targeted email campaigns to increase engagement",
                            "Offer entry-level products or discounts to encourage more purchases",
                            "Improve onboarding experience for better retention",
                            "Monitor for potential growth to Medium Value segment"
                        ],
                        1: [
                            "Offer exclusive premium products and early access",
                            "Implement VIP loyalty program with special benefits", 
                            "Provide personalized customer service and dedicated account manager",
                            "Request testimonials and referrals for marketing"
                        ],
                        2: [
                            "Cross-sell complementary products",
                            "Offer bundle deals to increase average order value",
                            "Implement moderate-level loyalty rewards",
                            "Monitor for progression to High Value segment"
                        ]
                    }
                    
                    for rec in recommendations[cluster]:
                        st.markdown(f"✅ {rec}")
                
                else:
                    st.error("Failed to make prediction. Please check the input values.")
            
            else:
                st.info("Please fill out the form and click 'Predict Customer Segment' to see the results.")
                
                # Show example predictions
                st.subheader("Example Customer Profiles")
                
                examples = [
                    {"order_count": 2, "quantity": 5, "order_amount": 1000, "order_duration": 1000, "location": "United States", "expected": "Low Value"},
                    {"order_count": 8, "quantity": 25, "order_amount": 15000, "order_duration": 5000, "location": "Singapore", "expected": "High Value"},
                    {"order_count": 5, "quantity": 15, "order_amount": 5000, "order_duration": 3000, "location": "Canada", "expected": "Medium Value"}
                ]
                
                for example in examples:
                    st.write(f"• {example['order_count']} orders, ${example['order_amount']:,.0f} spent → {example['expected']}")
    
    # Insights & Recommendations
    elif selected_section == "Insights & Recommendations":
        st.header("💡 Insights & Recommendations")
        
        st.subheader("Key Insights")
        
        insights = [
            "🚀 **Engagement Analysis**: Monitor daily engagement patterns to identify peak activity periods",
            "📸 **Product Performance**: Identify top-performing products and optimize their visibility",
            "⏱️ **Conversion Time**: Track order duration to optimize user experience",
            "❌ **Failure Analysis**: Investigate reasons for failed or cancelled orders",
            "💰 **Revenue Optimization**: Focus on high-value customer segments",
            "🌍 **Geographic Focus**: Identify top-performing regions for targeted marketing",
            "🎯 **Customer Segments**: Use segmentation to personalize marketing strategies"
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")
        
        st.subheader("Strategic Recommendations")
        
        recommendations = [
            "**Customer Segmentation**: Implement targeted strategies for each customer segment",
            "**Product Strategy**: Focus on high-performing products and improve underperformers",
            "**Customer Retention**: Develop loyalty programs for high-value customers",
            "**Geographic Expansion**: Allocate resources to high-performing regions",
            "**User Experience**: Streamline checkout process to reduce failures",
            "**Data-Driven Decisions**: Use analytics to guide marketing and product decisions",
            "**Continuous Monitoring**: Regularly review customer segments and adjust strategies"
        ]
        
        for rec in recommendations:
            st.markdown(f"✅ {rec}")

if __name__ == "__main__":
    main()