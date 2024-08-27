import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='Unegui cars', page_icon=':car:', layout='wide')

# Load data
df = pd.read_csv('data.csv')

units = {
    'engine_capacity': 'L',  # Liters
    'year_of_manufacture': 'Year',
    'price': 'сая ₮',
    'road_traveled': 'KM',
    'Road Traveled Binned': 'KM'
}

df_autox = df.groupby('manufacturer')['price'].mean().sort_values(ascending=False).reset_index()
df_autox.columns = ['Manufacturer', 'Average Price']

title_counts = df['manufacturer'].value_counts()
title_counts_df = pd.DataFrame(title_counts)

# Display the scrollable table
st.write('Машины үйлдэрлэгч болон түүний тоо')
st.dataframe(title_counts_df, height=400)

# Create an interactive bar chart using Plotly Express
fig = px.bar(
    df_autox,
    x='Manufacturer',
    y='Average Price',
    title='Average Price by Manufacturer',
    labels={'Manufacturer': 'Manufacturer', 'Average Price': 'Average Price'},
    template='plotly_dark'
)

# Update layout for better appearance
fig.update_layout(
    xaxis_title='Manufacturer',
    yaxis_title=f'Average Price {units["price"]}',
    title_font_size=18,
    title_font_family='Arial',
    title_font_color='white',
    xaxis_tickangle=-45
)

st.markdown('<h5 align=center>Average Price by Manufacturer</h5>', unsafe_allow_html=True)
st.write('Үйлдвэрлэгч болон тэдгээрин дундаж үнийг харьцуулсан график')
st.plotly_chart(fig, use_container_width=True) 
imported_year_counts = df['imported_year'].value_counts().reset_index()
imported_year_counts.columns = ['Imported Year', 'Count']

# Create an interactive bar chart using Plotly Express
fig = px.bar(
    imported_year_counts,
    x='Imported Year',
    y='Count',
    title='Count of Observations by Imported Year',
    labels={'Imported Year': 'Imported Year', 'Count': 'Count'},
    template='plotly_dark'
)

# Update layout for better appearance
fig.update_layout(
    xaxis_title='Imported Year',
    yaxis_title='Count',
    title_font_size=18,
    title_font_family='Arial',
    title_font_color='white',
    xaxis_tickangle=-45
)

st.markdown('<h5 align=center>Imported Year vs Count</h5>', unsafe_allow_html=True)
st.write('Машины орж ирсэн он болон түүний тооны график')
st.plotly_chart(fig, use_container_width=True)

transmission_counts = df['transmission'].value_counts()
fig1 = go.Figure(data=[go.Bar(x=transmission_counts.index, y=transmission_counts.values)])
fig1.update_layout(
    title='Нийтлэг transmission',
    xaxis_title='Хөтлөгчийн төрөл болон үнийн хамаарал',
    yaxis_title='Frequency',
    xaxis_tickangle=45
)

# Second interactive plot: Engine
engine_counts = df['engine'].value_counts()
fig2 = go.Figure(data=[go.Bar(x=engine_counts.index, y=engine_counts.values)])
fig2.update_layout(
    title='Шатхууны төрөл',
    xaxis_title='Шатхууны төрөл үнийн хамаарал',
    yaxis_title='Frequency',
    xaxis_tickangle=45
)

# Third interactive plot: Type
type_counts = df['type'].value_counts()
fig3 = go.Figure(data=[go.Bar(x=type_counts.index, y=type_counts.values)])
fig3.update_layout(
    title='Машины төрөл',
    xaxis_title='Машины биеийн төрөл үнийн хамаарал',
    yaxis_title='Frequency',
    xaxis_tickangle=45
)

# Display the plots in Streamlit
st.write('Хөтлөгчийн төрөл болон түүний тоо ширхэгээр харуулав.')
st.plotly_chart(fig1, use_container_width=True)
st.write('Хөдөлгүүрийн төрөл болон түүний тоо ширхэгээр хауулав.')
st.plotly_chart(fig2, use_container_width=True)
st.write('Машины биеийн төрөл болон тоог харуулав.')
st.plotly_chart(fig3, use_container_width=True) 



# Sidebar filters
with st.sidebar:
    with st.expander('Dataframe filter', expanded=False):
        st.header('Filter Options')
        
        mark = st.multiselect('Select a mark:', options=df['mark'].unique())
        manufacturer = st.multiselect('Select a manufacturer:', options=df['manufacturer'].unique())
        engine = st.multiselect('Select an engine:', options=df['engine'].unique())
        gearbox = st.multiselect('Select gearbox:', options=df['gearbox'].unique())
        ccolor = st.multiselect('Select color:', options=df['color'].unique())

# Apply filters
if mark:
    df = df[df['mark'].isin(mark)]
if manufacturer:
    df = df[df['manufacturer'].isin(manufacturer)]
if engine:
    df = df[df['engine'].isin(engine)]
if gearbox:
    df = df[df['gearbox'].isin(gearbox)]
if ccolor:
    df = df[df['color'].isin(ccolor)]

st.title(':bar_chart: Dashboard')
st.markdown('##')
st.dataframe(df)

# Apply additional filtering
mask = ~((df['road_traveled'] == 0) & (df['year_of_manufacture'] < 2023))
mask2 = ~((df['road_traveled'] == 0) & (df['condition'] != 'Шинэ'))
df = df[mask & mask2]

# Calculate z-scores for numerical columns
numerical_columns = df.select_dtypes(include='number').columns

# Ensure that numerical_columns is not empty
if not numerical_columns.empty:
    st.title('Z score graph')
    st.write('Z score нь өгөгдсөн тархалтын дунджаас хэдэн стандарт хазайлттай болохыг хэлж өгдөг. Хэрэв сөрөг байвал дунджаас доогуур, эерэг байвал дунджаас дээгүүр байгааг харуулна. ')
    with st.expander('Z-score Filter', expanded=False):
        selected_columns = st.multiselect(
            'Select columns to plot:',
            options=numerical_columns,
            default=numerical_columns[1] if len(numerical_columns) > 1 else []
        )

    if selected_columns:
        z_scores_all_columns = df[selected_columns].apply(lambda x: (x - x.mean()) / x.std())
        
        # Create interactive Plotly histograms
        for column in selected_columns:
            fig = px.histogram(
                z_scores_all_columns, 
                x=column, 
                title=f'{column.replace("_", " ").capitalize()} Z-score Distribution',
                nbins=100,
                template='plotly_dark'
            )
            fig.update_traces(marker_color='lightgreen', opacity=0.7)
            
            # Update layout to include units if applicable
            fig.update_layout(
                xaxis_title=f'{column.replace("_", " ").capitalize()} ({units.get(column, "")}) ',
                yaxis_title='Count',
                title_font_size=18,
                title_font_family='Arial',
                title_font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)

        # Plot scatter plots with outliers
        z_score_threshold = 4
        for column in selected_columns:
            # Create a mask for outliers
            outliers_mask = abs(z_scores_all_columns[column]) > z_score_threshold
            
            # Create a DataFrame for plotting
            plot_data = df.copy()
            plot_data['Z-Score'] = z_scores_all_columns[column]
            plot_data['Outlier'] = outliers_mask
            
            # Create an interactive scatter plot
            fig = px.scatter(
                plot_data,
                x=plot_data.index,
                y=column,
                color='Outlier',
                color_discrete_map={True: 'red', False: 'blue'},
                title=f'Scatter Plot of {column} with Z-Score Outliers (Threshold={z_score_threshold})',
                labels={'x': 'Index', column: column},
                template='plotly_dark'
            )
            
            fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
            
            # Show the plot in Streamlit
            st.markdown('<h3>Scatter Plot</h3>', unsafe_allow_html=True)
            st.write('z score-ийн цэгэн график. Хэрэв утга нь 4-ээс дээш байвал True болно. Энэ нь худал байх боломжтой утгуудыг харуулна. ')
            st.plotly_chart(fig, use_container_width=True)
        
        # Remove outliers from DataFrame
        for column in selected_columns:
            outliers_mask_subset = abs(z_scores_all_columns[column]) > z_score_threshold
            df = df[~outliers_mask_subset]

        def display_numerical_boxplots(df, target_column):
            # Create a Plotly box plot for each numerical column
            for column in selected_columns:
                fig = px.box(
                    df,
                    x=target_column,
                    y=column,
                    title=f'Box Plot of {column} by {target_column}',
                    template='plotly_dark'
                )
                fig.update_traces(marker_color='red', line=dict(color='DarkSlateGrey'))
                fig.update_layout(
                    xaxis_title=f'{target_column.replace("_", " ").capitalize()} ({units.get(target_column, "")})',
                    yaxis_title=f'{column.replace("_", " ").capitalize()} ({units.get(column, "")})'
                )
                st.write('Үнийг тоон багануудтай харьцуулсан график')
                st.plotly_chart(fig, use_container_width=True)  # Use st.pyplot to display Matplotlib plots in Streamlit

        target_column = 'price'
        display_numerical_boxplots(df, target_column)
else:
    st.write("No numerical columns available to plot.")
    
Q1 = df['road_traveled'].quantile(0.25)
Q3 = df['road_traveled'].quantile(0.75)
IQR = Q3 - Q1

outlier_threshold = 1.5

outlier_mask = (df['road_traveled'] < Q1 - outlier_threshold * IQR) | (df['road_traveled'] > Q3 + outlier_threshold * IQR)

df = df[~outlier_mask]
df = df.drop('mark', axis=1)
df_null = df.isna().mean().round(4) * 100

df_null.sort_values(ascending=False).head()

outliers = ['price']

# Create a Plotly box plot for the 'price' column
if df.empty:
    st.write("No data available to plot after filtering out outliers.")
else:
    # Create and display the plot
    fig = px.box(
        df,
        y='price',
        title='Үнэ хэдээс хэдийн хооронд тархсан байгааг харуулж байна.',
        labels={'price': 'Price'},
        template='plotly_dark'
    )

    fig.update_layout(
        yaxis_title=f'Price ({units["price"]})',
        xaxis_title='Price',
        title_font_size=14,
        title_font_family='Arial',
        title_font_color='black',
        xaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False
        )
    )
    st.write('Outliers Variable Distribution')
    st.plotly_chart(fig, use_container_width=True)

df['manufacturer'].unique()
df.drop_duplicates(inplace=True)

cat_col = df.select_dtypes(include=['object']).columns
num_col = df.select_dtypes(exclude=['object']).columns
df_cat = df[cat_col]
df_num = df[num_col]

# Calculate the value counts for the 'manufacturer' column

# Create an interactive distribution plot using Plotly
column = st.selectbox('Select column to visualize:', ['price', 'road_traveled'])
st.write('Тодорхой давтамжаар машинуудын тоог явсан зам/үнийн дүнгээр харуулж байна.')
# Create histogram based on the selected column
fig = px.histogram(
    df,
    x=column,
    nbins=30,
    title=f'{column.replace("_", " ").capitalize()} Distribution',
    template='plotly_dark',
    marginal='rug',
    color_discrete_sequence=['blue']
)

# Update layout
fig.update_layout(
    xaxis_title=f'{column.replace("_", " ").capitalize()} {units.get(column, "")}',
    yaxis_title='Count',
    title_font_size=18,
    title_font_family='Arial',
    title_font_color='white',
)

# Display the plot

st.plotly_chart(fig, use_container_width=True)

num_bins = 15

# Binning 'Price'
bin_edges_price = np.linspace(df['price'].min(), df['price'].max(), num_bins + 1)
df['Price Binned'] = pd.cut(df['price'], bins=bin_edges_price)

# Binning 'Road Traveled'
bin_edges_traveled = np.linspace(df['road_traveled'].min(), df['road_traveled'].max(), num_bins + 1)
df['Road Traveled Binned'] = pd.cut(df['road_traveled'], bins=bin_edges_traveled).astype(str)


# List of available columns for plotting
available_columns = [
    'engine_capacity', 'year_of_manufacture', 'gearbox', 'wheel_position', 
    'type', 'color', 'imported_year', 'engine', 'saloon_color', 
    'transmission', 'doors', 'Road Traveled Binned'
]

# Multi-select dropdown for column selection
selected_columns = st.multiselect('Select columns to view:', available_columns, default=available_columns[:1], key='multiselect1')

# Plotting based on selected columns
if selected_columns:
    for column in selected_columns:
        fig = px.box(
            df,
            x=column,
            y='price',
            title=f'Box Plot of {column} vs Price',
            template='plotly_dark'
        )
        
        # Customize the layout
        fig.update_layout(
            xaxis_title=f'{column.replace("_", " ").capitalize()} ({units.get(column, "")})',
            yaxis_title=f'Price ({units.get("price", "")})',
            title_font_size=18,
            title_font_family='Arial',
            title_font_color='white',
            xaxis_tickangle=60
        )
        
        # Display the interactive plot
        st.write('Сонгосон баганыг үнийн дүнтэй харьцуулж харуулна.')
        
        st.plotly_chart(fig, use_container_width=True)
    
available_columns2= ['engine_capacity', 'year_of_manufacture', 'gearbox', 'wheel_position', 'type', 'color', 'year_of_manufacture',
                      'imported_year', 'engine','saloon_color','transmission','doors', 'price_binned']

selected_columns2 = st.multiselect('Select columns to view:', available_columns2, default=available_columns[:1], key='multiselect2')

if selected_columns2:
    for column in selected_columns2:
        fig = px.box(
            df,
            x=column,
            y='road_traveled',
            title=f'Box Plot of {column} vs Road traveled',
            template='plotly_dark'
        )
        
        # Customize the layout
        fig.update_layout(
            xaxis_title=f'{column.replace("_", " ").capitalize()} ({units.get(column, "")})',
            yaxis_title=f'Road traveled ({units.get("road_traveled", "")})',
            title_font_size=18,
            title_font_family='Arial',
            title_font_color='white',
            xaxis_tickangle=90
        )
        
        # Display the interactive plot
        st.write('Сонгосон баганыг явсан замтай харьцуулж харуулна.')
        st.plotly_chart(fig, use_container_width=True)


# Calculate correlation
correlation = df['road_traveled'].corr(df['price'])

# Create scatter plot with a trendline
scatter_fig = px.scatter(
    df,
    x='road_traveled',
    y='price',
    title=f'Scatter Plot of Price by Road Traveled (Correlation: {correlation:.2f})',
    labels={'road_traveled': 'Road Traveled', 'price': 'Price'},
    template='plotly_dark',
    trendline='ols',
    color_discrete_sequence=['#1f77b4']  # Color for points
)

# Customize trendline color
for trace in scatter_fig.data:
    if trace.mode == 'lines':  # This identifies the trendline
        trace.update(line=dict(color='red'))  # Set trendline color to red

# Update layout for the scatter plot
scatter_fig.update_layout(
    xaxis_title=f'Road Traveled ({units["road_traveled"]})',
    yaxis_title=f'Price ({units["price"]})',
    title_font_size=18,
    title_font_family='Arial',
    title_font_color='white',
)



# Hexbin plot using Plotly Graph Objects
hexbin_fig = go.Figure(go.Histogram2d(
    x=df['road_traveled'],
    y=df['price'],
    colorscale='Blues',
    xbins=dict(size=(df['road_traveled'].max() - df['road_traveled'].min()) / 20),  # Adjust grid size
    ybins=dict(size=(df['price'].max() - df['price'].min()) / 20)  # Adjust grid size
))

# Update layout for the hexbin-like plot
hexbin_fig.update_layout(
    xaxis_title=f'Road Traveled ({units["road_traveled"]})',
    yaxis_title=f'Price ({units["price"]})',
    title='Hexbin-like Plot of Price vs Road Traveled',
    title_font_size=18,
    title_font_family='Arial',
    title_font_color='white',
    template='plotly_dark'
)

# Display both plots in Streamlit
st.markdown('<h5 align=center>Scatter plot of Price by Road Traveled</h5>', unsafe_allow_html=True)
st.write('Явсан замыг үнэтэй харьцуулсан график. Явсан зам ихсэх тусам үнэ буурах хандлагатай.')
st.plotly_chart(scatter_fig, use_container_width=True)
st.markdown('<h5 align=center>Hexbin plot of Price by Road Traveled</h5>', unsafe_allow_html=True)
st.write('Өнгө гүнзгийрэх тусам илүү их утгатай гэсэн үг.')
st.plotly_chart(hexbin_fig, use_container_width=True)



st.header('Correlation')
st.write('Үнийг бусад тоон баганатай харьцуулж коррелиацыг харуулав.')

# Select columns

col2 = st.selectbox('Select second column:', numerical_columns  , key='col2')

# Calculate correlation between the selected columns
if col2:
    correlation_value = df['price'].corr(df[col2])
    st.write(f"Correlation between **{'price'}** and **{col2}**: {correlation_value:.2f}")

    # Plot correlation scatter plot
    fig = go.Figure(data=go.Scatter(
        x=df['price'],
        y=df[col2],
        mode='markers',
        marker=dict(color='blue', size=5),
    ))

    # Add trendline (optional)
    fig.add_trace(go.Scatter(
        x=df['price'],
        y=df['price'] * correlation_value,
        mode='lines',
        line=dict(color='red'),
        name='Trendline'
    ))

    # Update layout for better appearance
    fig.update_layout(
        title=f'Scatter Plot of {'price'} vs {col2}',
        xaxis_title='price',
        yaxis_title=col2,
        title_font_size=18,
        title_font_family='Arial',
        title_font_color='white',
        template='plotly_dark'
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Please select both columns to view the correlation.")


df_clustering = df[['price', 'road_traveled']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clustering)

st.title('K-Means Clustering')

# Sidebar for user inputs
st.header('User Input Parameters')
n_clusters = st.slider('Number of Clusters (k)', min_value=2, max_value=10, value=3, step=1)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Create an interactive scatter plot using Plotly Express
fig = px.scatter(
    df,
    x='price',
    y='road_traveled',
    color='Cluster',
    title='K-Means Clustering of Cars',
    labels={'price': 'Price', 'road_traveled': 'Road Traveled'},
    template='plotly_dark'
)

# Update layout for better appearance
fig.update_layout(
    xaxis_title='Price',
    yaxis_title='Road Traveled',
    title_font_size=18,
    title_font_family='Arial',
    title_font_color='white'
)

# Display the interactive plot in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Display the cluster centers in the original feature space
cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers_original, columns=df_clustering.columns)
st.write("Cluster Centers in Original Feature Space:")
st.write(cluster_centers_df)
