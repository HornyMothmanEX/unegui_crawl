import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='Unegui apartment', page_icon=':House:', layout='wide')

df = pd.read_csv('20240820_160338_unegui_apartment.csv')
units = {
    'Price': 'сая ₮',
    'Area': 'м²'
}


df['Area'] = df['Area'].apply(lambda x: 500 if x > 500 else x)

st.title(':bar_chart: Dashboard')
st.markdown('##')

# Initialize session state for buttons
if 'button1_active' not in st.session_state:
    st.session_state.button1_active = False
if 'button2_active' not in st.session_state:
    st.session_state.button2_active = False

# Define callback functions to control button states
def activate_button1():
    st.session_state.button1_active = True
    st.session_state.button2_active = False

def activate_button2():
    st.session_state.button1_active = False
    st.session_state.button2_active = True

col1, col2 = st.columns([2, 2]) 

# Place buttons in the columns
with col1:
    button1 = st.button('м²-аар харах', on_click=activate_button1)
with col2:
    button2 = st.button('Нийт үнээр харах', on_click=activate_button2)

# Display which button is active
if st.session_state.button1_active:
    df['Price'] = df.apply(
    lambda row: (row['Price'] / row['Area']) if row['Area'] != 0 and row['Price'] > 15 else row['Price'],
    axis=1)
       
elif st.session_state.button2_active:
    df['Price'] = df.apply(
    lambda row: row['Price'] * row['Area'] if row['Price'] < 15 else row['Price'],
    axis=1,
    )
df['District'] = df['Location'].apply(lambda x: x.split(',')[0].strip())
df['Location'] = df['Location'].apply(lambda x: x.split(',', 1)[-1].strip())
columns = list(df.columns)
columns.insert(1, columns.pop(columns.index('District')))
df = df[columns]

st.dataframe(df) 


df_autox = df.groupby('Location')['Price'].mean().sort_values(ascending=False).reset_index()
df_autox.columns = ['Location', 'Average Price']
title_counts = df['Location'].value_counts()
title_counts_df = pd.DataFrame(title_counts)


# Display the scrollable table
st.write('Байршил болон түүн дээрх заруудын тоо', title_counts.sum())
st.dataframe(title_counts_df, height=400)

fig = px.bar(
    df_autox,
    x='Location',
    y='Average Price',
    title='Average Price by Location',
    labels={'Location': 'Location', 'Average Price': 'Average Price'},
    template='plotly_dark'
)

# Update layout for better appearance
fig.update_layout(
    xaxis_title='Location',
    yaxis_title=f'Average Price {units["Price"]}',
    title_font_size=18,
    title_font_family='Arial',
    title_font_color='white',
    xaxis_tickangle=-45
)

st.markdown('<h5 align=center>Average Price by Location</h5>', unsafe_allow_html=True)
st.write('Байршил болон тэдгээрийн дундаж үнийг харьцуулсан график')
st.plotly_chart(fig, use_container_width=True) 

completed_year_counts = df['Year of completion'].value_counts().reset_index()
completed_year_counts.columns = ['Year of completion', 'Count']

fig = px.bar(
    completed_year_counts,
    x='Year of completion',
    y='Count',
    title='Count of Observations by Completed Year',
    labels={'Completed Year': 'Completed year', 'Count': 'Count'},
    template='plotly_dark'
)

# Update layout for better appearance
fig.update_layout(
    xaxis_title='Completed Year',
    yaxis_title='Count',
    title_font_size=18,
    title_font_family='Arial',
    title_font_color='white',
    xaxis_tickangle=-45
)

st.markdown('<h5 align=center>Completed Year vs Count</h5>', unsafe_allow_html=True)
st.write('Орон сууц баригдсан он болон түүний тооны график')
st.plotly_chart(fig, use_container_width=True)

Room_counts = df['Rooms'].value_counts()
fig1 = go.Figure(data=[go.Bar(x=Room_counts.index, y=Room_counts.values)])
fig1.update_layout(
    title='Price and Rooms',
    xaxis_title='Өрөөний тоо ба давтамж',
    yaxis_title='Frequency',
    xaxis_tickangle=0
)

floor_counts = df['Which floor'].value_counts()
fig2 = go.Figure(data=[go.Bar(x=floor_counts.index, y=floor_counts.values)])
fig2.update_layout(
    title='Floor frequency',
    xaxis_title='Давхарын давтамж',
    yaxis_title='Frequency',
    xaxis_tickangle=0
)

garage_counts = df['Garage'].value_counts()
fig3 = go.Figure(data=[go.Bar(x=garage_counts.index, y=garage_counts.values)])
fig3.update_layout(
    title='Garage',
    xaxis_title='Гаражийн давтамж',
    yaxis_title='Frequency',
    xaxis_tickangle=0
)

st.write('Өрөөний тоо хэр давтагдаж байгааг харуулав.')
st.plotly_chart(fig1, use_container_width=True)
st.write('Хэдэн давхарын байрлал хэр давтагдаж байгааг харуулав.')
st.plotly_chart(fig2, use_container_width=True)
st.write('Гараж байгаа эсэх хэр давтагдаж байгааг харуулав.')
st.plotly_chart(fig3, use_container_width=True) 

df['Date'] = pd.to_datetime(df['Date'])

# Group by date and count the occurrences
date_counts = df['Date'].value_counts().sort_index()

# Create a histogram or line chart
fig = px.line(
    date_counts,
    x=date_counts.index,
    y=date_counts.values,
    title='Frequency of Data Points by Date',
    labels={'x': 'Date', 'y': 'Frequency'},
    template='plotly_dark'
)

# Customize the appearance of the chart
fig.update_traces(marker_color='lightblue')
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Frequency',
    title_font_size=18,
    title_font_family='Arial',
    title_font_color='white'
)

st.markdown('<span align=center>Зарууд орсон цаг хугацаа болон түүний давтамжийг харьцуулсан график</span>', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True)

df['Date'] = df['Date'].dt.date

# Create a box plot with Date on the x-axis and Price on the y-axis
fig = px.box(
    df,
    x='Date',
    y='Price',
    title='Price Distribution Over Time',
    labels={
        'Date': 'Date',
        'Price': 'Price (in millions)'  # Adjust the label according to your data
    },
    template='plotly_dark'
)

# Customize the chart (optional)
fig.update_traces(marker_color='lightblue')
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price',
    title_font_size=18,
    title_font_family='Arial',
    title_font_color='white',
    xaxis=dict(
        showgrid=True,
        zeroline=True
    ),
    yaxis=dict(showgrid=True, zeroline=True),
)

st.markdown('<span align=center>Орон сууцны зар орсон цаг хугацаа болон түүний үнийн мэдээлэлийг харьцуулсан график</span>', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True)


# with st.sidebar:
#     with st.expander('Dataframe filter', expanded=False):
#         st.header('Filter Options')
        
#         room = st.multiselect('Number of rooms:', options=df['Rooms'].unique())
#         year_completed = st.multiselect('Completed year:', options=df['Year of completion'].unique())
#         garage = st.multiselect('Have garage:', options=df['Garage'].unique())
#         area = st.multiselect('Area size:', options=df['Area'].unique())
#         location = st.multiselect('Select location:', options=df['Location'].unique())

# # Apply filters
# if room:
#     df = df[df['Room'].isin(room)]
# if year_completed:
#     df = df[df['Year of completion'].isin(year_completed)]
# if garage:
#     df = df[df['Garage'].isin(garage)]
# if area:
#     df = df[df['gearbox'].isin(area)]
# if location:
#     df = df[df['location'].isin(location)]

numerical_columns = df.select_dtypes(include='number').columns

if not numerical_columns.empty:
    st.title('Z score graph')
    st.write('Z score нь өгөгдсөн тархалтын дунджаас хэдэн стандарт хазайлттай болохыг хэлж өгдөг. Хэрэв сөрөг байвал дунджаас доогуур, эерэг байвал дунджаас дээгүүр байгааг харуулна.')
    with st.expander('Z-score Filter', expanded=False):
        selected_columns = st.multiselect(
            'Select columns to plot:',
            options=numerical_columns,
            default=numerical_columns[0]
        )

    if selected_columns:
        z_scores_all_columns = df[selected_columns].apply(lambda x: (x - x.mean()) / x.std())

        # Color palette for differentiation
        color_palette = px.colors.qualitative.Plotly  # A set of colors to use
        color_iterator = iter(color_palette)  # Create an iterator for colors

        for column in selected_columns:
            # Assign a unique color for each column
            color = next(color_iterator, random.choice(color_palette))

            fig = px.histogram(
                z_scores_all_columns,
                x=column,
                title=f'{column.replace("_", " ").capitalize()} Z-score Distribution',
                nbins=100,
                template='plotly_dark'
            )
            fig.update_traces(marker_color=color, opacity=0.7)

            fig.update_layout(
                xaxis_title=f'{column.replace("_", " ").capitalize()}',
                yaxis_title='Count',
                title_font_size=18,
                title_font_family='Arial',
                title_font_color='white'
            )

            st.plotly_chart(fig, use_container_width=True)
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
                    xaxis_title=f'{target_column.replace("_", " ").capitalize()} {units.get(target_column, "")}',
                    yaxis_title=f'{column.replace("_", " ").capitalize()} {units.get(column, "")}'
                )
                st.write('Үнийг тоон багануудтай харьцуулсан график')
                st.plotly_chart(fig, use_container_width=True)  # Use st.pyplot to display Matplotlib plots in Streamlit

        target_column = 'Price'
        display_numerical_boxplots(df, target_column)
else:
    st.write("No numerical columns available to plot.")

Q1 = df['Area'].quantile(0.25)
Q3 = df['Area'].quantile(0.75)
IQR = Q3 - Q1

outlier_threshold = 1.5

outlier_mask = (df['Area'] < Q1 - outlier_threshold * IQR) | (df['Area'] > Q3 + outlier_threshold * IQR)

df = df[~outlier_mask]
df_null = df.isna().mean().round(4) * 100

df_null.sort_values(ascending=False).head()

outliers = ['Price']

# Create a Plotly box plot for the 'price' column
if df.empty:
    st.write("No data available to plot after filtering out outliers.")
else:
    # Create and display the plot
    fig = px.box(
        df,
        y='Price',
        title='Үнэ хэдээс хэдийн хооронд тархсан байгааг харуулж байна.',
        labels={'Price': 'Price'},
        template='plotly_dark'
    )

    fig.update_layout(
        yaxis_title=f'Price ({units["Price"]})',
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

df['District'].unique()
df.drop_duplicates(inplace=True)

cat_col = df.select_dtypes(include=['object']).columns
num_col = df.select_dtypes(exclude=['object']).columns
df_cat = df[cat_col]
df_num = df[num_col]

column = st.selectbox('Select column to visualize:', ['Price', 'Area'])
st.write('Үнэ/Талбайн давтамж.')
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
bin_edges_price = np.linspace(df['Price'].min(), df['Price'].max(), num_bins + 1)
df['Price Binned'] = pd.cut(df['Price'], bins=bin_edges_price)

# Binning 'Road Traveled'
bin_edges_traveled = np.linspace(df['Area'].min(), df['Area'].max(), num_bins + 1)
df['Area Binned'] = pd.cut(df['Area'], bins=bin_edges_traveled).astype(str)


# List of available columns for plotting
available_columns = [
    'Location', 'District', 'Floor', 'Balcony', 'Year of completion', 
    'Garage', 'Window', 'Number of apartment floor', 'Door', 'Which floor', 
    'Number of windows', 'Area Binned'
]

# Multi-select dropdown for column selection
selected_columns = st.multiselect('Select columns to view:', available_columns, default=available_columns[:1], key='multiselect1')

# Plotting based on selected columns
if selected_columns:
    for column in selected_columns:
        fig = px.box(
            df,
            x=column,
            y='Price',
            title=f'Box Plot of {column} vs Price',
            template='plotly_dark'
        )
        
        # Customize the layout
        fig.update_layout(
            xaxis_title=f'{column.replace("_", " ").capitalize()} {units.get(column, "")}',
            yaxis_title=f'Price {units.get("price", "")}',
            title_font_size=18,
            title_font_family='Arial',
            title_font_color='white',
            xaxis_tickangle=60
        )
        
        # Display the interactive plot
        st.write('Сонгосон баганыг үнийн дүнтэй харьцуулж харуулна.')
        
        st.plotly_chart(fig, use_container_width=True)
        
available_columns2 = [
    'Location', 'District', 'Floor', 'Balcony', 'Year of completion', 
    'Garage', 'Window', 'Number of apartment floor', 'Door', 'Which floor', 
    'Number of windows', 'Price Binned'
]

selected_columns2 = st.multiselect('Select columns to view:', available_columns2, default=available_columns[:1], key='multiselect2')

if selected_columns2:
    for column in selected_columns2:
        fig = px.box(
            df,
            x=column,
            y='Area',
            title=f'Box Plot of {column} vs Area',
            template='plotly_dark'
        )
        
        # Customize the layout
        fig.update_layout(
            xaxis_title=f'{column.replace("_", " ").capitalize()} {units.get(column, "")}',
            yaxis_title=f'Area {units.get("Area", "")}',
            title_font_size=18,
            title_font_family='Arial',
            title_font_color='white',
            xaxis_tickangle=90
        )
        
        # Display the interactive plot
        st.write('Сонгосон баганыг талбайтай харьцуулж харуулна.')
        st.plotly_chart(fig, use_container_width=True)

correlation = df['Area'].corr(df['Price'])

# Create scatter plot with a trendline
scatter_fig = px.scatter(
    df,
    x='Area',
    y='Price',
    title=f'Scatter Plot of Price by Area (Correlation: {correlation:.2f})',
    labels={'Area': 'Area', 'Price': 'Price'},
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
    xaxis_title=f'Area ({units["Area"]})',
    yaxis_title=f'Price ({units["Price"]})',
    title_font_size=18,
    title_font_family='Arial',
    title_font_color='white',
)



# Hexbin plot using Plotly Graph Objects
hexbin_fig = go.Figure(go.Histogram2d(
    x=df['Area'],
    y=df['Price'],
    colorscale='Blues',
    xbins=dict(size=(df['Area'].max() - df['Area'].min()) / 20),  # Adjust grid size
    ybins=dict(size=(df['Price'].max() - df['Price'].min()) / 20)  # Adjust grid size
))

# Update layout for the hexbin-like plot
hexbin_fig.update_layout(
    xaxis_title=f'Area ({units["Area"]})',
    yaxis_title=f'Price ({units["Price"]})',
    title='Hexbin-like Plot of Price vs Area',
    title_font_size=18,
    title_font_family='Arial',
    title_font_color='white',
    template='plotly_dark'
)

# Display both plots in Streamlit
st.markdown('<h5 align=center>Scatter plot of Price by Area</h5>', unsafe_allow_html=True)
st.write('Талбайг үнэтэй харьцуулсан цэгэн график. Талбай ихсэх тусам үнэ өсөх хандлагатай.')
st.plotly_chart(scatter_fig, use_container_width=True)
st.markdown('<h5 align=center>Hexbin plot of Price by Area</h5>', unsafe_allow_html=True)
st.write('Өнгө гүнзгийрэх тусам илүү их утгатай гэсэн үг.')
st.plotly_chart(hexbin_fig, use_container_width=True)

st.header('Correlation')
st.write('Үнийг бусад тоон баганатай харьцуулж коррелиацыг харуулав.')

# Select columns

col2 = st.selectbox('Select second column:', numerical_columns  , key='col2')

# Calculate correlation between the selected columns
if col2:
    correlation_value = df['Price'].corr(df[col2])
    st.write(f"Correlation between **{'Price'}** and **{col2}**: {correlation_value:.2f}")

    # Plot correlation scatter plot
    fig = go.Figure(data=go.Scatter(
        x=df['Price'],
        y=df[col2],
        mode='markers',
        marker=dict(color='blue', size=5),
    ))

    # Add trendline (optional)
    fig.add_trace(go.Scatter(
        x=df['Price'],
        y=df['Price'] * correlation_value,
        mode='lines',
        line=dict(color='red'),
        name='Trendline'
    ))

    # Update layout for better appearance
    fig.update_layout(
        title=f'Scatter Plot of {'Price'} vs {col2}',
        xaxis_title=col2,
        yaxis_title='Price',
        title_font_size=18,
        title_font_family='Arial',
        title_font_color='white',
        template='plotly_dark'
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Please select both columns to view the correlation.")

df_clustering = df[['Price', 'Area']]
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
    x='Price',
    y='Area',
    color='Cluster',
    title='K-Means Clustering of Cars',
    labels={'price': 'Price', 'road_traveled': 'Road Traveled'},
    template='plotly_dark'
)

# Update layout for better appearance
fig.update_layout(
    xaxis_title='Price',
    yaxis_title='Area',
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

