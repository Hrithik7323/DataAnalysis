import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

# Load the Iris dataset from sklearn or from a CSV file
from sklearn.datasets import load_iris

# Load the dataset into a pandas DataFrame
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Preview the data
df.head()





# Create Dash application
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    # Dropdown for species selection
    html.H1("Iris Dataset Visualization Dashboard"),
    dcc.Dropdown(
        id='species-dropdown',
        options=[
            {'label': species, 'value': species}
            for species in df['species'].unique()
        ],
        value=df['species'].unique()[0],  # Default value
        multi=False
    ),
    
    # Scatter plot for Sepal Length vs Sepal Width
    dcc.Graph(id='scatter-plot'),
    
    # Histogram for Sepal Length
    dcc.Graph(id='histogram')
])

# Define the callback function for interactivity
@app.callback(
    [dash.dependencies.Output('scatter-plot', 'figure'),
     dash.dependencies.Output('histogram', 'figure')],
    [dash.dependencies.Input('species-dropdown', 'value')]
)
def update_graphs(species):
    # Filter data based on selected species
    filtered_df = df[df['species'] == species]
    
    # Create scatter plot
    scatter_fig = px.scatter(filtered_df, x=iris.feature_names[0], y=iris.feature_names[1],
                             color='species', title="Sepal Length vs Sepal Width")
    
    # Create histogram
    histogram_fig = px.histogram(filtered_df, x=iris.feature_names[0], nbins=20,
                                  title="Distribution of Sepal Length")
    
    return scatter_fig, histogram_fig

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)


python app.py

# Modify the layout to add another dropdown for selecting the X and Y axis
app.layout = html.Div([
    # Dropdown for species selection
    dcc.Dropdown(id='species-dropdown',
                 options=[{'label': species, 'value': species} for species in df['species'].unique()],
                 value=df['species'].unique()[0], multi=False),
    
    # Dropdown for feature selection
    dcc.Dropdown(id='x-axis-dropdown',
                 options=[{'label': feature, 'value': feature} for feature in iris.feature_names],
                 value=iris.feature_names[0], multi=False),
    
    dcc.Dropdown(id='y-axis-dropdown',
                 options=[{'label': feature, 'value': feature} for feature in iris.feature_names],
                 value=iris.feature_names[1], multi=False),
    
    # Scatter plot for dynamic feature selection
    dcc.Graph(id='scatter-plot'),
])

# Callback function for updating the scatter plot
@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [dash.dependencies.Input('species-dropdown', 'value'),
     dash.dependencies.Input('x-axis-dropdown', 'value'),
     dash.dependencies.Input('y-axis-dropdown', 'value')]
)
def update_scatter(species, x_axis, y_axis):
    # Filter data based on selected species
    filtered_df = df[df['species'] == species]
    
    # Create scatter plot with selected features for X and Y axes
    scatter_fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color='species', title=f"{x_axis} vs {y_axis}")
    
    return scatter_fig
