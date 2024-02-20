import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from visualize import read_data, convert_annots_to_matrix, tsne_on_annotations, colors

annots = read_data("../../data/youth/signature/all/all_signature.json")
annots_matrix, labels, metadata = convert_annots_to_matrix(annots)
S_t_sne = tsne_on_annotations(annots_matrix)
# Ensure you have the images in the 'assets' folder and update these paths accordingly
df = pd.DataFrame({
    'x': S_t_sne[:, 0],
    'y': S_t_sne[:, 1],
    'image_url': metadata
})

# Create a Plotly figure
fig = px.scatter(df, x='x', y='y', hover_data=['image_url'], color=[colors[p_color] for p_color in labels])

# Initialize Dash app
app = dash.Dash(__name__)


# App layout with 4 images in a 2x2 grid
app.layout = html.Div([
    dcc.Graph(id='scatter-plot', figure=fig),
    html.Div([
        html.Div([
            html.Img(id='image1', style={'width': '25%', 'display': 'inline-block'}),
            html.Img(id='image2', style={'width': '25%', 'display': 'inline-block'})
        ]),
        html.Div([
            html.Img(id='image3', style={'width': '25%', 'display': 'inline-block'}),
            html.Img(id='image4', style={'width': '25%', 'display': 'inline-block'})
        ])
    ])
])


# Callback to update image based on hover data
@app.callback(
    [Output('image1', 'src'),
     Output('image2', 'src'),
     Output('image3', 'src'),
     Output('image4', 'src')],
    [Input('scatter-plot', 'hoverData')]
)
def update_image(hover_data):
    if hover_data is not None:
        cam1_path = hover_data['points'][0]['customdata'][0]
        return [cam1_path, cam1_path.replace('cam1', 'cam2'), cam1_path.replace('cam1', 'cam3'), cam1_path.replace('cam1', 'cam4')]
    else:
        return ['', '', '', '']


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
