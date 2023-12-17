import pickle
from datetime import date

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from pandas import Timestamp
from sklearn.preprocessing import OneHotEncoder

df_clean = pd.read_csv("data/processed/CRS_data_cleaned.csv")
df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
df_dec5_pool = pd.read_csv(
    "data/processed/CRS_Scores_as_of_DEC5.csv"
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container(
    [
        dbc.Container(
            [
                html.H1(
                    "CRS Score Predictor",
                    className="display-3 fw-bold",
                ),
            ]
        ),
        dbc.Tabs(
            [
                dbc.Tab(label="Number of Draws per Year", tab_id="tab-1"),
                dbc.Tab(label="Number of Draws per Category", tab_id="tab-2"),
                dbc.Tab(label="Number of Invitations Per Category", tab_id="tab-3"),
                dbc.Tab(label="Number of invitations per year", tab_id="tab-4"),
                dbc.Tab(label="No Program Draw Trends", tab_id="tab-5"),
                dbc.Tab(label="Canadian Experience Class Draw Trends", tab_id="tab-6"),
                dbc.Tab(label="Gap Between Draws", tab_id="tab-7"),
                dbc.Tab(label="Express Entry Pool Overview", tab_id="tab-8"),
                dbc.Tab(label="Express Entry Pool Overview 2", tab_id="tab-9"),
                dbc.Tab(label="Prediction CRS or #Invitations", tab_id="tab-11"),
                # dbc.Tab(label="No Program Draw Trends", tab_id="tab-12"),
            ],
            id="tabs",
            active_tab="tab-1",
        ),
        html.Div(id="tab-content", className="px-4"),
    ]
)


@app.callback(Output("tab-content", "children"), Input("tabs", "active_tab"))
def render_content(tab):
    if tab == "tab-1":
        draws_per_year = df_clean.groupby("Year").size()
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=draws_per_year.index,
                    y=draws_per_year.values,
                    mode="lines+markers+text",
                    text=draws_per_year.values,
                    textposition="top center",
                )
            ]
        )
        fig.update_layout(
            title_text="Number of Draws per Year",
            xaxis_title="Year",
            yaxis_title="#Draws",
        )
        return dcc.Graph(figure=fig)
    elif tab == "tab-2":
        draws_per_category = df_clean.groupby("Draw_Category").size()
        fig = go.Figure(
            data=[
                go.Bar(
                    x=draws_per_category.index,
                    y=draws_per_category.values,
                    text=draws_per_category.values,
                    textposition="auto",
                    marker_color="Chartreuse",
                )
            ]
        )
        fig.update_layout(
            title_text="Number of Draws per Category",
            xaxis_title="Draw Category",
            yaxis_title="#Draws",
        )
        return dcc.Graph(figure=fig)
    elif tab == "tab-3":
        invitations_per_category = df_clean.groupby("Draw_Category")[
            "Invitations"
        ].sum()
        fig = go.Figure(
            data=[
                go.Bar(
                    x=invitations_per_category.index,
                    y=invitations_per_category.values,
                    text=invitations_per_category.values,
                    textposition="auto",
                    marker_color="Coral",
                )
            ]
        )
        fig.update_layout(
            title_text="Number of Invitations per Category",
            xaxis_title="Draw Category",
            yaxis_title="# Draws",
        )
        return dcc.Graph(figure=fig)
    elif tab == "tab-4":
        invitations_per_year = df_clean.groupby("Year")["Invitations"].sum()
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=invitations_per_year.index,
                    y=invitations_per_year.values,
                    mode="lines+markers+text",
                    text=invitations_per_year.values,
                    textposition="top center",
                    marker_color="Lime",
                )
            ]
        )
        fig.update_layout(
            title_text="Number of Invitations per Year",
            xaxis_title="Year",
            yaxis_title="# Invitations",
        )
        return dcc.Graph(figure=fig)
    elif tab == "tab-5":
        no_program_draws_min_monthly = (
            df_clean[df_clean["Draw_Category"] == "No Category Specified"]
            .groupby(["Year", "Month"])["CRS"]
            .min()
        )
        no_program_draws_min_monthly = no_program_draws_min_monthly.reset_index()
        no_program_draws_min_monthly["Month_Year"] = no_program_draws_min_monthly[
            ["Year", "Month"]
        ].apply(lambda x: f"{x[1]}-{x[0]}", axis=1)
        cec_min_monthly = (
            df_clean[df_clean["Draw_Category"] == "Canadian Experience Class only"]
            .groupby(["Year", "Month"])["CRS"]
            .min()
        )
        cec_min_monthly = cec_min_monthly.reset_index()
        cec_min_monthly["Month_Year"] = cec_min_monthly[["Year", "Month"]].apply(
            lambda x: f"{x[1]}-{x[0]}", axis=1
        )
        cec_min_monthly.rename(columns={"CRS": "CEC_CRS"}, inplace=True)
        monthly_trends = pd.merge(
            left=no_program_draws_min_monthly,
            right=cec_min_monthly,
            on=["Year", "Month"],
            how="left",
        )
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=monthly_trends["Month_Year_x"],
                    y=monthly_trends["CRS"],
                    mode="lines+markers+text",
                    # text=monthly_trends["CRS"],
                    # textposition="top center",
                    marker_color="Lime",
                )
            ]
        )
        fig.update_layout(
            title_text="No Program Draw Trends",
            xaxis_title="Month-Year",
            yaxis_title="CRS",
        )
        return dcc.Graph(figure=fig)

    elif tab == "tab-6":
        cec_min_monthly = (
            df_clean[df_clean["Draw_Category"] == "Canadian Experience Class only"]
            .groupby(["Year", "Month"])["CRS"]
            .min()
        )
        cec_min_monthly = cec_min_monthly.reset_index()
        cec_min_monthly["Month_Year"] = cec_min_monthly[["Year", "Month"]].apply(
            lambda x: f"{x[1]}-{x[0]}", axis=1
        )  # .assign(day=1))
        cec_min_monthly.rename(columns={"CRS": "CEC_CRS"}, inplace=True)

        # Create the line plot with markers and text
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=cec_min_monthly["Month_Year"],
                    y=cec_min_monthly["CEC_CRS"],
                    mode="lines+markers+text",  # creates a line plot with markers and text
                    # text=no_program_draws_min_monthly['CRS'],  # this will display the value at each marker
                    textposition="top center",
                    marker_color="SpringGreen",  # this will position the text above each marker
                )
            ]
        )

        # fig.add_trace(go.Scatter(x=monthly_trends['Month_Year_x'], y=monthly_trends['CEC_CRS'], mode='lines'))

        fig.update_layout(
            title_text="CRS Score Trend in Canadian Experience Class",
            xaxis_title="Year",  # this will add a label to the x-axis
            yaxis_title="CRS Score",
        )

        return dcc.Graph(figure=fig)

    elif tab == "tab-8":
        fig = px.pie(
            df_dec5_pool[df_dec5_pool["Total Range"] == 1],
            values="Number of candidates",
            names="CRS score range",
            title="Proportion of Candidates per CRS Score Range in Dec 5 Pool",
            hole=0.4,
        )
        fig.update_traces(textinfo="label+percent", insidetextorientation="radial")
        return dcc.Graph(figure=fig)

    elif tab == "tab-9":
        fig = px.line(
            df_dec5_pool[df_dec5_pool["Total Range"] == 0],
            x="CRS score range",
            y="Number of candidates",
            title="Number of Candidates per CRS Score Between 450 and 500 in Dec 5 Pool",
            markers=True,
        )
        fig.update_traces(line=dict(color="OrangeRed"))
        return dcc.Graph(figure=fig)
    elif tab == "tab-7":
        gap_bw_draws = 365 // df_clean.groupby("Year")["Date"].count()

        # Create the line plot with markers and text
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=gap_bw_draws.index,
                    y=gap_bw_draws.values,
                    mode="lines+markers+text",  # creates a line plot with markers and text
                    text=gap_bw_draws.values,  # this will display the value at each marker
                    textposition="top center",
                    marker_color="orange",  # this will position the text above each marker
                )
            ]
        )

        fig.update_layout(
            title_text=f"Gap between the Draws for each year, with a median of {np.median(gap_bw_draws)} days",
            xaxis_title="Year",  # this will add a label to the x-axis
            yaxis_title="# Gap in Days",
        )

        return dcc.Graph(figure=fig)

    elif tab == "tab-11":
        return html.Div(
            [
                html.Br(),
                html.H4("CRS Score Prediction"),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Label("Date", htmlFor="date-picker-single"), width=2
                        ),
                        dbc.Col(
                            dcc.DatePickerSingle(
                                id="date-picker-single", date=date.today()
                            ),
                            width=3,
                        ),
                        dbc.Col(html.Label("Category", htmlFor="dropdown"), width=2),
                        dbc.Col(
                            dcc.Dropdown(
                                id="dropdown",
                                options=[
                                    "No Category Specified",
                                    "Canadian Experience Class only",
                                    "Provincial nominees only",
                                    "Federal Skilled Trades candidates only",
                                    "Foreign Skilled Worker Program nominees only",
                                    "Healthcare occupations only",
                                    "STEM occupations only",
                                    "French language proficiency only",
                                    "Trades occupations only",
                                    "Transport occupations only",
                                    "Agriculture and Agri-Food occupations only",
                                ],
                                value="OPT1",
                            ),
                            width=3,
                            style={"color": "black"},
                        ),
                        dbc.Col(
                            html.Button("Predict", id="crs-predict-button", n_clicks=0),
                            width=2,
                        ),
                    ]
                ),
                html.Br(),
                html.Div(id="crs-output-container"),
                html.Br(),
                html.H4("Number of Invitations Prediction"),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Label("Date", htmlFor="date-picker-single-2"), width=2
                        ),
                        dbc.Col(
                            dcc.DatePickerSingle(
                                id="date-picker-single-2", date=date.today()
                            ),
                            width=3,
                        ),
                        dbc.Col(html.Label("Category", htmlFor="dropdown-2"), width=2),
                        dbc.Col(
                            dcc.Dropdown(
                                id="dropdown-2",
                                options=[
                                    "No Category Specified",
                                    "Canadian Experience Class only",
                                    "Provincial nominees only",
                                    "Federal Skilled Trades candidates only",
                                    "Foreign Skilled Worker Program nominees only",
                                    "Healthcare occupations only",
                                    "STEM occupations only",
                                    "French language proficiency only",
                                    "Trades occupations only",
                                    "Transport occupations only",
                                    "Agriculture and Agri-Food occupations only",
                                ],
                                value="OPT1",
                            ),
                            width=3,
                            style={"color": "black"},
                        ),
                        dbc.Col(
                            html.Button("Predict", id="inv-predict-button", n_clicks=0),
                            width=2,
                        ),
                    ]
                ),
                html.Br(),
                html.Div(id="inv-output-container"),
                html.Br(),
            ]
        )


encoder = OneHotEncoder(sparse=False)


@app.callback(
    Output("crs-output-container", "children"),
    Input("crs-predict-button", "n_clicks"),
    [
        State("date-picker-single", "date"),
        State("dropdown", "options"),
        State("dropdown", "value"),
    ],
)
def update_output(n, date, categories, category):
    if n >= 1:
        date = Timestamp(date)
        year = date.year
        month = date.month
        quarter = date.quarter

        # One-hot encode the category
        category_encoded = encoder.fit_transform(np.array([[category]]))

        data = {f"Draw_Category_{cat}": [0] for cat in categories}
        df = pd.DataFrame(data)
        df[f"Draw_Category_{category}"] = 1
        df_d = pd.DataFrame({"Year": [year], "Month": [month], "Quarter": [quarter]})
        x_test = pd.concat([df_d, df], axis=1)
        filename = "saved_models/CRS_pred_with_xgb.sav"
        model_v1_ = pickle.load(open(filename, "rb"))
        y_pred = model_v1_.predict([x_test.iloc[0, :].values])[0]

        # Return the array
        return html.Div(
            [
                html.H5("Predicted Minimum CRS Score : "),
                html.Strong(round(y_pred, 0), style={"font-size": "xx-large"}),
            ]
        )


@app.callback(
    Output("inv-output-container", "children"),
    Input("inv-predict-button", "n_clicks"),
    [
        State("date-picker-single-2", "date"),
        State("dropdown-2", "options"),
        State("dropdown-2", "value"),
    ],
)
def update_output(n, date, categories, category):
    if n >= 1:
        date = Timestamp(date)
        year = date.year
        month = date.month
        quarter = date.quarter

        # One-hot encode the category
        category_encoded = encoder.fit_transform(np.array([[category]]))

        data = {f"Draw_Category_{cat}": [0] for cat in categories}
        df = pd.DataFrame(data)
        df[f"Draw_Category_{category}"] = 1
        df_d = pd.DataFrame({"Year": [year], "Month": [month], "Quarter": [quarter]})
        x_test = pd.concat([df_d, df], axis=1)
        filename = "saved_models/Invitations_pred_with_xgb.sav"
        model_v1_ = pickle.load(open(filename, "rb"))
        y_pred = model_v1_.predict([x_test.iloc[0, :].values])[0]

        # Return the array
        return html.Div(
            [
                html.H5("Predicted Number of Invitations : "),
                html.Strong(round(y_pred, 0), style={"font-size": "xx-large"}),
            ]
        )


if __name__ == "__main__":
    app.run_server(debug=False)
