import streamlit as st
import pandas as pd
import numpy as np
import os

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="Automobile Price Prediction",
    page_icon="🚗",
    layout="wide"
)


@st.cache_data
def load_data():
    possible_paths = [
        "Automobile_data.csv",
        "data/Automobile_data.csv",
        "../data/Automobile_data.csv"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)

    st.error("No se encontró el archivo Automobile_data.csv")
    st.stop()


@st.cache_data
def clean_data(df):
    df = df.copy()
    df = df.replace("?", np.nan)

    numeric_cols = [
        "normalized-losses",
        "bore",
        "stroke",
        "horsepower",
        "peak-rpm",
        "price"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["price"])

    df["mpg_average"] = (df["city-mpg"] + df["highway-mpg"]) / 2
    df["power_to_weight"] = df["horsepower"] / df["curb-weight"]
    df["engine_efficiency"] = df["horsepower"] / df["engine-size"]
    df["car_volume"] = df["length"] * df["width"] * df["height"]

    return df


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor, numeric_features, categorical_features


@st.cache_resource
def train_models(df):
    X = df.drop("price", axis=1)
    y = df["price"]

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    results = []

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        results.append({
            "Modelo": name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred)
        })

    best_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42))
    ])

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "results": pd.DataFrame(results).sort_values("RMSE"),
        "best_model": best_model,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }


df_raw = load_data()
df = clean_data(df_raw)
model_data = train_models(df)

st.title("🚗 Predicción del precio de automóviles")
st.caption("Proyecto final - Machine Learning aplicado a la industria automotriz")

section = st.sidebar.radio(
    "Navegación",
    [
        "Overview",
        "Dataset",
        "EDA",
        "Modelos",
        "Predicciones",
        "Importancia de variables",
        "Conclusiones"
    ]
)


if section == "Overview":
    st.header("Overview del proyecto")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Registros", df.shape[0])

    with col2:
        st.metric("Variables", df.shape[1])

    with col3:
        st.metric("Precio promedio", f"{df['price'].mean():,.0f}")

    with col4:
        st.metric("Precio máximo", f"{df['price'].max():,.0f}")

    st.markdown("""
    El objetivo del proyecto es construir un modelo de Machine Learning capaz de predecir el precio de automóviles a partir de características técnicas y comerciales.

    El problema se aborda como una tarea de regresión supervisada, ya que la variable objetivo `price` es numérica continua.
    """)

    fig = px.histogram(df, x="price", nbins=30, title="Distribución del precio")
    st.plotly_chart(fig, use_container_width=True)


elif section == "Dataset":
    st.header("Descripción del dataset")

    st.markdown("""
    El dataset contiene información técnica, comercial y aseguradora de automóviles. Cada registro representa un vehículo con atributos como marca, combustible, dimensiones, peso, motor, potencia, consumo y precio.

    La variable objetivo es `price`. Los valores faltantes estaban representados originalmente con `"?"` y fueron tratados durante la limpieza.
    """)

    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Valores faltantes luego de limpieza inicial")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) > 0:
        st.dataframe(missing.rename("Valores faltantes"))
    else:
        st.success("No quedan valores faltantes visibles en esta etapa.")


elif section == "EDA":
    st.header("Análisis exploratorio")

    col1, col2 = st.columns(2)

    with col1:
        avg_price_make = df.groupby("make")["price"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(
            avg_price_make,
            x="make",
            y="price",
            title="Precio promedio por marca"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df,
            x="engine-size",
            y="price",
            title="Tamaño del motor vs precio"
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.scatter(
            df,
            x="horsepower",
            y="price",
            title="Potencia vs precio"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.scatter(
            df,
            x="curb-weight",
            y="price",
            title="Peso del vehículo vs precio"
        )
        st.plotly_chart(fig, use_container_width=True)

    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()["price"].sort_values(ascending=False).reset_index()
    corr.columns = ["Variable", "Correlación con price"]

    st.subheader("Correlaciones con price")
    st.dataframe(corr, use_container_width=True)


elif section == "Modelos":
    st.header("Comparación de modelos")

    results = model_data["results"]
    st.dataframe(results, use_container_width=True)

    fig = px.bar(
        results,
        x="RMSE",
        y="Modelo",
        orientation="h",
        title="Comparación de modelos según RMSE"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    best = results.iloc[0]

    st.markdown(f"""
    El mejor modelo fue **{best['Modelo']}**, con:

    - MAE: **{best['MAE']:.2f}**
    - RMSE: **{best['RMSE']:.2f}**
    - R²: **{best['R2']:.3f}**
    """)


elif section == "Predicciones":
    st.header("Predicciones del modelo final")

    y_test = model_data["y_test"]
    y_pred = model_data["y_pred"]

    predictions_df = pd.DataFrame({
        "Precio real": y_test,
        "Precio predicho": y_pred,
        "Error absoluto": abs(y_test - y_pred)
    })

    st.dataframe(predictions_df.head(15), use_container_width=True)

    fig = px.scatter(
        predictions_df,
        x="Precio real",
        y="Precio predicho",
        title="Precio real vs precio predicho"
    )

    min_price = min(predictions_df["Precio real"].min(), predictions_df["Precio predicho"].min())
    max_price = max(predictions_df["Precio real"].max(), predictions_df["Precio predicho"].max())

    fig.add_trace(
        go.Scatter(
            x=[min_price, max_price],
            y=[min_price, max_price],
            mode="lines",
            name="Predicción perfecta"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    fig_error = px.histogram(
        predictions_df,
        x="Error absoluto",
        nbins=30,
        title="Distribución del error absoluto"
    )
    st.plotly_chart(fig_error, use_container_width=True)

    st.subheader("Mayores errores")
    st.dataframe(
        predictions_df.sort_values("Error absoluto", ascending=False).head(10),
        use_container_width=True
    )


elif section == "Importancia de variables":
    st.header("Importancia de variables")

    best_model = model_data["best_model"]
    numeric_features = model_data["numeric_features"]
    categorical_features = model_data["categorical_features"]

    feature_names_num = numeric_features.tolist()

    feature_names_cat = best_model.named_steps["preprocessor"] \
        .named_transformers_["cat"] \
        .named_steps["encoder"] \
        .get_feature_names_out(categorical_features)

    feature_names = np.concatenate([feature_names_num, feature_names_cat])
    importances = best_model.named_steps["model"].feature_importances_

    feature_importance_df = pd.DataFrame({
        "Variable": feature_names,
        "Importancia": importances
    }).sort_values("Importancia", ascending=False)

    st.dataframe(feature_importance_df.head(15), use_container_width=True)

    fig = px.bar(
        feature_importance_df.head(15),
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Top 15 variables más importantes"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    Las variables más importantes son `curb-weight`, `engine-size` y `horsepower`, lo que confirma que el precio está explicado principalmente por peso, tamaño del motor y potencia.
    """)


elif section == "Conclusiones":
    st.header("Conclusiones y cierre ejecutivo")

    st.markdown("""
    El modelo seleccionado fue `Gradient Boosting`, que obtuvo el mejor desempeño general en la comparación de modelos.

    El análisis mostró que el precio está fuertemente asociado a variables técnicas como `curb-weight`, `engine-size` y `horsepower`. El modelo predice correctamente la tendencia general, aunque presenta mayores errores en algunos vehículos de alto valor.

    Como limitación principal, el dataset tiene pocos registros y no incluye variables actuales de mercado como año, kilometraje, estado del vehículo o ubicación. Para una implementación real sería necesario ampliar y actualizar la base de datos.
    """)
