# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier


# def prepare_data(df):
#     """Encode, split, and scale dataset."""
#     df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
#     df.drop(columns=["customerID", "tenure_group"], axis=1, inplace=True)

#     le = LabelEncoder()
#     for col in df.select_dtypes(include="object"):
#         df[col] = le.fit_transform(df[col].astype(str))

#     X = df.drop("Churn", axis=1)
#     y = df["Churn"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     return X_train, X_test, y_train, y_test


# def train_logistic_regression(X_train, y_train):
#     """Train Logistic Regression and return the fitted model."""
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train, y_train)
#     return model


# def train_random_forest(X_train, y_train):
#     """Train Random Forest and return the fitted model."""
#     model = RandomForestClassifier(n_estimators=300, random_state=42)
#     model.fit(X_train, y_train)
#     return model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


def prepare_data(df):
    df = df.copy()
    y = df["Churn_Yes"]
    X = df.drop(columns=["Churn_Yes", "tenure_group"], errors="ignore")

    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder="drop"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor




def train_logistic_regression(X_train, y_train, preprocessor):
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )
    model.fit(X_train, y_train)
    return model



def train_random_forest(X_train, y_train, preprocessor):
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ))
        ]
    )
    model.fit(X_train, y_train)
    return model

