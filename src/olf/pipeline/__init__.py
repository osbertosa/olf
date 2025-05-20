preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), Categorical_feature),
    ],
    remainder='passthrough',

)
model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),        
    ('smote', SMOTE(random_state=42)),     
    ('classifier', CatBoostClassifier(random_state=42, verbose=0, iterations=100, learning_rate=0.1, depth=6, l2_leaf_reg=3, eval_metric='F1', devices='0:1')),
])

model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),        
    ('smote', SMOTE(random_state=42)),     
    ('classifier', LogisticRegression(random_state=42, max_iter=1000)),
])


model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),        
    ('smote', SMOTE(random_state=42)),     
    ('classifier', DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=2, min_samples_leaf=1)),
])


model= ImbPipeline(steps=[
    ('preprocessor', preprocessor),        
    ('smote', SMOTE(random_state=42)),     
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1)),
])
