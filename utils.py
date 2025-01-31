import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (classification_report, confusion_matrix,
                             matthews_corrcoef, mean_squared_error, r2_score,
                             roc_auc_score, roc_curve)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   OneHotEncoder, RobustScaler, StandardScaler,
                                   scale)
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor

SEED = 42
ALPHA = 0.05


def apply_bonferroni_correction(p_values, num_tests, alpha=ALPHA):
    """Apply Bonferroni correction to p-values."""
    corrected_p_values = p_values * num_tests
    significant_features = corrected_p_values < alpha
    return corrected_p_values, significant_features


def encode_categorical_with_reference(df, column, reference):
    """
    Encode categorical variables with reference encoding.
    """
    categories = [reference] + [x for x in df[column].unique() if x != reference]
    return pd.Categorical(df[column], categories=categories).codes


def prepare_data_with_encoding(df):
    """
    Prepare data with categorical encoding.
    """
    df_encoded = df.copy()
    df_encoded['gender'] = encode_categorical_with_reference(df_encoded, 'gender',
                                                             'Man')
    df_encoded['race'] = encode_categorical_with_reference(df_encoded, 'race', 'White')
    df_encoded['ethnicity'] = encode_categorical_with_reference(df_encoded, 'ethnicity',
                                                                'Others')
    return df_encoded


def assemble_ml_dataset(emr_df, drug_effects_df, label_df, cv_splits_df,
                        embeddings_df=None):
    """
    assemble dataset for machine learning with CV splits and optional embeddings.

    args:
        emr_df: dataframe with demographic and comorbidity data
        drug_effects_df: dataframe with drug effect columns
        label_df: dataframe with target labels
        cv_splits_df: dataframe with CV split information
        embeddings_df: optional dataframe with drug embeddings (person_id and embed_* columns)
    """
    # start with EMR data
    ml_df = emr_df.copy()

    # apply encoding to categorical variables
    ml_df = prepare_data_with_encoding(ml_df)

    # merge with drug effects
    ml_df = pd.merge(ml_df, drug_effects_df, on='person_id', how='inner')

    # merge with embeddings if provided
    if embeddings_df is not None:
        ml_df = pd.merge(ml_df, embeddings_df, on='person_id', how='inner')

    # merge with labels
    ml_df = pd.merge(ml_df, label_df, on='person_id', how='inner')

    # add split column initialized as 'train'
    ml_df['split'] = 'train'

    # mark test splits for each fold
    for _, row in cv_splits_df.iterrows():
        fold = row['fold']
        test_ids = row['person_id']

        # create fold-specific split column
        split_col = f'split_fold{fold}'
        ml_df[split_col] = 'train'
        ml_df.loc[ml_df['person_id'].isin(test_ids), split_col] = 'test'

    return ml_df


def train_and_evaluate_cv_models(ml_df, unique_comorbidities, model_name, target_col,
                                 with_drug_effect=True, with_llm_cot=False,
                                 with_llm_no_cot=False,
                                 with_embeddings=False, embedding_prefix='embed_',
                                 robust_scaler=False):
    """
    train and evaluate models using cross-validation with optional drug embeddings.

    args:
        ml_df: main dataframe containing all features
        unique_comorbidities: list of comorbidity columns
        model_name: name of the model to use
        target_col: target column name
        with_drug_effect: whether to include original drug effect features
        with_llm_cot: whether to include LLM chain-of-thought probability
        with_llm_no_cot: whether to include LLM no chain-of-thought probability
        with_embeddings: whether to include drug embeddings
        embedding_prefix: prefix of embedding columns (e.g., 'embed_' for embed_0, embed_1, etc.)
        robust_scaler: whether to use robust scaler
    """
    # base features
    features = ['gender', 'race', 'ethnicity', 'age'] + list(unique_comorbidities)

    # add LLM features if specified
    if with_llm_cot:
        features.append('cot_combined')
    if with_llm_no_cot:
        features.append('no_cot_combined')

    # add embedding features if specified
    if with_embeddings:
        embedding_cols = [col for col in ml_df.columns if
                          col.startswith(embedding_prefix)]
        features.extend(embedding_cols)

    # results storage
    cv_results = {}

    # process each fold
    for fold in tqdm(range(1, 6), desc="Processing CV folds"):
        print(f"\nProcessing Fold {fold}")
        split_col = f'split_fold{fold}'

        # get fold-specific data
        train_df = ml_df[ml_df[split_col] == 'train']
        test_df = ml_df[ml_df[split_col] == 'test']

        # get fold-specific features
        fold_features = features.copy()
        if with_drug_effect:
            drug_cols = [col for col in ml_df.columns if f'_fold{fold}_' in col]
            fold_features.extend(drug_cols)

        # create temporary DataFrame for this fold
        fold_df = pd.concat([train_df, test_df])
        fold_df['split'] = fold_df[split_col]

        # train and evaluate using only fold-specific features
        y_test, y_pred, y_proba = train_and_evaluate_models(
            df=fold_df,
            model_name=model_name,
            target_col=target_col,
            features=fold_features,
            robust_scaler=robust_scaler
        )

        # store results
        cv_results[fold] = {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'features': fold_features
        }

    return cv_results


def train_and_evaluate_models(df, model_name, target_col,
                              features, robust_scaler=False):
    """
    Train and evaluate a single model.
    """
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    if robust_scaler:
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    # calculate the imbalance ratio
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    if model_name == 'RandomForest':
        classifier = RandomForestClassifier(
            n_estimators=100,  # reduced from 200 as feature space is smaller
            max_depth=10,  # add explicit depth control
            min_samples_leaf=20,  # reduced as we have <= 50 features
            class_weight='balanced',
            random_state=SEED
        )
    elif model_name == 'SVM':
        classifier = SVC(
            kernel='rbf',  # changed to rbf for better performance
            C=1.0,  # add explicit regularization
            probability=True,
            class_weight='balanced',
            random_state=SEED
        )
    elif model_name == 'LogisticRegression':
        classifier = None  # placeholder, we'll handle logistic regression separately
    elif model_name == "XGBoost":
        classifier = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,  # can use slightly higher learning rate
            max_depth=5,  # good depth for <50 features
            min_child_weight=3,  # add this for better generalization
            scale_pos_weight=imbalance_ratio,
            objective='binary:logistic',
            use_label_encoder=False,
            random_state=SEED
        )
    elif model_name == "MLP":
        classifier = MLPClassifier(
            hidden_layer_sizes=(64, 32),  # reduced architecture
            activation='relu',
            max_iter=10000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=SEED
        )
    else:
        raise ValueError(f'Illegal model {model_name}')

    # preprocessing pipeline for numerical data
    numerical_pipeline = Pipeline([('scaler', scaler)])

    # scale it
    X_train_processed = numerical_pipeline.fit_transform(X_train)
    X_test_processed = numerical_pipeline.transform(X_test)

    # convert processed features back to DataFrame
    X_train_processed = pd.DataFrame(X_train_processed, columns=features)
    X_test_processed = pd.DataFrame(X_test_processed, columns=features)

    if model_name == 'LogisticRegression':
        # add intercept term
        X_train_with_const = sm.add_constant(X_train_processed)
        X_test_with_const = sm.add_constant(X_test_processed)

        model = sm.Logit(y_train, X_train_with_const).fit(disp=False)
        y_proba_test = model.predict(X_test_with_const)

        # determine optimal threshold
        fpr, tpr, thresholds = roc_curve(y_test, y_proba_test)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(f'Optimal classification threshold: {optimal_threshold:.4f}')

        y_pred_test = y_proba_test >= optimal_threshold

        # print metrics
        print(f'Model: {model_name}')
        print(classification_report(y_test, y_pred_test))
        auc = roc_auc_score(y_test, y_proba_test)
        print(f'ROC AUC Score: {auc:.5g}')
        mcc = matthews_corrcoef(y_test, y_pred_test)
        print(f'Matthews Correlation Coefficient: {mcc:.5g}')

        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred_test))

        # get coefficients and p-values
        coef = model.params[1:]
        p_values = model.pvalues[1:]

        corrected_p_values, significant_features = apply_bonferroni_correction(
            p_values.values, len(coef), alpha=ALPHA)

        print('Feature Coefficients and p-values:')
        for feature, coef_value, p_value, corrected_p_value, significant in zip(
                coef.index, coef.values, p_values.values, corrected_p_values,
                significant_features
        ):
            if corrected_p_value <= 0.05:
                print(
                    f'\t✅ {feature}: Coef = {coef_value:.5g}, p-value = {p_value:.5g}, '
                    f'Bonferroni-corrected p-value = {corrected_p_value:.5g}')
            else:
                print(
                    f'\t❌ {feature}: Coef = {coef_value:.5g}, p-value = {p_value:.5g}, '
                    f'Bonferroni-corrected p-value = {corrected_p_value:.5g}')
    else:
        pipeline = Pipeline([('classifier', classifier)])
        pipeline.fit(X_train_processed, y_train)

        if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
            y_proba_test = pipeline.predict_proba(X_test_processed)[:, 1]
        else:
            y_proba_test = pipeline.decision_function(X_test_processed)
            y_proba_test = (y_proba_test - y_proba_test.min()) / (
                        y_proba_test.max() - y_proba_test.min())

        fpr, tpr, thresholds = roc_curve(y_test, y_proba_test)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(f'Optimal classification threshold: {optimal_threshold:.4f}')

        y_pred_test = y_proba_test >= optimal_threshold

        print(f'Model: {model_name}')
        print(classification_report(y_test, y_pred_test))
        auc = roc_auc_score(y_test, y_proba_test)
        print(f'ROC AUC Score: {auc:.5g}')
        mcc = matthews_corrcoef(y_test, y_pred_test)
        print(f'Matthews Correlation Coefficient: {mcc:.5g}')

        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred_test))

        if model_name == 'SVM':
            print('Feature Coefficients:')
            coef = pipeline.named_steps['classifier'].coef_[0]
            for feature, coef_value in zip(features, coef):
                print(f'\t{feature}: Coef = {coef_value:.5g}')
        elif model_name in ['RandomForest', 'XGBoost']:
            print('Feature Importances:')
            importances = pipeline.named_steps['classifier'].feature_importances_
            for feature, importance in zip(features, importances):
                print(f'\t{feature}: {importance:.5g}')

    return y_test, y_pred_test, y_proba_test


def analyze_cv_results(cv_results):
    """
    Analyze results across all CV folds.
    """
    print("\nOverall Cross-validation Results:")

    metrics = {
        'auc': [],
        'mcc': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for fold, results in cv_results.items():
        y_test = results['y_test']
        y_pred = results['y_pred']
        y_proba = results['y_proba']

        metrics['auc'].append(roc_auc_score(y_test, y_proba))
        metrics['mcc'].append(matthews_corrcoef(y_test, y_pred))

        fold_report = classification_report(y_test, y_pred, output_dict=True)
        metrics['precision'].append(fold_report['1']['precision'])
        metrics['recall'].append(fold_report['1']['recall'])
        metrics['f1'].append(fold_report['1']['f1-score'])

    for metric, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric.upper()}: {mean_val:.3f} ± {std_val:.3f}")


def combine_drug_probabilities(drug_probs, baseline_prob=0.116):
    """
    Combine multiple drug-T2D probabilities using log odds ratios.
    Handle both very high (near 1) and very low (near 0) probabilities.

    Args:
        drug_probs (list): List of probabilities p(T2D|drug) for each drug
        baseline_prob (float): Baseline T2D prevalence in population (default: 0.116)

    Returns:
        float: Combined log odds ratio score
    """
    # handle extreme probabilities
    cleaned_probs = []
    for p in drug_probs:
        if p >= 1 - 1e-10:  # handle values too close to 1
            p = 1 - 1e-10
        elif p <= 1e-10:  # handle values too close to 0
            p = 1e-10
        cleaned_probs.append(p)

    try:
        # convert each probability to odds ratio relative to baseline
        odds_ratios = [(p / (1 - p)) / (baseline_prob / (1 - baseline_prob)) for p in
                       cleaned_probs]

        # convert to log odds, with handling for very small odds ratios
        log_odds = []
        for or_ in odds_ratios:
            if or_ <= 1e-10:  # if odds ratio is extremely small
                log_odds.append(-10)  # set a floor for extremely negative log odds
            else:
                log_odds.append(np.log2(or_))

        # combine by taking the average
        combined_log_odds = sum(log_odds) / len(log_odds)

        return combined_log_odds

    except Exception as e:
        print(f"Error processing probabilities: {cleaned_probs}")
        print(f"Error: {str(e)}")
        return np.nan


def prepare_drug_probabilities(df_drugs, df_assoc, prob_col='probability'):
    """
    Prepare and clean drug probability mapping.
    """
    # merge drug names with their probabilities
    merged_df = pd.merge(
        df_drugs[['standard_concept_name']],
        df_assoc[[prob_col]],
        left_index=True,
        right_index=True
    )

    # clean the data
    merged_df = merged_df.dropna()
    merged_df = merged_df[merged_df[prob_col] <= 1.0]

    # create a dictionary for faster lookup
    drug_prob_dict = dict(zip(merged_df['standard_concept_name'], merged_df[prob_col]))

    return drug_prob_dict


def calculate_combined_probabilities(ml_df, drug_prob_dict,
                                     target_col='combined_log_odds',
                                     baseline_prob=0.116, imputation_method='mean'):
    """
    Calculate combined log odds ratios for each patient's drug combination.

    Args:
        ml_df: Machine learning dataframe with patient drug lists
        drug_prob_dict: Dictionary mapping drugs to their probabilities
        target_col: Name of column to store combined log odds ratios
        baseline_prob: Baseline probability for T2D in population
        imputation_method: Method to impute missing values ('mean', 'zero', or 'median')

    Returns:
        Modified dataframe with combined log odds ratios and imputed missing values
    """
    # initialize the target column
    ml_df[target_col] = np.nan  # ensure missing values are properly marked

    # iterate through each patient's drug records
    for idx, row in tqdm(ml_df.iterrows(), total=len(ml_df),
                         desc=f"Calculating {target_col}"):
        drug_str = row['standard_concept_name']
        if not drug_str:
            continue  # Leave as NaN

        # collect probabilities for all drugs the patient is taking
        probs = [drug_prob_dict[drug] for drug in drug_str.split('|') if
                 drug in drug_prob_dict]

        # calculate combined log odds if there are valid probabilities
        if probs:
            ml_df.at[idx, target_col] = combine_drug_probabilities(probs, baseline_prob)

    # count missing values
    n_missing = ml_df[target_col].isna().sum()

    # handle missing values based on chosen imputation method
    if imputation_method == 'mean':
        column_mean = ml_df[target_col].mean()
        ml_df[target_col] = ml_df[target_col].fillna(column_mean)
        if n_missing > 0:
            print(
                f"Imputed {n_missing} missing values with column mean: {column_mean:.4f}")

    elif imputation_method == 'zero':
        ml_df[target_col] = ml_df[target_col].fillna(0)
        if n_missing > 0:
            print(f"Imputed {n_missing} missing values with 0.")

    elif imputation_method == 'median':
        column_median = ml_df[target_col].median()
        ml_df[target_col] = ml_df[target_col].fillna(column_median)
        if n_missing > 0:
            print(
                f"Imputed {n_missing} missing values with column median: {column_median:.4f}")

    else:
        raise ValueError(
            "Invalid imputation method. Choose 'mean', 'zero', or 'median'.")

    return ml_df


def process_drug_combinations(ml_df, df_drugs, df_assoc, prob_col='probability',
                              target_col='combined_log_odds', baseline_prob=0.116,
                              imputation_method='mean'):
    """
    Process drug combinations and calculate combined log odds ratios.
    """
    # work on a copy to prevent modifications to the original ml_df
    ml_df = ml_df.copy()

    # prepare drug probability dictionary
    drug_prob_dict = prepare_drug_probabilities(df_drugs, df_assoc, prob_col)

    # calculate combined probabilities and impute missing values
    ml_df = calculate_combined_probabilities(ml_df, drug_prob_dict, target_col,
                                             baseline_prob, imputation_method)

    return ml_df

