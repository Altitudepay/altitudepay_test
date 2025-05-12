# retrain_model.py

import pandas as pd
import pickle
import os
import json
import shutil
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from datetime import datetime

def run_retraining_pipeline(csv_path="transaction.csv"):
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")

        expected_order = ['bin', 'merchant', 'card_type', 'status', 'is_3d', 'currency', 'processor_name', 'project_name']
        df = df[expected_order]

        # Encoding helper
        def customEncoder(column_name_to_encode, data):
            mapping_file = f"{column_name_to_encode}_mapping.json"
            if column_name_to_encode in data.columns:
                most_common = data[column_name_to_encode].mode()[0]
                data[column_name_to_encode] = data[column_name_to_encode].fillna(most_common)

                if os.path.exists(mapping_file):
                    with open(mapping_file, 'r') as f:
                        mapping = json.load(f)
                else:
                    mapping = {}

                mapping = {str(k): v for k, v in mapping.items()}
                current_max = max(mapping.values(), default=-1)
                new_keys_added = False

                for val in sorted(data[column_name_to_encode].unique()):
                    key = str(val)
                    if key not in mapping:
                        current_max += 1
                        mapping[key] = current_max
                        new_keys_added = True

                data[f"{column_name_to_encode}_encoded"] = data[column_name_to_encode].apply(lambda x: mapping[str(x)])

                if new_keys_added or not os.path.exists(mapping_file):
                    with open(mapping_file, 'w') as f:
                        json.dump(mapping, f)

                data.drop(column_name_to_encode, axis=1, inplace=True)

        for col in ['merchant', 'card_type', 'status', 'is_3d', 'currency', 'processor_name', 'project_name']:
            customEncoder(col, df)

        df['success_flag'] = df['status_encoded'].apply(lambda x: 1 if x == 0 else 0)
        df['bin_prefix'] = df['bin'] // 1000
        df['bin_suffix'] = df['bin'] % 1000

        bin_stats = df.groupby('bin')['success_flag'].mean().reset_index(name='bin_success_rate_check')
        filtered_bins = bin_stats[(bin_stats['bin_success_rate_check'] > 0.10)]
        df = df[df['bin'].isin(filtered_bins['bin'])]

        valid_processors = df['processor_name_encoded'].value_counts()
        valid_processors = valid_processors[valid_processors >= 10].index
        df = df[df['processor_name_encoded'].isin(valid_processors)]

        bin_tx = df.groupby('bin').size().reset_index(name='bin_tx_count')
        bin_success = df.groupby('bin')['success_flag'].mean().reset_index(name='bin_success_rate')
        proc_success = df.groupby('processor_name_encoded')['success_flag'].mean().reset_index(name='processor_success_rate')

        bin_proc_group = df.groupby(['bin', 'processor_name_encoded']).agg(
            bin_processor_tx_count=('success_flag', 'count'),
            bin_processor_success_count=('success_flag', 'sum')
        ).reset_index()
        bin_proc_group['bin_processor_success_rate'] = (
            bin_proc_group['bin_processor_success_count'] / bin_proc_group['bin_processor_tx_count']
        )

        df = df.merge(bin_tx, on='bin', how='left')
        df = df.merge(bin_success, on='bin', how='left')
        df = df.merge(proc_success, on='processor_name_encoded', how='left')
        df = df.merge(bin_proc_group, on=['bin', 'processor_name_encoded'], how='left')
        df = df.fillna(0)


        features = [
            'bin', 'bin_prefix', 'bin_suffix', 'is_3d_encoded',
            'bin_tx_count', 'bin_success_rate', 'processor_success_rate',
            'bin_processor_tx_count', 'bin_processor_success_count', 'bin_processor_success_rate'
        ]
        X = df[features]
        y = df['success_flag']

        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        with open("models/model_latest.pkl", "rb") as f:
            prev_model = pickle.load(f)
        booster = prev_model.get_booster()

        y_holdout_pred_prev = prev_model.predict(X_holdout)
        old_acc = accuracy_score(y_holdout, y_holdout_pred_prev)

        scale_ratio = (len(y_train) - sum(y_train)) / sum(y_train)
        model = XGBClassifier(
            scale_pos_weight=scale_ratio,
            eval_metric='logloss',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train, xgb_model=booster)

        y_pred = model.predict(X_holdout)
        y_prob = model.predict_proba(X_holdout)[:, 1]
        new_model_accuracy = accuracy_score(y_holdout, y_pred)

        msg = f"Old Accuracy: {old_acc * 100:.2f}%, New Accuracy: {new_model_accuracy * 100:.2f}%\n"
        # msg += f"AUC: {roc_auc_score(y_holdout, y_prob):.4f}\n"
        # msg += classification_report(y_holdout, y_pred, zero_division=0)

        if new_model_accuracy >= 0.87:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy("models/model_latest.pkl", f"models/model_backup_{timestamp}.pkl")
            shutil.copy("stats/processor_success_stats_latest.pkl", f"stats/stats_backup_{timestamp}.pkl")

            with open("models/model_latest.pkl", "wb") as f:
                pickle.dump(model, f)

            with open("stats/processor_success_stats_latest.pkl", "rb") as f:
                old_stats = pickle.load(f)
            combined_processors = list(set(old_stats["all_processors"]).union(set(df['processor_name_encoded'].unique())))

            updated_stats = {
                "bin_tx": bin_tx.set_index("bin").to_dict(orient="index"),
                "bin_success": bin_success.set_index("bin").to_dict(orient="index"),
                "proc_success": proc_success.set_index("processor_name_encoded").to_dict(orient="index"),
                "bin_proc_stats": bin_proc_group.set_index(['bin', 'processor_name_encoded']).to_dict(orient="index"),
                "all_processors": combined_processors
            }
            with open("stats/processor_success_stats_latest.pkl", "wb") as f:
                pickle.dump(updated_stats, f)

            msg += "\n[SUCCESS] Model updated (accuracy >= 87%)"
        else:
            msg += f"\n[WARNING] Model NOT updated (accuracy {new_model_accuracy * 100:.2f}%) - below threshold"

        return msg, old_acc, new_model_accuracy

    except Exception as e:
        return f"[ERROR] Retraining failed: {str(e)}", None, None
