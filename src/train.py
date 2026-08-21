import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65
REFERENCE_POSITIVE_RATIO = 0.248  # 24.8% theo dataset Adult goc


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.
    Tich hop Bonus 2 (Threshold tuning), Bonus 3 (Detailed report), Bonus 5 (Data drift check).

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout tai nguong 0.5.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # -------------------------------------------------------------
    # BONUS 5: Cảnh Báo Lệch Lạc Dữ Liệu (Data Drift / Class Distribution)
    # -------------------------------------------------------------
    pos_count = int((y_train == 1).sum())
    total_count = len(y_train)
    pos_ratio = float(pos_count / total_count) if total_count > 0 else 0.0
    drift_diff = abs(pos_ratio - REFERENCE_POSITIVE_RATIO)
    drift_warning = bool(drift_diff > 0.05)

    print(f"--- DATA DISTRIBUTION CHECK ---")
    print(f"Total training samples: {total_count}")
    print(f"Positive class ratio (target=1): {pos_ratio:.4f} ({pos_ratio * 100:.2f}%) | Ref: {REFERENCE_POSITIVE_RATIO * 100:.2f}%")
    if drift_warning:
        print(f"[WARNING] ⚠️ Data Drift Alert: Ty le lop duong lech {drift_diff * 100:.2f}% (> 5%) so voi ty le tham chieu!")
    else:
        print("[INFO] Data distribution is within normal range (diff <= 5%).")

    # Thiet lap tracking URI neu chua set bien moi truong
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("positive_class_ratio", pos_ratio)

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Danh gia mac dinh tai nguong 0.5
        preds_default = model.predict(X_eval)
        f1_default = float(f1_score(y_eval, preds_default))
        acc_default = float(accuracy_score(y_eval, preds_default))

        mlflow.log_metric("f1_score", f1_default)
        mlflow.log_metric("accuracy", acc_default)

        # -------------------------------------------------------------
        # BONUS 2: Dieu Chinh Nguong Quyet Dinh (Threshold Tuning)
        # -------------------------------------------------------------
        # Du doan xac suat cua lop duong (target = 1)
        probs_eval = model.predict_proba(X_eval)[:, 1]
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]
        best_threshold = 0.50
        best_f1 = f1_default
        threshold_results = {}

        for t in thresholds:
            preds_t = (probs_eval >= t).astype(int)
            # Neu tap du doan khong co lop duong nao thi f1 = 0
            f1_t = float(f1_score(y_eval, preds_t, zero_division=0))
            threshold_results[f"{t:.2f}"] = round(f1_t, 4)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = t

        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_f1", best_f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"--- EVALUATION RESULTS ---")
        print(f"Default (threshold=0.50): F1: {f1_default:.4f} | Accuracy: {acc_default:.4f}")
        print(f"Optimal (threshold={best_threshold:.2f}): F1: {best_f1:.4f} (Delta F1: {best_f1 - f1_default:+.4f})")

        # -------------------------------------------------------------
        # BONUS 3: Báo Cáo Precision / Recall Tự Động & Confusion Matrix
        # -------------------------------------------------------------
        cm = confusion_matrix(y_eval, preds_default)
        # sklearn confusion matrix: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        p0 = float(precision_score(y_eval, preds_default, pos_label=0, zero_division=0))
        r0 = float(recall_score(y_eval, preds_default, pos_label=0, zero_division=0))
        p1 = float(precision_score(y_eval, preds_default, pos_label=1, zero_division=0))
        r1 = float(recall_score(y_eval, preds_default, pos_label=1, zero_division=0))

        clf_report = classification_report(y_eval, preds_default, target_names=["<=50K (0)", ">50K (1)"], digits=4)

        os.makedirs("outputs", exist_ok=True)
        detail_path = "outputs/detail.txt"
        with open(detail_path, "w", encoding="utf-8") as f_det:
            f_det.write("======================================================================\n")
            f_det.write("            AI IN ACTION - LAB 21: DETAILED MODEL REPORT              \n")
            f_det.write("======================================================================\n\n")
            f_det.write(f"1. OVERALL METRICS (Default Threshold = 0.50):\n")
            f_det.write(f"   - F1-Score (Positive Class >50K): {f1_default:.4f}\n")
            f_det.write(f"   - Accuracy:                       {acc_default:.4f}\n\n")
            f_det.write(f"2. CONFUSION MATRIX:\n")
            f_det.write(f"   - True Negatives  (TN - Thu nhap thap doan dung): {tn}\n")
            f_det.write(f"   - False Positives (FP - Thu nhap thap doan nham): {fp}\n")
            f_det.write(f"   - False Negatives (FN - Thu nhap cao bo sot)   : {fn}\n")
            f_det.write(f"   - True Positives  (TP - Thu nhap cao doan dung): {tp}\n\n")
            f_det.write(f"3. CLASS-WISE PRECISION & RECALL:\n")
            f_det.write(f"   - Class 0 (<=50K): Precision = {p0:.4f} | Recall = {r0:.4f}\n")
            f_det.write(f"   - Class 1 (>50K) : Precision = {p1:.4f} | Recall = {r1:.4f}\n\n")
            f_det.write(f"4. SKLEARN CLASSIFICATION REPORT:\n")
            f_det.write(f"{clf_report}\n\n")
            f_det.write(f"5. THRESHOLD TUNING ANALYSIS (Bonus 2):\n")
            f_det.write(f"   - Default Threshold: 0.50 -> F1 = {f1_default:.4f}\n")
            f_det.write(f"   - Best Threshold:    {best_threshold:.2f} -> F1 = {best_f1:.4f}\n")
            f_det.write(f"   - Threshold Scan Results: {json.dumps(threshold_results, indent=2)}\n\n")
            f_det.write(f"6. DATA DISTRIBUTION & DRIFT CHECK (Bonus 5):\n")
            f_det.write(f"   - Total Samples:        {total_count}\n")
            f_det.write(f"   - Positive Ratio (Obs): {pos_ratio:.4f} ({pos_ratio * 100:.2f}%)\n")
            f_det.write(f"   - Positive Ratio (Ref): {REFERENCE_POSITIVE_RATIO:.4f} ({REFERENCE_POSITIVE_RATIO * 100:.2f}%)\n")
            f_det.write(f"   - Drift Alert Status:   {'WARNING (Drift > 5%)' if drift_warning else 'PASSED (Normal)'}\n")
            f_det.write("======================================================================\n")

        report_data = {
            "f1_score": round(f1_default, 4),
            "accuracy": round(acc_default, 4),
            "best_threshold": round(best_threshold, 2),
            "best_f1": round(best_f1, 4),
            "positive_class_ratio": round(pos_ratio, 4),
            "drift_warning": drift_warning,
        }
        with open("outputs/report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1_default



if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)

