# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| Thông tin | Chi tiết |
|---|---|
| Họ và tên | Ngô Hoàng Phú |
| MSSV | 2A202601244 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/phunh1901/Track2_Day21_2A202601244_NgoHoangPhu |
| Ngày nộp | 21/8/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |
| 2 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 3 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Cấu hình này có F1 cao nhất (0.7149) và vượt quality gate 0.65. Lần chạy 2 có accuracy cao hơn nhưng F1 thấp hơn, cho thấy accuracy bị lớp đa số chi phối; lần chạy 3 bị underfit.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Chỉ 24.8% mẫu thuộc lớp >50K. Mô hình luôn đoán “thu nhập thấp” vẫn đạt accuracy 75.2% nhưng F1 lớp dương bằng 0. Vì vậy quality gate dùng F1 lớp dương—trung bình điều hòa của precision và recall—thay vì accuracy hoặc F1 weighted/macro bị lớp đa số kéo lên.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow thiếu `pkg_resources` | Lệch phiên bản setuptools | Cố định `setuptools<70` |
| DVC `AccessDenied` | IAM thiếu quyền S3 | Cấp quyền bucket cho CI user |
| EC2 không load model | scikit-learn lệch phiên bản | Cố định `scikit-learn==1.4.2` |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Thêm 22.361 mẫu giúp F1 tăng 0.0205 và accuracy tăng 0.0080. Commit con trỏ DVC đã tự kích hoạt đủ Unit Test → Train → Quality Gate → Release.

---

## 5. Thách Thức Nâng Cao Đã Hoàn Thành (Bonus Challenges)

* **Bonus 2 – Threshold tuning:** Quét 0.10–0.90; ngưỡng 0.30 đạt F1 0.7537, tăng 0.0183 so với ngưỡng 0.50.
* **Bonus 3 – Báo cáo chi tiết:** `outputs/detail.txt` chứa confusion matrix, precision và recall, được lưu bằng Actions Artifact. FN gây bỏ sót khách hàng giá trị cao nên được xem là tốn kém hơn FP.
* **Bonus 4 – Safety guard:** Model mới được upload vào vùng `candidates/<commit>`; pipeline chỉ promote sang `artifacts/current/` khi F1 ≥ 0.65 và không thấp hơn model current. Nếu không đạt, Release bị chặn và model đang chạy không bị ghi đè.
* **Bonus 5 – Drift:** Tỷ lệ dương 24.78% so với mốc 24.80%; pipeline ghi tỷ lệ vào MLflow/report và cảnh báo khi lệch quá 5 điểm phần trăm.


