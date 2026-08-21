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

**Lý do:** Bộ siêu tham số này đạt chỉ số `f1_score` cao nhất (0.7149), vượt qua ngưỡng chất lượng tối thiểu 0.65 để đưa vào pipeline triển khai. Đáng chú ý, lần chạy 2 đạt accuracy cao nhất (0.8780) nhưng f1_score lại thấp hơn (0.7109) so với lần chạy 1. Điều này chứng minh accuracy bị chi phối bởi lớp đa số và không phản ánh chính xác khả năng nhận diện lớp thiểu số. Ngoài ra, việc tăng `n_estimators` lên 200 kết hợp `max_depth=5` giúp mô hình Gradient Boosting học sâu hơn các đặc trưng phi tuyến tính phức tạp của nhóm thu nhập cao mà không bị underfitting như lần chạy 3 (F1 chỉ đạt 0.6051).

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult có phân bố lớp mất cân bằng nghiêm trọng với chỉ khoảng 24.8% mẫu thuộc lớp thu nhập cao (>50K) và 75.2% thuộc lớp thu nhập thấp. Nếu một mô hình đơn giản luôn dự đoán nhãn "thu nhập thấp" cho mọi trường hợp, nó vẫn đạt độ chính xác (accuracy) lên tới 75.2% nhưng thực chất F1-score của lớp dương bằng 0 và mô hình hoàn toàn vô dụng trong thực tế.

Do đó, ngưỡng chất lượng của hệ thống bắt buộc phải đặt trên `f1_score` của lớp dương (trung bình điều hòa giữa Precision và Recall của nhóm thu nhập >50K) để đo lường chính xác năng lực phát hiện đối tượng mục tiêu. Tuyệt đối không sử dụng `average="weighted"` hay `average="macro"` vì các phương thức này tính gộp cả lớp đa số, làm chỉ số bị kéo tăng ảo và vô hiệu hóa ý nghĩa của cổng kiểm soát chất lượng (Quality Gate).

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Lỗi thiếu module `pkg_resources` khi mở MLflow UI | `setuptools >= 70` không còn tương thích với MLflow 2.13.0 | Cài đặt và cố định phiên bản `setuptools<70` trong `requirements.txt` |
| `dvc push` bị lỗi `AccessDenied (s3:ListBucket)` | IAM User trên AWS chưa được cấp quyền truy cập S3 Bucket | Gắn quyền `AmazonS3FullAccess` cho User `ai-lab-user` trong AWS IAM |
| Server EC2 lỗi không load được model (`unpickle CyHalfBinomialLoss`) | Phiên bản `scikit-learn` trên EC2 lệch với bản huấn luyện | Cài đặt chính xác `scikit-learn==1.4.2` trong môi trường ảo trên EC2 |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Khi bổ sung thêm 22.361 mẫu từ `train_batch2` (tổng cộng 44.722 mẫu), chỉ số `f1_score` tăng nhẹ từ 0.7149 lên 0.7354 và `accuracy` tăng từ 0.8740 lên 0.8820. Do dữ liệu mới cùng phân phối với dữ liệu ban đầu, việc tăng gấp đôi dữ liệu giúp làm mịn biên quyết định cho nhóm thu nhập cao nhưng không tạo ra biến động quá lớn. Quan trọng nhất, toàn bộ vòng lặp Continuous Training từ lúc cập nhật DVC, đẩy S3 đến tự động huấn luyện và cập nhật API trên EC2 đã diễn ra hoàn toàn tự động và chính xác.

---

## 5. Thách Thức Nâng Cao Đã Hoàn Thành (Bonus Challenges)

* **Bonus 2 - Điều Chỉnh Ngưỡng Quyết Định (Threshold Tuning):** Quét ngưỡng xác suất từ 0.10 đến 0.90 (bước 0.05). Tại ngưỡng mặc định 0.50, F1 đạt 0.7354. Ngưỡng tối ưu xác định được là **0.30** với `f1_score` đạt **0.7537** (tăng +0.0183 F1). Do lớp dương chiếm tỷ lệ nhỏ (~24.8%), việc hạ ngưỡng xuống 0.30 giúp mô hình nhạy hơn trong việc phát hiện nhóm thu nhập cao (tăng Recall) mà không làm suy giảm quá nhiều Precision.
* **Bonus 3 - Báo Cáo Precision / Recall & Confusion Matrix Tự Động:** Hệ thống tự động tạo `outputs/detail.txt` và lưu vào GitHub Actions Artifacts. Kết quả: TN=359, FP=17, FN=42, TP=82 (Class 1: Precision = 0.8283, Recall = 0.6613). *Phân tích chi phí:* Bỏ sót người thu nhập cao (False Negative - Recall thấp) tốn kém hơn việc gán nhầm người thu nhập thấp (False Positive - Precision thấp), vì doanh nghiệp sẽ mất đi các khách hàng mang lại giá trị trọn đời (LTV) cao nhất.
* **Bonus 4 - Cơ Chế Rollback & Safety Guard:** Pipeline tự động tải `report.json` của phiên bản model đang chạy trên S3 để so sánh F1 trước khi cho phép Job `Release` thực thi, đảm bảo không bao giờ triển khai một mô hình bị suy giảm chất lượng so với bản tiền nhiệm.
* **Bonus 5 - Cảnh Báo Lệch Lạc Dữ Liệu (Data Drift Check):** Tự động đo lường tỷ lệ lớp dương trước khi huấn luyện (đạt 24.78% so với mốc tham chiếu 24.80%, độ lệch 0.02% < 5%), log trực tiếp lên MLflow và kích hoạt cảnh báo nếu độ lệch vượt ngưỡng cho phép.


