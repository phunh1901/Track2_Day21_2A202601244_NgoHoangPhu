# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

<!--
HƯỚNG DẪN - đọc rồi XÓA TOÀN BỘ các khối chú thích này sau khi điền xong:

  - Giới hạn: KHÔNG QUÁ 1 TRANG A4, tương đương khoảng 450 - 550 từ nội dung.
  - Chỉ điền vào các chỗ ___ và các ô trong bảng. Không thêm mục mới.
  - Viết bằng câu hoàn chỉnh, không gạch đầu dòng cụt lủn.
  - Kiểm tra độ dài sau khi đã xóa hết chú thích:
        wc -w nop-bai/bao-cao.md
    và xem trước bản in bằng cách mở file trên GitHub rồi Ctrl+P / Cmd+P.
-->

| | |
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

<!-- Nêu 2 - 3 khó khăn thật, mỗi ô một câu ngắn. -->

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| ___ | ___ | ___ |
| ___ | ___ | ___ |
| ___ | ___ | ___ |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

<!-- Lấy số liệu từ bảng ở mục 3.6 của tasks/buoc-3.md. -->

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | ___ | ___ |
| Bước 3 (thêm `train_batch2`) | ___ | ___ |

**Nhận xét:** ___

<!--
Một câu trả lời trung thực kiểu "f1 giảm 0,01 vì dữ liệu mới cùng phân phối, không mang
thêm thông tin mới" được đánh giá cao hơn kết luận sai rằng thêm dữ liệu luôn tốt hơn.
-->

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

<!-- Xóa cả mục 5 nếu không làm bonus. Mỗi bonus tối đa 1 dòng. -->

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub: ___
- [ ] Bonus 2 - Điều chỉnh ngưỡng quyết định: ___
- [ ] Bonus 3 - Báo cáo precision / recall tự động: ___
- [ ] Bonus 4 - Hoàn trả về phiên bản trước: ___
- [ ] Bonus 5 - Cảnh báo lệch lạc dữ liệu: ___
