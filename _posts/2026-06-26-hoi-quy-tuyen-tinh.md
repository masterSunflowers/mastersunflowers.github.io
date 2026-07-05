---
title: Hồi quy tuyến tính
author: Thieu Luu
date: '2026-06-26'
category: LLM Foundations
layout: post
---

# Tổng quan

**Hồi quy tuyến tính (linear regression)** là mô hình học có giám sát (supervised learning) cơ bản nhất cho bài toán **hồi quy** — dự đoán một giá trị liên tục $y \in \mathbb{R}$ từ một vector đặc trưng (features) $\mathbf{x} \in \mathbb{R}^d$.

Giả định cốt lõi: quan hệ giữa đầu vào và đầu ra là **tuyến tính** theo các tham số. Mô hình học một tổ hợp tuyến tính có trọng số của các đặc trưng:

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d = \mathbf{w}^\top \mathbf{x} + b$$

- $w_1, \dots, w_d$ là **trọng số (weights / coefficients)** — đo mức độ ảnh hưởng của từng đặc trưng.
- $b$ (hay $w_0$) là **bias / intercept** — giá trị dự đoán khi mọi đặc trưng bằng 0.

Mẹo gọn: thêm cột hằng số 1 vào $\mathbf{x}$ để gộp bias vào vector trọng số, khi đó $\hat{y} = \mathbf{w}^\top \mathbf{x}$.

Dù đơn giản, hồi quy tuyến tính là nền tảng để hiểu các mô hình phức tạp hơn: hồi quy logistic chỉ là linear regression bọc thêm hàm sigmoid, và mỗi neuron trong mạng nơ-ron đều là một phép biến đổi tuyến tính $\mathbf{w}^\top \mathbf{x} + b$ rồi qua hàm kích hoạt (xem LLM architecture).

# Hàm mất mát (Loss function)

Để "huấn luyện", ta cần đo độ sai lệch giữa dự đoán $\hat{y}$ và giá trị thật $y$. Hàm mất mát chuẩn là **sai số bình phương trung bình (Mean Squared Error — MSE)**:

$$J(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \mathbf{w}^\top \mathbf{x}_i \right)^2$$

- Bình phương sai số → phạt nặng các lỗi lớn (nhạy với **outlier**).
- Hàm là **lồi (convex)** theo $\mathbf{w}$ → có nghiệm cực tiểu toàn cục duy nhất, không kẹt ở cực tiểu cục bộ.
- Nếu muốn ít nhạy với outlier hơn, có thể dùng **MAE** (trung bình trị tuyệt đối sai số) hoặc **Huber loss** (lai giữa MSE và MAE).

# Cách tìm nghiệm

## 1. Nghiệm đóng — Ordinary Least Squares (OLS)

Vì $J(\mathbf{w})$ lồi, đặt đạo hàm bằng 0 ta có **phương trình chuẩn (normal equation)** với nghiệm tường minh:

$$\mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

trong đó $\mathbf{X} \in \mathbb{R}^{n \times d}$ là ma trận thiết kế (mỗi hàng là một mẫu).

- **Ưu:** cho ngay nghiệm chính xác, không cần chọn learning rate hay lặp.
- **Nhược:** phải nghịch đảo ma trận $\mathbf{X}^\top \mathbf{X}$ ($d \times d$) — chi phí $O(d^3)$, không khả thi khi số đặc trưng $d$ rất lớn. Ngoài ra nếu các đặc trưng **đa cộng tuyến (multicollinearity)** thì $\mathbf{X}^\top \mathbf{X}$ gần suy biến, nghịch đảo không ổn định.

## 2. Gradient Descent

Khi dữ liệu lớn, ta tối ưu lặp bằng **gradient descent** — đi ngược hướng gradient của hàm mất mát:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \, \nabla_{\mathbf{w}} J(\mathbf{w}), \qquad \nabla_{\mathbf{w}} J = -\frac{2}{n} \mathbf{X}^\top (\mathbf{y} - \mathbf{X}\mathbf{w})$$

với $\eta$ là **learning rate**. Các biến thể: Batch GD (dùng toàn bộ dữ liệu), **SGD** (một mẫu mỗi bước), Mini-batch GD (cân bằng — phổ biến nhất trong deep learning). Đây chính là thuật toán tối ưu nền tảng dùng cho cả việc huấn luyện mạng nơ-ron và LLM.

# Các giả định của hồi quy tuyến tính

Để mô hình tuyến tính cho kết quả đáng tin (đặc biệt khi suy luận thống kê), thường giả định:

1. **Tuyến tính:** quan hệ giữa $\mathbf{x}$ và $y$ thực sự tuyến tính theo tham số.
2. **Độc lập:** các phần dư (residuals) độc lập với nhau.
3. **Phương sai đồng nhất (homoscedasticity):** phần dư có phương sai không đổi.
4. **Phân phối chuẩn của phần dư:** sai số tuân theo phân phối chuẩn (quan trọng cho khoảng tin cậy/kiểm định).
5. **Ít/không đa cộng tuyến:** các đặc trưng không tương quan tuyến tính mạnh với nhau.

# Regularization — chống quá khớp (overfitting)

Khi nhiều đặc trưng hoặc dữ liệu ít, mô hình dễ overfit. Ta thêm **số hạng phạt (penalty)** vào hàm mất mát để giữ trọng số nhỏ:

- **Ridge (L2):** $J(\mathbf{w}) + \lambda \sum_j w_j^2$ — co (shrink) trọng số về gần 0, ổn định khi đa cộng tuyến.
- **Lasso (L1):** $J(\mathbf{w}) + \lambda \sum_j \mid w_j\mid $ — đẩy một số trọng số về đúng 0 → **chọn lọc đặc trưng (feature selection)**.
- **Elastic Net:** kết hợp cả L1 và L2.

$\lambda$ là siêu tham số kiểm soát độ mạnh của regularization: $\lambda$ lớn → mô hình đơn giản hơn (bias cao, variance thấp).

# Đánh giá mô hình

| Chỉ số | Ý nghĩa |
|---|---|
| **MSE / RMSE** | Sai số bình phương trung bình (RMSE = căn của MSE, cùng đơn vị với $y$). Càng nhỏ càng tốt. |
| **MAE** | Trung bình trị tuyệt đối sai số — ít nhạy outlier hơn RMSE. |
| **$R^2$ (hệ số xác định)** | Tỷ lệ phương sai của $y$ được mô hình giải thích, $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$. Nằm trong $(-\infty, 1]$; càng gần 1 càng tốt. |
| **Adjusted $R^2$** | $R^2$ có hiệu chỉnh theo số đặc trưng — tránh "ảo giác" tăng khi thêm biến vô ích. |

# Ưu / nhược điểm

**Ưu điểm:**
- Đơn giản, nhanh, dễ huấn luyện kể cả trên dữ liệu lớn.
- **Dễ diễn giải (interpretable):** mỗi trọng số cho biết trực tiếp ảnh hưởng của một đặc trưng → quan trọng trong các lĩnh vực cần giải thích (y tế, tài chính).
- Là baseline tốt để so sánh trước khi thử mô hình phức tạp.

**Nhược điểm:**
- Chỉ nắm bắt quan hệ **tuyến tính** — thua khi dữ liệu phi tuyến (có thể giảm nhẹ bằng polynomial features hoặc feature engineering).
- Nhạy với outlier (do MSE bình phương lỗi).
- Giả định khá chặt; vi phạm giả định làm suy luận thống kê sai lệch.

# Liên hệ

- **Hồi quy đa thức (polynomial regression):** vẫn là linear regression nhưng trên các đặc trưng mở rộng $x, x^2, x^3, \dots$ → mô hình hóa phi tuyến mà vẫn tuyến tính theo tham số.
- **Hồi quy logistic:** áp hàm sigmoid lên đầu ra tuyến tính để giải bài toán phân loại.
- **Deep learning & LLM:** mỗi lớp tuyến tính (linear/dense layer) trong Transformer là phép $\mathbf{W}\mathbf{x} + \mathbf{b}$; gradient descent / MSE ở đây là nền tảng tối ưu cho toàn bộ ngành — xem LLM Foundations.
