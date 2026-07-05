---
title: Reinforcement learning
author: Thieu Luu
date: '2026-06-03'
category: Training And Reinforcement Learning
layout: post
---

# Tổng quan
Artificial intelligence: To be able to learn to make decisions to achieve goals
Reinforcement learning: 
- Interacting with our environment and learning to make decisions from interaction
- Sequential: future interactions can depend on earlier ones
- Goal-directed
- Reward hypothesis: any goal can be formalized as the outcome of maximizing a cumulative reward
- Use for finding solution or adapting online, deal with unforeseen circumstances
- Require us to think about time, (long-term) consequences of actions, actively gathering experience, predicting the future, dealing with uncertainty, etc.
- Huge potential scope

Tại thời điểm t, agent quan sát được môi trường đang ở trạng thái $O_t$ và có phần thưởng $R_t$, agent đưa ra hành động $A_t$. Sau khi môi trường nhận action của agent thì agent sẽ quan sát được trạng thái $O_{t+1}$ và tính được phần thưởng $R_{t+1}$.

Công việc của agent là đưa ra hành động $A_t$ để tối đa hóa ***return**:
return $G_t = R_{t+1} + R_{t+2} + R_{t+3} + ...$

**value** là **return** mong đợi khi agent ở trạng thái $S_t$
$$v(s) = E[G_t \mid  S_t=s] = E[R_{t+1} + R_{T+2} + R_{t+3} + ... \mid  S_t = s, A_t = a]$$
Hành động có thể có kết quả lâu dài (mang tính dài hạn), phần thưởng có thể đến muộn, có thể hi sinh phần thưởng ngay lập tức để lấy nhiều phần thưởng hơn về lâu dài.

Một ánh xạ từ trạng thái của môi trường đến hành động của agent gọi là 'chính sách': **policy**

# Các dạng bài toán reinforcement learning
## Fully observable environment 

Fully observable environment là môi trường mà agent có thể nhận thức mọi thuộc tính của môi trường, phân biệt với partial observation environment (khó hơn, agent không thể biết hết về môi trường, các những hidden state)

Full observation environment => $O_t = Se_t \quad \forall t$ 
=> Markov decision processes là một cách thiết kế bài toán hữu dụng.
Từ đây, tôi dùng kí hiệu $Se_t$ là trạng thái của môi trường tại thời điểm t và $S_t$ là trạng thái của agent

Một quá trình ra quyết định là quá trình ra quyết định Markov (MDP) nếu:
$$p(r,s \mid  S_t, A_t) = p(r,s \mid  H_t, A_t)$$
nghĩa là trạng thái hiện tại của agent bao gồm tất cả các thông tin cần thiết từ lịch sử, nghĩa rằng lịch sử trạng thái của agent trong quá khứ không ảnh hưởng tới việc chuyển trạng thái từ $S_t$ sang $S_{t+1}$ nếu hành động $A_t$ của Agent xảy ra, đồng thời phần thưởng cũng vậy. 

## Partially observable environment

Partially observable environment là môi trường mà agent không thể 'nhìn thấy' tất cả thuộc tính của môi trường.
VD: 
- một robot với tầm nhìn của 1 chiếc camera không thể nói lên vị trí tuyệt đối của nó trong không gian
- một agent chơi poker chỉ nhìn thấy những là bài của mình và những là bài trên sân
Đây gọi là Partially observable Markov decision process (POMDP), trạng thái của môi trường có thể vẫn tuân theo Markov, nhưng agent không biết điều đó và chúng ta vẫn có thể tạo ra Markov agent state

# Thành phần của một agent

1. **Agent State**

$$S_{t+1} = u(S_{t}, A_{t}, R_{t+1}, O_{t+1})$$
$u$ là hàm cập nhật trạng thái của agent

2. **Policy**
Policy là luật ánh xạ hành động của agent dựa trên agent state:
Với luật tất định (deterministic) ta có $A = \pi(S)$ 
Với luật ngẫu nhiên (stochastic), dựa theo xác suất: $\pi(A\mid S) = p(A\mid S)$

3. **Value Function**
$$v_\pi(s) = E[G_t \mid  S_t = s, \pi] = E[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... \mid  S_t=s, \pi]$$
trong đó thể hiện việc các giá trị $R_{t+1}$, $R_{t+2}$, ... và $\gamma \in [0, 1]$ phụ thuộc vào không chỉ state $S_t$ mà còn phụ thuộc vào policy $\pi$, $\gamma$ là hệ số chiết khấu (discount factor) được hiểu nôm na là các phần thưởng các xa thì có độ đóng góp vào hàm giá trị không, ít hơn hoặc bằng các phần thưởng ngay lập tức.

Value Function có tính chất đệ quy, ta có $G_t = R_{t+1} + \gamma G_{t+1}$
tương tự ta có:
$$v_\pi(s) = E[R_{t+1} + \gamma G_{t+1} \mid  S_t = s, A_{t} \sim \pi(s)] = E[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid  S_t = s, A_{t} \sim \pi(s)]$$
trong đó $A_{t} \sim \pi(s)$ nghĩa là với policy $\pi$ thì hành động $A_t$ đươc chọn tại agent state $s$
đây là đẳng thức Bellman, một đăng thức tương tự cho giá trị tối ưu (optimal value) là:
$$v_*(s) = \max_a E[R_{t+1} + \gamma v_*(S_{t+1}) \mid  S_t = s, A_t = a]$$
đẳng thức này không phụ thuộc vào policy và được khai thác triệt để để tạo các thuật toán.
4. Model
Model dự đoán trạng thái tiếp theo của agent và phần thưởng dựa trên trạng thái hiện tại của agent và hành động của agent.

Trạng thái tiếp theo:
$$P(s, a, s') \approx p(S_{t+1} = s' \mid  S_t=s, A_t=a)$$
Phần thưởng ngay lập tức tiếp theo:
$$R(s,a) \approx E[R_{t+1}\mid S_t=s, A_t=a]$$
# Phân loại Agent

Cách chia dựa trên value, policy
1. Value based: chỉ dựa trên hàm value để ra quyết định
2. Policy based: chỉ dựa trên policy để ra quyết định
3. Actor Critic: dựa trên cả policy và value để ra quyết định, trong đó actor sử dụng policy và phần critic sử dụng value

Một các chia khác dựa trên việc agent có sử dụng model hay không
1. Model free
2. Model based

# Ví dụ bài toán Multi-Armed Bandit

Ví dụ ta có 1 agent giúp ta đánh bạc hiệu quả trong các Casino, agent này có thể chọn các máy chơi từ 1->n các máy này có tỉ lệ đánh thắng là {$R_a \mid  a \in A$}
Trong đó:
- $A$ là tập các máy trong Casino
- $R_a$ là tỉ lệ đánh thắng khi chơi máy a
- Tại mỗi thời điểm t, agent chọn 1 máy để chơi $A_t \in A$
- Khi đó phần thưởng của agent chính là tỉ lệ đánh thắng của máy $A_t$ là $R_t \approx R_{A_t}$ 
- Mục tiêu là tối đa hóa tỉ lệ đánh thắng $\sum^t_{i=1} R_i$
- Chúng ta cần huấn luyện agent học một policy: xác suất lựa chọn chơi các máy trong tập máy $A$

Đây là một ví dụ điển hình để biểu diễn sự đánh đổi giữa Exploration (khám phá kiến thức mới) và Exploitation (khai thác trên kiến thức đã biết).

Khái niệm giá trị (values) và 'sự nuối tiêct' (regret)
- Giá trị hành động của hành động a (lựa chọn máy a để chơi) có phần thưởng kỳ vọng (tỷ lệ thắng) là: $$q(a) = E[R_t \mid  A_t=a]$$
- Giá trị tối ưu là: $$v_* = \max_{a \in A}q(a) = \max_aE[R_t \mid  A_t = a]$$
- Regret của hành động a là: $$\Delta_a = v_* - q(a)$$
- Ta muốn tối ưu tỉ lệ chiến thắng lũy kế theo thời gian (các lần chơi) tương ứng với việc tối thiểu hóa regret sau các lần chơi: $$L_t = \sum_{n=1}^t v_* - q(A_n) = \sum_{n=1}^t \Delta_{A_n}$$
