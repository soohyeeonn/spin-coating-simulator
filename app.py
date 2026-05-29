import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. EBP & Meyerhofer 계산 함수 (아까 만든 물리 엔진)
def calculate_spin_coating(rpm, h0, E_rate):
    omega = rpm * (2 * np.pi / 60)
    rho = 1000
    eta_0 = 0.05
    dt = 0.1
    t_array = np.arange(0, 100, dt)
    
    # EBP 모델 (증발 X)
    h_ebp = h0 / np.sqrt(1 + (4 * rho * omega**2 * h0**2 * t_array) / (3 * eta_0))
    
    # Meyerhofer 모델 (증발 O)
    h_meyer = np.zeros_like(t_array)
    h_meyer[0] = h0
    
    def get_eta(t):
        return eta_0 * np.exp(0.1 * t) 
        
    def dh_dt(h, t):
        spin_off = (2 * rho * omega**2 * h**3) / (3 * get_eta(t))
        evaporation = E_rate
        return -spin_off - evaporation

    t_gel = None
    for i in range(1, len(t_array)):
        t_n = t_array[i-1]
        h_n = h_meyer[i-1]
        
        k1 = dh_dt(h_n, t_n)
        k2 = dh_dt(h_n + 0.5 * dt * k1, t_n + 0.5 * dt)
        k3 = dh_dt(h_n + 0.5 * dt * k2, t_n + 0.5 * dt)
        k4 = dh_dt(h_n + dt * k3, t_n + dt)
        
        h_next = h_n + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        if h_next < 0: h_next = 0
        h_meyer[i] = h_next
        
        spin_off_rate = (2 * rho * omega**2 * h_next**3) / (3 * get_eta(t_array[i]))
        if t_gel is None and spin_off_rate < E_rate:
            t_gel = t_array[i]

    return t_array, h_ebp, h_meyer, t_gel, h_meyer[-1]

# 2. 웹 대시보드 화면 구성 (Streamlit)
st.title("Spin Coating Simulator 💿")
st.markdown("Emslie-Bonner-Peck (EBP) 이론과 Meyerhofer 모델 비교")

# 사이드바에 입력 슬라이더 만들기
st.sidebar.header("Process Parameters")
rpm = st.sidebar.slider("Rotation Speed (RPM)", 1000, 6000, 3000, step=100)
h0_um = st.sidebar.slider("Initial Thickness (um)", 10.0, 200.0, 100.0, step=10.0)
E_rate_ums = st.sidebar.slider("Evaporation Rate (um/s)", 0.1, 5.0, 1.0, step=0.1)

# 단위를 미터(m)로 변환
h0 = h0_um * 1e-6
E_rate = E_rate_ums * 1e-6

# 계산 함수 실행
t, h_ebp, h_meyer, t_gel, h_final = calculate_spin_coating(rpm, h0, E_rate)

# 결과 텍스트 출력
st.subheader("Simulation Results")
col1, col2 = st.columns(2)
col1.metric("Final Thickness (Meyerhofer)", f"{h_final * 1e6:.2f} um")
col2.metric("Gel Point (t_gel)", f"{t_gel:.1f} s" if t_gel else "N/A")

# 3. 그래프 그리기 (Matplotlib)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t, h_ebp * 1e6, label='EBP Theory (No Evaporation)', linestyle='--')
ax.plot(t, h_meyer * 1e6, label='Meyerhofer Model (With Evaporation)', linewidth=2)

if t_gel:
    ax.axvline(x=t_gel, color='red', linestyle=':', label=f'Gel Point ({t_gel:.1f}s)')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Thickness (um)')
ax.set_title('Film Thickness over Time')
ax.legend()
ax.grid(True)

# 웹 화면에 그래프 띄우기
st.pyplot(fig)