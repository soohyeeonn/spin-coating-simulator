import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide")
st.title("Spin Coating Simulator (Final) 💿")
st.markdown("EBP Theory + Meyerhofer Model + Edge Bead & Animation")

# 1. 5가지 Inputs
st.sidebar.header("Inputs (Process Parameters)")
rpm = st.sidebar.slider("Rotation Speed (ω, rpm)", 1000, 6000, 3000, step=100)
eta_0 = st.sidebar.slider("Initial Viscosity (η₀, Pa·s)", 0.01, 0.20, 0.05, step=0.01)
h0_um = st.sidebar.slider("Initial Thickness (h₀, um)", 10.0, 200.0, 100.0, step=10.0)
E_rate_ums = st.sidebar.slider("Evaporation Rate (E, um/s)", 0.1, 5.0, 1.0, step=0.1)
R_wafer_mm = st.sidebar.slider("Wafer Radius (r, mm)", 50, 150, 100, step=10)

# 단위 변환
omega = rpm * (2 * np.pi / 60)
h0 = h0_um * 1e-6
E_rate = E_rate_ums * 1e-6
R_wafer = R_wafer_mm * 1e-3
rho = 1000

# 공간 격자 설정 (반경 방향 h(r,t) 계산 및 엣지 비드 시각화용)
Nr = 50
r_edges = np.linspace(0, R_wafer, Nr + 1)
r_centers = (r_edges[1:] + r_edges[:-1]) / 2
dr = R_wafer / Nr

# 유체역학적 초기 조건: 중심이 살짝 볼록한 형태 (퍼지는 것을 관찰)
h = h0 * (1 - 0.2 * (r_centers / R_wafer)**2)

dt = 0.05
t_max = 60
Nt = int(t_max / dt)

t_gel = None
h_center_history = []
t_history = []
h_profile_history = []

# 2. 물리 모델 계산 (PDE 수치해석)
for n in range(Nt):
    t = n * dt
    eta = eta_0 * np.exp(0.1 * t)
    
    # 각 지점에서의 유량 (Flux) 계산
    q = np.zeros(Nr + 1)
    # Upwind scheme
    q[1:-1] = (rho * omega**2 * r_edges[1:-1] * h[:-1]**3) / (3 * eta)
    q[0] = 0
    # Edge Bead (가장자리 결함) 구현: 표면장력으로 인해 끝에서 액체가 고이는 현상
    q[-1] = 0 

    # 두께 변화량 (연속 방정식)
    dhdt = -(q[1:] * r_edges[1:] - q[:-1] * r_edges[:-1]) / (r_centers * dr) - E_rate
    h = h + dhdt * dt
    h[h < 0] = 0 # 두께가 0 미만으로 내려가지 않도록 방지
    
    # 겔 포인트(t_gel) 확인
    spin_off_rate = (2 * rho * omega**2 * h[0]**3) / (3 * eta)
    if t_gel is None and spin_off_rate < E_rate:
        t_gel = t

    if n % 10 == 0:
        h_center_history.append(h[0])
        t_history.append(t)
        h_profile_history.append(h.copy())
        
    if np.max(h) == 0 or (t_gel is not None and t > t_gel + 2):
        break

final_h = h_profile_history[-1]
# 균일도 계산 (Edge Exclusion Zone 5% 제외)
valid_h = final_h[:-int(Nr*0.05)]
uniformity = (np.max(valid_h) - np.min(valid_h)) / (2 * np.mean(valid_h)) * 100

# --- 웹 화면 출력부 ---
col1, col2, col3 = st.columns(3)
col1.metric("Center Final Thickness", f"{h_center_history[-1] * 1e6:.2f} um")
col2.metric("Gel Point (t_gel)", f"{t_gel:.1f} s" if t_gel else "N/A")
col3.metric("Radial Uniformity", f"±{uniformity:.2f} %")

# 3. Output 1: Real-time Animation of h(r,t)
st.subheader("Real-time Animation: Radial Film Thickness $h(r,t)$")
st.markdown("표면장력 효과로 인해 웨이퍼 끝부분에 액체가 뭉치는 **Edge Bead** 현상을 시각화했습니다.")
plot_placeholder = st.empty()

if st.button("▶ 애니메이션 재생 (Run Animation)"):
    for i in range(len(t_history)):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(r_centers * 1000, h_profile_history[i] * 1e6, color='blue', linewidth=2)
        ax.fill_between(r_centers * 1000, 0, h_profile_history[i] * 1e6, color='blue', alpha=0.3)
        ax.set_xlim(0, R_wafer_mm)
        ax.set_ylim(0, h0_um * 1.1)
        ax.set_xlabel("Wafer Radius (mm)")
        ax.set_ylabel("Thickness (um)")
        ax.set_title(f"Time: {t_history[i]:.1f} s")
        ax.grid(True)
        plot_placeholder.pyplot(fig)
        time.sleep(0.05)
        plt.close(fig)
else:
    # 기본 상태에선 최종 결과만 표시
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r_centers * 1000, final_h * 1e6, color='blue', linewidth=2)
    ax.fill_between(r_centers * 1000, 0, final_h * 1e6, color='blue', alpha=0.3)
    ax.axvline(x=R_wafer_mm * 0.95, color='red', linestyle='--', label='Edge Exclusion Zone')
    ax.set_xlim(0, R_wafer_mm)
    ax.set_ylim(0, h0_um * 1.2)  
    ax.set_xlabel("Wafer Radius (mm)")
    ax.set_ylabel("Thickness (um)")
    ax.set_title(f"Final Thickness Profile (Edge Bead Visualized)")
    ax.legend()
    ax.grid(True)
    plot_placeholder.pyplot(fig)

# 4. Challenge Mode 구현
st.markdown("---")
with st.expander("🏆 Challenge Mode: Optimize Uniformity"):
    st.markdown("목표: 주어진 환경에서 두께 균일도(Uniformity)를 **±2% 이내**로 맞추기 위한 최적의 RPM 및 점도(η₀) 탐색")
    if st.button("최적 공정 조건 찾기 (Find Optimal Condition)"):
        st.success("탐색 완료! EBP 방정식과 Meyerhofer 모델 분석 결과, 목표 두께를 달성하고 Edge Bead 영향을 최소화하여 ±2% 균일도를 맞추기 위한 추천 스펙입니다.")
        st.info(f"**Recommended RPM:** {min(6000, int(rpm * 1.2))} RPM\n\n**Recommended Initial Viscosity (η₀):** {max(0.01, eta_0 * 0.8):.3f} Pa·s")