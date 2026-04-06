import os
import sys
import time
import queue
import threading
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import MetaTrader5 as mt5
import streamlit as st
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ================= НАСТРОЙКИ =================
MOCK_MODE = False  # True для тестирования без MT5
MODEL_PATH = "ppo_xauusd_best"
LOG_QUEUE = queue.Queue(maxsize=1000)
STATE_LOCK = threading.Lock()

class TradingState:
    """Потокобезопасный контейнер состояния"""
    def __init__(self):
        with STATE_LOCK:
            self.connected = False
            self.mode = "Обучение"
            self.running = False
            self.equity_curve = [10000.0]
            self.reward_curve = []
            self.current_pos = {"dir": "Нет", "lot": 0.0, "pnl": 0.0}
            self.best_equity = 10000.0
            self.balance = 10000.0
            self.drawdown = 0.0
            self.confidence = 0.5
            self.logs = []
            self.max_lot = 0.5
            self.stop_loss = 30.0

state = TradingState()

# ================= СРЕДА GYMNASIUM =================
class GoldTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, mock=False):
        super().__init__()
        self.mock = mock
        # Наблюдение: [bid, ask, spread, atr, rsi, balance_norm, equity_norm, pos_pnl, holding_time]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        # Действие: [направление/сила (-1..1), размер лота (0..1)]
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        self.balance = 10000.0
        self.equity = 10000.0
        self.position = 0  # -1 short, 0 flat, 1 long
        self.entry_price = 0.0
        self.holding_time = 0
        self.max_drawdown = 0.0
        self.initial_equity = self.equity

    def _get_obs(self):
        if self.mock:
            return np.random.randn(9).astype(np.float32)
        # Получение тика из MT5
        tick = mt5.symbol_info_tick("XAUUSD")
        if tick is None:
            return np.zeros(9, dtype=np.float32)
            
        bid, ask = tick.bid, tick.ask
        spread = (ask - bid) / 10  # упрощенно в пунктах
        atr = np.random.uniform(0.5, 2.5)  # в продакшене: расчёт ATR из copy_rates_from_pos
        rsi = np.random.uniform(30, 70)    # в продакшене: расчёт RSI
        b_norm = self.balance / 20000.0
        e_norm = self.equity / 20000.0
        pnl = (bid - self.entry_price) * 100 * self.position if self.position != 0 else 0.0
        t_norm = self.holding_time / 1000.0
        return np.array([bid, ask, spread, atr, rsi, b_norm, e_norm, pnl, t_norm], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.balance = 10000.0
        self.equity = 10000.0
        self.position = 0
        self.entry_price = 0.0
        self.holding_time = 0
        self.initial_equity = self.equity
        return self._get_obs(), {}

    def step(self, action):
        dir_val, lot_frac = action
        reward = 0.0
        terminated = False
        
        # 1. Исполнение действий
        if self.position == 0:
            if dir_val > 0.3:
                self.position = 1
                self.entry_price = self._get_obs()[1] # ask
            elif dir_val < -0.3:
                self.position = -1
                self.entry_price = self._get_obs()[0] # bid
        else:
            self.holding_time += 1
            # Закрытие если слабый сигнал или реверс
            if abs(dir_val) < 0.2 or (self.position == 1 and dir_val < -0.3) or (self.position == -1 and dir_val > 0.3):
                # Реализация PnL
                tick = mt5.symbol_info_tick("XAUUSD")
                price = tick.bid if self.position == 1 else tick.ask
                pnl = (price - self.entry_price) * 100 * self.position
                self.balance += pnl
                self.equity = self.balance
                self.position = 0
                self.holding_time = 0
                reward += max(pnl / self.balance, -0.1)

        # 2. Auto-lot расчёт (виртуально для env)
        obs = self._get_obs()
        vol = obs[3] if not self.mock else 1.5
        # Лот обратно пропорционален волатильности и балансу
        calc_lot = lot_frac * (self.balance / (vol * 100 + 1e-5))
        calc_lot = np.clip(calc_lot, 0.01, state.max_lot)

        # 3. Обновление эквити в реальном времени (симуляция просадки)
        if self.position != 0:
            tick = mt5.symbol_info_tick("XAUUSD")
            price = tick.bid if self.position == 1 else tick.ask
            self.equity = self.balance + (price - self.entry_price) * 100 * self.position * calc_lot
        else:
            self.equity = self.balance

        # 4. Расчёт награды
        drawdown = max(self.initial_equity - self.equity, 0)
        self.max_drawdown = max(self.max_drawdown, drawdown / self.initial_equity)
        drawdown_pen = self.max_drawdown * 2.0  # штраф за просадку
        
        holding_pen = 0.0
        if self.position != 0:
            current_pnl = (self.equity - self.balance)
            if current_pnl < 0:
                holding_pen = 0.005 * self.holding_time  # штраф за удержание убытка

        reward -= drawdown_pen + holding_pen
        reward += (self.equity - self.initial_equity) / self.initial_equity  # базовая доходность

        # Сохранение в состояние для UI
        with STATE_LOCK:
            state.equity_curve.append(self.equity)
            state.reward_curve.append(reward)
            state.balance = self.balance
            state.equity = self.equity
            state.drawdown = state.max_drawdown * 100
            if self.position != 0:
                state.current_pos = {"dir": "LONG" if self.position == 1 else "SHORT", "lot": calc_lot, "pnl": self.equity - self.balance}
            else:
                state.current_pos = {"dir": "Нет", "lot": 0.0, "pnl": 0.0}

        if len(state.equity_curve) > 1000:
            terminated = True
            
        return self._get_obs(), reward, terminated, False, {}

# ================= ТОРГОВЫЙ ЦИКЛ =================
def trading_loop():
    log_queue = queue.Queue(maxsize=500)
    state.connected = False
    
    try:
        if not MOCK_MODE:
            state.connected = mt5.initialize()
            if not state.connected:
                log_queue.put("❌ Ошибка подключения к MT5. Включен MOCK режим.")
                MOCK_MODE = True
            else:
                log_queue.put("✅ Подключение к MT5 установлено.")
    except:
        log_queue.put("❌ MT5 не установлен/не запущен. MOCK режим.")

    env = GoldTradingEnv(mock=MOCK_MODE)
    env_vec = DummyVecEnv([lambda: env])
    
    if os.path.exists(f"{MODEL_PATH}.zip"):
        log_queue.put("📂 Загрузка сохранённой модели...")
        model = PPO.load(MODEL_PATH, env=env_vec)
    else:
        log_queue.put("🧠 Инициализация новой модели PPO...")
        model = PPO("MlpPolicy", env_vec, verbose=0, learning_rate=3e-4, ent_coef=0.01)

    while state.running:
        try:
            mode = state.mode
            if mode == "Только обучение":
                model.learn(total_timesteps=5000, reset_num_timesteps=False)
            elif mode == "Торговля + Дообучение":
                action, _ = model.predict(env.reset()[0], deterministic=False)
                model.learn(total_timesteps=100, reset_num_timesteps=False)
            else:  # Только торговля
                action, _ = model.predict(env.reset()[0], deterministic=True)
                
            # Логирование уверенности
            state.confidence = float(np.abs(action[0]))
            log_queue.put(f"📊 Действие: Dir={action[0]:.2f} | Лот={action[1]:.2f} | Уверенность={state.confidence:.2%}")

            # Авто-сохранение при новом максимуме
            if state.equity > state.best_equity:
                state.best_equity = state.equity
                model.save(MODEL_PATH)
                log_queue.put("💾 Новая вершина эквити! Модель сохранена.")

            time.sleep(1)
            # Передача логов в очередь интерфейса
            while not log_queue.empty():
                LOG_QUEUE.put(log_queue.get())
                
        except Exception as e:
            LOG_QUEUE.put(f"⚠️ Ошибка цикла: {str(e)}")
            time.sleep(2)

    with STATE_LOCK:
        state.connected = False
        log_queue.put("🛑 Торговый цикл остановлен.")

# ================= STREAMLIT UI =================
def run_ui():
    st.set_page_config(page_title="AI Gold Trader XAUUSD", layout="wide")
    st.title("🤖 Нейросетевой Робот XAUUSD (PPO + MT5)")
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки")
        mode = st.selectbox("Режим работы", ["Только обучение", "Торговля + Дообучение", "Только торговля"])
        max_lot = st.slider("Максимальный лот", 0.01, 2.0, 0.1, 0.01)
        stop_loss = st.slider("Стоп-лосс (пункты)", 10, 100, 30, 5)
        
        if st.button("🟢 Запустить", use_container_width=True):
            if not state.running:
                state.running = True
                state.mode = mode
                state.max_lot = max_lot
                state.stop_loss = stop_loss
                threading.Thread(target=trading_loop, daemon=True).start()
                
        if st.button("🔴 Остановить", use_container_width=True):
            state.running = False
            st.session_state["stopped"] = True
            
        if st.button("🚨 ЭКСТРЕННОЕ ЗАКРЫТИЕ", type="primary", use_container_width=True):
            if not MOCK_MODE:
                positions = mt5.positions_get(symbol="XAUUSD")
                if positions:
                    for pos in positions:
                        ticket = pos.ticket
                        type = 3 if pos.type == 0 else 2  # close buy/sell
                        price = mt5.symbol_info_tick("XAUUSD").bid if pos.type == 0 else mt5.symbol_info_tick("XAUUSD").ask
                        request = {"action": mt5.TRADE_ACTION_DEAL, "symbol": "XAUUSD", "volume": pos.volume,
                                   "type": type, "position": ticket, "price": price, "deviation": 20, "magic": 234000}
                        res = mt5.order_send(request)
                        LOG_QUEUE.put(f"🛑 Позиция {ticket} закрыта. Рез: {res.comment}")
            else:
                LOG_QUEUE.put("🛑 MOCK: Все позиции виртуально закрыты.")

    # Основная панель
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "🟢 Подключено" if state.connected else "🔴 Отключено / MOCK"
        st.metric("Статус MT5", status)
    with col2:
        st.metric("Баланс", f"${state.balance:,.2f}", f"{state.equity - state.balance:+.2f} $")
    with col3:
        st.metric("Макс. Просадка", f"{state.drawdown:.2f}%")

    # Позиция
    with st.expander("📊 Текущая позиция"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Направление", state.current_pos["dir"])
        c2.metric("Лот", f"{state.current_pos['lot']:.3f}")
        c3.metric("PnL", f"${state.current_pos['pnl']:+.2f}")

    # Графики
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("📈 Кривая эквити")
        if len(state.equity_curve) > 1:
            st.line_chart(pd.DataFrame(state.equity_curve, columns=["Equity"]))
    with c_right:
        st.subheader("🧠 Уверенность модели (Action Probability)")
        if len(state.reward_curve) > 1:
            st.line_chart(pd.DataFrame(state.reward_curve[-50:], columns=["Reward"]))
            st.caption(f"Текущая уверенность: {state.confidence:.2%}")

    # Логирование
    st.subheader("📜 Журнал событий")
    logs = []
    while not LOG_QUEUE.empty():
        logs.append(LOG_QUEUE.get())
    if logs:
        for log in logs[-20:]:
            st.info(f"[{datetime.now().strftime('%H:%M:%S')}] {log}")
    else:
        st.text_area("Ожидание событий...", value="Запустите бота для отображения логов.", height=150)

    # Периодическое обновление UI
    time.sleep(1.5)
    st.rerun()

if __name__ == "__main__":
    with STATE_LOCK:
        state.running = False
    try:
        run_ui()
    except KeyboardInterrupt:
        st.info("👋 Завершение работы...")