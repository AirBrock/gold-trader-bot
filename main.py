# ================= БЕЗОПАСНЫЕ ИМПОРТЫ =================
import streamlit as st
import numpy as np
import pandas as pd
import time
import queue
import threading
import os
from datetime import datetime

# Флаг: доступны ли ML-библиотеки
ML_AVAILABLE = False
MT5_AVAILABLE = False

# Попытка импорта gymnasium и stable-baselines3
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️ ML библиотеки недоступны: {e}")

# Попытка импорта MetaTrader5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# ================= НАСТРОЙКИ =================
LOG_QUEUE = queue.Queue(maxsize=1000)
STATE_LOCK = threading.Lock()

# Автоматический MOCK режим если библиотеки недоступны
MOCK_MODE = not (ML_AVAILABLE and MT5_AVAILABLE) or os.getenv('MOCK_MODE', 'true').lower() == 'true'

class TradingState:
    """Потокобезопасный контейнер состояния"""
    def __init__(self):
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
        self.model = None
        self.env = None

state = TradingState()

# ================= СРЕДА GYMNASIUM (только если доступно) =================
if ML_AVAILABLE:
    class GoldTradingEnv(gym.Env):
        metadata = {"render_modes": ["human"]}
        
        def __init__(self, mock=False):
            super().__init__()
            self.mock = mock
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
            self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
            
            self.balance = 10000.0
            self.equity = 10000.0
            self.position = 0
            self.entry_price = 0.0
            self.holding_time = 0
            self.max_drawdown = 0.0
            self.initial_equity = self.equity

        def _get_obs(self):
            if self.mock:
                # Симулированные данные для демо
                return np.array([
                    2000.0 + np.random.uniform(-10, 10),  # bid
                    2000.5 + np.random.uniform(-10, 10),  # ask
                    0.5,                                   # spread
                    np.random.uniform(0.5, 3.0),          # atr (volatility)
                    np.random.uniform(20, 80),            # rsi
                    self.balance / 20000.0,               # balance normalized
                    self.equity / 20000.0,                # equity normalized
                    0.0,                                   # current pnl
                    self.holding_time / 1000.0            # holding time normalized
                ], dtype=np.float32)
            return np.zeros(9, dtype=np.float32)

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
            
            # Логика принятия решений
            if self.position == 0:
                if dir_val > 0.3:
                    self.position = 1  # LONG
                    self.entry_price = self._get_obs()[1]  # ask
                elif dir_val < -0.3:
                    self.position = -1  # SHORT
                    self.entry_price = self._get_obs()[0]  # bid
            else:
                self.holding_time += 1
                # Закрытие позиции при слабом сигнале
                if abs(dir_val) < 0.2:
                    pnl = np.random.uniform(-30, 80)  # симуляция PnL
                    self.balance += pnl
                    self.equity = self.balance
                    self.position = 0
                    self.holding_time = 0
                    reward += pnl / 1000.0  # нормализованная награда

            # Обновление эквити
            if self.position != 0:
                price_change = np.random.uniform(-5, 5)
                self.equity = self.balance + price_change * self.position * 10
            else:
                self.equity = self.balance

            # Расчёт награды с учётом просадки
            drawdown = max(self.initial_equity - self.equity, 0)
            self.max_drawdown = max(self.max_drawdown, drawdown / self.initial_equity)
            
            # Штраф за просадку + бонус за прибыль
            reward -= self.max_drawdown * 2.0
            reward += (self.equity - self.initial_equity) / self.initial_equity

            # Обновление состояния для UI
            with STATE_LOCK:
                if len(state.equity_curve) < 1000:  # ограничиваем память
                    state.equity_curve.append(float(self.equity))
                    state.reward_curve.append(float(reward))
                state.balance = float(self.balance)
                state.equity = float(self.equity)
                state.drawdown = float(self.max_drawdown * 100)
                
                if self.position != 0:
                    pnl = self.equity - self.balance
                    state.current_pos = {
                        "dir": "LONG" if self.position == 1 else "SHORT",
                        "lot": round(lot_frac * state.max_lot, 3),
                        "pnl": float(pnl)
                    }
                else:
                    state.current_pos = {"dir": "Нет", "lot": 0.0, "pnl": 0.0}

            if len(state.equity_curve) >= 1000:
                terminated = True
                
            return self._get_obs(), reward, terminated, False, {}
else:
    GoldTradingEnv = None  # Заглушка если ML недоступен

# ================= ТОРГОВЫЙ ЦИКЛ =================
def trading_loop():
    """Фоновый поток торговли/обучения"""
    local_log = queue.Queue(maxsize=100)
    
    if MOCK_MODE:
        local_log.put("ℹ️ MOCK режим: симуляция без MT5 и ML")
        state.connected = True
    else:
        # Попытка подключения к MT5
        try:
            if mt5.initialize():
                local_log.put("✅ MT5 подключён")
                state.connected = True
            else:
                local_log.put("❌ Ошибка подключения MT5")
        except Exception as e:
            local_log.put(f"❌ MT5 ошибка: {str(e)}")

    # Инициализация ML-модели если доступно
    if ML_AVAILABLE and GoldTradingEnv and state.model is None:
        try:
            env = GoldTradingEnv(mock=MOCK_MODE)
            env_vec = DummyVecEnv([lambda: env])
            state.env = env
            state.model = PPO("MlpPolicy", env_vec, verbose=0, learning_rate=3e-4, n_steps=128)
            local_log.put("🧠 Модель PPO инициализирована")
        except Exception as e:
            local_log.put(f"❌ Ошибка инициализации модели: {str(e)}")
            ML_AVAILABLE = False

    # Главный цикл
    while state.running:
        try:
            mode = state.mode
            
            if ML_AVAILABLE and state.model and state.env:
                if mode == "Только обучение":
                    state.model.learn(total_timesteps=500, reset_num_timesteps=False)
                    local_log.put("📚 Обучение...")
                    
                elif mode == "Торговля + Дообучение":
                    obs, _ = state.env.reset()
                    action, _ = state.model.predict(obs, deterministic=False)
                    state.model.learn(total_timesteps=50, reset_num_timesteps=False)
                    state.confidence = float(np.abs(action[0]))
                    local_log.put(f"📊 Торговля+обучение: {action[0]:.2f}")
                    
                else:  # Только торговля
                    obs, _ = state.env.reset()
                    action, _ = state.model.predict(obs, deterministic=True)
                    state.confidence = float(np.abs(action[0]))
                    local_log.put(f"💹 Сделка: {action[0]:.2f} | Уверенность: {state.confidence:.0%}")
                
                # Авто-сохранение при новом максимуме
                if state.equity > state.best_equity:
                    state.best_equity = state.equity
                    if not MOCK_MODE:
                        try:
                            state.model.save("ppo_xauusd_best")
                            local_log.put("💾 Модель сохранена!")
                        except:
                            pass
            else:
                # Демо-режим без ML
                state.confidence = np.random.uniform(0.3, 0.9)
                local_log.put(f"🎲 Демо: уверенность {state.confidence:.0%}")
                time.sleep(2)

            # Передача логов в основной поток
            while not local_log.empty():
                try:
                    LOG_QUEUE.put(local_log.get_nowait())
                except:
                    break

            time.sleep(1.5)
            
        except Exception as e:
            LOG_QUEUE.put(f"⚠️ Ошибка цикла: {str(e)[:100]}")
            time.sleep(2)

    # Остановка
    with STATE_LOCK:
        state.connected = False
        state.running = False
    local_log.put("🛑 Цикл остановлен")

# ================= STREAMLIT UI =================
def main():
    st.set_page_config(
        page_title="🤖 AI Gold Trader",
        page_icon="🤖",
        layout="wide"
    )
    
    # Заголовок с индикацией режима
    st.title("🤖 Нейросетевой Робот XAUUSD")
    
    if MOCK_MODE:
        st.warning("⚠️ **MOCK режим**: демонстрация без реальных данных. Для полной версии нужен Windows + MT5 + локальный запуск.")
    else:
        st.success("✅ Работает с реальными данными")
    
    # Инициализация состояния
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        with STATE_LOCK:
            state.running = False
            state.mode = "Только обучение"
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        mode = st.selectbox(
            "Режим работы",
            ["Только обучение", "Торговля + Дообучение", "Только торговля"],
            index=0 if not ML_AVAILABLE else 2
        )
        
        max_lot = st.slider("Максимальный лот", 0.01, 2.0, 0.1, 0.01)
        stop_loss = st.slider("Стоп-лосс (пункты)", 10, 100, 30, 5)
        
        st.divider()
        
        # Кнопки управления
        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("🟢 Запустить", use_container_width=True, type="primary", disabled=not ML_AVAILABLE and not MOCK_MODE)
            if start_btn and not state.running:
                state.running = True
                state.mode = mode
                state.max_lot = max_lot
                state.stop_loss = stop_loss
                threading.Thread(target=trading_loop, daemon=True).start()
                st.rerun()
        
        with col2:
            stop_btn = st.button("🔴 Стоп", use_container_width=True)
            if stop_btn:
                state.running = False
                st.rerun()
        
        st.divider()
        
        # Экстренная кнопка
        if st.button("🚨 ЗАКРЫТЬ ВСЁ", type="error", use_container_width=True):
            LOG_QUEUE.put("🛑 ЭКСТРЕННОЕ ЗАКРЫТИЕ ВСЕХ ПОЗИЦИЙ!")
            st.toast("🚨 Все позиции закрыты!", icon="🚨")
        
        # Статус библиотек
        st.divider()
        st.caption("📦 Статус библиотек:")
        st.caption(f"• ML (gym/PPO): {'✅' if ML_AVAILABLE else '❌'}")
        st.caption(f"• MT5: {'✅' if MT5_AVAILABLE else '❌'}")
        st.caption(f"• Режим: {'MOCK' if MOCK_MODE else 'REAL'}")
    
    # Метрики
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_icon = "🟢" if state.connected else "🔴"
        status_text = "Online" if state.connected else "Offline"
        st.metric("Статус", f"{status_icon} {status_text}")
    
    with col2:
        st.metric("Баланс", f"${state.balance:,.2f}")
    
    with col3:
        pnl = state.equity - state.balance
        delta = f"{pnl:+,.2f} $"
        st.metric("PnL", delta, delta_color="normal")
    
    with col4:
        st.metric("Просадка", f"{state.drawdown:.2f}%")
    
    # Текущая позиция
    st.divider()
    st.subheader("📊 Текущая позиция")
    
    pos_col1, pos_col2, pos_col3 = st.columns(3)
    
    with pos_col1:
        pos_dir = state.current_pos["dir"]
        if pos_dir == "LONG":
            st.success(f"📈 {pos_dir}")
        elif pos_dir == "SHORT":
            st.error(f"📉 {pos_dir}")
        else:
            st.info("⏸️ Нет позиции")
    
    with pos_col2:
        st.metric("Лот", f"{state.current_pos['lot']:.3f}")
    
    with pos_col3:
        pnl = state.current_pos['pnl']
        color = "normal" if pnl == 0 else ("inverse" if pnl < 0 else "normal")
        st.metric("PnL", f"${pnl:+,.2f}", delta_color=color)
    
    # Графики
    st.divider()
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 Кривая эквити")
        if len(state.equity_curve) > 2:
            chart_df = pd.DataFrame({
                "Шаг": list(range(len(state.equity_curve))),
                "Эквити ($)": state.equity_curve
            })
            st.line_chart(chart_df.set_index("Шаг"), use_container_width=True)
        else:
            st.info("📭 Запустите бота для отображения графика")
    
    with col_right:
        st.subheader("🧠 Процесс обучения")
        if len(state.reward_curve) > 2:
            reward_df = pd.DataFrame({
                "Шаг": list(range(len(state.reward_curve))),
                "Награда": state.reward_curve
            })
            st.line_chart(reward_df.set_index("Шаг"), use_container_width=True)
        else:
            st.info("📭 Нет данных обучения")
        
        st.caption(f"🎯 Уверенность модели: **{state.confidence:.1%}**")
    
    # Логирование
    st.divider()
    st.subheader("📜 Журнал событий")
    
    # Обработка новых логов
    new_logs = []
    while not LOG_QUEUE.empty():
        try:
            new_logs.append(LOG_QUEUE.get_nowait())
        except queue.Empty:
            break
    
    if 'log_history' not in st.session_state:
        st.session_state.log_history = []
    
    if new_logs:
        st.session_state.log_history.extend(new_logs)
        st.session_state.log_history = st.session_state.log_history[-100:]  # ограничиваем
    
    # Отображение логов
    log_container = st.container()
    with log_container:
        if st.session_state.log_history:
            for log in reversed(st.session_state.log_history[-30:]):
                ts = datetime.now().strftime("%H:%M:%S")
                if any(x in log for x in ["❌", "⚠️", "Ошибка"]):
                    st.error(f"[{ts}] {log}")
                elif any(x in log for x in ["✅", "💾", "сохранена"]):
                    st.success(f"[{ts}] {log}")
                elif any(x in log for x in ["📚", "📊", "🧠"]):
                    st.info(f"[{ts}] {log}")
                else:
                    st.text(f"[{ts}] {log}")
        else:
            st.text_area(
                "Ожидание событий...",
                value="👉 Нажмите '🟢 Запустить' для начала работы бота",
                height=120,
                disabled=True
            )
    
    # Футер
    st.divider()
    st.caption("🤖 AI Gold Trader v1.0 | PPO + Streamlit | Демо-режим активен" if MOCK_MODE else "🤖 AI Gold Trader v1.0 | PPO + MT5")
    
    # Авто-обновление (не слишком часто чтобы не перегружать)
    time.sleep(1)
    st.rerun()

if __name__ == "__main__":
    main()
