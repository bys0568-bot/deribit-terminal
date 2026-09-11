from datetime import datetime, timezone, timedelta
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# 페이지 설정 (와이드 모드)
st.set_page_config(
    page_title="Deribit Quant Terminal", page_icon="📈", layout="wide"
)

# 영문 통일 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def calculate_zero_threshold(df_target, col_name):
  if df_target.empty:
    return 0
  df_sorted = df_target.sort_values('strike_grouped').reset_index(drop=True)
  strikes = df_sorted['strike_grouped'].values
  vals = df_sorted[col_name].values
  cum_vals = np.cumsum(vals)

  for i in range(len(cum_vals) - 1):
    if cum_vals[i] * cum_vals[i + 1] <= 0:
      s1, s2 = strikes[i], strikes[i + 1]
      c1, c2 = cum_vals[i], cum_vals[i + 1]
      if c1 != c2:
        return s1 - c1 * (s2 - s1) / (c2 - c1)
      return s1

  idx = np.argmin(np.abs(cum_vals))
  return strikes[idx] if len(strikes) > 0 else 0


def get_monthly_expiration(valid_expirations):
  if len(valid_expirations) > 2:
    for ts in valid_expirations:
      dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
      if dt.weekday() == 4 and (dt - datetime.now(timezone.utc)).days >= 4:
        return ts
  return (
      valid_expirations[-1]
      if len(valid_expirations) > 1
      else valid_expirations[0]
  )


def run_dashboard():
  st.sidebar.header('⚙️ Terminal Settings')
  currency = st.sidebar.selectbox('Select Asset', ['BTC', 'ETH'], index=0)

  count = st_autorefresh(interval=300000, limit=None, key="refresh_counter")
  if st.sidebar.button('🔄 Refresh Now'):
    st.rerun()
  st.sidebar.markdown(f'⏱️ Auto-refresh count: **{count}**')

  url = 'https://www.deribit.com/api/v2/public/get_instruments'
  params = {'currency': currency, 'kind': 'option', 'expired': 'false'}

  with st.spinner(f'🔄 Fetching {currency} options data...'):
    try:
      response = requests.get(url, params=params, timeout=10)
      data = response.json()
      if 'error' in data:
        st.error(f"⚠️ API Error: {data['error'].get('message')}")
        return
      instruments = data['result']

      ticker_res = requests.get(
          'https://www.deribit.com/api/v2/public/ticker',
          params={'instrument_name': f'{currency}-PERPETUAL'},
          timeout=10,
      ).json()
      spot_price = ticker_res['result']['index_price']

      now_utc = datetime.now(timezone.utc)
      dvol_val = 0.0
      try:
        dvol_url = (
            'https://www.deribit.com/api/v2/public/get_volatility_index_data'
        )
        end_ts = int(now_utc.timestamp() * 1000)
        start_ts = end_ts - (86400 * 1000 * 5)
        dvol_res = requests.get(
            dvol_url,
            params={
                'currency': currency.lower(),
                'start_timestamp': start_ts,
                'end_timestamp': end_ts,
                'resolution': '1D',
            },
            timeout=5,
        ).json()
        res_data = dvol_res.get('result', {})
        if isinstance(res_data, dict) and 'data' in res_data:
          candles = res_data['data']
          if candles:
            dvol_val = candles[-1][4]
        elif isinstance(res_data, list) and res_data:
          dvol_val = res_data[-1][4]
      except Exception:
        pass

      expirations = sorted(
          list(set(item['expiration_timestamp'] for item in instruments))
      )
      valid_expirations = [
          ts for ts in expirations if ts > now_utc.timestamp() * 1000
      ]
      if not valid_expirations:
        st.warning('⚠️ No valid option expirations found.')
        return

      nearest_ts = valid_expirations[0]
      # 주간물: 0DTE 다음 만기 (존재할 경우)
      weekly_ts = (
          valid_expirations[1] if len(valid_expirations) > 1 else nearest_ts
      )
      monthly_ts = get_monthly_expiration(valid_expirations)

      targets = {
          '0DTE / Micro': nearest_ts,
          'Weekly (주간물)': weekly_ts,
          'Monthly / Macro': monthly_ts,
      }

      tab1, tab2, tab3 = st.tabs(
          ['⚡ 0DTE / Micro', '📅 Weekly (주간물)', '🏛️ Monthly / Macro']
      )
      tabs_dict = {
          '0DTE / Micro': tab1,
          'Weekly (주간물)': tab2,
          'Monthly / Macro': tab3,
      }

      for label, target_ts in targets.items():
        with tabs_dict[label]:
          target_dt_utc = datetime.fromtimestamp(
              target_ts / 1000, tz=timezone.utc
          )
          target_date = target_dt_utc.date()

          target_instruments = [
              item
              for item in instruments
              if abs(item['expiration_timestamp'] - target_ts) < 60000
          ]
          if not target_instruments:
            st.info(f'No contracts for {target_date}')
            continue

          time_to_expiry_seconds = (target_dt_utc - now_utc).total_seconds()
          if time_to_expiry_seconds < 0:
            time_to_expiry_seconds = 1
          time_to_expiry_years = time_to_expiry_seconds / (365.25 * 24 * 3600)
          expected_move_1sig = (
              spot_price * (dvol_val / 100) * np.sqrt(time_to_expiry_years)
          )
          upper_1sig = spot_price + expected_move_1sig
          lower_1sig = spot_price - expected_move_1sig

          chart_data = []
          for inst in target_instruments:
            ins_name = inst['instrument_name']
            strike = inst['strike']
            opt_type = inst['option_type']

            t_res = requests.get(
                'https://www.deribit.com/api/v2/public/ticker',
                params={'instrument_name': ins_name},
                timeout=5,
            ).json()
            if 'result' in t_res:
              t_info = t_res['result']
              greeks = t_info.get('greeks', {})
              gamma = greeks.get('gamma', 0) if greeks else 0
              delta = greeks.get('delta', 0) if greeks else 0
              oi = t_info.get('open_interest', 0)
              iv = t_info.get('mark_iv', 0)

              net_gex = (
                  gamma
                  * oi
                  * (spot_price**2)
                  * 0.01
                  * (1 if opt_type == 'call' else -1)
              )
              net_dex = delta * oi * spot_price

              chart_data.append({
                  'instrument_name': ins_name,
                  'strike': strike,
                  'net_gex': net_gex,
                  'net_dex': net_dex,
                  'open_interest': oi,
                  'option_type': opt_type,
                  'iv': iv,
              })

          df = pd.DataFrame(chart_data)
          if df.empty:
            continue

          otm_puts = df[
              (df['option_type'] == 'put') & (df['strike'] < spot_price)
          ]
          otm_calls = df[
              (df['option_type'] == 'call') & (df['strike'] > spot_price)
          ]
          avg_put_iv = otm_puts['iv'].mean() if not otm_puts.empty else 0.0
          avg_call_iv = otm_calls['iv'].mean() if not otm_calls.empty else 0.0
          skew_val = avg_put_iv - avg_call_iv

          unique_strikes = df['strike'].unique()
          pain_dict = {}
          for s in unique_strikes:
            call_pain = (
                np.maximum(
                    0, s - df[df['option_type'] == 'call']['strike']
                )
                * df[df['option_type'] == 'call']['open_interest']
            ).sum()
            put_pain = (
                np.maximum(
                    0, df[df['option_type'] == 'put']['strike'] - s
                )
                * df[df['option_type'] == 'put']['open_interest']
            ).sum()
            pain_dict[s] = call_pain + put_pain
          max_pain_price = min(pain_dict, key=pain_dict.get)

          total_oi = df['open_interest'].sum()
          total_puts = df[df['instrument_name'].str.endswith('-P')][
              'open_interest'
          ].sum()
          total_calls = df[df['instrument_name'].str.endswith('-C')][
              'open_interest'
          ].sum()
          pc_ratio = total_puts / total_calls if total_calls > 0 else 0

          total_net_gex_m = df['net_gex'].sum() / 1e6
          total_net_dex_m = df['net_dex'].sum() / 1e6

          if currency == 'BTC':
            df['strike_grouped'] = (df['strike'] / 500).round() * 500
            width_val = 350
          else:
            df['strike_grouped'] = df['strike']
            width_val = 15

          grouped = (
              df.groupby('strike_grouped')
              .sum(numeric_only=True)
              .reset_index()
          )

          min_strike = spot_price * 0.93
          max_strike = spot_price * 1.07
          filtered_df = grouped[
              (grouped['strike_grouped'] >= min_strike)
              & (grouped['strike_grouped'] <= max_strike)
          ]
          if filtered_df.empty:
            filtered_df = grouped

          zero_gamma_val = calculate_zero_threshold(filtered_df, 'net_gex')
          zero_delta_val = calculate_zero_threshold(filtered_df, 'net_dex')

          fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
          strikes = filtered_df['strike_grouped']
          gex_vals = filtered_df['net_gex'] / 1e6
          dex_vals = filtered_df['net_dex'] / 1e6

          gex_colors = ['green' if x >= 0 else 'red' for x in gex_vals]
          dex_colors = ['green' if x >= 0 else 'red' for x in dex_vals]

          for ax in [ax1, ax2]:
            ax.axvspan(
                lower_1sig,
                upper_1sig,
                color='orange',
                alpha=0.08,
                label='1σ Expected Move (68%)',
            )
            ax.axvline(
                upper_1sig,
                color='orange',
                linestyle='-.',
                linewidth=1.2,
                alpha=0.7,
            )
            ax.axvline(
                lower_1sig,
                color='orange',
                linestyle='-.',
                linewidth=1.2,
                alpha=0.7,
            )

          # GEX Chart
          ax1.bar(strikes, gex_vals, width=width_val, color=gex_colors, alpha=0.8)
          ax1.axvline(
              spot_price,
              color='blue',
              linestyle='--',
              label=f'Spot: {spot_price:,.2f}',
          )
          ax1.axvline(
              max_pain_price,
              color='purple',
              linestyle=':',
              linewidth=2,
              label=f'Max Pain: {max_pain_price:,.0f}',
          )
          if zero_gamma_val > 0:
            ax1.axvline(
                zero_gamma_val,
                color='darkorange',
                linestyle='-',
                linewidth=1.8,
                label=f'Zero Gamma (Threshold): {zero_gamma_val:,.0f}',
            )

          ax1.set_title(
              f'[{label}] Net GEX | Total: {total_net_gex_m:+,.2f}M | Expiry:'
              f' {target_date}',
              fontsize=10,
          )
          ax1.set_ylabel('Net GEX ($M)')
          ax1.grid(True, alpha=0.3)

          # 범례 박스 내 P/C, OI 통합
          handles, labels_leg = ax1.get_legend_handles_labels()
          dummy_handle = mlines.Line2D([0], [0], color='none')
          handles.extend([dummy_handle, dummy_handle])
          labels_leg.extend([
              f'P/C Ratio: {pc_ratio:.2f}',
              f'Total OI: {total_oi:,.0f}',
          ])
          ax1.legend(
              handles=handles,
              labels=labels_leg,
              loc='upper left',
              fontsize=8,
              ncol=3,
              framealpha=0.9,
          )

          # DEX Chart
          ax2.bar(strikes, dex_vals, width=width_val, color=dex_colors, alpha=0.8)
          ax2.axvline(
              spot_price,
              color='blue',
              linestyle='--',
              label=f'Spot: {spot_price:,.2f}',
          )
          if zero_delta_val > 0:
            ax2.axvline(
                zero_delta_val,
                color='deeppink',
                linestyle='-',
                linewidth=1.8,
                label=f'Zero Delta (Threshold): {zero_delta_val:,.0f}',
            )

          ax2.set_title(
              f'[{label}] Net DEX: {total_net_dex_m:+,.2f}M  |  DVOL:'
              f' {dvol_val:.1f}  |  Skew: {skew_val:+.2f}%',
              fontsize=10,
          )
          ax2.set_ylabel('Net DEX ($M)')
          ax2.grid(True, alpha=0.3)

          if currency == 'BTC':
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(500))
            ax2.xaxis.set_major_formatter(
                ticker.FuncFormatter(
                    lambda x, pos: (
                        f'{int(x/1000)}k' if x % 1000 == 0 else f'{x/1000:g}k'
                    )
                )
            )
          ax2.set_xlabel('Strike Price (USD)')

          plt.tight_layout()
          st.pyplot(fig)
          plt.close(fig)

    except Exception as e:
      st.error(f'❌ Error loading data: {e}')


if __name__ == '__main__':
  run_dashboard()