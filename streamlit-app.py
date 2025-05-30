import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="트레이딩 모니터링 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


class TradingDashboard:
    def __init__(self, db_path="trading.sqlite"):
        self.db_path = db_path

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def load_trading_data(self, limit=100):
        """거래 내역 데이터 로드"""
        conn = self.get_connection()
        query = """
                SELECT id, timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price
                FROM trading_history
                ORDER BY timestamp DESC
                    LIMIT ? \
                """
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['total_value'] = df['btc_balance'] * df['btc_krw_price'] + df['krw_balance']
            df['profit_loss'] = df['btc_balance'] * (df['btc_krw_price'] - df['btc_avg_buy_price'])

        return df

    def load_reflection_data(self, limit=50):
        """반성 일기 데이터 로드"""
        conn = self.get_connection()
        query = """
                SELECT r.*, h.decision, h.percentage, h.btc_krw_price, h.timestamp as trade_timestamp
                FROM trading_reflection r
                         JOIN trading_history h ON r.trading_id = h.id
                ORDER BY r.reflection_date DESC LIMIT ? \
                """
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()

        if not df.empty:
            df['reflection_date'] = pd.to_datetime(df['reflection_date'])
            df['trade_timestamp'] = pd.to_datetime(df['trade_timestamp'])

        return df


def main():
    st.title("🚀 자동매매 모니터링 대시보드")
    st.markdown("---")

    # 대시보드 인스턴스 생성
    dashboard = TradingDashboard()

    # 사이드바 설정
    st.sidebar.header("📊 대시보드 설정")

    # 데이터 로드 옵션
    data_limit = st.sidebar.selectbox("표시할 거래 내역 수", [50, 100, 200, 500], index=1)
    auto_refresh = st.sidebar.checkbox("자동 새로고침 (30초)", value=False)

    if auto_refresh:
        st.sidebar.info("30초마다 자동으로 데이터를 새로고침합니다.")
        # 실제 운영시에는 st.rerun()을 30초마다 실행하는 로직 추가 필요

    # 데이터 로드
    try:
        trading_df = dashboard.load_trading_data(data_limit)
        reflection_df = dashboard.load_reflection_data()

        if trading_df.empty:
            st.warning("거래 데이터가 없습니다. 데이터베이스를 확인해주세요.")
            return

        # 메인 메트릭스
        col1, col2, col3, col4 = st.columns(4)

        latest_trade = trading_df.iloc[0] if not trading_df.empty else None

        with col1:
            st.metric(
                label="현재 BTC 잔고",
                value=f"{latest_trade['btc_balance']:.8f} BTC" if latest_trade is not None else "N/A"
            )

        with col2:
            st.metric(
                label="현재 KRW 잔고",
                value=f"₩{latest_trade['krw_balance']:,.0f}" if latest_trade is not None else "N/A"
            )

        with col3:
            total_value = latest_trade['total_value'] if latest_trade is not None else 0
            st.metric(
                label="총 자산 가치",
                value=f"₩{total_value:,.0f}"
            )

        with col4:
            profit_loss = latest_trade['profit_loss'] if latest_trade is not None else 0
            st.metric(
                label="현재 손익",
                value=f"₩{profit_loss:,.0f}",
                delta=f"{profit_loss:,.0f}"
            )

        st.markdown("---")

        # 탭 생성
        tab1, tab2, tab3, tab4 = st.tabs(["📈 거래 현황", "💹 수익률 분석", "📝 반성 일기", "📊 상세 통계"])

        with tab1:
            st.subheader("최근 거래 내역")

            # 거래 결정 분포
            col1, col2 = st.columns([1, 2])

            with col1:
                decision_counts = trading_df['decision'].value_counts()
                fig_pie = px.pie(
                    values=decision_counts.values,
                    names=decision_counts.index,
                    title="거래 결정 분포"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                # 시간별 거래 현황
                fig_timeline = px.scatter(
                    trading_df.head(20),
                    x='timestamp',
                    y='btc_krw_price',
                    color='decision',
                    size='percentage',
                    hover_data=['reason', 'btc_balance', 'krw_balance'],
                    title="최근 거래 타임라인"
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

            # 거래 내역 테이블
            st.subheader("거래 내역 상세")
            display_df = trading_df[['timestamp', 'decision', 'percentage', 'reason',
                                     'btc_krw_price', 'btc_balance', 'krw_balance']].copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['btc_krw_price'] = display_df['btc_krw_price'].apply(lambda x: f"₩{x:,.0f}")
            display_df['btc_balance'] = display_df['btc_balance'].apply(lambda x: f"{x:.8f}")
            display_df['krw_balance'] = display_df['krw_balance'].apply(lambda x: f"₩{x:,.0f}")

            st.dataframe(display_df, use_container_width=True)

        with tab2:
            st.subheader("수익률 분석")

            col1, col2 = st.columns(2)

            with col1:
                # 자산 가치 변화
                fig_value = px.line(
                    trading_df,
                    x='timestamp',
                    y='total_value',
                    title="총 자산 가치 변화",
                    labels={'total_value': '자산 가치 (KRW)', 'timestamp': '시간'}
                )
                st.plotly_chart(fig_value, use_container_width=True)

            with col2:
                # 손익 변화
                fig_pl = px.bar(
                    trading_df.head(20),
                    x='timestamp',
                    y='profit_loss',
                    color='profit_loss',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    title="거래별 손익"
                )
                st.plotly_chart(fig_pl, use_container_width=True)

            # 수익률 통계
            if len(trading_df) > 1:
                initial_value = trading_df.iloc[-1]['total_value']
                current_value = trading_df.iloc[0]['total_value']
                total_return = ((current_value - initial_value) / initial_value) * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("총 수익률", f"{total_return:.2f}%")
                with col2:
                    avg_trade_size = trading_df['percentage'].mean()
                    st.metric("평균 거래 비율", f"{avg_trade_size:.1f}%")
                with col3:
                    trade_count = len(trading_df)
                    st.metric("총 거래 횟수", f"{trade_count}회")

        with tab3:
            st.subheader("AI 거래 반성 일기")

            if not reflection_df.empty:
                # 성공률 트렌드
                fig_success = px.line(
                    reflection_df,
                    x='reflection_date',
                    y='success_rate',
                    title="거래 성공률 트렌드",
                    labels={'success_rate': '성공률 (%)', 'reflection_date': '날짜'}
                )
                st.plotly_chart(fig_success, use_container_width=True)

                # 반성 일기 상세
                st.subheader("최근 반성 일기")
                for idx, row in reflection_df.head(5).iterrows():
                    with st.expander(
                            f"📝 {row['reflection_date'].strftime('%Y-%m-%d')} - {row['decision']} ({row['percentage']:.1f}%)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**시장 상황:**")
                            st.write(row['market_condition'])
                            st.write("**결정 분석:**")
                            st.write(row['decision_analysis'])
                        with col2:
                            st.write("**개선점:**")
                            st.write(row['improvement_point'])
                            st.write("**학습 포인트:**")
                            st.write(row['learning_points'])
                        st.metric("성공률", f"{row['success_rate']:.1f}%")
            else:
                st.info("반성 일기 데이터가 없습니다.")

        with tab4:
            st.subheader("상세 통계")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**거래 통계**")
                stats = {
                    "총 거래 수": len(trading_df),
                    "매수 거래": len(trading_df[trading_df['decision'] == 'buy']),
                    "매도 거래": len(trading_df[trading_df['decision'] == 'sell']),
                    "보유 거래": len(trading_df[trading_df['decision'] == 'hold']),
                    "평균 거래 비율": f"{trading_df['percentage'].mean():.2f}%",
                    "최대 거래 비율": f"{trading_df['percentage'].max():.2f}%"
                }

                for key, value in stats.items():
                    st.write(f"• {key}: {value}")

            with col2:
                st.write("**자산 통계**")
                asset_stats = {
                    "현재 BTC 보유량": f"{latest_trade['btc_balance']:.8f} BTC",
                    "현재 KRW 잔고": f"₩{latest_trade['krw_balance']:,.0f}",
                    "평균 매수가": f"₩{latest_trade['btc_avg_buy_price']:,.0f}",
                    "현재 BTC 가격": f"₩{latest_trade['btc_krw_price']:,.0f}",
                    "총 자산 가치": f"₩{latest_trade['total_value']:,.0f}"
                }

                for key, value in asset_stats.items():
                    st.write(f"• {key}: {value}")

            # BTC 가격 차트
            st.subheader("BTC 가격 변화")
            fig_btc = px.line(
                trading_df,
                x='timestamp',
                y='btc_krw_price',
                title="BTC/KRW 가격 변화",
                labels={'btc_krw_price': 'BTC 가격 (KRW)', 'timestamp': '시간'}
            )
            st.plotly_chart(fig_btc, use_container_width=True)

        # 마지막 업데이트 시간
        st.markdown("---")
        st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {str(e)}")
        st.info("데이터베이스 파일 경로와 테이블 구조를 확인해주세요.")


if __name__ == "__main__":
    main()