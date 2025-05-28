import traceback
import pandas as pd

def perform_ma_analysis(df):
    """종합적인 이동평균선 분석 수행"""
    try:
        print("3-3. 이동평균선 분석 중...")

        if len(df) == 0:
            return {}

        current = df.iloc[-1]
        current_price = current['close']

        # 현재 이동평균선 값들
        current_ma5 = current['sma_5']
        current_ma20 = current['sma_20']
        current_ma60 = current['sma_60']
        current_ma120 = current['sma_120']

        # 이전 값들 (트렌드 분석용)
        if len(df) >= 2:
            prev = df.iloc[-2]
            prev_ma5 = prev['sma_5']
            prev_ma20 = prev['sma_20']
            prev_ma60 = prev['sma_60']
            prev_ma120 = prev['sma_120']
        else:
            prev_ma5 = prev_ma20 = prev_ma60 = prev_ma120 = 0

        # 크로스 신호 분석
        ma_cross_signals = analyze_ma_cross_signals(df)

        # 정배열/역배열 분석
        ma_alignment = analyze_ma_alignment(df)

        # 지지/저항 분석
        support_resistance = analyze_ma_support_resistance(df)

        # 이동평균선 분석 결과 구성
        ma_analysis = {
            "current_price": current_price,
            "moving_averages": {
                "sma_5": current_ma5,
                "sma_20": current_ma20,
                "sma_60": current_ma60,
                "sma_120": current_ma120
            },
            "price_vs_ma": {
                "vs_sma5_percent": ((current_price - current_ma5) / current_ma5 * 100) if current_ma5 else 0,
                "vs_sma20_percent": ((current_price - current_ma20) / current_ma20 * 100) if current_ma20 else 0,
                "vs_sma60_percent": ((current_price - current_ma60) / current_ma60 * 100) if current_ma60 else 0,
                "vs_sma120_percent": (
                        (current_price - current_ma120) / current_ma120 * 100) if current_ma120 else 0,
            },
            "ma_trends": {
                "sma5_trend": "상승" if current_ma5 > prev_ma5 else "하락" if current_ma5 < prev_ma5 else "보합",
                "sma20_trend": "상승" if current_ma20 > prev_ma20 else "하락" if current_ma20 < prev_ma20 else "보합",
                "sma60_trend": "상승" if current_ma60 > prev_ma60 else "하락" if current_ma60 < prev_ma60 else "보합",
                "sma120_trend": "상승" if current_ma120 > prev_ma120 else "하락" if current_ma120 < prev_ma120 else "보합",
            },
            "cross_signals": ma_cross_signals,
            "alignment": ma_alignment,
            "support_resistance": support_resistance
        }

        # 결과 출력
        print(f"\n=== Moving Average Analysis ===")
        print(f"현재가: {current_price:,.0f}원")
        print(f"5일 이평: {current_ma5:,.0f}원 ({ma_analysis['ma_trends']['sma5_trend']})")
        print(f"20일 이평: {current_ma20:,.0f}원 ({ma_analysis['ma_trends']['sma20_trend']})")
        print(f"60일 이평: {current_ma60:,.0f}원 ({ma_analysis['ma_trends']['sma60_trend']})")
        print(f"120일 이평: {current_ma120:,.0f}원 ({ma_analysis['ma_trends']['sma120_trend']})")
        print(f"정배열 상태: {ma_alignment['status']} (강도: {ma_alignment['strength']})")

        if ma_cross_signals:
            print(f"크로스 신호: {', '.join(ma_cross_signals)}")

        if support_resistance.get('nearest_support'):
            nearest_support = support_resistance['nearest_support']
            print(
                f"가장 가까운 지지선: {nearest_support['name']} ({nearest_support['level']:,.0f}원, -{nearest_support['distance_percent']:.2f}%)")

        if support_resistance.get('nearest_resistance'):
            nearest_resistance = support_resistance['nearest_resistance']
            print(
                f"가장 가까운 저항선: {nearest_resistance['name']} ({nearest_resistance['level']:,.0f}원, +{nearest_resistance['distance_percent']:.2f}%)")

        return ma_analysis

    except Exception as e:
        print(f"Error in perform_ma_analysis: {e}")
        traceback.print_exc()
        return {}

# 이동평균선 크로스 신호 분석
def analyze_ma_cross_signals(df):
    """이동평균선 크로스 신호를 분석하여 골든크로스/데드크로스 감지"""
    try:
        ma_cross_signals = []

        # 최근 2일 데이터로 크로스 확인
        if len(df) >= 2:
            # 현재와 이전 데이터
            current = df.iloc[-1]
            previous = df.iloc[-2]

            # 5일선과 20일선 크로스
            if (previous['sma_5'] <= previous['sma_20'] and
                    current['sma_5'] > current['sma_20']):
                ma_cross_signals.append("골든크로스(5일선이 20일선 상향돌파)")

            elif (previous['sma_5'] >= previous['sma_20'] and
                  current['sma_5'] < current['sma_20']):
                ma_cross_signals.append("데드크로스(5일선이 20일선 하향돌파)")

            # 20일선과 60일선 크로스
            if (previous['sma_20'] <= previous['sma_60'] and
                    current['sma_20'] > current['sma_60']):
                ma_cross_signals.append("중기 골든크로스(20일선이 60일선 상향돌파)")

            elif (previous['sma_20'] >= previous['sma_60'] and
                  current['sma_20'] < current['sma_60']):
                ma_cross_signals.append("중기 데드크로스(20일선이 60일선 하향돌파)")

        return ma_cross_signals

    except Exception as e:
        print(f"Error in analyze_ma_cross_signals: {e}")
        return []

# 이동평균선 정배열/역배열 분석
def analyze_ma_alignment(df):
    """이동평균선 정배열/역배열 상태 분석"""
    try:
        if len(df) == 0:
            return {"status": "데이터 부족", "strength": 0}

        current = df.iloc[-1]
        sma5 = current['sma_5']
        sma20 = current['sma_20']
        sma60 = current['sma_60']
        sma120 = current['sma_120']
        current_price = current['close']

        # 정배열 체크 (단기 > 중기 > 장기)
        is_bullish_alignment = (sma5 > sma20 > sma60 > sma120)

        # 역배열 체크 (단기 < 중기 < 장기)
        is_bearish_alignment = (sma5 < sma20 < sma60 < sma120)

        # 현재가 위치 분석
        price_above_all_ma = (current_price > sma5 > sma20 > sma60 > sma120)
        price_below_all_ma = (current_price < sma5 < sma20 < sma60 < sma120)

        if is_bullish_alignment and price_above_all_ma:
            return {"status": "강한 상승 정배열", "strength": 5}
        elif is_bullish_alignment:
            return {"status": "상승 정배열", "strength": 4}
        elif is_bearish_alignment and price_below_all_ma:
            return {"status": "강한 하락 역배열", "strength": -5}
        elif is_bearish_alignment:
            return {"status": "하락 역배열", "strength": -4}
        else:
            return {"status": "혼재", "strength": 0}

    except Exception as e:
        print(f"Error in analyze_ma_alignment: {e}")
        return {"status": "분석 오류", "strength": 0}

# 이동평균선 지지/저항 분석
def analyze_ma_support_resistance(df):
    """이동평균선의 지지/저항 역할 분석"""
    try:
        if len(df) == 0:
            return {}

        current = df.iloc[-1]
        current_price = current['close']

        support_levels = []
        resistance_levels = []

        ma_levels = {
            'sma_5': current['sma_5'],
            'sma_20': current['sma_20'],
            'sma_60': current['sma_60'],
            'sma_120': current['sma_120']
        }

        for ma_name, ma_value in ma_levels.items():
            if pd.notna(ma_value):
                if current_price > ma_value:
                    # 현재가가 이평선 위에 있으면 지지선 역할
                    support_levels.append({
                        'level': ma_value,
                        'name': ma_name,
                        'distance_percent': ((current_price - ma_value) / ma_value) * 100
                    })
                else:
                    # 현재가가 이평선 아래에 있으면 저항선 역할
                    resistance_levels.append({
                        'level': ma_value,
                        'name': ma_name,
                        'distance_percent': ((ma_value - current_price) / current_price) * 100
                    })

        # 가장 가까운 지지/저항선 찾기
        nearest_support = min(support_levels,
                              key=lambda x: x['distance_percent']) if support_levels else None
        nearest_resistance = min(resistance_levels,
                                 key=lambda x: x['distance_percent']) if resistance_levels else None

        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance
        }

    except Exception as e:
        print(f"Error in analyze_ma_support_resistance: {e}")
        return {}