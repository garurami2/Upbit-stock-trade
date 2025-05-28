import os
import traceback
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# 업비트 거래소의 API를 쉽게 사용할 수 있도록 도워주는 파이썬 라이브러리
# 상세한 pypupbit 라이브러리 사용법 링크
# github.com/sharebook-kr/pyupbit
def ai_trading():
    import json
    try:
        ####### 차트 데이터뿐만 아니라 뉴스, 공포탐욕지수 등 다양한 데이터를 입력하는 구간 #######
        import pyupbit

        # .env 파일에 저장한 access key와 secret key를 가져온 다음 access key를 가져와서 간단히 로그인
        access = os.getenv("UPBIT_ACCESS_KEY")
        secret = os.getenv("UPBIT_SECRET_KEY")
        upbit = pyupbit.Upbit(access, secret)

        print("=== 데이터 수집 시작 ===")

        # 1. 현재 투자 상태 조회
        print("1. 투자 상태 조회 중...")
        balances = upbit.get_balances()
        my_krw = upbit.get_balance("KRW")
        my_btc = upbit.get_balance("KRW-BTC")
        current_btc_price = pyupbit.get_current_price("KRW-BTC")

        # 투자 상태 정보 구성
        investment_status = {
            "krw_balance": my_krw,
            "btc_balance": my_btc,
            "btc_current_price": current_btc_price,
            "btc_value_krw": my_btc * current_btc_price if my_btc else 0,
            "total_value_krw": my_krw + (my_btc * current_btc_price if my_btc else 0),
            "btc_ratio": (my_btc * current_btc_price) / (
                        my_krw + (my_btc * current_btc_price if my_btc else 0)) * 100 if (my_krw + (
                my_btc * current_btc_price if my_btc else 0)) > 0 else 0
        }

        print(f"   - KRW 잔고: {my_krw:,.0f}원")
        print(f"   - BTC 잔고: {my_btc:.8f} BTC")
        print(f"   - BTC 평가금액: {investment_status['btc_value_krw']:,.0f}원")
        print(f"   - 총 자산: {investment_status['total_value_krw']:,.0f}원")
        print(f"   - BTC 비중: {investment_status['btc_ratio']:.1f}%")

        # 2. 오더북(호가) 데이터 가져오기
        print("2. 오더북 데이터 수집 중...")
        orderbook = pyupbit.get_orderbook("KRW-BTC")
        if not orderbook or len(orderbook) == 0:
            raise ValueError("Orderbook 데이터를 불러오지 못했습니다.")

        # 3. 차트 데이터 가져오기 - 30일 일봉
        print("3. 30일 일봉 데이터 수집 중...")
        df_daily = pyupbit.get_ohlcv("KRW-BTC", count=30, interval="day")

        # 4. 차트 데이터 가져오기 - 24시간 시간봉
        print("4. 24시간 시간봉 데이터 수집 중...")
        df_hourly = pyupbit.get_ohlcv("KRW-BTC", count=24, interval="minute60")

        # 5. 이동평균선 계산 (5일, 20일)
        print("5. 이동평균선 계산 중...")
        df_daily_ma = df_daily.copy()
        df_daily_ma['MA5'] = df_daily_ma['close'].rolling(window=5).mean()
        df_daily_ma['MA20'] = df_daily_ma['close'].rolling(window=20).mean()

        # DataFrame을 dict로 변환 시 datetime index 처리
        # 30일봉
        df_data_dict = []
        for index, row in df_daily.iterrows():
            day_date = row.to_dict()
            day_date['date'] = index.strftime('%Y-%m-%d')
            df_data_dict.append(day_date)

        # 24시간 봉
        df_hourly_dict = []
        for index, row in df_hourly.iterrows():
            hour_date = row.to_dict()
            hour_date['date'] = index.strftime('%Y-%m-%d')
            df_hourly_dict.append(hour_date)

        df_date_hourly_dict = {
            "daily_date": df_data_dict,
            "hourly_date": df_hourly_dict,
        }

        print("\n=== OHLCV Data ===")
        print(json.dumps(df_date_hourly_dict, indent=2))

        # 현재 가격과 이동평균선 비교
        current_price = df_daily_ma['close'].iloc[-1]
        current_ma5 = df_daily_ma['MA5'].iloc[-1]
        current_ma20 = df_daily_ma['MA20'].iloc[-1]

        # 이전 값들과 비교하여 추세 파악
        prev_ma5 = df_daily_ma['MA5'].iloc[-2] if len(df_daily_ma) > 1 else current_ma5
        prev_ma20 = df_daily_ma['MA20'].iloc[-2] if len(df_daily_ma) > 1 else current_ma20

        # 골든크로스/데드크로스 확인 (최근 5일간)
        ma_cross_signals = []
        for i in range(max(1, len(df_daily_ma) - 5), len(df_daily_ma)):
            if i > 0 and not pd.isna(df_daily_ma['MA5'].iloc[i]) and not pd.isna(df_daily_ma['MA20'].iloc[i]):
                if (df_daily_ma['MA5'].iloc[i] > df_daily_ma['MA20'].iloc[i] and
                        df_daily_ma['MA5'].iloc[i - 1] <= df_daily_ma['MA20'].iloc[i - 1]):
                    ma_cross_signals.append(f"골든크로스 발생 ({df_daily_ma.index[i].strftime('%Y-%m-%d')})")
                elif (df_daily_ma['MA5'].iloc[i] < df_daily_ma['MA20'].iloc[i] and
                      df_daily_ma['MA5'].iloc[i - 1] >= df_daily_ma['MA20'].iloc[i - 1]):
                    ma_cross_signals.append(f"데드크로스 발생 ({df_daily_ma.index[i].strftime('%Y-%m-%d')})")

        # 이동평균선 분석 정보
        ma_analysis = {
            "current_price": current_price,
            "ma5_current": current_ma5,
            "ma20_current": current_ma20,
            "price_vs_ma5": ((current_price - current_ma5) / current_ma5 * 100) if current_ma5 else 0,
            "price_vs_ma20": ((current_price - current_ma20) / current_ma20 * 100) if current_ma20 else 0,
            "ma5_trend": "상승" if current_ma5 > prev_ma5 else "하락" if current_ma5 < prev_ma5 else "보합",
            "ma20_trend": "상승" if current_ma20 > prev_ma20 else "하락" if current_ma20 < prev_ma20 else "보합",
            "ma5_vs_ma20": "위" if current_ma5 > current_ma20 else "아래" if current_ma5 < current_ma20 else "동일",
            "cross_signals": ma_cross_signals,
            "support_resistance": {
                "ma5_support": current_ma5 if current_price > current_ma5 else None,
                "ma20_support": current_ma20 if current_price > current_ma20 else None,
                "ma5_resistance": current_ma5 if current_price < current_ma5 else None,
                "ma20_resistance": current_ma20 if current_price < current_ma20 else None
            }
        }

        print(f"   - 현재가: {current_price:,.0f}원")
        print(f"   - 5일 이평: {current_ma5:,.0f}원 ({ma_analysis['ma5_trend']})")
        print(f"   - 20일 이평: {current_ma20:,.0f}원 ({ma_analysis['ma20_trend']})")
        print(f"   - 5일선 대비: {ma_analysis['price_vs_ma5']:+.2f}%")
        print(f"   - 20일선 대비: {ma_analysis['price_vs_ma20']:+.2f}%")
        print(f"   - 5일선은 20일선 {ma_analysis['ma5_vs_ma20']}에 위치")
        if ma_cross_signals:
            print(f"   - 크로스 신호: {', '.join(ma_cross_signals)}")

        print("=== 데이터 수집 완료 ===\n")

        ####### 투자 판단에 대한 전략과 성향을 설정하는 알고리즘 #######
        # 6. OpenAI에게 데이터 제공하고 판단받기
        print("6. AI 투자 판단 요청 중...")
        from openai import OpenAI
        client = OpenAI()

        # AI에게 제공할 데이터 구성
        ai_prompt_data = {
            "current_investment_status": investment_status,
            "orderbook_info": {
                "timestamp": orderbook['timestamp'],
                "total_ask_size": orderbook['total_ask_size'],
                "total_bid_size": orderbook['total_bid_size'],
                "best_ask_price": orderbook['orderbook_units'][0]['ask_price'],
                "best_bid_price": orderbook['orderbook_units'][0]['bid_price'],
                "ask_bid_spread": orderbook['orderbook_units'][0]['ask_price'] - orderbook['orderbook_units'][0][
                    'bid_price'],
            },
            "moving_averages_analysis": ma_analysis,
            "daily_chart_30d": df_daily.to_json(),
            "hourly_chart_24h": df_hourly.to_json()
        }

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": """You are an expert Bitcoin trading analyst. Analyze the provided data and make a trading decision.

                                        Consider the following factors:
                                        1. Current investment status (cash vs BTC ratio, total portfolio value)
                                        2. Orderbook data (bid/ask spread, market depth, buying/selling pressure)
                                        3. Moving averages analysis (5-day and 20-day MA trends, golden/dead cross signals)
                                        4. 30-day daily chart patterns and trends
                                        5. 24-hour hourly chart for short-term momentum
                                        6. Technical indicators (price vs MA levels, support/resistance)
                                        7. Risk management principles
                                        
                                        Pay special attention to:
                                        - Golden Cross (5MA crossing above 20MA) = Bullish signal
                                        - Dead Cross (5MA crossing below 20MA) = Bearish signal
                                        - Price position relative to moving averages
                                        - Moving average trend directions
                                        - Support and resistance levels from MAs
                                        
                                        Provide your decision in JSON format with detailed reasoning.
                                        
                                        Response format:
                                        {
                                          "decision": "buy" | "sell" | "hold",
                                          "reason": "detailed technical and fundamental analysis including MA analysis",
                                          "confidence": 1-10,
                                          "suggested_action": "specific action recommendation",
                                          "risk_assessment": "risk level and management strategy",
                                          "technical_summary": "summary of key technical indicators and MA signals"
                                        }"""
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Please analyze this Bitcoin trading data and provide your recommendation:\n\n{ai_prompt_data}"
                        }
                    ]
                }
            ],
            response_format={
                "type": "json_object"
            },
            temperature=0.7,
            max_completion_tokens=1000
        )

        print(f"response : {response}")
        result = response.choices[0].message.content

        import json
        result = json.loads(result)

        print("=== AI 투자 판단 결과 ===")
        print(f"### AI Decision: {result['decision'].upper()} ###")
        print(f"### Confidence: {result.get('confidence', 'N/A')}/10 ###")
        print(f"### Reason: {result['reason']} ###")
        print(f"### Suggested Action: {result.get('suggested_action', 'N/A')} ###")
        print(f"### Risk Assessment: {result.get('risk_assessment', 'N/A')} ###")
        print(f"### Technical Summary: {result.get('technical_summary', 'N/A')} ###")
        print("=" * 50)

        ####### 실제 거래 실행 #######
        if result['decision'] == 'buy':
            # 매수 - 현재 보유 현금의 95%만 사용 (안전마진 확보)
            available_krw = my_krw * 0.95
            if available_krw > 5000:
                print(f"### Buy Order Executed: {available_krw:,.0f}원 ###")
                order_result = upbit.buy_market_order("KRW-BTC", available_krw)
                print(f"주문 결과: {order_result}")
                print(f"매수 이유: {result['reason']}")
            else:
                print(f"### Buy Order Failed: Insufficient KRW (보유: {my_krw:,.0f}원, 필요: 5,000원 이상) ###")

        elif result['decision'] == 'sell':
            # 매도 - 보유 BTC의 95%만 매도 (안전마진 확보)
            available_btc = my_btc * 0.95
            btc_value = available_btc * current_btc_price

            if btc_value > 5000:
                print(f"### Sell Order Executed: {available_btc:.8f} BTC (약 {btc_value:,.0f}원) ###")
                order_result = upbit.sell_market_order("KRW-BTC", available_btc)
                print(f"주문 결과: {order_result}")
                print(f"매도 이유: {result['reason']}")
            else:
                print(
                    f"### Sell Order Failed: Insufficient BTC (보유: {my_btc:.8f} BTC, 가치: {my_btc * current_btc_price:,.0f}원) ###")

        elif result['decision'] == 'hold':
            # 보유
            print("### Hold Order Executed ###")
            print(f"보유 이유: {result['reason']}")

        print("\n" + "=" * 70 + "\n")

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        traceback.print_exc()
        print("10초 후 다시 시도합니다...")


# 메인 실행 루프
if __name__ == "__main__":
    print("Bitcoin AI Trading Bot 시작")
    print("Ctrl+C로 종료할 수 있습니다.")
    print("=" * 70)

    try:
        # 재귀적으로 계속 실행
        print("Starting Bitcoin Trading Bot...")
        ai_trading()
        # while True:
        #
        #
        #     ai_trading()
        #     ####### 실행주기 - 1분마다 실행 (API 제한 고려) #######
        #     print("1분 후 다음 분석을 시작합니다...")
        #     time.sleep(60)  # 1분 간격으로 실행

    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 종료되었습니다.")
    except Exception as e:
        print(f"예상치 못한 오류: {str(e)}")
        time.sleep(60)  # 에러 발생 시에도 1분 대기