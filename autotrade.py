import os
import traceback
import time
import pyupbit
import ta
import json
import moving_aver as ma
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

class CryptoDataCollector:
    def __init__(self, ticker="KRW-BTC"):
        self.ticker = ticker
        # .env 파일에 저장한 access key와 secret key를 가져온 다음 access key를 가져와서 간단히 로그인
        self.access = os.getenv("UPBIT_ACCESS_KEY")
        self.secret = os.getenv("UPBIT_SECRET_KEY")
        self.upbit = pyupbit.Upbit(self.access, self.secret)
        self.client = OpenAI()

    # 현재 투자 상태 조회
    def get_current_status(self):
        try:

            print("1. 투자 상태 조회 중...")
            my_krw = float(self.upbit.get_balance("KRW")) # 보유 현금
            my_btc = float(self.upbit.get_balance(self.ticker)) # 보유 암호
            avg_buy_price = float(self.upbit.get_avg_buy_price(self.ticker)) # 평균
            current_btc_price = pyupbit.get_current_price(self.ticker) #  현재가
            btc_value_krw = my_btc * current_btc_price if my_btc else 0 # BTC 평가 금액
            total_value_krw = my_krw + (my_btc * current_btc_price if my_btc else 0) # 총 자산
            btc_ratio = (my_btc * current_btc_price) / (
                        my_krw + (my_btc * current_btc_price if my_btc else 0)) * 100 if (my_krw + (
                        my_btc * current_btc_price if my_btc else 0)) > 0 else 0

            print(f"   - KRW 잔고: {my_krw:,.0f}원")
            print(f"   - BTC 잔고: {my_btc:.8f} BTC")
            print(f"   - BTC 평가금액: {btc_value_krw:,.0f}원")
            print(f"   - 총 자산: {total_value_krw:,.0f}원")
            print(f"   - BTC 비중: {btc_ratio:.1f}%")

            # 투자 상태 정보 구성
            return{
                "my_krw" : my_krw,
                "my_btc" : my_btc,
                "avg_buy_price" : avg_buy_price,
                "current_btc_price" : current_btc_price,
                "total_value_krw" : total_value_krw,
                "unrealized_profit" : ((current_btc_price - avg_buy_price) * my_btc) if my_btc else 0,
            }
        except Exception as e:
            print(f"Error in get_current_status: {e}")
            traceback.print_exc()
            return None

    # 호가 데이터 조회
    def get_orderbook_data(self):
        try:
            print("2. 호가 데이터 수집중 ...")
            orderbook = pyupbit.get_orderbook(self.ticker)

            print(f"호가데이터 :: {orderbook}")

            if not orderbook or len(orderbook) == 0:
                raise ValueError("Orderbook 데이터를 불러오지 못했습니다.")

            ask_prices = []
            ask_sizes = []
            bid_prices = []
            bid_sizes = []

            for unit in orderbook['orderbook_units'][:5]:
                ask_prices.append(unit['ask_price'])
                ask_sizes.append(unit['ask_size'])
                bid_prices.append(unit['bid_price'])
                bid_sizes.append(unit['bid_size'])
            return {
                "timestamp": datetime.fromtimestamp(orderbook['timestamp'] / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                "total_ask_size": float(orderbook['total_ask_size']),
                "total_bid_size": float(orderbook['total_bid_size']),
                "ask_price": ask_prices,
                "ask_sizes": ask_sizes,
                "bid_price": bid_prices,
                "bid_sizes": bid_sizes
            }
        except Exception as e:
            print(f"Error in get_orderbook_data: {e}")
            traceback.print_exc()
            return None

    # 차트 데이터 수집
    def get_ohlcv_data(self):
        try:
            print("3. 차트 데이터 수집 ...")

            # 3-1. 차트 데이터 가져오기 - 30일 일봉
            print("3-1. 30일 일봉 데이터 수집 중...")
            df_daily = pyupbit.get_ohlcv("KRW-BTC", count=30, interval="day")
            df_daily = self.add_technical_indicators(df_daily)
            print(f"30일 일봉 데이터 :: {df_daily}")

            # 3-2. 차트 데이터 가져오기 - 24시간 시간봉
            print("3-2. 24시간 시간봉 데이터 수집 중...")
            df_hourly = pyupbit.get_ohlcv("KRW-BTC", count=24, interval="minute60")
            df_hourly = self.add_technical_indicators(df_hourly)
            print(f"24시간 시간봉 데이터 :: {df_hourly}")

            # 3-3. 이동평균선 계산 (5일, 20일, 60일, 120일)
            ma_analysis = ma.perform_ma_analysis(df_daily)
            # print("5. 이동평균선 계산 중...")
            # df_daily_ma = df_daily.copy()
            # df_daily_ma['MA5'] = df_daily_ma['close'].rolling(window=5).mean()
            # df_daily_ma['MA20'] = df_daily_ma['close'].rolling(window=20).mean()

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
                hour_date['date'] = index.strftime('%Y-%m-%d %H:%M:%S')
                df_hourly_dict.append(hour_date)

            # 최신 기술적 지표 출력
            print("\n=== Last Technical Indicators ===")
            print(f"RSI: {df_daily['rsi'].iloc[-1]:.2f}")
            print(f"MACD: {df_daily['macd'].iloc[-1]:.2f}")
            print(f"BB Position: {df_daily['bb_pband'].iloc[-1]:.2f}")

            # 이동평균선 분석 정보
            # ma_analysis = {
            #     "current_price": current_price,
            #     "ma5_current": current_ma5,
            #     "ma20_current": current_ma20,
            #     "price_vs_ma5": ((current_price - current_ma5) / current_ma5 * 100) if current_ma5 else 0,
            #     "price_vs_ma20": ((current_price - current_ma20) / current_ma20 * 100) if current_ma20 else 0,
            #     "ma5_trend": "상승" if current_ma5 > prev_ma5 else "하락" if current_ma5 < prev_ma5 else "보합",
            #     "ma20_trend": "상승" if current_ma20 > prev_ma20 else "하락" if current_ma20 < prev_ma20 else "보합",
            #     "ma5_vs_ma20": "위" if current_ma5 > current_ma20 else "아래" if current_ma5 < current_ma20 else "동일",
            #     "cross_signals": ma_cross_signals,
            #     "support_resistance": {
            #         "ma5_support": current_ma5 if current_price > current_ma5 else None,
            #         "ma20_support": current_ma20 if current_price > current_ma20 else None,
            #         "ma5_resistance": current_ma5 if current_price < current_ma5 else None,
            #         "ma20_resistance": current_ma20 if current_price < current_ma20 else None
            #     }
            # }
            #
            # print(f"   - 현재가: {current_price:,.0f}원")
            # print(f"   - 5일 이평: {current_ma5:,.0f}원 ({ma_analysis['ma5_trend']})")
            # print(f"   - 20일 이평: {current_ma20:,.0f}원 ({ma_analysis['ma20_trend']})")
            # print(f"   - 5일선 대비: {ma_analysis['price_vs_ma5']:+.2f}%")
            # print(f"   - 20일선 대비: {ma_analysis['price_vs_ma20']:+.2f}%")
            # print(f"   - 5일선은 20일선 {ma_analysis['ma5_vs_ma20']}에 위치")
            # if ma_cross_signals:
            #     print(f"   - 크로스 신호: {', '.join(ma_cross_signals)}")

            print("=== 데이터 수집 완료 ===\n")

            return {
                "daily_date": df_data_dict,
                "hourly_date": df_hourly_dict,
                "ma_analysis": ma_analysis
            }

        except Exception as e:
            print(f"Error in get_ohlcv_data: {e}")
            traceback.print_exc()
            return None

    def add_technical_indicators(self, df):

        try:
            print("3-3. 이동평균선 계산 중...")

            # 기술적 분석 지표 추가
            # 볼린저 밴드
            indicator_bb = ta.volatility.BollingerBands(close=df['close'])
            df['bb_high'] = indicator_bb.bollinger_hband()
            df['bb_mid'] = indicator_bb.bollinger_mavg()
            df['bb_low'] = indicator_bb.bollinger_lband()
            df['bb_pband'] = indicator_bb.bollinger_pband()

            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()

            # MACD
            macd = ta.trend.MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()

            # 이동평균선
            df['sma_5'] = ta.trend.SMAIndicator(close=df['close'], window=5).sma_indicator()
            df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
            df['sma_60'] = ta.trend.SMAIndicator(close=df['close'], window=60).sma_indicator()
            df['sma_120'] = ta.trend.SMAIndicator(close=df['close'], window=120).sma_indicator()

            # ATR
            df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()

            return df
        except Exception as e:
            print(f"Error in add_technical_indicators: {e}")
            traceback.print_exc()
            return None

    def perform_ma_analysis(self, df):
        """종합적인 이동평균선 분석 수행"""
        try:
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
            ma_cross_signals = self.analyze_ma_cross_signals(df)

            # 정배열/역배열 분석
            ma_alignment = self.analyze_ma_alignment(df)

            # 지지/저항 분석
            support_resistance = self.analyze_ma_support_resistance(df)

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

    # AI 분석 및 매매 신호 생성
    def get_ai_analysis(self, analysis_data):

        try:

            # AI에게 제공할 데이터 구성
            ai_prompt_data = {
                "current_investment_status": analysis_data['current_status'],
                "orderbook_info": {
                    "timestamp": analysis_data['orderbook_data']['timestamp'],
                    "total_ask_size": analysis_data['orderbook_data']['total_ask_size'],
                    "total_bid_size": analysis_data['orderbook_data']['total_bid_size'],
                    "best_ask_price": analysis_data['orderbook_data']['ask_price'][0],
                    "best_bid_price": analysis_data['orderbook_data']['bid_price'][0],
                    "ask_bid_spread": analysis_data['orderbook_data']['ask_price'][0] - analysis_data['orderbook_data']['bid_price'][0],
                },
                "moving_averages_analysis": analysis_data['ma_analysis'],
                "ohlcv": analysis_data['ohlcv_data']
            }

            response = self.client.chat.completions.create(
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
                                                    3. Moving averages analysis with detailed signals:
                                                       - Golden Cross (5MA crossing above 20MA) = Strong Bullish signal
                                                       - Dead Cross (5MA crossing below 20MA) = Strong Bearish signal
                                                       - MA alignment (정배열/역배열) strength
                                                       - Support/Resistance levels from MAs
                                                       - Price position relative to each MA
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
                                                      "confidence_score": 0-100,
                                                      "rist_level": "low/medium/high",
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
            # result = response.choices[0].message.content

            result = json.loads(response.choices[0].message.content)

            print("=== AI 투자 판단 결과 ===")
            print(f"### AI Decision: {result['decision'].upper()} ###")
            print(f"### Confidence: {result.get('confidence', 'N/A')}/10 ###")
            print(f"### Reason: {result['reason']} ###")
            print(f"### Suggested Action: {result.get('suggested_action', 'N/A')} ###")
            print(f"### Risk Assessment: {result.get('risk_assessment', 'N/A')} ###")
            print(f"### Technical Summary: {result.get('technical_summary', 'N/A')} ###")
            print("=" * 50)

            return result
        except Exception as e:
            print(f"Error in get_ai_analysis: {e}")
            traceback.print_exc()
            return None


    # 매매 실행
    def execute_trade(self, decision, confidence_score):
        try:

            ####### 실제 거래 실행 #######
            if decision == 'buy' and confidence_score > 70:
                krw = self.upbit.get_balance("KRW")
                # 매수 - 현재 보유 현금의 95%만 사용 (안전마진 확보)
                available_krw = krw * 0.95
                if available_krw > 5000:
                    print(f"### Buy Order Executed: {available_krw:,.0f}원 ###")
                    order_result = self.upbit.buy_market_order("KRW-BTC", available_krw)
                    print("\n=== Buy Order Executed ===")
                    print(f"주문 결과: {json.dumps(order_result, indent=2)}")
                else:
                    print(
                        f"### Buy Order Failed: Insufficient KRW (보유: {available_krw:,.0f}원, 필요: 5,000원 이상) ###")

            elif decision == 'sell' and confidence_score > 70:
                btc = self.upbit.get_balance(self.ticker)
                current_price = pyupbit.get_current_price(self.ticker)

                # 매도 - 보유 BTC의 95%만 매도 (안전마진 확보)
                # available_btc = current_status["my_btc"] * 0.95
                # btc_value = available_btc * current_status["current_btc_price"]

                if btc * current_price > 5000:
                    print(f"### Sell Order Executed: {btc:.8f} BTC (약 {current_price:,.0f}원) ###")
                    order_result = self.upbit.sell_market_order("KRW-BTC", btc)
                    print("\n=== Sell Order Executed ===")
                    print(f"주문 결과: {json.dumps(order_result, indent=2)}")
                else:
                    print(
                        f"### Sell Order Failed: Insufficient BTC (보유: {btc:.8f} BTC, 가치: {btc * current_price:,.0f}원) ###")
        except Exception as e:
            print(f"Error in execute_trade: {e}")
            traceback.print_exc()
            return None
# 업비트 거래소의 API를 쉽게 사용할 수 있도록 도워주는 파이썬 라이브러리
# 상세한 pypupbit 라이브러리 사용법 링크
# github.com/sharebook-kr/pyupbit
def ai_trading():
    import json
    try:
        collector = CryptoDataCollector("KRW-BTC")
        ####### 차트 데이터뿐만 아니라 뉴스, 공포탐욕지수 등 다양한 데이터를 입력하는 구간 #######
        # 1. 현재 투자 상태 조회
        current_status = collector.get_current_status()
        print("\n=== Current Investment Status ===")
        print(json.dumps(current_status, indent=2))

        # 2. 호가 데이터 조회
        orderbook_data = collector.get_orderbook_data()
        print("\n=== Current Orderbook ===")
        print(json.dumps(orderbook_data, indent=2))

        # 3. 차트 데이터 수집
        ohlcv_data = collector.get_ohlcv_data()
        print("\n=== OHLCV Data ===")
        print(json.dumps(ohlcv_data, indent=2))

        ####### 투자 판단에 대한 전략과 성향을 설정하는 알고리즘 #######
        # 4. AI 분석을 위한 데이터 준비
        if all([current_status, orderbook_data, ohlcv_data]):
            analysis_data = {
                "current_status" : current_status,
                "orderbook_data" : orderbook_data,
                "ohlcv_data" : ohlcv_data
            }

            # 5. AI 분석 실행
            ai_result = collector.get_ai_analysis(analysis_data)

            if ai_result:
                print("\n=== AI Analysis Results ===")
                print(json.dumps(ai_result, indent=2))

            # 6. 매매 실행
            collector.execute_trade(ai_result['decision'], ai_result['confidence_score'])


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