import os
import traceback
import time
import pyupbit
import ta
import json
import moving_aver as ma
import requests
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
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.upbit = pyupbit.Upbit(self.access, self.secret)
        self.client = OpenAI()
        self.fear_greed_api = "https://api.alternative.me/fng/"

    # 비트코인 관련 최신 뉴스 조회
    def get_crypto_news(self):
        try:
            base_url = "https://serpapi.com/search.json"
            params = {
                "engine": "google_news",
                "q": "bitcoin crypto trading",
                "api_key": self.serpapi_key,
                "gl": "us", # 미국 뉴스
                "hl": "en"  # 영어 뉴스
            }

            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                news_data = response.json()

                if 'news_results' not in news_data:
                    return None

                processed_news = []
                for news in news_data['news_results'][:5]:  # 상위 5개의 뉴스만 처리
                    processed_news.append({
                        'title': news.get('title', ''),
                        'link': news.get('link', ''),
                        'source': news.get('source', {}).get('name', ''),
                        'date': news.get('date', ''),
                        'snippet': news.get('snippet', ''),
                    })
                print("\n=== Latest Crypto News ===")
                for news in processed_news:
                    print(f"\nTitle: {news['title']}")
                    print(f"Source: {news['source']}")
                    print(f"Date: {news['date']}")

                return processed_news

            return None

        except Exception as e:
            print(f"Error in get_crypto_news: {e}")
            traceback.print_exc()
            return None

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

    # 이동 평균선 계산
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

    # 공포탐욕지수 데이터 조회
    def get_fear_greed_index(self, limit=7):
        try:
            response = requests.get(f"{self.fear_greed_api}?limit={limit}")
            if response.status_code == 200:
                data = response.json()

                # 최신 공포탐욕지수 출력
                latest = data['data'][0]
                print("\n=== Fear and Greed Index ===")
                print(f"Current Value: {latest['value']}({latest['value_classification']})")

                # 7일간의 데이터 가공
                processed_data = []
                for item in data['data']:
                    processed_data.append({
                        'data': datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d'),
                        'value': int(item['value']),
                        'classification': item['value_classification'],
                    })

                # 추세 분석
                values = [int(item['value']) for item in data['data']]
                avg_value = sum(values) / len(values)
                trend = 'Improving' if values[0] > avg_value else 'Deteriorating'

                print(values)

                return {
                    'current': {
                        'value': int(latest['value']),
                        'classification': latest['value_classification']
                    },
                    'history': processed_data,
                    'trend': trend,
                    'average': avg_value
                }
        except Exception as e:
            print(f"Error in get_fear_greed_index: {e}")
            traceback.print_exc()
            return None

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
                "moving_averages_analysis": analysis_data['ohlcv_data']['ma_analysis'],
                "ohlcv": analysis_data['ohlcv_data'],
                "fear_greed": analysis_data['fear_greed'],
                "news": analysis_data['news']
            }

            # 어떤식으로 질문을 할지 설정
            prompt = """You are an expert Bitcoin trading analyst. Analyze the provided data and make a trading decision.
    
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
                        8. Recent News Sentiment
                        
                        Please consider the following key points:
                        - Fear & Greed Index below 20 (Extreme Fear) may present buying
                        opportunities
                        - Fear & Greed Index above 80 (Extreme Greed) may present selling
                        opportunities
                        - The trend of the Fear & Greed Index is also a crucial indicator 

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

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": f"Market datafor analysis: {json.dumps(ai_prompt_data)}"
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
    def execute_trade(self, decision, confidence_score, fear_greed_value):
        print(f"fear_greed_value(공포탐욕지수): {fear_greed_value}")
        try:
            ####### 실제 거래 실행 #######
            if decision == 'buy':
                # 공포탐욕지수에 따른 매매 비율 조정
                # 극도의 공포 상태(0-25)에서는 더 과감한 매수
                if fear_greed_value <= 25:
                    trade_ratio = 0.9995 # 최대 매수
                elif fear_greed_value <= 40:
                    trade_ratio = 0.7 # 중간 매수
                else:
                    trade_ratio = 0.5 # 소액 매수

                # 신뢰도 70 이상
                if confidence_score >= 70:
                    krw = self.upbit.get_balance("KRW")
                    if krw > 5000:
                        order_result = self.upbit.buy_market_order(self.ticker, krw * trade_ratio)
                        print(f"### Buy Order Executed: {krw * trade_ratio:,.0f}원 ###")
                        print("\n=== Buy Order Executed ===")
                        print(f"Trade Ratio: {trade_ratio * 100}%")
                        print(f"주문 결과: {json.dumps(order_result, indent=2)}")
                    else:
                        print(f"### Buy Order Failed: Insufficient KRW (보유: {krw:,.0f}원, 필요: 5,000원 이상) ###")

            elif decision == 'sell' and confidence_score > 70:

                # 극도의 탐욕 상태(75-100)에서는 더 과감한 매도
                if fear_greed_value >= 75:
                    trade_ratio = 1.0 # 전량 매도
                elif fear_greed_value >= 60:
                    trade_ratio = 0.7 # 일부 매도
                else:
                    trade_ratio = 0.5 # 소량 매도

                # 신뢰도 70 이상
                if confidence_score >= 70:
                    btc = self.upbit.get_balance(self.ticker)
                    current_price = pyupbit.get_current_price(self.ticker)

                    # 매도 - 보유 BTC의 95%만 매도 (안전마진 확보)
                    # available_btc = current_status["my_btc"] * 0.95
                    # btc_value = available_btc * current_status["current_btc_price"]

                    if btc * current_price > 5000:
                        sell_amount = btc * trade_ratio
                        print(f"### Sell Order Executed: {btc:.8f} BTC (약 {sell_amount:,.0f}원) ###")
                        order_result = self.upbit.sell_market_order(self.ticker, sell_amount)
                        print("\n=== Sell Order Executed ===")
                        print(f"Trade Ratio: {trade_ratio * 100}%")
                        print(f"주문 결과: {json.dumps(order_result, indent=2)}")
                    else:
                        print(
                            f"### Sell Order Failed: Insufficient BTC (보유: {btc:.8f} BTC, 가치: {btc * current_price:,.0f}원) ###")

        except Exception as e:
            print(f"Error in execute_trade: {e}")
            traceback.print_exc()

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

        # 4. 공포탐욕지수 조회
        fear_greed_data = collector.get_fear_greed_index()

        # 5. 뉴스 데이터 조회
        news_data = collector.get_crypto_news()

        ####### 투자 판단에 대한 전략과 성향을 설정하는 알고리즘 #######
        # 6. AI 분석을 위한 데이터 준비
        if all([current_status, orderbook_data, ohlcv_data, fear_greed_data, news_data]):
            analysis_data = {
                "current_status" : current_status,
                "orderbook_data" : orderbook_data,
                "ohlcv_data" : ohlcv_data,
                "fear_greed": fear_greed_data,
                "news": news_data
            }

            # 7. AI 분석 실행
            ai_result = collector.get_ai_analysis(analysis_data)

            if ai_result:
                print("\n=== AI Analysis Results ===")
                print(json.dumps(ai_result, indent=2))

            # 8. 매매 실행
            collector.execute_trade(ai_result['decision'], ai_result['confidence_score'], fear_greed_data['current']['value'])


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
        while True:

            ai_trading()
            ####### 실행주기 - 10분마다 실행 (API 제한 고려) #######
            print("10분 후 다음 분석을 시작합니다...")
            time.sleep(600)  # 10분 간격으로 실행

    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 종료되었습니다.")
    except Exception as e:
        print(f"예상치 못한 오류: {str(e)}")
        time.sleep(60)  # 에러 발생 시에도 1분 대기

