import os
import traceback
import time
import pyupbit
import ta
import json
import requests
import base64
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# 별도로 뺀 분석
import moving_aver as ma  # 이동평균선
import youtube_api  # 유튜브 분석
import charts  # 지표 분석
import DatabaseManager as dm

from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()


@dataclass
class TradingConfig:
    """거래 설정을 관리하는 데이터클래스"""
    MIN_TRADE_AMOUNT = 5000
    SAFETY_MARGIN = 0.95
    EXTREME_FEAR_THRESHOLD = 25
    EXTREME_GREED_THRESHOLD = 75
    MODERATE_FEAR_THRESHOLD = 40
    MODERATE_GREED_THRESHOLD = 60
    HIGH_CONFIDENCE_THRESHOLD = 70

    # 공포탐욕지수에 따른 거래 비율 조정 계수
    FEAR_BUY_MULTIPLIER = 1.2
    FEAR_SELL_MULTIPLIER = 0.5
    GREED_BUY_MULTIPLIER = 0.5
    GREED_SELL_MULTIPLIER = 1.2
    MODERATE_MULTIPLIER = 0.8


class CryptoDataCollector:
    def __init__(self, ticker="KRW-BTC"):
        self.ticker = ticker
        self.config = TradingConfig()
        self._init_apis()
        self.dbmanager = dm.DatabaseManager()

    # 과거 거래 분석 및 반성
    def analyze_past_decisions(self):
        try:
            # 최근 거래 내역 조회
            recent_trades = self.dbmanager.get_recent_trades(10)
            recent_reflections = self.dbmanager.get_reflection_history(5)

            # 현재 시장 상태 조회
            current_market = {
                "price": float(pyupbit.get_current_price(self.ticker)),
                "status": self.get_current_status(),
                "fear_greed": self.get_fear_greed_index(),
                "technical": self.get_ohlcv_data()
            }

            # AI에 분석 요청
            reflection_prompt = {
                "recent_trades": recent_trades,
                "recent_reflections": recent_reflections,
                "current_market": current_market
            }

            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an AI trading advisor. Provide your analysis in JSON format with these exact fields:
                        {
                            "market_condition": "Current market state analysis",
                            "decision_analysis": "Analysis of past trading decisions",
                            "improvement_points": "Points to improve",
                            "success_rate": numeric value between 0-100,
                            "learning_points": "Key lessons learned"
                        }"""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze these trading records and market conditions and provide response in JSON formant:\n{json.dumps(reflection_prompt, indent=2)}"
                    }
                ],
                response_format={
                    "type": "json_object"
                }
            )

            reflection = json.loads(response.choices[0].message.content)

            # 반성 일기 저장
            reflection_data = {
                'trading_id': recent_trades[0][0], # 최근 거래 ID
                'reflection_date': datetime.now(),
                'market_condition': reflection['market_condition'],
                'decision_analysis': reflection['decision_analysis'],
                'improvement_points': reflection['improvement_points'],
                'success_rate': reflection['success_rate'],
                'learning_points': reflection['learning_points']
            }

            self.dbmanager.add_reflection(reflection_data)

            return reflection

        except Exception as e:
            print(f"Error in analyze_past_decisions: {e}")
            traceback.print_exc()
            return None

    def _init_apis(self):
        """API 초기화를 별도 메서드로 분리"""
        self.access = os.getenv("UPBIT_ACCESS_KEY")
        self.secret = os.getenv("UPBIT_SECRET_KEY")
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.upbit = pyupbit.Upbit(self.access, self.secret)
        self.client = OpenAI()
        self.fear_greed_api = "https://api.alternative.me/fng/"
        self.youtube_channels = ["3XbtEX3jUv4"]

    def _handle_request_error(self, func_name: str, e: Exception) -> None:
        """공통 에러 처리 메서드"""
        print(f"Error in {func_name}: {e}")
        traceback.print_exc()

    def get_crypto_news(self) -> Optional[List[Dict]]:
        """비트코인 관련 최신 뉴스 조회"""
        try:
            params = {
                "engine": "google_news",
                "q": "bitcoin crypto trading",
                "api_key": self.serpapi_key,
                "gl": "us",
                "hl": "en"
            }

            response = requests.get("https://serpapi.com/search.json", params=params)
            if response.status_code != 200:
                return None

            news_data = response.json()
            if 'news_results' not in news_data:
                return None

            processed_news = []
            for news in news_data['news_results'][:5]:
                processed_news.append({
                    'title': news.get('title', ''),
                    'link': news.get('link', ''),
                    'source': news.get('source', {}).get('name', ''),
                    'date': news.get('date', ''),
                    'snippet': news.get('snippet', ''),
                })

            self._print_news_summary(processed_news)
            return processed_news

        except Exception as e:
            self._handle_request_error("get_crypto_news", e)
            return None

    def _print_news_summary(self, news_list: List[Dict]) -> None:
        """뉴스 요약 출력 (중복 제거)"""
        print("\n=== Latest Crypto News ===")
        for news in news_list:
            print(f"\nTitle: {news['title']}")
            print(f"Source: {news['source']}")
            print(f"Date: {news['date']}")

    def get_current_status(self) -> Optional[Dict]:
        """현재 투자 상태 조회"""
        try:
            print("1. 투자 상태 조회 중...")

            # 잔고 정보 한 번에 조회
            balances = self.upbit.get_balances()
            my_krw = float(self.upbit.get_balance("KRW"))
            my_btc = float(self.upbit.get_balance(self.ticker))
            avg_buy_price = float(self.upbit.get_avg_buy_price(self.ticker))
            current_btc_price = pyupbit.get_current_price(self.ticker)

            # 계산 로직 통합
            btc_value_krw = my_btc * current_btc_price if my_btc else 0
            total_value_krw = my_krw + btc_value_krw
            btc_ratio = (btc_value_krw / total_value_krw * 100) if total_value_krw > 0 else 0

            status_info = {
                "my_krw": my_krw,
                "my_btc": my_btc,
                "avg_buy_price": avg_buy_price,
                "current_btc_price": current_btc_price,
                "total_value_krw": total_value_krw,
                "unrealized_profit": (current_btc_price - avg_buy_price) * my_btc if my_btc else 0,
            }

            self._print_status_summary(my_krw, my_btc, btc_value_krw, total_value_krw, btc_ratio)
            return status_info

        except Exception as e:
            self._handle_request_error("get_current_status", e)
            return None

    def _print_status_summary(self, krw: float, btc: float, btc_value: float, total: float, ratio: float) -> None:
        """투자 상태 요약 출력"""
        print(f"   - KRW 잔고: {krw:,.0f}원")
        print(f"   - BTC 잔고: {btc:.8f} BTC")
        print(f"   - BTC 평가금액: {btc_value:,.0f}원")
        print(f"   - 총 자산: {total:,.0f}원")
        print(f"   - BTC 비중: {ratio:.1f}%")

    def get_orderbook_data(self) -> Optional[Dict]:
        """호가 데이터 조회"""
        try:
            print("2. 호가 데이터 수집중 ...")
            orderbook = pyupbit.get_orderbook(self.ticker)

            if not orderbook or len(orderbook) == 0:
                raise ValueError("Orderbook 데이터를 불러오지 못했습니다.")

            # 호가 데이터 추출 최적화
            units = orderbook['orderbook_units'][:5]
            ask_prices = [unit['ask_price'] for unit in units]
            ask_sizes = [unit['ask_size'] for unit in units]
            bid_prices = [unit['bid_price'] for unit in units]
            bid_sizes = [unit['bid_size'] for unit in units]

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
            self._handle_request_error("get_orderbook_data", e)
            return None

    def get_ohlcv_data(self) -> Optional[Dict]:
        """차트 데이터 수집"""
        try:
            print("3. 차트 데이터 수집 ...")

            # 차트 데이터 한 번에 수집
            df_daily = self._get_chart_data("day", 30, "30일 일봉")
            df_hourly = self._get_chart_data("minute60", 24, "24시간 시간봉")

            if df_daily is None or df_hourly is None:
                return None

            # 이동평균선 계산
            ma_analysis = ma.perform_ma_analysis(df_daily)

            # DataFrame to dict 변환 최적화
            daily_data = self._df_to_dict(df_daily, '%Y-%m-%d')
            hourly_data = self._df_to_dict(df_hourly, '%Y-%m-%d %H:%M:%S')

            self._print_technical_indicators(df_daily)
            print("=== 데이터 수집 완료 ===\n")

            return {
                "daily_date": daily_data,
                "hourly_date": hourly_data,
                "ma_analysis": ma_analysis
            }

        except Exception as e:
            self._handle_request_error("get_ohlcv_data", e)
            return None

    def _get_chart_data(self, interval: str, count: int, description: str):
        """차트 데이터 수집 공통 메서드"""
        print(f"3-{interval}. {description} 데이터 수집 중...")
        df = pyupbit.get_ohlcv("KRW-BTC", count=count, interval=interval)
        return self.add_technical_indicators(df) if df is not None else None

    def _df_to_dict(self, df, date_format: str) -> List[Dict]:
        """DataFrame을 딕셔너리 리스트로 변환"""
        result = []
        for index, row in df.iterrows():
            data = row.to_dict()
            data['date'] = index.strftime(date_format)
            result.append(data)
        return result

    def _print_technical_indicators(self, df) -> None:
        """기술적 지표 출력"""
        print("\n=== Last Technical Indicators ===")
        print(f"RSI: {df['rsi'].iloc[-1]:.2f}")
        print(f"MACD: {df['macd'].iloc[-1]:.2f}")
        print(f"BB Position: {df['bb_pband'].iloc[-1]:.2f}")

    def add_technical_indicators(self, df):
        """기술적 지표 추가 (최적화된 버전)"""
        try:
            print("3-3. 이동평균선 계산 중...")

            close_price = df['close']
            high_price = df['high']
            low_price = df['low']

            # 볼린저 밴드
            bb = ta.volatility.BollingerBands(close=close_price)
            df['bb_high'] = bb.bollinger_hband()
            df['bb_mid'] = bb.bollinger_mavg()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_pband'] = bb.bollinger_pband()

            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(close=close_price).rsi()

            # MACD
            macd = ta.trend.MACD(close=close_price)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()

            # 이동평균선 (반복 최소화)
            for window in [5, 20, 60, 120]:
                df[f'sma_{window}'] = ta.trend.SMAIndicator(close=close_price, window=window).sma_indicator()

            # ATR
            df['atr'] = ta.volatility.AverageTrueRange(high=high_price, low=low_price,
                                                       close=close_price).average_true_range()

            return df
        except Exception as e:
            self._handle_request_error("add_technical_indicators", e)
            return None

    def get_fear_greed_index(self, limit=7) -> Optional[Dict]:
        """공포탐욕지수 데이터 조회"""
        try:
            response = requests.get(f"{self.fear_greed_api}?limit={limit}")
            if response.status_code != 200:
                return None

            data = response.json()
            latest = data['data'][0]

            # 데이터 가공
            processed_data = [
                {
                    'data': datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d'),
                    'value': int(item['value']),
                    'classification': item['value_classification'],
                }
                for item in data['data']
            ]

            # 추세 분석
            values = [int(item['value']) for item in data['data']]
            avg_value = sum(values) / len(values)
            trend = 'Improving' if values[0] > avg_value else 'Deteriorating'

            print("\n=== Fear and Greed Index ===")
            print(f"Current Value: {latest['value']}({latest['value_classification']})")

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
            self._handle_request_error("get_fear_greed_index", e)
            return None

    def capture_and_analyze_chart(self) -> Optional[str]:
        """차트 캡처 및 분석"""
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = f"chart_{current_time}.png"

        try:
            url = f"https://upbit.com/full_chart?code=CRIX.UPBIT.{self.ticker}"
            capture_success = charts.capture_full_page(url, screenshot_path)

            if not capture_success:
                return None

            # 이미지 분석
            with open(screenshot_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this cryptocurrency chart and provide insights about: 
                                    1) Current trend
                                    2) Key support/resistance levels 
                                    3) Technical indicator signals
                                    4) Notable patterns"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }],
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            self._handle_request_error("capture_and_analyze_chart", e)
            return None
        finally:
            # 임시 파일 정리
            if os.path.exists(screenshot_path):
                os.remove(screenshot_path)

    def get_ai_analysis(self, analysis_data: Dict) -> Optional[Dict]:
        """AI 분석 및 매매 신호 생성"""
        try:
            # 차트 및 유튜브 분석
            chart_analysis = self.capture_and_analyze_chart()
            youtube_analysis = youtube_api.get_youtube_analysis(self.youtube_channels)

            # 과거 반성 일기 분석 추가
            past_reflections = self.dbmanager.get_reflection_history(5)

            # AI 프롬프트 데이터 구성
            ai_prompt_data = self._build_ai_prompt_data(analysis_data, chart_analysis, youtube_analysis, past_reflections)

            # AI 분석 실행
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": self._get_ai_system_prompt()},
                    {"role": "user", "content": f"Market data for analysis: {json.dumps(ai_prompt_data)}"}
                ],
                response_format=self._get_response_format()
            )

            result = json.loads(response.choices[0].message.content)
            self._print_ai_results(result)
            return result

        except Exception as e:
            self._handle_request_error("get_ai_analysis", e)
            return None

    def _build_ai_prompt_data(self, analysis_data: Dict, chart_analysis: str, youtube_analysis: str, past_reflections: str) -> Dict:
        """AI 프롬프트 데이터 구성"""
        orderbook = analysis_data['orderbook_data']
        return {
            "current_investment_status": analysis_data['current_status'],
            "orderbook_info": {
                "timestamp": orderbook['timestamp'],
                "total_ask_size": orderbook['total_ask_size'],
                "total_bid_size": orderbook['total_bid_size'],
                "best_ask_price": orderbook['ask_price'][0],
                "best_bid_price": orderbook['bid_price'][0],
                "ask_bid_spread": orderbook['ask_price'][0] - orderbook['bid_price'][0],
            },
            "moving_averages_analysis": analysis_data['ohlcv_data']['ma_analysis'],
            "ohlcv": analysis_data['ohlcv_data'],
            "fear_greed": analysis_data['fear_greed'],
            "news": analysis_data['news'],
            "chart_analysis": chart_analysis,
            "youtube_analysis": youtube_analysis,
            "past_reflections": past_reflections
        }

    def _get_ai_system_prompt(self) -> str:
        """AI 시스템 프롬프트 반환"""
        return """You are an expert Bitcoin trading analyst. Analyze the provided data and make a trading decision.

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
        9. Visual Chart Analysis Results
        10. YouTube Content Analysis

        Please consider the following key points:
        - Fear & Greed Index below 20 (Extreme Fear) may present buying opportunities
        - Fear & Greed Index above 80 (Extreme Greed) may present selling opportunities
        - The trend of the Fear & Greed Index is also a crucial indicator 

        Pay special attention to:
        - Golden Cross (5MA crossing above 20MA) = Bullish signal
        - Dead Cross (5MA crossing below 20MA) = Bearish signal
        - Price position relative to moving averages
        - Moving average trend directions
        - Support and resistance levels from MAs

        Provide your decision in JSON format with detailed reasoning."""

    def _get_response_format(self) -> Dict:
        """AI 응답 형식 정의"""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "trading_decision",
                "description": "Trading decision based on market analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "percentage": {
                            "type": "integer",
                            "description": "Percentage value for the trading decision (0-100)"
                        },
                        "decision": {
                            "type": "string",
                            "description": "Trading decision to make",
                            "enum": ['buy', 'sell', 'hold']
                        },
                        "confidence_score": {
                            "type": "integer",
                            "description": "Confidence level of the trading decision (0-100)"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Detailed explanation for the decision"
                        },
                        "reflection_based_adjustments":{
                            "type": "object",
                            "properties": {
                                "risk_adjustment": {"type": "string"},
                                "strategy_improvement": {"type": "string"},
                                "confidence_factors": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["risk_adjustment", "strategy_improvement", "confidence_factors"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["percentage", "decision", "confidence_score", "reason", "reflection_based_adjustments"],
                    "additionalProperties": False
                }
            }
        }

    def _print_ai_results(self, result: Dict) -> None:
        """AI 분석 결과 출력"""
        print("=== AI 투자 판단 결과 ===")
        print(f"### AI Decision: {result['decision'].upper()} ###")
        print(f"### Confidence: {result.get('confidence_score', 'N/A')}/100 ###")
        print(f"### Reason: {result['reason']} ###")
        print("=" * 50)

    def execute_trade(self, decision: str, percentage: int, confidence_score: int, fear_greed_value: int,
                      reason: str) -> None:
        """매매 실행 (최적화된 버전)"""
        print(f"fear_greed_value(공포탐욕지수): {fear_greed_value}")

        try:
            if confidence_score < self.config.HIGH_CONFIDENCE_THRESHOLD:
                print(f"### 신뢰도 부족으로 거래 미실행 (현재: {confidence_score}, 필요: {self.config.HIGH_CONFIDENCE_THRESHOLD}) ###")


            trade_ratio = self._calculate_trade_ratio(percentage, decision, fear_greed_value)

            if decision == 'buy':
                self._execute_buy_order(trade_ratio, percentage, fear_greed_value)
            elif decision == 'sell':
                self._execute_sell_order(trade_ratio, percentage, fear_greed_value)

            # 거래 기록
            self._record_trade_result(decision.upper(), percentage, reason)

        except Exception as e:
            self._handle_request_error("execute_trade", e)

    def _calculate_trade_ratio(self, percentage: int, decision: str, fear_greed_value: int) -> float:
        """공포탐욕지수에 따른 거래 비율 계산"""
        base_ratio = percentage / 100.0
        config = self.config

        if decision == 'buy':
            if fear_greed_value <= config.EXTREME_FEAR_THRESHOLD:
                return min(base_ratio * config.FEAR_BUY_MULTIPLIER, 1.0)
            elif fear_greed_value <= config.MODERATE_FEAR_THRESHOLD:
                return base_ratio * config.MODERATE_MULTIPLIER
            else:
                return base_ratio * config.GREED_BUY_MULTIPLIER
        else:  # sell
            if fear_greed_value >= config.EXTREME_GREED_THRESHOLD:
                return min(base_ratio * config.GREED_SELL_MULTIPLIER, 1.0)
            elif fear_greed_value >= config.MODERATE_GREED_THRESHOLD:
                return base_ratio * config.MODERATE_MULTIPLIER
            else:
                return base_ratio * config.FEAR_SELL_MULTIPLIER

    def _execute_buy_order(self, trade_ratio: float, original_percentage: int, fear_greed_value: int) -> None:
        """매수 주문 실행"""
        krw = self.upbit.get_balance("KRW")
        if krw <= self.config.MIN_TRADE_AMOUNT:
            print(
                f"### Buy Order Failed: Insufficient KRW (보유: {krw:,.0f}원, 필요: {self.config.MIN_TRADE_AMOUNT:,}원 이상) ###")
            return

        order_amount = krw * trade_ratio
        if order_amount >= self.config.MIN_TRADE_AMOUNT:
            order_result = self.upbit.buy_market_order(self.ticker, order_amount)
            self._print_order_result("Buy", order_amount, trade_ratio, original_percentage, fear_greed_value,
                                     order_result)

    def _execute_sell_order(self, trade_ratio: float, original_percentage: int, fear_greed_value: int) -> None:
        """매도 주문 실행"""
        btc = self.upbit.get_balance(self.ticker)
        current_price = pyupbit.get_current_price(self.ticker)
        btc_value = btc * current_price

        if btc_value <= self.config.MIN_TRADE_AMOUNT:
            print(f"### Sell Order Failed: Insufficient BTC (보유: {btc:.8f} BTC, 가치: {btc_value:,.0f}원) ###")
            return

        sell_amount = btc * trade_ratio
        if sell_amount * current_price >= self.config.MIN_TRADE_AMOUNT:
            order_result = self.upbit.sell_market_order(self.ticker, sell_amount)
            self._print_order_result("Sell", sell_amount, trade_ratio, original_percentage, fear_greed_value,
                                     order_result, is_btc=True)

    def _print_order_result(self, order_type: str, amount: float, trade_ratio: float,
                            original_percentage: int, fear_greed_value: int, order_result: Dict,
                            is_btc: bool = False) -> None:
        """주문 결과 출력"""
        print(f"\n=== {order_type} Order Executed ===")
        if is_btc:
            print(f"Trade Amount: {amount:.8f} BTC (약 {trade_ratio * 100:.1f}%)")
        else:
            print(f"Trade Amount: {amount:,.0f} KRW ({trade_ratio * 100:.1f}%)")
        print(f"Original AI Suggestion: {original_percentage}%")
        print(f"Fear & Greed Index: {fear_greed_value}")
        print(f"주문 결과: {json.dumps(order_result, indent=2)}")

    def _record_trade_result(self, decision: str, percentage: int, reason: str) -> None:
        """거래 결과 기록"""
        try:
            btc_balance = float(self.upbit.get_balance(self.ticker))
            krw_balance = float(self.upbit.get_balance("KRW"))
            btc_avg_buy_price = float(self.upbit.get_avg_buy_price(self.ticker))
            btc_krw_price = float(pyupbit.get_current_price(self.ticker))

            self.dbmanager.record_trade(decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price,
                                   btc_krw_price)
        except Exception as e:
            self._handle_request_error("_record_trade_result", e)


def ai_trading():
    """AI 거래 메인 함수"""
    try:
        collector = CryptoDataCollector("KRW-BTC")

        # 과거 거래 분석 및 반성 수행
        reflection = collector.analyze_past_decisions()
        if reflection:
            print("\n=== Trading Reflection ===")
            print(json.dumps(reflection, indent=2))

        # 데이터 수집
        print("\n=== 데이터 수집 시작 ===")
        data_collection_tasks = [
            ("Current Investment Status", collector.get_current_status),
            ("Current Orderbook", collector.get_orderbook_data),
            ("OHLCV Data", collector.get_ohlcv_data),
            ("Fear Greed Data", collector.get_fear_greed_index),
            ("Crypto News", collector.get_crypto_news)
        ]

        collected_data = {}
        for task_name, task_func in data_collection_tasks:
            print(f"\n=== {task_name} start ===")
            result = task_func()
            if result is None:
                print(f"### {task_name} 수집 실패 ###")
                return
            collected_data[task_name.lower().replace(' ', '_')] = result
            print(f"=== {task_name} end ===")

        # AI 분석 및 거래 실행
        analysis_data = {
            "current_status": collected_data["current_investment_status"],
            "orderbook_data": collected_data["current_orderbook"],
            "ohlcv_data": collected_data["ohlcv_data"],
            "fear_greed": collected_data["fear_greed_data"],
            "news": collected_data["crypto_news"]
        }

        ai_result = collector.get_ai_analysis(analysis_data)
        if ai_result:
            print("\n=== AI Analysis Results ===")
            print(json.dumps(ai_result, indent=2))

            # 반성 기반 조정 사항 출력
            print("\n=== Reflection-based Adjustments ===")
            print(json.dumps(ai_result["reflection_based_adjustments"], indent=2))

            collector.execute_trade(
                ai_result['decision'],
                ai_result['percentage'],
                ai_result['confidence_score'],
                collected_data["fear_greed_data"]['current']['value'],
                ai_result['reason']
            )

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        traceback.print_exc()
        print("10초 후 다시 시도합니다...")


def main():
    """메인 실행 함수"""
    print("Bitcoin AI Trading Bot 시작")
    print("Ctrl+C로 종료할 수 있습니다.")
    print("=" * 70)

    try:
        print("Starting Bitcoin Trading Bot...")
        while True:
            ai_trading()
            print("10분 후 다음 분석을 시작합니다...")
            time.sleep(600)  # 10분 간격으로 실행


    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 종료되었습니다.")
    except Exception as e:
        print(f"예상치 못한 오류: {str(e)}")
        traceback.print_exc()
        time.sleep(60)  # 에러 발생 시에도 1분 대기

if __name__ == "__main__":
    main()