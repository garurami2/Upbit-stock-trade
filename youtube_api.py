import json
import traceback
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


# 유튜브 영상 자막 분석
def get_youtube_analysis(channelsId):
    try:
        client = OpenAI()

        all_transcripts = []
        for video_id in channelsId:
            try:
                transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                try:
                    transcript = transcripts.find_transcript(['ko'])
                except NoTranscriptFound:
                    transcript = transcripts.find_generated_transcript(['ko'])  # ✅ fetch() 사용하여 JSON 직렬화 가능 객체 반환
                txt_data = transcript.fetch()
                if not txt_data:
                    print(f"[SKIP] 자막 데이터가 비어 있음: {video_id}")
                    continue
                text = ' '.join(entry.text for entry in txt_data)
                all_transcripts.append({
                    "video_id": video_id,
                    "content": text
                })
            except TranscriptsDisabled:
                print(f"자막이 비활성화된 영상입니다: {video_id}")
                traceback.print_exc()
                continue
            except NoTranscriptFound:
                print(f"'{video_id}' 영상에 해당 언어 자막이 없습니다.")
                traceback.print_exc()
                continue
            except Exception as e:
                print(f"예상치 못한 오류: {e}")
                traceback.print_exc()
                continue

        if not all_transcripts:
            return None

        # 한국어 분석을 위한 시스템 메시지
        system_message = """
                            You are an expert cryptocurrency trading analyst. Analyze Korean YouTube content related to cryptocurrency
                            trading and provide insights. Focus on analyzing these key aspets from the korean transcripts:
                            1. Trading Strategy
                            - Entry/exit points
                            - Risk management methods
                            - Trading patterns
                            
                            2. Market Analysis
                            - Market sentiment
                            - Important price levels
                            - Potential scenarios
                            
                            3. Risk Factors
                            - Market risks
                            - Technical risks
                            - External risks
                            
                            4. Technical Analytics
                            - Technical indicators
                            - Chart patterns
                            - Key price levels
                            
                            5. Market Impact Factors
                            - Econonmic factors
                            - News and events
                            - Market trends
                            
                            Provide analysis in JSON format with confidence scores.
                         """

        # OpenAI API를 사용하여 자막 내용 분석
        analysis_prompt = """
                                Analyze the cryptocurrency-related video transcripts and provide a JSON response with the following keys:
                                - key_insights: array of market insights and predictions
                                - sentiment: "bullish" or "bearish"
                                - main_arguments: array of key arguments
                                - technical_analysis: description of technical analysis mentions
                                - risk_factors: array of potential risks
                                
                                Respond ONLY in strict JSON format.
                          """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role" : "system", "content": system_message
                },
                {
                    "role" : "user", "content": f"{analysis_prompt}\n\n. Transcripts: {json.dumps(all_transcripts)}. Make sure to include json in your response."
                }
            ],
            response_format={"type": "json_object"}
        )

        # JSON 디코딩
        try:
            analysis_result = json.loads(response.choices[0].message.content)
            print(f"analysis_result: {analysis_result}")
            return analysis_result
        except Exception as e:
            print(f"JSON Parsing Error: {e}")
            print("Original response:", response.choices[0].message.content)
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"Error in get_youtube_analysis: {e}")
        traceback.print_exc()
        return None