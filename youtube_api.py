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
                transcript = transcripts.find_transcript(['en']).fetch()  # ✅ fetch() 사용하여 JSON 직렬화 가능 객체 반환
                text = ' '.join(entry.text for entry in transcript)
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
                    "role" : "system", "content":"You are a helpful assistant that responds in JSON format."
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