import json
import traceback
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi


# 유튜브 영상 자막 분석
def get_youtube_analysis(channelsId):
    try:
        client = OpenAI()

        all_transcripts = []
        for video_id in channelsId:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                text = ' '.join(entry['text'] for entry in transcript)
                all_transcripts.append({
                    "video_id": video_id,
                    "content": text
                })
            except Exception as e:
                print(f"Error getting transcript for video {video_id}: {e}")
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