import os
import requests
from openai import OpenAI
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydub import AudioSegment
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key="")

CLIENT_ID = os.getenv('CLIENT_ID')
print(CLIENT_ID)
AUTH_TOKEN = os.getenv('AUTH_TOKEN')
print(AUTH_TOKEN)


class URLRequest(BaseModel):
    url: str


@app.post("/soap-notes")
async def process_audio(request: Request, url_request: URLRequest):
    client_id = request.headers.get('client_id')
    auth_token = request.headers.get('auth_token')

    if not client_id or not auth_token:
        raise HTTPException(status_code=401, detail="Error: Client ID or Authorization token is missing")
    if client_id != CLIENT_ID or auth_token != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Error: Authentication failed")

    url = url_request.url
    file_path = await download_file(url)
    chunk_length = 600000
    if not file_path:
        raise HTTPException(status_code=400, detail="Error: File download failed")
    try:
        audio = AudioSegment.from_file(file_path)
    except Exception:
        raise HTTPException(status_code=400, detail="Error loading audio file")
    duration = len(audio)
    print(duration)
    text = ""
    all_segments = []
    last_end = 0
    try:
        if duration <= chunk_length:
            transcription = transcribe_audio(file_path)
            if transcription.language == "english":
                all_segments = transcription.segments
                text = transcription.text
            else:
                text = translate_audio(file_path)

        else:
            for start in range(0, duration, chunk_length):
                chunk = audio[start:start + chunk_length]
                chunk_name = f"temp_chunk_{start // chunk_length}.mp3"
                chunk.export(chunk_name, format="mp3")
                transcription = transcribe_audio(chunk_name)
                if transcription.language == "english":
                    for segment in transcription.segments:
                        # Use attribute access instead of subscript
                        segment.start += last_end
                        segment.end += last_end
                        keys_to_exclude = {"id", "seek", "tokens", "temperature", "avg_logprob", "compression_ratio",
                                           "no_speech_prob"}
                        filtered_segment = {k: getattr(segment, k) for k in segment.__dict__ if
                                            k not in keys_to_exclude}
                        all_segments.append(filtered_segment)
                    if transcription.segments:
                        last_end = transcription.segments[-1].end

                    text += transcription.text + " "
                else:
                    text += translate_audio(chunk_name) + " "
                os.remove(chunk_name)

        response = generate_soap_notes(text)
        data = json.loads(response)

        soap_note = data["soap_note"]
        cpt_codes = data["cpt_codes"]
        modifiers = data["modifiers"]
        tags = data["tags"]
        return {"soap_note": soap_note, "tags": tags, 'cpt_codes': cpt_codes,
                'modifiers': modifiers, "transcription": all_segments}
    except Exception as e:
        print(f"Error processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing request")


async def download_file(url: str, tmp_folder: str = "tmp", file_name: str = "audio_file.mp3"):
    os.makedirs(tmp_folder, exist_ok=True)
    file_path = os.path.join(tmp_folder, file_name)
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded and saved to {file_path}")
        return file_path
    except requests.exceptions.RequestException as e:
        print(f"Failed to download file: {e}")
        return None


def transcribe_audio(file_path: str):
    try:
        with open(file_path, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        return transcription
    except Exception as e:
        print(f"Transcription error: {e}")


def translate_audio(file_path: str):
    try:
        with open(file_path, 'rb') as audio_file:
            transcription = client.audio.translations.create(
                model="whisper-1",
                file=audio_file,
                prompt="If the audio contains another language, it should be translated into English. The final "
                       "output must be in English. Ignor the silence")
        return transcription.text
    except Exception as e:
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Error during transcription")


def generate_soap_notes(transcription_text: str):
    prompt = """You are an expert American physician.
            Generate a separate SOAP note from the following transcript.
            The SOAP note should be concise and utilize bullet point format.
            Include ICD-10 and CPT codes in parentheses next to the diagnosis and services.
            Generate a detailed Subjective section, with all diagnoses separated.
            Include plans for each assessment.
            Include all relevant information discussed in the transcript.
            Use double asterisks for all the headings.
            Write Subjective, Objective, Assessment, and Plan on separate lines using the newline character (/n)."""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": transcription_text}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "soapNote_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "soap_note": {
                                "description": "Generated SOAP note from the transcript.",
                                "type": "string"
                            },
                            "cpt_codes": {
                                "description": "List of CPT codes related to the services mentioned in the transcript.",
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "modifiers": {
                                "description": "List of modifiers related to the services mentioned in the transcript.",
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "tags": {
                                "description": "List of diseases or disorders mentioned in the transcript.",
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": ["soap_note", "cpt_codes", "modifiers"],
                        "additionalProperties": False
                    }
                }
            }
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating SOAP notes: {e}")
        raise HTTPException(status_code=500, detail="Error generating SOAP notes")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))