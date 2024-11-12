import os

import httpx
import requests
from openai import OpenAI
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydub import AudioSegment
import pusher
import json
import re

pusher_client = pusher.Pusher(
    app_id='1835059',
    key='f5d7f359a1682ae86bd6',
    secret='9a0384a6d345adb01574',
    cluster='mt1',
    ssl=True
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=""
)

CLIENT_ID = os.getenv('CLIENT_ID')
print(CLIENT_ID)
AUTH_TOKEN = os.getenv('AUTH_TOKEN')
print(AUTH_TOKEN)


class URLRequest(BaseModel):
    url: str
    id: str


@app.post("/soap-notes")
async def process_audio(request: Request, url_request: URLRequest):
    client_id = request.headers.get('client_id')
    auth_token = request.headers.get('auth_token')

    if not client_id or not auth_token:
        raise HTTPException(status_code=401, detail="Error: Client ID or Authorization token is missing")
    if client_id != CLIENT_ID or auth_token != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Error: Authentication failed")

    url = url_request.url
    id = url_request.id
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

    def filter_segment(segment):
        keys_to_exclude = {"id", "seek", "tokens", "temperature", "avg_logprob", "compression_ratio", "no_speech_prob"}
        return {k: getattr(segment, k) for k in segment.__dict__ if k not in keys_to_exclude}

    try:
        if duration <= chunk_length:
            transcription = transcribe_audio(file_path)
            print(transcription)
            # Translate if not in English
            if transcription.language != "english":
                text = translate_audio(file_path)
            else:
                text = transcription.text
                all_segments = [filter_segment(segment) for segment in transcription.segments]

        else:
            for start in range(0, duration, chunk_length):
                chunk = audio[start:start + chunk_length]
                chunk_name = f"temp_chunk_{start // chunk_length}.mp3"
                chunk.export(chunk_name, format="mp3")
                transcription = transcribe_audio(chunk_name)

                # Check if the chunk needs translation
                if transcription.language != "english":
                    text += translate_audio(chunk_name) + " "
                else:
                    for segment in transcription.segments:
                        segment.start += last_end
                        segment.end += last_end
                        all_segments.append(filter_segment(segment))
                    if transcription.segments:
                        last_end = transcription.segments[-1].end
                    text += transcription.text + " "
                os.remove(chunk_name)

        response = generate_soap_notes(text)
        data = json.loads(response)

        soap_note = data["soap_note"]
        cpt_codes = data["cpt_codes"]
        modifiers = data["modifiers"]
        tags = data["tags"]
        icd_10_codes = data["icd_10_codes"]

        valid_transcription = valid_json(str(all_segments))

        await send_soap_note(id, soap_note, tags, cpt_codes, modifiers, valid_transcription, icd_10_codes)
    except Exception as e:
        print(f"Error processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing request")



def valid_json(json_str):
    try:
        # Step 1: Replace single quotes around keys
        formatted_str = re.sub(r"(?<=\{|\s|,)'([^']+)':", r'"\1":', json_str)

        # Step 2: Replace single quotes around values (excluding 'text' fields)
        formatted_str = re.sub(
            r'(?<!text):\s\'([^\']*)\'(?!\s*[,\}])', r': "\1"', formatted_str
        )

        # Step 3: Ensure 'text' values are enclosed in double quotes if they aren’t already
        # This pattern specifically looks for 'text' field values and ensures double quotes
        formatted_str = re.sub(
            r'"text":\s*\'([^\']*)\'', r'"text": "\1"', formatted_str
        )
        return formatted_str
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None


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
        print(f"Error in download_file: {e}")
        raise HTTPException(status_code=500, detail="Error downloading file")


async def send_soap_note(id, soap_note, tags, cpt_codes, modifiers, valid_transcription, icd_10_codes):
    try:
        url = 'https://azzportal.com/admin/public/api/v2/add-soapnotes'
        data = {
            "id": id,
            "soap_note": str(soap_note),
            "tags": str(tags),
            "cpt_codes": str(cpt_codes),
            "modifiers": str(modifiers),
            "transcription": valid_transcription,
            "icd_10_codes": icd_10_codes
        }

        headers = {
            'client_id': CLIENT_ID,
            'auth_token': AUTH_TOKEN
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
        else:
            response_data = {
                "error": "Failed to send SOAP note",
                "status": response.status_code,
                "message": response.text
            }
        pusher_client.trigger('soap-note-channel', 'soap-note-event', response_data)
        print(response_data)
        return response_data
    except Exception as e:
        print(f"Error in send_soap_note: {str(e)}")
        raise HTTPException(status_code=500, detail="Error sending SOAP note")


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
        raise HTTPException(status_code=500, detail="Error during transcribing")


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
        raise HTTPException(status_code=500, detail="Error during translation")


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
                                "description": "List of CPT codes related to the services mentioned in the transcript, along with reasons.",
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "code": {
                                            "type": "string",
                                            "description": "CPT code."
                                        },
                                        "reason": {
                                            "type": "string",
                                            "description": "Reason for the CPT code."
                                        }
                                    },
                                    "required": ["code", "reason"]
                                }
                            },
                            "modifiers": {
                                "description": "List of modifiers related to the services mentioned in the transcript, along with reasons.",
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "modifier": {
                                            "type": "string",
                                            "description": "Modifier code."
                                        },
                                        "reason": {
                                            "type": "string",
                                            "description": "Reason for the modifier."
                                        }
                                    },
                                    "required": ["modifier", "reason"]
                                }
                            },
                            "tags": {
                                "description": "List of diseases or disorders mentioned in the transcript.",
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "icd_10_codes": {
                                "description": "List of ICD-10 codes related to the diseases or disorders mentioned in the transcript, with their names.",
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "code": {
                                            "type": "string",
                                            "description": "ICD-10 code."
                                        },
                                        "name": {
                                            "type": "string",
                                            "description": "Name of the disease or disorder for the ICD-10 code."
                                        }
                                    },
                                    "required": ["code", "name"]
                                }
                            }
                        },
                        "required": ["soap_note", "cpt_codes", "modifiers", "icd_10_codes"],
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

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
