import datetime
import io
import pathlib
import sys
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import requests
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from PIL import Image
from transformers import AutoTokenizer, VitsModel

app = FastAPI()

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", force_reload=True)


@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")

    img_np = np.array(image)

    results = model(img_np)

    results.render()

    img_with_boxes = Image.fromarray(results.ims[0])

    img_byte_arr = io.BytesIO()
    img_with_boxes.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/jpeg")


USERNAME = "ahmed_mohib"
PASSWORD = "M00h4qLkMn"


LOCATION = "30.1864,71.4886"

PHYTOS_THRESHOLDS = {"Low Risk": 30, "Medium Risk": 60}


def number_to_urdu(n):
    urdu_numbers = {
        0: "صفر",
        1: "ایک",
        2: "دو",
        3: "تین",
        4: "چار",
        5: "پانچ",
        6: "چھے",
        7: "سات",
        8: "آٹھ",
        9: "نو",
        10: "دس",
        11: "گیارہ",
        12: "بارہ",
        13: "تیرہ",
        14: "چودہ",
        15: "پندرہ",
        16: "سولہ",
        17: "سترہ",
        18: "اٹھارہ",
        19: "انیس",
        20: "بیس",
        21: "اکیس",
        22: "بائیس",
        23: "تئیس",
        24: "چوبیس",
        25: "پچیس",
        26: "چھبیس",
        27: "ستائیس",
        28: "اٹھائیس",
        29: "انتیس",
        30: "تیس",
        31: "اکتیس",
        32: "بتیس",
        33: "تینتیس",
        34: "چونتیس",
        35: "پینتیس",
        36: "چھتیس",
        37: "ستتیس",
        38: "اڑتیس",
        39: "انتالیس",
        40: "چالیس",
        48: "اڑتالیس",
    }
    return urdu_numbers.get(n, str(n))


def get_phytophthora_insights():
    today = datetime.now(timezone.utc)
    start_date = today - timedelta(days=10)
    end_date = today + timedelta(days=10)

    print(f"Fetching Phytophthora Negative Prognosis data from {start_date.date()} to {end_date.date()}...\n")

    parameter = "phytophthora_negative:idx"
    interval = "P1D"

    url = (
        f"https://api.meteomatics.com/"
        f"{start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}--{end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}:"
        f"{interval}/{parameter}/{LOCATION}/json"
    )

    try:
        response = requests.get(url, auth=(USERNAME, PASSWORD))
        response.raise_for_status()
        prognosis_data = response.json()
    except requests.exceptions.RequestException as e:
        return f"میٹیومیٹکس API سے ڈیٹا حاصل کرنے میں خرابی ہوئی: {e}"

    try:
        data_entries = prognosis_data["data"][0]["coordinates"][0]["dates"]
    except (KeyError, IndexError) as e:
        return f"ڈیٹا پارس کرنے میں خرابی ہوئی: {e}"

    risk_counts = {"Low Risk": 0, "Medium Risk": 0, "High Risk": 0}

    for entry in data_entries:
        value = entry.get("value")
        if value is None:
            continue

        if value < PHYTOS_THRESHOLDS["Low Risk"]:
            risk_counts["Low Risk"] += 1
        elif PHYTOS_THRESHOLDS["Low Risk"] <= value <= PHYTOS_THRESHOLDS["Medium Risk"]:
            risk_counts["Medium Risk"] += 1
        else:
            risk_counts["High Risk"] += 1

    total_days = sum(risk_counts.values())
    if total_days == 0:
        return "تجزیہ کے لئے کوئی درست ڈیٹا دستیاب نہیں ہے۔"

    low_risk_days = number_to_urdu(risk_counts["Low Risk"])
    medium_risk_days = number_to_urdu(risk_counts["Medium Risk"])
    high_risk_days = number_to_urdu(risk_counts["High Risk"])

    summary = (
        f"گزشتہ دس دنوں سے لے کر اگلے دس دنوں تک، آلووں میں فائیٹوفیتھرا کے امراض کا خطرہ مندرجہ ذیل ہے: "
        f"{low_risk_days} دن کم خطرے کے ساتھ، {medium_risk_days} دن درمیانے خطرے کے ساتھ، "
        f"اور {high_risk_days} دن زیادہ خطرے کے ساتھ۔ "
    )

    if risk_counts["High Risk"] > 0:
        summary += (
            "زیادہ خطرے والے دنوں کے لئے، فصلوں کی حفاظت کے لئے اگلے اڑتالیس گھنٹوں کے اندر فنگسائڈ لگانے کی سفارش کی جاتی ہے۔ "
            "درمیانے خطرے والے دنوں پر احتیاطی اقدامات جیسے باقاعدہ نگرانی اور ممکنہ فنگسائڈ کا استعمال کرنے پر غور کریں۔ "
            "کم خطرے والے دنوں پر باقاعدہ نگرانی جاری رکھیں تاکہ کسی بھی ممکنہ مسئلے کی جلد شناخت ہو سکے۔"
        )
    else:
        summary += "مجموعی طور پر، آلووں میں فائیٹوفیتھرا کا خطرہ قابل انتظام ہے۔ صحت مند فصلوں کو برقرار رکھنے کے لئے باقاعدہ نگرانی جاری رکھیں اور حسب ضرورت احتیاطی اقدامات کریں۔"

    return summary


# Initialize the TTS model and tokenizer
try:
    tts_model = VitsModel.from_pretrained("facebook/mms-tts-urd-script_arabic")
    tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-urd-script_arabic")
    tts_model.eval()
except Exception as e:
    tts_model = None
    tts_tokenizer = None
    print(f"Error loading TTS model: {e}")


@app.get(
    "/phytophthora_insights", response_class=PlainTextResponse, summary="فائیٹوفیتھرا کے خطرے کی بصیرتیں فراہم کریں"
)
async def phytophthora_insights():
    insights = get_phytophthora_insights()
    return insights


@app.post("/text_to_audio", summary="Convert Urdu text to audio")
async def text_to_audio(request: Request):
    """
    Converts provided Urdu text to audio and returns the audio file.

    Args:
        request (Request): The incoming request containing Urdu text in JSON.

    Returns:
        StreamingResponse: The generated audio file in WAV format.
    """
    if tts_model is None or tts_tokenizer is None:
        raise HTTPException(status_code=500, detail="TTS model is not loaded.")

    try:
        data = await request.json()
        text = data.get("text")
        if not text:
            raise HTTPException(status_code=400, detail="No text provided.")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON input.")

    try:
        # Tokenize the input text
        inputs = tts_tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            # Generate the waveform
            output = tts_model(**inputs).waveform

        # Squeeze to remove batch dimension and convert to numpy
        waveform = output.squeeze().cpu().numpy()

        # Write the waveform to a bytes buffer as WAV
        buffer = io.BytesIO()
        sf.write(buffer, waveform, samplerate=22050, format="WAV")
        buffer.seek(0)

        return StreamingResponse(
            buffer, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=output.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {e}")


@app.get("/insights_audio", summary="Get Phytophthora insights as audio")
async def insights_audio():
    """
    Fetches Phytophthora insights and returns the audio version.

    Returns:
        StreamingResponse: The generated audio file in WAV format.
    """
    insights = get_phytophthora_insights()

    if "خرابی" in insights or "درست ڈیٹا" in insights:
        raise HTTPException(status_code=500, detail=insights)

    if tts_model is None or tts_tokenizer is None:
        raise HTTPException(status_code=500, detail="TTS model is not loaded.")

    try:
        # Tokenize the insights text
        inputs = tts_tokenizer(insights, return_tensors="pt")

        with torch.no_grad():
            # Generate the waveform
            output = tts_model(**inputs).waveform

        # Squeeze to remove batch dimension and convert to numpy
        waveform = output.squeeze().cpu().numpy()

        # Write the waveform to a bytes buffer as WAV
        buffer = io.BytesIO()
        sf.write(buffer, waveform, samplerate=22050, format="WAV")
        buffer.seek(0)

        return StreamingResponse(
            buffer, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=insights.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
