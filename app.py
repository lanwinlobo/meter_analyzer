from fastapi import FastAPI, File, UploadFile 
from fastapi.responses import JSONResponse
import openai
import base64
from PIL import Image
from ultralytics import YOLO
import io
import tempfile
import re 
import json
import os

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")
model = "gpt-4o"

model1 = YOLO("https://huggingface.co/lanwinlobo/973update.pt/resolve/main/973update.pt")
model2 = YOLO("https://huggingface.co/lanwinlobo/yolo_meter/blob/main/model_23_01_2023_yolov9.pt")

class_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
    8: '8', 9: '9', 10: 'A', 11: 'R', 12: 'V', 13: 'W', 14: 'd',
    15: '.', 16: 'h', 17: 'k', 18: 'r'
}
units_set = {"A", "k", "V", "W", "d", "h", "r", "R"}

def encode_image_base64(file: bytes):
    return base64.b64encode(file).decode("utf-8")

@app.post("/analyze-meter/")
async def analyze_meter(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        base_64_image = encode_image_base64(file_bytes)
        
        prompt = """..."""  # (Keep the prompt as-is)

        chat_response = openai.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base_64_image}"}}
                ]
            }]
        )
        raw_content = chat_response.choices[0].message.content.strip()
        json_str = re.sub(r"^```json|```$", "", raw_content).strip("` \n")
        meter_info = json.loads(json_str)

        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            img.save(tmp_file.name)
            results1 = model1(tmp_file.name)

        cropped_box_img = None
        for result in results1:
            for box in result.boxes:
                if box.cls[0].item() == 19:
                    bbox = box.xyxy[0].tolist()
                    cropped_box_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                    break
            if cropped_box_img:
                break

        if cropped_box_img:
            results2 = model2(cropped_box_img)
            predictions = []
            for result in results2:
                for box in result.boxes:
                    predictions.append({
                        "class_id": int(box.cls[0].item()),
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": box.conf[0].item()
                    })

            predictions = sorted(predictions, key=lambda x: x['bbox'][0])
            mapped_values = [class_mapping[p["class_id"]] for p in predictions]
            reading = ''.join(v for v in mapped_values if v not in units_set)
            unit = ''.join(v for v in mapped_values if v in units_set) or "kwh"
        else:
            reading = "NA"
            unit = "NA"

        reading_info = {
            "reading": reading,
            "unit": unit
        }

        return JSONResponse(content={
            "meter_info": meter_info,
            "reading_info": reading_info
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)})
