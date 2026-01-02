# save as vote_ab.py
from openai import OpenAI
import base64, pathlib, mimetypes

def data_url(path):
    p = pathlib.Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"

imgB = "/data/wenjie_jacky_mo/imagenet/1000_img/n01491361__ILSVRC2012_val_00003204.JPEG"
imgA = "/data/wenjie_jacky_mo/imagenet/1000_img/n02085620__ILSVRC2012_val_00027452.JPEG"

client = OpenAI(api_key="0", base_url="http://127.0.0.1:8000/v1")

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text":
         "Which image do you prefer looking at?\nOption A:"},
        {"type": "image_url", "image_url": {"url": data_url(imgA)}},
        {"type": "text", "text": "Option B:"},
        {"type": "image_url", "image_url": {"url": data_url(imgB)}},
        {"type": "text", "text": 'Please respond with only "A" or "B".'}
    ]
}]

resp = client.chat.completions.create(
    model="/data/wenjie_jacky_mo/models/Qwen2.5-VL-72B-Instruct",  # 确保和服务端加载的一致
    messages=messages,
    temperature=1.0,
    max_tokens=100
)

with open("output.txt", "w") as f:
    f.write(resp.choices[0].message.content.strip() + "\n")
print(resp.choices[0].message.content.strip())
