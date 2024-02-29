import time
import requests
import json
import base64
import pdb

start = time.time()
host = "https://a3juid7agss9ns-8888.proxy.runpod.net"

def inpaint_outpaint(params: dict) -> dict:
    """
    example for inpaint outpaint v2
    """
    response = requests.post(url=f"{host}/v2/generation/image-prompt",
                        data=json.dumps(params),
                        headers={"Content-Type": "application/json"})
    pdb.set_trace()
    return response.json()


source = open("images/1.png", "rb").read()
mask = open("images/light.png", "rb").read()
cn = open("images/out.png","rb").read()
result = inpaint_outpaint(params={
                            "prompt": "perfect teeth, shiny teeth,veneer teeth, super white teeth, perfect shape of teeth",
                            "negative_prompt" : "imperfect teeth",
                            "style_selections":["Fooocus V2,Fooocus Enhance,Fooocus Sharp, Fooocus Negative"],
                            "input_image": "https://teethe.s3.ap-south-1.amazonaws.com/image_1708207213_9AHcEBev.png",
                            "input_mask": "https://teethe.s3.ap-south-1.amazonaws.com/image_1708207222_Nam4eFsu.png",
                            "image_prompts": [
                            #   {
                            # "cn_img": "https://teethe.s3.ap-south-1.amazonaws.com/image_1708207220_Xw0WrwUw.png",
                            # "cn_stop": 0.6,
                            # "cn_weight": 1,
                            #  "cn_type": "PyraCanny"
                            #   }
                              ],
                            "require_base64":True,
                            "async_process": False})
# print(json.dumps(result, indent=4, ensure_ascii=False))
pdb.set_trace()
base = result[0]['base64']
# Decode the base64 string
image_data = base64.b64decode(base)
end = time.time()
# Write the decoded data to a file
with open("image2.png", "wb") as f:
    f.write(image_data)
final = end - start
print(final)    