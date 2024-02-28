import os
import json

def find_jpg_images(root_folder):
    jpg_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))

    # sort the list of files
    jpg_files.sort()
    return jpg_files

def generate_commands(jpg_files, output_script):
    with open(output_script, 'w') as script_file:
        for image_path in jpg_files:
            img_subpath = os.path.relpath(image_path, root_folder)
            print(f'Processing {img_subpath}')
            init_image = "/src/chebel/" + img_subpath
            payload = {
                "input": {
                    "mode": "controlnet",
                    "control_image": init_image,
                    "controlnet_type": "canny-edge",
                    "control_image_strength": 0.85,
                    "steps": 42,
                    "width": 1280,
                    "height": 1280,
                    "seed": 54320614,
                    "lora": "https://edenartlab-prod-data.s3.us-east-1.amazonaws.com/431ff8fb8edf1fcf8d1bc1ddcc2662479ced491c6b98784cdb4b0aa6d70cd09c.tar",
                    "lora_scale": 0.6,
                    "text_input": "oil paint, soft pastel color tones, women, dance, brush strokes",
                    "uc_text": "frame, text, watermark, saturated, low-quality, signature, padding, margins, white borders, padded border, moiré pattern, downsampling, aliasing, distorted, blurry, blur, jpeg artifacts, compression artifacts, poorly drawn, low-resolution, bad, grainy, error, bad-contrast"
                }
            }
            # Convert the dictionary to a JSON string
            json_payload = json.dumps(payload)
            command = f'curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" -H \'Content-Type: application/json\' "http://0.0.0.0:5000/predictions" -d \'{json_payload}\'\n'
            script_file.write(command)

if __name__ == '__main__':
    root_folder = '/data/xander/Projects/cog/eden-sd-pipelines/chebel'
    output_script = 'chebel_api_requests.sh'
    jpg_files = find_jpg_images(root_folder)
    generate_commands(jpg_files, output_script)
    print(f'API requests saved to {output_script}')
