curl -s -X POST \
  -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "generate", "text_input": "the world"}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

curl -s -X POST \
  -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "real2real", "text_input": "real2real", "interpolation_seeds": "5|8", "interpolation_init_images":"https://generations.krea.ai/images/928271c8-5a8e-4861-bd57-d1398e8d9e7a.webp|https://generations.krea.ai/images/865142e2-8963-47fb-bbe9-fbe260271e00.webp", "seed": 4, "width": 512, "height": 512, "steps": 15, "n_film": 1, "n_frames": 20, "smooth": 1, "loop": 1}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"
  
curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "generate", "text_input": "A picture of <person1> wearing a French beret", "seed": 15, "width": 768, "height": 768, "steps": 100, "lora": "https://minio.aws.abraham.fun/creations-stg/c91f7a73aeaeca746f5137fa03a9b3993ba165322c0398e12ceae2f85f2f3c5a.txt", "lora_scale": 1.0}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "interpolate", "text_input": "lerp", "interpolation_seeds": "5|8|13", "interpolation_texts": "hello|world|cats and dogs", "seed": 3, "width": 512, "height": 512, "n_film": 1, "n_frames": 20, "steps": 15, "smooth": 1, "loop": 1, "stream": 1, "stream_every": 1}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "remix", "text_input": "Remix example", "init_image_data": "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp", "seed": 5, "width": 512, "height": 512, "n_samples": 4}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "interrogate", "text_input": "interrogate", "init_image_data": "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp"}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

