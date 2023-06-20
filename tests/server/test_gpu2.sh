curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "interpolate", "text_input": "lerp", "interpolation_seeds": "5|8|13", "interpolation_texts": "hello|world|cats and dogs", "seed": 3, "width": 512, "height": 512, "n_film": 1, "n_frames": 20, "steps": 15, "smooth": 1, "loop": 1, "stream": 1, "stream_every": 1}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "interpolate 0 request fired!!!"

curl -s -X POST \
  -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "generate", "text_input": "the world", "steps": 30, "width": 1024, "height": 1024}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Generate 0 request fired!!!"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "interpolate", "text_input": "lerp", "interpolation_seeds": "5|8|13", "interpolation_texts": "hello|world|cats and dogs", "seed": 3, "width": 512, "height": 512, "n_film": 1, "n_frames": 20, "steps": 15, "smooth": 1, "loop": 1, "stream": 1, "stream_every": 1}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "interpolate 1 request fired!!!"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "remix", "text_input": "Remix example", "init_image_data": "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp", "seed": 5, "width": 512, "height": 768, "n_samples": 2}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Remix 0 crash request fired!!!"

curl -s -X POST \
  -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "generate", "text_input": "the world", "steps": 30, "width": 1024, "height": 1024}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Generate 1 request fired!!!"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "interpolate", "text_input": "lerp", "interpolation_seeds": "5|8|13", "interpolation_texts": "hello|world|cats and dogs", "seed": 3, "width": 512, "height": 512, "n_film": 1, "n_frames": 20, "steps": 15, "smooth": 1, "loop": 1, "stream": 1, "stream_every": 1}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "interpolate 2 request fired!!!"

curl -s -X POST \
  -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "generate", "text_input": "the world", "steps": 30, "width": 1024, "height": 1024}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Generate 2 request fired!!!"