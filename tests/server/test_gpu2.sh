curl -s -X POST \
  -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "generate", "text_input": "the world", "steps": 30, "width": 768, "height": 768}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Generate 0 request fired!!!"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "interpolate", "text_input": "lerp", "interpolation_seeds": "5|8|13", "interpolation_texts": "hello|world|cats and dogs", "seed": 3, "width": 640, "height": 512, "n_film": 1, "n_frames": 20, "steps": 15, "smooth": 1, "loop": 1, "stream": 1, "stream_every": 1}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "interpolate 0 request fired!!!"

curl -s -X POST \
  -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "generate", "text_input": "the world", "steps": 30, "width": 1024, "height": 1512}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Generate 1 request fired!!!"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "interpolate", "text_input": "lerp", "interpolation_seeds": "5|8|13", "interpolation_texts": "hello|world|cats and dogs", "seed": 3, "width": 512, "height": 640, "n_film": 1, "n_frames": 20, "steps": 15, "smooth": 1, "loop": 1, "stream": 1, "stream_every": 1}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "interpolate 1 request fired!!!"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "remix", "text_input": "Remix example", "init_image_data": "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp", "seed": 5, "width": 1024, "height": 768, "n_samples": 2}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Remix 0 crash request fired!!!"

curl -s -X POST \
  -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "generate", "text_input": "the world", "steps": 30, "width": 1024, "height": 1512}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Generate 1 request fired!!!"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "interpolate", "text_input": "lerp", "interpolation_seeds": "5|8|13", "interpolation_texts": "hello|world|cats and dogs", "seed": 3, "width": 640, "height": 578, "n_film": 1, "n_frames": 20, "steps": 15, "smooth": 1, "loop": 1, "stream": 1, "stream_every": 1}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "interpolate 2 request fired!!!"

curl -s -X POST \
  -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "generate", "text_input": "the world", "steps": 30, "width": 1512, "height": 1024}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Generate 2 request fired!!!"

curl -s -X POST \
  -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "generate", "text_input": "the world", "steps": 30, "width": 640, "height": 960}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Generate 0 request fired!!!"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "remix", "text_input": "Remix example", "init_image_data": "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp", "seed": 5, "width": 512, "height": 768, "n_samples": 2}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Remix 0 request fired!!!"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "interrogate", "text_input": "interrogate", "init_image_data": "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp"}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Interrogate 0 request fired!!!"

curl -s -X POST \
  -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "generate", "text_input": "the world", "steps": 30, "width": 640, "height": 640}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Generate 1 request fired!!!"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "interrogate", "text_input": "interrogate", "init_image_data": "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp"}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Interrogate 2 request fired!!!"

curl -s -X POST \
  -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "generate", "text_input": "the world", "steps": 30, "width": 1024, "height": 1512}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Generate 3 request fired!!!"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "remix", "text_input": "Remix example", "init_image_data": "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp", "seed": 5, "width": 1512, "height": 1024, "n_samples": 1}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Remix 3 request fired!!!"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "interrogate", "text_input": "interrogate", "init_image_data": "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp"}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Interrogate 3 request fired!!!"


curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "remix", "text_input": "Remix example", "init_image_data": "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp", "seed": 5, "width": 1024, "height": 512, "n_samples": 2}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Remix 4 request fired!!!"

curl -s -X POST \
  -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "generate", "text_input": "the world", "steps": 30, "width": 1768, "height": 960}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Generate request fired!!!"

curl -s -X POST \
  -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "generate", "text_input": "the world", "steps": 30, "width": 1256, "height": 1256, "n_samples": 1}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Generate 2x request fired!!!"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "remix", "text_input": "Remix example", "init_image_data": "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp", "seed": 5, "width": 1512, "height": 1024, "n_samples": 2}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Remix request fired!!!"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "interrogate", "text_input": "interrogate", "init_image_data": "https://generations.krea.ai/images/3cd0b8a8-34e5-4647-9217-1dc03a886b6a.webp"}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "Interrogate request fired!!!"

curl -s -X POST -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "interpolate", "text_input": "lerp", "interpolation_seeds": "5|8|13", "interpolation_texts": "hello|world|cats and dogs", "seed": 3, "width": 640, "height": 640, "n_film": 1, "n_frames": 20, "steps": 15, "smooth": 1, "loop": 1, "stream": 1, "stream_every": 1}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "interpolate request fired!!!"

curl -s -X POST \
  -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "real2real", "text_input": "real2real", "interpolation_seeds": "5|8", "interpolation_init_images":"https://generations.krea.ai/images/928271c8-5a8e-4861-bd57-d1398e8d9e7a.webp|https://generations.krea.ai/images/865142e2-8963-47fb-bbe9-fbe260271e00.webp", "seed": 4, "width": 640, "height": 960, "steps": 30, "n_film": 1, "n_frames": 20, "smooth": 1, "loop": 1}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"

echo "real2real request fired!!!"

