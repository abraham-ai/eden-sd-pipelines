curl -s -X POST \
  -d '{"version": "fef11678ae5dc4cc024f8d3b4860c65124118a015fecac10fb2f0d652c7538d4", "input": {"mode": "real2real", "text_input": "real2real", "interpolation_seeds": "5|8", "interpolation_init_images":"https://generations.krea.ai/images/928271c8-5a8e-4861-bd57-d1398e8d9e7a.webp|https://generations.krea.ai/images/865142e2-8963-47fb-bbe9-fbe260271e00.webp", "seed": 4, "width": 512, "height": 512, "steps": 30, "n_film": 1, "n_frames": 20, "smooth": 1, "loop": 1}}' \
  -H "Authorization: Token $REPLICATE_API_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://0.0.0.0:5000/predictions"