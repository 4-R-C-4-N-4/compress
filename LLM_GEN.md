export LLM_BASE_URL=https://openrouter.ai/api/v1
export LLM_API_KEY=<your openrouter key>
export LLM_MODEL=nvidia/nemotron-3-super-120b-a12b:free

  .venv/bin/python3 scripts/llm_seed.py \
      --type "JSON API responses with nested objects, arrays, and mixed types" \
      --seed-id 3 --name json_llm \
      --samples 20 --order 4 \
      --output seeds/json_llm.seedmodel \
      --save-corpus /tmp/json_llm_corpus.txt
