export OPENAI_API_KEY="sk-..."

python run_batch_cot.py \
    --llm openaillm \
    --input inputs/prompts.csv \
    --threshold 75 \
    --max-iterations 3 \
    --bias-top-k 25 \
    --cultural-top-k 25 \
    --cache-file embeddings.pkl \
    --openai-model gpt-4o \

# python run_batch_cot.py \
#     --llm openaillm \
#     --input inputs/prompts.csv \
#     --threshold 75 \
#     --max-iterations 3 \
#     --bias-top-k 25 \
#     --cultural-top-k 25 \
#     --cache-file embeddings.pkl \
#     --openai-model gpt-4o \
#     --openai-embed-model text-embedding-3-small \
#     --openai-temperature 0.7 \
#     --openai-max-tokens 1024