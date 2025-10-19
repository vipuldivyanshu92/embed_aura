# Install vLLM from pip:
pip install vllm

# Load and run the model:
vllm serve "unsloth/Qwen2.5-3B-Instruct"

# Call the server using curl:
curl -X POST "http://localhost:8000/v1/chat/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "unsloth/Qwen2.5-3B-Instruct",
		"messages": [
			{
				"role": "user",
				"content": "What is the capital of France?"
			}
		]
	}'