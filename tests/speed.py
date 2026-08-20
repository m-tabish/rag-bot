import time

import ollama

# We use a massive text prompt to force the model to output a long paragraph response
test_prompt = "Write a highly detailed, 5-paragraph historical breakdown explaining the evolution of computer graphics processors and microarchitectures from 1990 to 2026."

print("Initializing connection to Llama 3.2 3B via Ollama...")
print("Analyzing generation speed (processing tokens)... Please wait...\n")

start_time = time.time()

try:
    # Query your local active model
    response = ollama.chat(
        model='qwen2.5-coder:1.5b',
        messages=[{'role': 'user', 'content': test_prompt}]
    )

    end_time = time.time()
    total_duration = end_time - start_time

    # Ollama returns exact metadata stats inside its response structure
    # Number of generated tokens
    eval_count = response.get('eval_count', 0)
    # Number of prompt tokens parsed
    prompt_count = response.get('prompt_count', 0)

    if eval_count > 0:
        tokens_per_second = eval_count / total_duration
        print("==============================")
        print("   HARDWARE SPEED BENCHMARK   ")
        print("==============================")
        print(f"• Prompt Processing:  {prompt_count} tokens loaded")
        print(f"• Tokens Generated:   {eval_count} tokens created")
        print(f"• Total Stream Time:  {total_duration:.2f} seconds")
        print(f"• Actual Speed:       {tokens_per_second:.2f} tokens/sec")
        print("==============================")

        # Performance Evaluation check for your laptop setup
        if tokens_per_second >= 30:
            print("STATUS: Excellent! Your RTX 3050 is handling 100% of the workload.")
        elif tokens_per_second >= 15:
            print(
                "STATUS: Moderate performance. Background processes might be sharing VRAM.")
        else:
            print(
                "STATUS: Warning. Low token speed detected. The model is likely offloading to your CPU.")

    else:
        print("Error: Could not retrieve native metrics from Ollama. Make sure the 'ollama run llama3.2' background process is active.")

except Exception as e:
    print(f"An execution error occurred: {e}")
    print("Please make sure Ollama is open and running in your taskbar tray.")
