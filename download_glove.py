import requests
import os

url = "https://huggingface.co/datasets/Jay-Mayekar/glove-vectors/resolve/main/glove.6B.50d.txt"
output_path = "glove.6B.50d.txt"
target_lines = 50000

print(f"Starting streamed download of GloVe vectors (retaining top {target_lines} words)...")

response = requests.get(url, stream=True)
if response.status_code != 200:
    print(f"Error: Received status code {response.status_code} from server.")
    exit(1)

lines_written = 0
buffer = ""

with open(output_path, "w", encoding="utf-8") as out_file:
    # Read response stream in chunks
    for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
        if not chunk:
            continue
        
        buffer += chunk
        # Split by newline
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            out_file.write(line + "\n")
            lines_written += 1
            
            if lines_written % 10000 == 0:
                print(f"Downloaded and wrote {lines_written} words...")
                
            if lines_written >= target_lines:
                break
        
        if lines_written >= target_lines:
            break

print(f"SUCCESS: Wrote {lines_written} words to {output_path} (File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB)")
