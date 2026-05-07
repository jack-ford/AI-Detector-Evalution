import os
import requests

def split_data_into_chunks(data: str, chunk_size: int):
    """Splits the input data into chunks of specified size."""
    words = data.split(" ")
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks    

def read_all_files_in_directory(directory: str):
    """Reads all text files in the specified directory and returns their contents as a dictionary with filename as key and file content as value."""
    file_contents = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                content = content.replace('\n', '').replace('\t', '').replace('\r', '')
                file_contents[filename] = content
    return file_contents

def run_ai_detector(text: str, apiKey: str):
    """Sends text to the Winston AI detector and returns the response payload."""

    if (not text):
        return {
            "error": 0,
            "message": "Please provide text to analyze."
        }

    response = requests.post(
        "https://api.sapling.ai/api/v1/aidetect",
        headers={
            "Content-Type": "application/json",
        },
        json={
            "key": apiKey,
            "text": text,
            "sent_scores": False
        },
    )

    if 200 <= response.status_code < 300:
        return response.json()

    return {
        "error": response.status_code,
        "message": response.text,
    }

def save_data(save_dir, text_id, text, ai_score, chunk_num):
    """Saves the analyzed data to a file in the stat_results directory."""
    result = {
        "chunk_id": get_chunk_id(text_id, chunk_num),
        "word_count": len(text.split(" ")),
        "text": text,
        "ai_score": ai_score
    }

    filename = f"{result['chunk_id']}.txt"
    with open(os.path.join(save_dir, filename), 'w', encoding='utf-8') as result_file:
        result_file.write(str(result))

def get_chunk_id(text_id, chunk_num):
    """Generates a unique chunk ID based on the text ID and chunk number."""
    return f"{text_id}-{chunk_num + 1}"

def main():
    data = read_all_files_in_directory("stat_data")
    for filename, content in data.items():
        chunks = split_data_into_chunks(content, 100) 
        for i, chunk in enumerate(chunks[:6]):
            chunk_id = get_chunk_id(filename.replace(".txt", ""), i)
            print(f"Analyzing {chunk_id}...")

            # Check if result already exists for this chunk
            result_path = os.path.join("stat_results", f"{chunk_id}.txt")
            if os.path.exists(result_path):
                print(f"Result already exists for {chunk_id}, skipping analysis.")
                continue
                
            key = os.getenv("API_KEY")

            # Run AI detector and handle response
            response = run_ai_detector(chunk, key)

            if "error" in response:
                if response["error"] != 0:
                    print(f"Error analyzing {chunk_id}: {response['message']}")
                continue

            ai_score = response["score"]

            save_data("detection_results", filename.replace(".txt", ""), chunk, ai_score, i)

            print(f"Finished analyzing {chunk_id}. AI Score: {ai_score:.4f}\n")

if __name__ == "__main__":
    main()