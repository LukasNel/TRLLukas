import json
import os
from datetime import datetime

def format_response(response):
    """Format the response text by handling think blocks and code blocks."""
    # Split response into think and actual response parts
    parts = response.split('</think>')
    
    formatted_parts = []
    for part in parts:
        if '<think>' in part:
            # Extract think block
            think_content = part.split('<think>')[1].strip()
            formatted_parts.append(f"**Thinking Process:**\n{think_content}\n")
        else:
            # Handle the actual response
            if part.strip():
                # Handle code blocks
                if '```' in part:
                    formatted_parts.append(part.strip())
                else:
                    formatted_parts.append(f"**Response:**\n{part.strip()}\n")
    
    return '\n'.join(formatted_parts)

def convert_logs_to_markdown(json_file):
    """Convert test results JSON to markdown format."""
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create markdown content
    md_content = []
    
    # Add header
    md_content.append(f"# Test Results Report")
    md_content.append(f"\n**Timestamp:** {data['timestamp']}")
    md_content.append(f"**Test Type:** {data['test_type']}")
    md_content.append(f"**Model:** {data['model']}\n")
    
    # Add test results
    md_content.append("## Test Results\n")
    
    for i, result in enumerate(data['results'], 1):
        md_content.append(f"### Test Case {i}")
        md_content.append(f"\n**Query:** {result['query']}\n")
        
        # Format the response
        formatted_response = format_response(result['response'])
        md_content.append(formatted_response)
        
        # Add success status
        status = "✅ Success" if result['success'] else "❌ Failed"
        md_content.append(f"\n**Status:** {status}\n")
        md_content.append("---\n")
    
    # Generate output filename
    output_file = json_file.replace(".json", ".md")
    
    # Write to markdown file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    
    return output_file

if __name__ == "__main__":
    # Get the most recent test results file
    log_dir = "logs"
    json_files = [f for f in os.listdir(log_dir) if f.startswith("test_results_") and f.endswith(".json")]
    if not json_files:
        print("No test results files found in logs directory")
        exit(1)
    for file in json_files:
        print(file)
        json_path = os.path.join(log_dir, file)
        # Convert to markdown
        output_file = convert_logs_to_markdown(json_path)
        print(f"Converted {json_path} to {output_file}") 
    