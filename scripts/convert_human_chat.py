#!/usr/bin/env python3
"""
Script to convert human_chat.txt to takeTurnConv.json format and append it.
"""

import json
import re
from pathlib import Path

def parse_human_chat(input_file):
    """
    Parse human_chat.txt and convert to takeTurnConv.json format.
    Each conversation turn (Human 1 -> Human 2) becomes one JSON line.
    """
    conversations = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_user_msg = None
    current_assistant_msg = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Match "Human 1: message" or "Human 2: message"
        match = re.match(r'Human (\d+):\s*(.+)', line)
        if not match:
            continue
        
        human_num = match.group(1)
        message = match.group(2).strip()
        
        if human_num == '1':
            # If we have a pending conversation turn, save it
            if current_user_msg is not None and current_assistant_msg is not None:
                conversations.append({
                    "message": [
                        {"role": "user", "content": current_user_msg},
                        {"role": "assistant", "content": current_assistant_msg}
                    ]
                })
                current_user_msg = None
                current_assistant_msg = None
            
            # Start new user message
            current_user_msg = message
        
        elif human_num == '2':
            # This is the assistant response
            if current_user_msg is not None:
                current_assistant_msg = message
            else:
                # If there's no user message, skip this (orphaned assistant message)
                continue
    
    # Don't forget the last conversation if it ends with Human 2
    if current_user_msg is not None and current_assistant_msg is not None:
        conversations.append({
            "message": [
                {"role": "user", "content": current_user_msg},
                {"role": "assistant", "content": current_assistant_msg}
            ]
        })
    
    return conversations

def append_to_jsonl(output_file, conversations):
    """
    Append conversations to the JSONL file (one JSON object per line).
    """
    with open(output_file, 'a', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

def main():
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    
    input_file = data_dir / "human_chat.txt"
    output_file = data_dir / "takeTurnConv.json"
    
    print(f"Reading from: {input_file}")
    print(f"Appending to: {output_file}")
    
    # Parse the human chat file
    conversations = parse_human_chat(input_file)
    print(f"Parsed {len(conversations)} conversation turns")
    
    # Append to the output file
    append_to_jsonl(output_file, conversations)
    print(f"Successfully appended {len(conversations)} conversations to {output_file}")

if __name__ == "__main__":
    main()

