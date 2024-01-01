#!/bin/bash

# Check if JSON string is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <json_string>"
    exit 1
fi

# Remove whitespaces from input JSON string
json_string_no_whitespace=$(echo "$1" | tr -d '[:space:]')

# Export each key-value pair as an environment variable
while IFS=':' read -r key value; do
    export "$key"="${value//\"/}"
done < <(echo "$json_string_no_whitespace" | jq -r "to_entries|map(\"\(.key):\(.value)\")|.[]")
