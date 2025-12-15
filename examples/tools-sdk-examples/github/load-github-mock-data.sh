#!/bin/bash
# Load comprehensive GitHub mock data into the mock tools server
#
# Usage: ./load-github-mock-data.sh [base_url]
# Default base_url: http://localhost:3000

set -e

BASE_URL="${1:-http://localhost:3000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_FILE="$SCRIPT_DIR/github-mock-data.json"

if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Mock data file not found at $DATA_FILE"
    exit 1
fi

echo "Loading GitHub mock data from $DATA_FILE"
echo "Target server: $BASE_URL"
echo ""

# Function to load mock response for a tool
load_mock_response() {
    local tool_name="$1"
    local response="$2"

    echo "Loading mock response for: $tool_name"

    curl -s -X POST "$BASE_URL/mock/update-response" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$tool_name\", \"mockResponse\": $response}" > /dev/null

    if [ $? -eq 0 ]; then
        echo "  Done"
    else
        echo "  Failed!"
    fi
}

# Function to add a fixture
add_fixture() {
    local tool_name="$1"
    local match="$2"
    local response="$3"

    echo "Adding fixture for: $tool_name (match: $match)"

    curl -s -X POST "$BASE_URL/mock/add-fixture" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$tool_name\", \"match\": $match, \"response\": $response}" > /dev/null

    if [ $? -eq 0 ]; then
        echo "  Done"
    else
        echo "  Failed!"
    fi
}

echo "=========================================="
echo "Loading default mock responses..."
echo "=========================================="

# Load default mock responses using jq
for tool in $(jq -r '.mockResponses | keys[]' "$DATA_FILE"); do
    response=$(jq -c ".mockResponses.\"$tool\".defaultResponse" "$DATA_FILE")
    if [ "$response" != "null" ]; then
        load_mock_response "$tool" "$response"
    fi
done

echo ""
echo "=========================================="
echo "Loading fixtures..."
echo "=========================================="

# Load fixtures for each tool
for tool in $(jq -r '.fixtures | keys[]' "$DATA_FILE"); do
    fixture_count=$(jq -r ".fixtures.\"$tool\" | length" "$DATA_FILE")

    for i in $(seq 0 $((fixture_count - 1))); do
        match=$(jq -c ".fixtures.\"$tool\"[$i].match" "$DATA_FILE")
        response=$(jq -c ".fixtures.\"$tool\"[$i].response" "$DATA_FILE")
        add_fixture "$tool" "$match" "$response"
    done
done

echo ""
echo "=========================================="
echo "Mock data loaded successfully!"
echo "=========================================="
echo ""
echo "Test commands:"
echo ""
echo "# Get file contents"
echo "curl -X POST $BASE_URL/mock/execute -H 'Content-Type: application/json' \\"
echo "  -d '{\"name\": \"get_file_contents\", \"arguments\": {\"owner\": \"acme-corp\", \"repo\": \"web-platform\", \"path\": \"README.md\"}}'"
echo ""
echo "# List issues"
echo "curl -X POST $BASE_URL/mock/execute -H 'Content-Type: application/json' \\"
echo "  -d '{\"name\": \"list_issues\", \"arguments\": {\"owner\": \"acme-corp\", \"repo\": \"web-platform\"}}'"
echo ""
echo "# List pull requests"
echo "curl -X POST $BASE_URL/mock/execute -H 'Content-Type: application/json' \\"
echo "  -d '{\"name\": \"list_pull_requests\", \"arguments\": {\"owner\": \"acme-corp\", \"repo\": \"web-platform\"}}'"
