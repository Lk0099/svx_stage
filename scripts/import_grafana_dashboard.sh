#!/usr/bin/env bash
# =============================================================================
# SmartVision-X — Import Grafana Dashboard into Existing DGX Grafana
# =============================================================================
# The DGX already has grafana running. This script imports the SmartVision-X
# dashboard into it via the Grafana HTTP API.
#
# Usage:
#   bash scripts/import_grafana_dashboard.sh [grafana-url] [user] [password]
#
# Defaults:
#   grafana-url = http://localhost:3000
#   user        = admin
#   password    = admin
# =============================================================================

GRAFANA_URL="${1:-http://localhost:3000}"
GRAFANA_USER="${2:-admin}"
GRAFANA_PASS="${3:-admin}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DASHBOARD_FILE="$PROJECT_ROOT/infra/grafana/dashboards/smartvision_main.json"

echo "=== Importing SmartVision-X Dashboard to Grafana ==="
echo "    URL:  $GRAFANA_URL"
echo "    User: $GRAFANA_USER"
echo ""

# ── Step 1: Check Grafana is reachable ─────────────────────────────────────
echo "[1/3] Checking Grafana health..."
if ! curl -sf "$GRAFANA_URL/api/health" -u "$GRAFANA_USER:$GRAFANA_PASS" > /dev/null; then
    echo "  ❌ Cannot reach Grafana at $GRAFANA_URL"
    echo "     Check the correct port with: docker inspect grafana"
    exit 1
fi
echo "  ✅ Grafana reachable"

# ── Step 2: Ensure Prometheus datasource exists ────────────────────────────
echo "[2/3] Verifying Prometheus datasource..."
DS_CHECK=$(curl -sf "$GRAFANA_URL/api/datasources/name/Prometheus" \
    -u "$GRAFANA_USER:$GRAFANA_PASS" 2>/dev/null)

if [ -z "$DS_CHECK" ]; then
    echo "  → Creating Prometheus datasource..."
    curl -sf -X POST "$GRAFANA_URL/api/datasources" \
        -u "$GRAFANA_USER:$GRAFANA_PASS" \
        -H "Content-Type: application/json" \
        -d '{
          "name":      "Prometheus",
          "type":      "prometheus",
          "url":       "http://prometheus:9090",
          "access":    "proxy",
          "isDefault": true
        }' > /dev/null
    echo "  ✅ Prometheus datasource created"
else
    echo "  ✅ Prometheus datasource already exists"
fi

# ── Step 3: Import dashboard ───────────────────────────────────────────────
echo "[3/3] Importing SmartVision-X dashboard..."

if [ ! -f "$DASHBOARD_FILE" ]; then
    echo "  ❌ Dashboard file not found: $DASHBOARD_FILE"
    exit 1
fi

# Wrap dashboard JSON for the import API
IMPORT_PAYLOAD=$(python3 -c "
import json, sys
with open('$DASHBOARD_FILE') as f:
    dash = json.load(f)
# Reset id so Grafana assigns a new one
dash['id'] = None
payload = {
    'dashboard': dash,
    'overwrite': True,
    'folderId':  0,
    'inputs':    [{'name': 'DS_PROMETHEUS', 'type': 'datasource',
                   'pluginId': 'prometheus', 'value': 'Prometheus'}]
}
print(json.dumps(payload))
")

RESULT=$(curl -sf -X POST "$GRAFANA_URL/api/dashboards/import" \
    -u "$GRAFANA_USER:$GRAFANA_PASS" \
    -H "Content-Type: application/json" \
    -d "$IMPORT_PAYLOAD" 2>&1)

if echo "$RESULT" | grep -q '"status":"success"'; then
    DASH_URL=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('importedUrl',''))" 2>/dev/null)
    echo "  ✅ Dashboard imported successfully"
    echo ""
    echo "=== Done ==="
    echo "  Open: $GRAFANA_URL$DASH_URL"
else
    echo "  ❌ Import failed: $RESULT"
    echo ""
    echo "  Manual import:"
    echo "    1. Open $GRAFANA_URL"
    echo "    2. Dashboards → Import → Upload JSON file"
    echo "    3. Select: $DASHBOARD_FILE"
    exit 1
fi
