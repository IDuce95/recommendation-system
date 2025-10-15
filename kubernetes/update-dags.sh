#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DAGS_DIR="$PROJECT_ROOT/app/airflow_dags/dags"
CONFIGMAP_FILE="$SCRIPT_DIR/airflow-dags-configmap.yaml"

echo "Updating Airflow DAGs ConfigMap..."

cat > "$CONFIGMAP_FILE" << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: airflow-dags
  namespace: recommendation-system
data:
EOF

for dag_file in "$DAGS_DIR"/*.py; do
    if [ -f "$dag_file" ]; then
        filename=$(basename "$dag_file")
        echo "  $filename: |" >> "$CONFIGMAP_FILE"
        
        while IFS= read -r line; do
            echo "    $line" >> "$CONFIGMAP_FILE"
        done < "$dag_file"
    fi
done

echo "DAGs ConfigMap updated successfully!"
echo "Apply with: kubectl apply -f $CONFIGMAP_FILE"
