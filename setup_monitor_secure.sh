#!/bin/bash
# Alternative setup using environment file (more secure)

ENV_FILE="/home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/.monitor_env"

if [ -f "$ENV_FILE" ]; then
    echo "⚠️  Environment file already exists: $ENV_FILE"
    echo "To update, delete the file and run this script again"
    exit 1
fi

# Get Gmail app password
read -sp "Enter Gmail App Password: " APP_PASSWORD
echo

# Create environment file with restricted permissions
cat > "$ENV_FILE" << EOF
export GMAIL_APP_PASSWORD="$APP_PASSWORD"
EOF

chmod 600 "$ENV_FILE"
echo "✅ Environment file created with restricted permissions (600)"

# Add to crontab
CRON_JOB="*/30 * * * * source /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/.monitor_env && /usr/bin/python3 /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/monitor_training.py >> /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/log/monitor.log 2>&1"

if crontab -l 2>/dev/null | grep -q "monitor_training.py"; then
    echo "⚠️  Cron job already exists"
    crontab -l
else
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "✅ Cron job added successfully"
    echo "📋 Monitor will run every 30 minutes"
    echo
    crontab -l
fi

echo
echo "🧹 To remove the cron job later:"
echo "   crontab -e  (and delete the monitor_training line)"
echo "   rm $ENV_FILE"
