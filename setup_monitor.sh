#!/bin/bash
# Setup training monitor with cron job

# Get Gmail app password (user must provide this)
read -sp "Enter Gmail App Password: " APP_PASSWORD
echo

# Export to environment
export GMAIL_APP_PASSWORD="$APP_PASSWORD"

# Add to crontab to run every 30 minutes
CRON_JOB="*/30 * * * * export GMAIL_APP_PASSWORD='$APP_PASSWORD' && /usr/bin/python3 /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/monitor_training.py >> /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/log/monitor.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "monitor_training.py"; then
    echo "⚠️  Cron job already exists"
    echo "Listing current cron jobs:"
    crontab -l
else
    # Add new cron job
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "✅ Cron job added successfully"
    echo "📋 New cron job will run every 30 minutes"
    echo
    echo "Current cron jobs:"
    crontab -l
fi

echo
echo "📊 To manually test the monitor right now:"
echo "   export GMAIL_APP_PASSWORD='$APP_PASSWORD'"
echo "   python3 /home/swatson/work/MachineLearning/kaggle/vesuvius-challenge-surface-detection/monitor_training.py"
